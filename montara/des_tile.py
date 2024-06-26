import os
import shutil
from collections import OrderedDict
import subprocess
import logging

import fitsio
import galsim
import numpy as np
import astropy.io.fits as pyfits
from galsim.config.output import OutputBuilder
from eastlake.fits import writeMulti
from eastlake.rejectlist import RejectList
from .hexalattice import create_hex_grid

from .utils import safe_mkdir, get_truth_from_image_file
from eastlake.des_files import Tile, read_pizza_cutter_yaml
from eastlake.utils import unpack_fits_file_if_needed

MODES = ["single-epoch", "coadd"]  # beast too?


class ChipNoiseBuilder(galsim.config.NoiseBuilder):
    """Build noise for a given image using input images. Optionally, a
    background is added as well.

    NOTE: This class is used internally by the DESTile output type below.

    Example
    -------
    You can use this class directly in a galsim config file like this

    ```YAML
    modules:
      - montara

    image:
      noise:
        type: ChipNoise

        # this is the path to the original image whose noise field is to be
        # used for the modeling
        orig_image_path: '/path/to/image/with/data'

        # specify how to use the original image noise field here
        # see the description of the values below
        noise_mode: 'skysigma'

        # optional value by which the overall variance in the image is scaled
        noise_fac: 1.0

        # optional path to a background image to add to the output image
        # you can specify the hdu via bkg_hdu
        # the background is assumed to be in HDU 1 of the FITS file if not
        # specified
        bkg_filename: '/path/to/image/with/background'
        bkg_hdu: 1
    ```

    The noise is pulled from the original image in various ways controlled by
    the parameter 'noise_mode'. Valid values are

        skysigma : use the SKYSIGMA value from the image header - implies
            constant noise and only valid for coadd images
        inverse_median_weight : use the inverse of the median of the
            weight map where the bit mask is zero - implies constant noise
        median_inverse_weight : use the median of the inverse of the
            weight map where the bit mask is zero - implies constant noise
        from_weight : use the weight map directly - implies varying noise.
            where the image is masked, the weight map may imply crazy noise
            values. So use the median in those areas.
    """
    req = {"orig_image_path": str, "noise_mode": str}
    opt = {"bkg_filename": str, "noise_fac": float, "bkg_hdu": int, "_zero_bkg": bool}

    def addNoise(
            self, config, base, im, rng, current_var, draw_method, logger):
        params, safe = galsim.config.GetAllParams(
            config, base, req=self.req, opt=self.opt)

        var = self.getNoiseVariance(config, base)
        if np.isscalar(var):
            noise = galsim.noise.GaussianNoise(rng, np.sqrt(var))
        else:
            noise = galsim.noise.VariableGaussianNoise(rng, var)
        im.addNoise(noise)

        if "bkg_filename" in params:
            if params["_zero_bkg"]:
                # for w/e reason fitsio cannot overwrite file in place so we
                # unpack, zero and then repack
                unpacked_fname, _ = unpack_fits_file_if_needed(params["bkg_filename"], "sci")
                with fitsio.FITS(unpacked_fname, "rw") as fits:
                    _im = fits["sci"].read()
                    _im[:, :] = 0.0
                    fits["sci"].write(_im)
                subprocess.check_output(
                    "rm -f %s && fpack %s" % (
                        os.path.basename(params["bkg_filename"]),
                        os.path.basename(unpacked_fname),
                    ),
                    shell=True,
                    cwd=os.path.dirname(unpacked_fname),
                )

            bkg_image = self.getBkg(config, base)
            if params["_zero_bkg"]:
                assert bkg_image.array.mean() == 0, "bkg not zero when it should be!"

            logger.debug("adding bkg with mean %.2e from file %s" % (
                (bkg_image.array).mean(), params["bkg_filename"]))
            im += bkg_image
        else:
            logger.debug("no bkg added")

    def getBkg(self, config, base):
        params, safe = galsim.config.GetAllParams(
            config, base, req=self.req, opt=self.opt)
        bkg_image = galsim.fits.read(
            params["bkg_filename"], hdu=params.get("bkg_hdu", 1))
        return bkg_image

    def getNoiseVariance(self, config, base, full=None):
        params, safe = galsim.config.GetAllParams(
            config, base, req=self.req, opt=self.opt)
        orig_im_fits = pyfits.open(params["orig_image_path"])
        if params["noise_mode"] == "skysigma":
            skysigma = orig_im_fits[1].header["SKYSIGMA"]
            var = skysigma**2
        elif params["noise_mode"] == "inverse_median_weight":
            var = 1. / np.median(orig_im_fits[3].data[orig_im_fits[2].data == 0])
        elif params["noise_mode"] == "median_inverse_weight":
            var = np.median(
                1. / (orig_im_fits[3].data[orig_im_fits[2].data == 0]))
        elif params["noise_mode"] == "from_weight":
            inv_var = orig_im_fits[3].data
            unmasked = orig_im_fits[2].data == 0
            var = np.zeros_like(inv_var)
            var[unmasked] = 1. / inv_var[unmasked]
            # weight map values in masked areas may be crazy
            # so assign the median here
            med = np.median(var[unmasked])
            var[~unmasked] = med
        else:
            raise ValueError(
                "%s not a valid noise_mode" % params["noise_mode"])
        if "noise_fac" in config:
            if config["noise_fac"] is not None:
                var *= config["noise_fac"]
        return var


def _set_catalog_sampler_rng_num(cfg, rng_num):
    "recursive function to set rng_num for catalog sampler"
    if isinstance(cfg, dict):
        for k in list(cfg.keys()):
            cfg[k] = _set_catalog_sampler_rng_num(cfg[k], rng_num)

        if cfg.get("type", None) == "catalog_sampler_value":
            cfg["rng_num"] = rng_num

    elif isinstance(cfg, list):
        for i in range(len(cfg)):
            cfg[i] = _set_catalog_sampler_rng_num(cfg[i], rng_num)

    return cfg


class DESTileBuilder(OutputBuilder):
    """Implements the DESTile custom output type.

    This type models a full focal plane including multiple CCD images using
    coherent patterns for things like the PSF and sky level.

    The wcs is taken from a reference wcs (e.g. from a set of Fits files),
    but can reset the pointing position to a different location on the sky.

    Example
    -------
    You can use this class directly in a galsim config file like this

    ```YAML
    # TODO - write example YAML
    ```

    """

    def setup(self, config, base, file_num, logger):
        logger.debug("Start DESTileBuilder setup file_num=%d" % file_num)

        # We'll be editing things in the config file by hand here, which
        # circumvents GalSim's
        # detection of when it is safe to reuse a current item.  The easiest
        # fix is to just
        # wipe out all the saved current items now.
        galsim.config.RemoveCurrent(base)

        # Make sure tile_num, band_num, exp_num, chip_num are considered
        # valid index_keys

        for obj_type in ["gal", "star"]:
            if obj_type in base:
                if "mag" in base[obj_type]:
                    if "flux" in base[obj_type]:
                        raise ValueError(
                            "both flux and mag found for object type %s" % obj_type)
                    mag = base[obj_type].pop("mag")
                    base[obj_type]["flux"] = {"type": "Eval",
                                              "str": "10**( 0.4 * (mag_zp - mag))",
                                              "fmag": mag}

        if "band_num" not in galsim.config.process.valid_index_keys:
            galsim.config.valid_index_keys += [
                "band_num", "exp_num", "chip_num"]
            galsim.config.eval_base_variables += [
                "band_num", "exp_num", "chip_num",
                "tile_start_obj_num", "nfiles", "tilename", "band",
                "file_path", "imsim_data", "desrun", "object_type_list",
                "is_rejectlisted", "coadd_wcs", "ccdnum",
            ]

        # Now, if we haven't already, we need to read in some things which
        # determine what images to simulate
        tilename = config["tilename"]
        bands = config["bands"]

        # get the simulation mode and check that it is valid
        mode = config.get("mode", "single-epoch")
        assert mode in MODES

        # Do some setup for the tile
        # Save all this to _tile_setup key of the config dict
        # to avoid doing it more than once.
        # Also build file lists that will be saved to the output directory
        # we'll need the path to the original DES data, and the run
        if "imsim_data" in config:
            imsim_data = galsim.config.ParseValue(config, "imsim_data", base, str)[0]
        else:
            imsim_data = os.environ["IMSIM_DATA"]
        base["imsim_data"] = imsim_data
        desrun = config["desrun"]
        base["desrun"] = desrun

        # Now deal with where the output files go
        # Use the dir entry in the config file as a base directory
        # for all tiles. Then use the path to the original image
        # as a template for the simulated one.
        if "base_dir" not in galsim.config.eval_base_variables:
            galsim.config.eval_base_variables += ["base_dir"]
        if "base_dir" not in base:
            if "dir" in config:
                d = galsim.config.ParseValue(config, "dir", base, str)[0]
            else:
                d = "."
            base["base_dir"] = d

        if "_tile_setup" not in config:

            if "n_se_test" in config:
                n_se_test = galsim.config.ParseValue(config, "n_se_test", base, int)[0]
                logger.warning(
                    "Using only %d images for tile %s!", n_se_test, tilename,
                )
            else:
                n_se_test = None

            # The Tile class from .tile_setup collects a load
            # of juicy information for each tile
            tile_info = Tile.from_tilename(
                tilename, bands=bands, desrun=desrun, n_se_test=n_se_test
            )

            # Here we just need to set a few more things
            # like the name of the output files
            image_paths_from_imsim_data = [
                os.path.relpath(f, imsim_data)
                for f in tile_info["image_files"]]
            file_names_with_fz = [
                os.path.join(base["base_dir"], f)
                for f in image_paths_from_imsim_data]
            # remove .fz
            output_file_names = [f[:-3] if f.endswith(".fits.fz") else f
                                 for f in file_names_with_fz]

            # Now for the coadds
            orig_coadd_files = tile_info["coadd_file_list"]
            coadd_paths_from_imsim_data = [
                os.path.relpath(f, imsim_data)
                for f in orig_coadd_files]
            coadd_paths_with_fz = [
                os.path.join(base["base_dir"], f)
                for f in coadd_paths_from_imsim_data]
            coadd_output_filenames = [
                (f[:-3] if f.endswith(".fits.fz") else f) for f in coadd_paths_with_fz]

            config["_tile_setup"] = {}
            tile_setup = config["_tile_setup"]
            tile_setup.update(tile_info)
            # also add output filenames
            tile_setup["output_file_list"] = output_file_names
            tile_setup["coadd_output_file_list"] = coadd_output_filenames

            # Check if we're rejectlisting stuff
            if "rejectlist_file" in config:
                rejectlist = RejectList.from_file(config["rejectlist_file"])
                # Make a list of booleans, one for each simulated image, True if an
                # image is rejectlisted
                is_rejectlisted_list = [
                    rejectlist.img_file_is_rejectlisted(os.path.basename(f))
                    for f in image_paths_from_imsim_data
                ]
                for i, is_rejectlisted in enumerate(is_rejectlisted_list):
                    if is_rejectlisted:
                        logger.debug("PSF for output file %s is rejectlisted" % (
                            output_file_names[i]))
                    logger.debug(
                        "%d/%d piff files for tile %s rejectlisted" % (
                            is_rejectlisted_list.count(True),
                            len(is_rejectlisted_list),
                            tilename,
                        )
                    )
                tile_setup["is_rejectlisted_list"] = is_rejectlisted_list
        else:
            # use the cached value
            tile_setup = config["_tile_setup"]

        # Set some eval_variables that can be used in the config file.
        # In particular:
        # - input image filename for wcs
        # - input image psfex filename
        # - tile bounds
        # - magnitude zeropoint (for converting mags to fluxes)
        # - band
        # - probably other stuff
        galsim.config.eval_base_variables += ["psfex_path", "orig_image_path"]

        mode = config.get("mode", "single-epoch")

        if mode == "single-epoch":
            galsim.config.eval_base_variables += ["piff_path"]

        if "eval_variables" not in base:
            base["eval_variables"] = OrderedDict()
        if mode == "single-epoch":
            orig_psfex_path = tile_setup["psfex_files"][file_num]
            orig_piff_path = tile_setup["piff_files"][file_num]
            orig_bkg_path = tile_setup["bkg_files"][file_num]
            orig_head_path = tile_setup["head_files"][file_num]
            base["orig_image_path"] = tile_setup["image_files"][file_num]
            base["psfex_path"] = orig_psfex_path
            base["piff_path"] = orig_piff_path
            base["head_path"] = orig_head_path
            base["ccdnum"] = tile_setup["ccdnum_list"][file_num]
            base["eval_variables"]["sband"] = tile_setup["band_list"][file_num]
            base["eval_variables"]["fmag_zp"] = tile_setup["mag_zp_list"][file_num]
            base["eval_variables"]["iccdnum"] = tile_setup["ccdnum_list"][file_num]
            band = tile_setup["band_list"][file_num]
            file_name = tile_setup["output_file_list"][file_num]
            if "rejectlist_file" in config:
                base["is_rejectlisted"] = tile_setup["is_rejectlisted_list"][file_num]

        elif (mode == "coadd"):
            # For coadd/meds modes we just have one file per band per tile, so
            # most of the below is the same, except for the file_name
            orig_psfex_path = tile_setup["coadd_psfex_files"][file_num]
            base["orig_image_path"] = tile_setup["coadd_file_list"][file_num]
            base["psfex_path"] = orig_psfex_path
            base["eval_variables"]["sband"] = tile_setup["coadd_band_list"][file_num]
            base["eval_variables"]["fmag_zp"] = tile_setup["coadd_mag_zp_list"][file_num]
            band = tile_setup["coadd_band_list"][file_num]
            file_name = tile_setup["coadd_output_file_list"][file_num]
            orig_piff_path = None
            orig_bkg_path = None

        # these are not put into eval_variables because we do not
        # expect to use them in the galsim config
        base["tilename"] = tilename
        base["band_num"] = bands.index(band)
        base["band"] = band
        base["coadd_wcs"] = tile_setup["coadd_wcs"]

        base["eval_variables"]["fra_min_deg"] = tile_setup["tile_ra_ranges_deg"][0]
        base["eval_variables"]["fra_max_deg"] = tile_setup["tile_ra_ranges_deg"][1]
        base["eval_variables"]["fdec_min_deg"] = tile_setup["tile_dec_ranges_deg"][0]
        base["eval_variables"]["fdec_max_deg"] = tile_setup["tile_dec_ranges_deg"][1]
        base["eval_variables"]["fcoadd_ra_min_deg"] = tile_setup["coadd_ra_ranges_deg"][0]
        base["eval_variables"]["fcoadd_ra_max_deg"] = tile_setup["coadd_ra_ranges_deg"][1]
        base["eval_variables"]["fcoadd_dec_min_deg"] = tile_setup["coadd_dec_ranges_deg"][0]
        base["eval_variables"]["fcoadd_dec_max_deg"] = tile_setup["coadd_dec_ranges_deg"][1]
        logger.debug(
            "ra min/max for tile %s: %f,%f degrees" % (
                tilename, base["eval_variables"]["fra_min_deg"],
                base["eval_variables"]["fra_max_deg"]))
        logger.debug(
            "dec min/max for tile %s: %f,%f degrees" % (
                tilename, base["eval_variables"]["fdec_min_deg"],
                base["eval_variables"]["fdec_max_deg"]))

        # Now set some fields for the sim
        base["image"]["wcs"] = {}
        se_wcs_type = config.get("se_wcs", "head")
        if se_wcs_type == "image" or mode == "coadd":
            base["image"]["wcs"]["type"] = "Fits"
            base["image"]["wcs"]["file_name"] = base["orig_image_path"]
        elif se_wcs_type == "head":
            base["image"]["wcs"]["type"] = "Fits"
            base["image"]["wcs"]["file_name"] = base["head_path"]
            base["image"]["wcs"]["text_file"] = True
        else:
            raise RuntimeError("SE WCS type %s not recognized!" % se_wcs_type)

        # set file_name in config
        config["file_name"] = file_name
        config["truth"]["file_name"] = get_truth_from_image_file(file_name, tilename)

        # make sure we're not overwriting the original image somehow.
        assert config["file_name"] != base["orig_image_path"]

        # Set the image size
        # we checked mode above so no need to do so here
        if mode == "single-epoch":
            base["image"]["xsize"] = 2048
            base["image"]["ysize"] = 4096
        elif mode == "coadd":
            base["image"]["xsize"] = 10000
            base["image"]["ysize"] = 10000

        # Noise
        # Some convenient options for noise
        if "noise_mode" in config:
            add_bkg = config.get("add_bkg", True)
            if mode == "coadd":
                try:
                    # because coadd files don't have a skysigma entry
                    # understandably.
                    assert config["noise_mode"] != "skysigma"
                except AssertionError as e:
                    logger.error(
                        "You can't use noise_mode=skysigma in coadd mode, "
                        "because the coadds don't have a skysigma.")
                    raise e
                if add_bkg:
                    logger.debug("add_bkg is set to True, but we're in "
                                 "coadd mode, setting to False")
                add_bkg = False

            base["image"]["noise"] = {
                "type": "ChipNoise",
                "orig_image_path": base["orig_image_path"],
                "noise_mode": config["noise_mode"],
                "noise_fac": config.get("noise_fac", None),
            }

            # also copy background file
            output_bkg_path = os.path.join(
                base["base_dir"],
                os.path.relpath(orig_bkg_path, imsim_data),
            )
            output_bkg_dir = os.path.dirname(output_bkg_path)
            if not os.path.isdir(output_bkg_dir):
                safe_mkdir(output_bkg_dir)
            shutil.copyfile(orig_bkg_path, output_bkg_path)
            base["image"]["noise"]["bkg_filename"] = output_bkg_path

            if add_bkg:
                base["image"]["noise"]["_zero_bkg"] = False
            else:
                base["image"]["noise"]["_zero_bkg"] = True

        elif "noise" in config and "add_bkg" in config["noise"]:
            raise ValueError(
                "Can't use add_bkg option without noise_mode at present")
        else:
            raise RuntimeError("You must use `noise_mode` in the `output` section!")

        if base["psf"]["type"] in ("DES_PSFEx", "DES_PSFEx_perturbed"):
            # If using psfex PSF, copy file to output directory and file_info
            output_psfex_path = os.path.join(
                base["base_dir"],
                os.path.relpath(orig_psfex_path, imsim_data),
            )
            output_psfex_dir = os.path.dirname(output_psfex_path)
            if not os.path.isdir(output_psfex_dir):
                safe_mkdir(output_psfex_dir)
            shutil.copyfile(orig_psfex_path, output_psfex_path)

            # make sure the draw method is correct for PSFEx
            assert base['stamp']['draw_method'] == 'no_pixel'
        elif base["psf"]["type"] in ["DES_Piff", "DES_SmoothPiff"]:
            output_piff_path = os.path.join(
                base["base_dir"],
                os.path.relpath(orig_piff_path, imsim_data),
            )
            output_piff_dir = os.path.dirname(output_piff_path)
            if not os.path.isdir(output_piff_dir):
                safe_mkdir(output_piff_dir)
            shutil.copyfile(orig_piff_path, output_piff_path)

            # make sure the draw method is correct for Piff
            if base["psf"].get("depixelize", False) or base["psf"]["type"] == "DES_SmoothPiff":
                assert base['stamp']['draw_method'] == 'auto'
            else:
                assert base['stamp']['draw_method'] == 'no_pixel'
        else:
            # we assume that anything else is ok to draw with auto
            if 'draw_method' in base['stamp']:
                logger.debug(
                    "I found draw method '%s' for psf type '%s'"
                    " - I hope this is OK.",
                    base['stamp']['draw_method'], base["psf"]["type"])
            else:
                logger.debug(
                    "No draw method found in config - galsim will use"
                    "draw_method=auto, I hope this is OK.")

        # We'll be setting the random number seed to repeat for each band,
        # which requires querying the number of objects in the exposure.
        # This however leads to a logical infinite loop if the number of
        # objects is a random variate.  So to make this work, we first get the
        # number of objects in each exposure using a well-defined rng, and
        # save those values to a list, which is then fully deterministic
        # for all other uses.
        if 'nobjects' not in base['image']:
            raise ValueError("nobjects required for output type DESTile")

        nobjects = base['image']['nobjects']
        # For DES sims, it's a bit awkward to get the total number of objects,
        # since we may want to simulate e.g. a fixed number of stars from
        # one catalog, and randomly sample a random number of galaxies from
        # a different catalog. So allow the user to specify an nobjects
        # type MixedNObjects, which has the following options:
        # - ngalaxies: int
        #    # of galaxies
        # - use_all_stars: (bool)
        #    simulated all the stars in the input star catalog (default=True)
        # - nstars: int
        #    if use_all_stars=False, simulate this many stars
        if isinstance(nobjects, dict):
            # ^ this should return True for both dict and OrderedDict
            if base['image']['nobjects']['type'] == "MixedNObjects":
                if nobjects.get("use_all_gals", False):
                    # if we use all of the galaxies, then we look at the input desgals
                    # object to get the number of galaxies
                    galsim.config.input.SetupInput(base, logger=logger)
                    key = 'desgal'
                    field = base['input'][key]
                    loader = galsim.config.input.valid_input_types[key]
                    gal_input = galsim.config.GetInputObj("desgal", config, base, "desgal")
                    if gal_input is None:
                        kwargs, safe = loader.getKwargs(field, base, logger)
                        kwargs['_nobjects_only'] = True
                        gal_input = loader.init_func(**kwargs)
                    ngalaxies = gal_input.getNObjects()
                else:
                    # First get the number of galaxies. Either this will be an int, in
                    # which case
                    # ParseValue will work straightaway, or a random variable, in which
                    # case we'll
                    # need to initalize the rng and then try ParseValue again.
                    try:
                        ngalaxies = galsim.config.ParseValue(
                            nobjects, 'ngalaxies', base, int)[0]
                    except TypeError:
                        seed = galsim.config.ParseValue(
                            base['image'], 'random_seed', base, int)[0]
                        try:
                            assert (isinstance(seed, int) and (seed != 0))
                        except AssertionError as e:
                            logger.critical(
                                "image.random_seed must be set to a non-zero integer for "
                                "output type DES_Tile")
                            raise e
                        base['rng'] = galsim.BaseDeviate(seed)
                        ngalaxies = galsim.config.ParseValue(nobjects, 'ngalaxies',
                                                            base, int)[0]
                logger.log(logging.CRITICAL, "simulating %d galaxies" % ngalaxies)

                if nobjects.get("use_all_stars", True):
                    # Now the stars. In this case
                    # use all the stars in the star input catalog. We need
                    # to process the inputs to find this number. The below
                    # is adapted from galsim.config.input.ProcessInputNObjects
                    galsim.config.input.SetupInput(base, logger=logger)
                    key = 'desstar'
                    field = base['input'][key]
                    loader = galsim.config.input.valid_input_types[key]
                    star_input = galsim.config.GetInputObj("desstar", config, base, "desstar")
                    if star_input is None:
                        kwargs, safe = loader.getKwargs(field, base, logger)
                        kwargs['_nobjects_only'] = True
                        star_input = loader.init_func(**kwargs)
                    nstars = star_input.getNObjects()
                elif 'nstars' in nobjects:
                    # use_all_stars is False so check if nstars is specified and if
                    # so use this many stars
                    nstars = galsim.config.ParseValue(nobjects, 'nstars', base, int)[0]
                else:
                    # If use_all_stars is False, and nstars is not specified, then
                    # set nstars to zero. No stars for you.
                    nstars = 0
                logger.log(logging.CRITICAL, "simulating %d stars" % nstars)

                nobj = nstars + ngalaxies
                # Save an object_type_list as a base_eval_variable, this can be used
                # with MixedScene to specify whether to draw a galaxy or star.
                obj_type_list = ['star'] * nstars + ['gal'] * ngalaxies
                base["object_type_list"] = obj_type_list
            else:
                # If we're not using MixedNObjects, parse the nobjects as usual
                # (as above we may need to initialize base['rng'] first).
                try:
                    nobj = galsim.config.ParseValue(
                        base['image'], 'nobjects', base, int)[0]
                except TypeError:
                    seed = galsim.config.ParseValue(
                        base['image'], 'random_seed', base, int)[0]
                    try:
                        assert (isinstance(seed, int) and (seed != 0))
                    except AssertionError as e:
                        logger.critical(
                            "image.random_seed must be set to a non-zero integer for "
                            "output type DES_Tile")
                        raise e
                    base['rng'] = galsim.BaseDeviate(seed)
                    nobj = galsim.config.ParseValue(
                        base['image'], 'nobjects', base, int)[0]
        else:
            # if not a dict, should be an int (this will be checked below though).
            nobj = nobjects

            # we can correct whole floats so do that
            if int(nobj) == nobj:
                nobj = int(nobj)

        # Check that we now have an integer nobj
        try:
            assert isinstance(nobj, int)
        except AssertionError as e:
            logger.critical("found non-integer nobj:", nobj)
            raise e
        base['image']['nobjects'] = nobj
        logger.debug(
            'nobjects = %s',
            galsim.config.CleanConfig(base['image']['nobjects']))

        # Set the random numbers to repeat for the objects so we get the same
        # objects in the field each time. In fact what we do is generate four
        # sets of random seeds:
        # 0: Sequence of seeds that iterates with obj_num i.e. no repetetion.
        #    Used for noise
        # 1: Sequence of seeds that starts with the first object number for a
        #    given tile, then iterate with the obj_num minus the first object
        #    number for that band, intended for quantities that should be the
        #    same between bands for a given tile.
        # 2: Sequence of seeds that iterates with image_num

        rs = base['image']['random_seed']
        if not isinstance(rs, list):
            first = galsim.config.ParseValue(
                base['image'], 'random_seed', base, int)[0]

            # launder through the RNG to randomize
            first = galsim.BaseDeviate(first).raw()

            base['image']['random_seed'] = []
            # The first one is the original random_seed specification,
            # used for noise, since that should be different for each band,
            # and probably most things in input, output, or image.
            if isinstance(rs, int):
                base['image']['random_seed'].append(
                    {'type': 'Sequence', 'index_key':
                     'obj_num', 'first': first})
            else:
                base['image']['random_seed'].append(rs)

            # The second one is used for the galaxies and repeats
            # through the same set of seed values for each band in a tile.
            # Here are some notes on how this works.
            # Galsim seeds the whole run with first.
            # Then obj_num is incremented each time any object is drawn (across
            # any band, image etc).
            # So tile_start_obj_num is the first obj_num that is encountered
            # for a given tile, the quantity
            # (obj_num - tile_start_obj_num) % nobjects is a unique id for
            # each object in the tile and is used as the seed for drawing it.
            # we then add first and tile_start_obj_num to this seed to make
            # sure the objects in each tile are unique.
            if nobj > 0:
                base['image']['random_seed'].append({
                    'type': 'Eval',
                    'str': (
                        'first + tile_start_obj_num + '
                        '(obj_num - tile_start_obj_num) % nobjects'),
                    'ifirst': first,
                    'inobjects': {'type': 'Current', 'key': 'image.nobjects'}
                })
            else:
                base['image']['random_seed'].append(
                    base['image']['random_seed'][0])

            # The third iterates per image
            base['image']['random_seed'].append(
                {'type': 'Sequence', 'index_key': 'image_num', 'first': first})

            if 'gal' in base:
                base['gal']['rng_num'] = 1
            if 'star' in base:
                base['star']['rng_num'] = 1
            if base["psf"]["type"] in ["DES_Piff", "DES_SmoothPiff"]:
                base["psf"] = _set_catalog_sampler_rng_num(base["psf"], 1)
            if 'stamp' in base:
                base['stamp']['rng_num'] = 1
            if 'image_pos' in base['image']:
                base['image']['image_pos']['rng_num'] = 1
            if 'world_pos' in base['image']:
                base['image']['world_pos']['rng_num'] = 1
            if "truth" in base["output"]:
                base["output"]["truth"] = _set_catalog_sampler_rng_num(base["output"]["truth"], 1)

        logger.debug(
            'random_seed = %s',
            galsim.config.CleanConfig(base['image']['random_seed']))

        # Random seed stuff needs tile_num and tile_start_obj_num
        nobjects = galsim.config.ParseValue(
            base["image"], "nobjects", base, int)[0]
        # some comments on the fields below:
        # tile_start_obj_num:
        #  is the object number of the first object in the tile
        #  computed from
        #    start_obj_num (first object in file) -
        #    file_num * nobjects
        #        (objects already drawn for files in this tile)
        base["tile_start_obj_num"] = (
            base['start_obj_num'] - file_num * nobjects)

        # For debugging purposes, it's useful to generate objects on a grid.
        # These positions need to be re-generated every time the tilename
        # changes.
        if config.get("grid_objects", False):
            if "world_pos" not in base["image"]:
                base["image"]["world_pos"] = {}
            if not base["image"]["world_pos"].get("_setup_as_list", False):
                if 'grid_border' in config:
                    border = galsim.config.ParseValue(config, 'grid_border', base, float)[0]
                else:
                    border = 0  # 0 pixels

                logger.debug(
                    "generating gridded objects positions with dither %s and border %s",
                    config.get("dither_scale", 0.5), border,
                )
                L = 10000  # tile length in pixels
                width = L - 2 * border
                x_pos_list = []
                y_pos_list = []
                uniform = galsim.UniformDeviate(first)

                if config.get("grid_objects", False) == "hex":
                    # fudge factor approximates the interobject spacing that best fills the tile
                    spacing = width / np.sqrt(nobjects) * np.sqrt(1.15)
                    nx = int(np.ceil(np.sqrt(nobjects) * np.sqrt(2) * 2))
                    # the factor of 0.866 makes sure the grid is square-ish
                    ny = int(np.ceil(np.sqrt(nobjects) * np.sqrt(2) / 0.8660254 * 2))

                    # here the spacing between grid centers is 1
                    hg, _ = create_hex_grid(nx=nx, ny=ny, rotate_deg=uniform() * 360)

                    # convert the spacing to right number of pixels
                    # we also recenter the grid since it comes out centered at 0,0
                    nprng = np.random.RandomState(seed=int(uniform()*100_000_000 + 1))
                    hg *= spacing
                    hxpos = hg[:, 0].ravel() + L/2
                    hypos = hg[:, 1].ravel() + L/2

                    # we randomly sort the positions so we fill the grid at random instead of ordered
                    # that way when we cut it down it doesn't have weird missing edges
                    rinds = nprng.choice(hxpos.shape[0], size=hxpos.shape[0], replace=False).astype(int)
                    hxpos = hxpos[rinds]
                    hypos = hypos[rinds]

                    ndone = 0
                    for hx, hy in zip(hxpos, hypos):
                        offset_x = 2 * (uniform() - 0.5) * config.get("dither_scale", 0.5)
                        offset_y = 2 * (uniform() - 0.5) * config.get("dither_scale", 0.5)
                        hx += offset_x
                        hy += offset_y
                        if (
                            hx > border
                            and hx < L - border
                            and hy > border
                            and hy < L - border
                        ):
                            ndone += 1
                            x_pos_list.append(hx)
                            y_pos_list.append(hy)
                            if ndone == nobjects:
                                break

                    assert ndone == nobjects, (
                        "Hex grid was not big enough to hold all of the objects! "
                        "Try decreasing output.grid_hex_spacing_fudge_factor to generate a denser grid!"
                    )
                else:
                    # in this case we want to use a grid of objects positions.
                    # compute this grid in X,Y for the coadd,
                    # then convert to world position
                    nobj_per_row = int(np.ceil(np.sqrt(nobjects)))
                    object_sep = width / nobj_per_row
                    for i in range(nobjects):
                        offset_x = 2 * (uniform() - 0.5) * config.get("dither_scale", 0.5)
                        offset_y = 2 * (uniform() - 0.5) * config.get("dither_scale", 0.5)
                        x_pos_list.append(
                            (object_sep / 2. + object_sep * (i % nobj_per_row) + offset_x)
                            + border)
                        y_pos_list.append(
                            (object_sep / 2. + object_sep * (i // nobj_per_row) + offset_y)
                            + border
                        )
                coadd_wcs = tile_setup["coadd_wcs"]
                world_pos_list = [
                    coadd_wcs.toWorld(galsim.PositionD(x, y))
                    for (x, y) in zip(x_pos_list, y_pos_list)]
                ra_list = [(p.ra / galsim.degrees)
                           for p in world_pos_list]
                dec_list = [(p.dec / galsim.degrees)
                            for p in world_pos_list]

                # add positions to galsim
                base["image"]["world_pos"] = {
                    "type": "RADec",
                    "ra": {
                        'type': 'Degrees',
                        'theta': {
                            'type': 'List',
                            'items': ra_list,
                            'index': "$obj_num - start_obj_num",
                            '_setup_as_list': True
                        }
                    },
                    "dec": {
                        'type': 'Degrees',
                        'theta': {
                            'type': 'List',
                            'items': dec_list,
                            'index': "$obj_num - start_obj_num",
                            '_setup_as_list': True
                        }
                    },
                    '_setup_as_list': True
                }

        logger.debug(
            "nobjects config: %s",
            galsim.config.CleanConfig(base['image']['nobjects']))
        logger.debug("nobjects: %d", nobjects)
        logger.debug("file_num: %d", file_num)
        logger.debug("start_obj_num: %d", base["start_obj_num"])
        logger.debug("tile_start_obj_num: %d", base["tile_start_obj_num"])

        logger.debug(
            'file_num, band = %s, %s',
            file_num, base["eval_variables"]["sband"])

        # This sets up the RNG seeds.
        # We are making sure to call the OutputBuilder from galsim
        # and not some other parent in the class hierarchy. This is
        # why we are not using super(...).
        OutputBuilder.setup(self, config, base, file_num, logger)

    def getNFilesTile(self, config, tilename):
        bands = config["bands"]
        if "imsim_data" in config:
            imsim_data = galsim.config.ParseValue(config, "imsim_data", {}, str)[0]
        else:
            imsim_data = os.environ["IMSIM_DATA"]
        desrun = config["desrun"]
        nfiles = 0
        mode = config.get("mode", "single-epoch")
        for band in bands:
            if mode == "single-epoch":
                if "n_se_test" in config:
                    n_se_test = galsim.config.ParseValue(config, "n_se_test", {}, int)[0]
                else:
                    n_se_test = None

                pyml = read_pizza_cutter_yaml(
                    imsim_data, desrun, tilename, band, n_se_test=n_se_test
                )
                nfiles += len(pyml["src_info"])
            elif mode == "coadd":
                nfiles += 1
            else:
                raise ValueError(
                    "invalid mode, should be either 'single-epoch' or 'coadd'")
        return nfiles

    def getNFiles(self, config, base, logger=None):
        """Returns the number of files to be built.

        As far as the config processing is concerned, this is the number of
        times it needs to call buildImages, regardless of how many physical
        files are actually written to disk. So this corresponds to output.nexp
        for the FocalPlane output type.

        @param config           The configuration dict for the output field.
        @param base             The base configuration dict.

        @returns the number of "files" to build.
        """
        # This function gets called early, before the setup function, as
        # the number of files is required to split up jobs sensibly. So we need
        # to read in some information about the tiles being simulated here
        tilename = config["tilename"]
        return self.getNFilesTile(config, tilename)

    def buildImages(
            self, config, base, file_num, image_num, obj_num, ignore, logger):
        """Build the images

        @param config           The configuration dict for the output field.
        @param base             The base configuration dict.
        @param file_num         The current file_num.
        @param image_num        The current image_num.
        @param obj_num          The current obj_num.
        @param ignore           A list of parameters that are allowed to be in
                                config that we can ignore here. i.e. it won't
                                be an error if they are present.
        @param logger           If given, a logger object to log progress.

        @returns a list of the images built
        """
        logger.info('Starting buildImages')
        logger.info('file_num: %d' % base['file_num'])
        logger.info('image_num: %d' % base['image_num'])

        ignore += ['tilename', 'bands', 'desrun', 'imsim_data', 'noise_mode',
                   'add_bkg', 'noise_fac', 'mode', 'grid_objects',
                   'rejectlist_file', 'dither_scale', 'coadd_wcs', 'n_se_test',
                   'grid_border']
        ignore += ['file_name', 'dir']
        ignore += ['analyze_with_interpimage_psf', 'analyze_with_interpimage_psf_s2n']
        logger.debug("current mag_zp: %f" % base["eval_variables"]["fmag_zp"])

        # We are making sure to call the OutputBuilder from galsim
        # and not some other parent in the class hieratrchy. This is
        # why we are not using super(...).
        images = OutputBuilder.buildImages(
            self, config, base, file_num, image_num, obj_num, ignore, logger)
        return images

    def writeFile(self, data, file_name, config, base, logger):
        """Write the data to a file.

        @param data             The data to write. Usually a list of
                                images returned by buildImages, but possibly
                                with extra HDUs tacked onto the end from the
                                extra output items.
        @param file_name        The file_name to write to.
        @param config           The configuration dict for the output field.
        @param base             The base configuration dict.
        @param logger           If given, a logger object to log progress.
        """
        try:
            fp = pyfits.open(base["orig_image_path"])
            header_list = [fp[i].header for i in [1, 2, 3]]
        finally:
            fp.close()
        writeMulti(data, file_name, header_list=header_list)


# hooks for the galsim config parser
galsim.config.process.top_level_fields += ['meta_params']
galsim.config.output.RegisterOutputType('DESTile', DESTileBuilder())
galsim.config.RegisterNoiseType('ChipNoise', ChipNoiseBuilder())
