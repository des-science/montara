from __future__ import print_function, absolute_import
import os
import pprint
import glob

import numpy as np
import galsim
import galsim.config
import fitsio
import eastlake

from .multiband_meds import MultibandMEDSBuilder
from eastlake.step import Step
from eastlake.utils import get_logger
from .utils import safe_mkdir
from eastlake.rejectlist import RejectList
from .tile_setup import (
    get_source_list_files, get_tile_center,
    get_truth_from_image_file, get_output_coadd_path)


class MontaraGalSimRunner(Step):
    """
    Pipeline step which runs galsim

    The config attribute is a little different here, since it is updated when
    running GalSim
    """

    def __init__(
        self, config, base_dir, name="galsim", logger=None, verbosity=0, log_file=None
    ):
        super().__init__(
            config, base_dir, name=name, logger=logger, verbosity=verbosity,
            log_file=log_file)
        self.config['output']['dir'] = base_dir

        if self.config["output"]["type"] == "MultibandMEDS":
            if "truth" in self.config["output"]:
                if "dir" in self.config["output"]["truth"]:
                    if not os.path.isabs(
                            self.config["output"]["truth"]["dir"]):
                        self.config["output"]["truth"]["dir"] = os.path.join(
                            base_dir, self.config["output"]["truth"]["dir"])

        # For the downstream interpretation of these sims, it's going to be
        # quite important to have various quantities saved in the truth files.
        # And that the column names correspond to what we expect them to...so
        # do a bit of enforcement of that here.
        output = self.config["output"]
        if output["type"] in ["DESTile", "MultibandMEDS"]:
            if "truth" not in output:
                output["truth"] = {}
                output["truth"]["colnames"] = {}
            x_str = "image_pos.x"
            y_str = "image_pos.y"
            p_str = "image_pos"
            if "stamp" in self.config:
                if "offset" in self.config["stamp"]:
                    x_str += " + (@stamp.offset).x"
                    y_str += " + (@stamp.offset).y"
                    p_str += " + (@stamp.offset)"

            add_to_truth = {
                "id": "$tile_start_obj_num + obj_num - start_obj_num",
                "flux": "$float((@current_obj).flux)",
                "mag": "$-2.5*np.log10((@current_obj).flux) + mag_zp",
                "x": "$%s" % x_str,
                "y": "$%s" % y_str,
                "ra": {
                    "type": "Eval",
                    "str": "'%.12e' % (ra_val)",
                    "fra_val": "$(@image.wcs).toWorld(%s).ra / galsim.degrees" % p_str},
                "dec": {
                    "type": "Eval",
                    "str": "'%.12e' % (dec_val)",
                    "fdec_val": "$(@image.wcs).toWorld(%s).dec / galsim.degrees" % p_str},
                "x_coadd": {
                    "type": "Eval",
                    "str": "'%.12e' % (x_coadd_val)",
                    "fx_coadd_val": "$coadd_wcs.toImage((@image.wcs).toWorld(%s)).x" % p_str},
                "y_coadd": {
                    "type": "Eval",
                    "str": "'%.12e' % (y_coadd_val)",
                    "fy_coadd_val": "$coadd_wcs.toImage((@image.wcs).toWorld(%s)).y" % p_str},
            }
            if "stamp" in self.config:
                if "objects" in self.config["stamp"]:
                    add_to_truth["obj_type_index"] = "@current_obj_type_index"
            if "catalog_sampler" in self.config["input"]:
                add_to_truth["gal_catalog_row"] = {
                    "type": "Eval",
                    "str": "-1 if @current_obj_type=='star' else int(gal_catalog_row)",  # noqa
                    "fgal_catalog_row": {
                        "type": "catalog_sampler_value",
                        "col": "catalog_row"}
                }
            if "desstar" in self.config["input"]:
                add_to_truth["star_catalog_row"] = {
                    "type": "Eval",
                    "str": "-1 if @current_obj_type=='gal' else int(star_catalog_row)",  # noqa
                    "fstar_catalog_row": {
                        "type": "DESStarValue",
                        "col": "catalog_row"}
                }
            if output["type"] == "MultibandMEDS":
                # no ra and dec
                add_to_truth.pop("ra")
                add_to_truth.pop("dec")
                # id column
                add_to_truth["id"] = (
                    "$tile_start_obj_num + (obj_num -start_obj_num) "
                    "// @output.nstamps_per_object")
                if "star_catalog_row" in add_to_truth:
                    add_to_truth["star_catalog_row"]["fstar_catalog_row"]["index"] = "$star_index"  # noqa

            for col in add_to_truth:
                if col in output["truth"]["columns"]:
                    self.logger.error(
                        "column %s already in truth.columns specified in "
                        "config file, overwriting since this column needs "
                        "to be a specific thing for downstream "
                        "processing" % col)
                output["truth"]["columns"][col] = add_to_truth[col]

        self.config_orig = galsim.config.CopyConfig(self.config)

    def execute(self, stash, new_params=None, except_abort=False, verbosity=1.,
                log_file=None, comm=None):

        if comm is not None:
            rank = comm.Get_rank()
        else:
            rank = 0

        if new_params is not None:
            galsim.config.UpdateConfig(self.config, new_params)

        # Make a copy of original config
        config = galsim.config.CopyConfig(self.config)
        if rank == 0:
            self.logger.debug(
                "Process config dict: \n%s", pprint.pformat(config))

        if self.name not in stash:
            stash[self.name] = {}

        # Get the tilename
        stash["tilenames"] = [config["output"]["tilename"]]

        galsim.config.Process(config, self.logger, except_abort=except_abort)

        self.update_stash(config, stash)

        # Return status and stash
        return 0, stash

    def update_stash(self, config, stash):
        # Update the stash with information on image files etc. required by
        # following steps.

        # Get the output type and number of files
        image_type = config["output"]["type"]
        bands = config["output"]["bands"]
        nbands = len(bands)
        tilenames = stash["tilenames"]
        tilename = tilenames[0]
        assert len(tilenames) == 1
        ntiles = 1

        self.logger.error(
            "Simulated tile %s in bands %s" % (
                tilename, str(bands)))
        stash["nbands"] = nbands
        stash["bands"] = bands

        # Add the rejectlist
        if "rejectlist_file" in config["output"]:
            rejectlist = RejectList.from_file(config["output"]["rejectlist_file"])
            stash["rejectlist"] = rejectlist.rejectlist_data

        # Add the PSF config
        stash["psf_config"] = config["psf"]
        # add draw_method if present
        if "draw_method" in config["stamp"]:
            stash["draw_method"] = config["stamp"]["draw_method"]
        else:
            stash["draw_method"] = "auto"

        if image_type == "DESTile":
            desrun = galsim.config.GetCurrentValue(
                "desrun", config["output"], str, config)
            try:
                desdata = galsim.config.GetCurrentValue(
                    "desdata", config["output"], str, config)
            except KeyError:
                desdata = os.environ['DESDATA']
            mode = config["output"].get("mode", "single-epoch")
            stash["desrun"] = desrun
            stash["desdata"] = desdata
            base_dir = self.base_dir

            # get source list files if running in single-epoch mode
            if mode == "single-epoch":
                stash["tile_info"] = {}
                for tilename in tilenames:
                    source_list_files = get_source_list_files(
                        base_dir, desrun, tilename, bands)
                    stash["tile_info"][tilename] = {}
                    for band in bands:
                        if band not in stash["tile_info"][tilename]:
                            stash["tile_info"][tilename][band] = {}
                        # get source list files
                        stash["tile_info"][tilename][band][
                            "source_list_files"] \
                            = source_list_files[band]
                        # and read this info into stash for handy downstream
                        # use...
                        (img_list_file, wgt_list_file, msk_list_file,
                         mag_zp_list_file) = source_list_files[band]

                        def read_source_list(
                                filename, default_ext, output_type=str):
                            with open(filename, 'r') as f:
                                lines = f.readlines()
                            output_list, exts = [], []
                            if len(lines) == 0:
                                raise ValueError(
                                    "Unexpectedly found a zero line "
                                    "source list file %s" % filename)
                            for ln in lines:
                                ext = default_ext
                                s = (ln.strip()).split(" ")[0]
                                if s[-1] == "]":
                                    ext = int(s[-2])
                                    s = s[:-3]
                                output_list.append(output_type(s))
                                exts.append(ext)
                            # make sure all exts are the same
                            assert len(set(exts)) <= 1
                            return output_list, exts[0]

                        image_files, image_ext = read_source_list(
                            img_list_file, 0)
                        print('image file', image_files)
                        print('image_ext', image_ext)
                        wgt_files, wgt_ext = read_source_list(
                            wgt_list_file, 2)
                        msk_files, msk_ext = read_source_list(
                            msk_list_file, 1)
                        mag_zps, _ = read_source_list(
                            mag_zp_list_file, None, float)

                        stash.set_filepaths(
                            "img_files", image_files, tilename, band=band)
                        stash["tile_info"][tilename][band]["img_ext"] \
                            = image_ext
                        stash.set_filepaths(
                            "wgt_files", wgt_files, tilename, band=band)
                        stash["tile_info"][tilename][band]["wgt_ext"] = wgt_ext
                        stash.set_filepaths(
                            "msk_files", msk_files, tilename, band=band)
                        stash["tile_info"][tilename][band]["msk_ext"] = msk_ext
                        stash["tile_info"][tilename][band]["mag_zps"] = mag_zps

                        # truth
                        truth_files = [
                            get_truth_from_image_file(f, tilename)
                            for f in image_files]
                        stash.set_filepaths(
                            "truth_files", truth_files, tilename, band=band)

                    # also get tile center
                    tile_center = get_tile_center(
                        desdata, desrun, tilename, bands[0])
                    stash["tile_info"][tilename]["tile_center"] = tile_center

                    # if doing gridded objects, save the true position data
                    # to a fits file
                    if config['output'].get('grid_objects', False):
                        # build up the unique truth entries via looking at the entries
                        # for each CCD
                        fnames = glob.glob(
                            os.path.join(base_dir, desrun, tilename, "**/truth_*.dat"),
                            recursive=True,
                        )
                        data = []
                        for fname in fnames:
                            if os.path.getsize(fname):
                                _d = np.atleast_1d(np.genfromtxt(fname, names=True))
                                data.append(_d)

                        if len(data) == 0:
                            raise RuntimeError(
                                "No objects drawn for tile %s when using a grid!" % tilename
                            )

                        data = np.concatenate(data)
                        uids, uinds = np.unique(data["id"], return_index=True)
                        n_pos_data = len(uids)
                        _pos_data = np.zeros(n_pos_data, dtype=[
                                ('ra', 'f8'), ('dec', 'f8'),
                                ('x', 'f8'), ('y', 'f8'),
                                ('id', 'i8')])
                        _pos_data['id'] = data['id'][uinds]
                        _pos_data['ra'] = data['ra'][uinds]
                        _pos_data['dec'] = data['dec'][uinds]
                        _pos_data['x'] = data['x_coadd'][uinds]
                        _pos_data['y'] = data['y_coadd'][uinds]

                        # we'll stash this for later
                        truepos_filename = os.path.join(
                            base_dir,
                            "true_positions",
                            "%s-truepositions.fits" % tilename,
                        )
                        safe_mkdir(os.path.dirname(truepos_filename))
                        self.logger.error(
                            "writing true position data to %s" % truepos_filename)
                        fitsio.write(truepos_filename, _pos_data, clobber=True)
                        stash.set_filepaths("truepositions_file",
                                            truepos_filename,
                                            tilename)

            elif mode == "coadd":
                # set the coadd filenames
                stash["tile_info"] = {}
                for tilename in tilenames:
                    stash["tile_info"][tilename] = {}
                    # add tile center
                    tile_center = get_tile_center(
                        desdata, desrun, tilename, bands[0])
                    stash["tile_info"][tilename]["tile_center"] = tile_center
                    for band in bands:
                        if band not in stash["tile_info"][tilename]:
                            stash["tile_info"][tilename][band] = {}
                        band_file_info = stash["tile_info"][tilename][band]

                        # Get coadd file names etc. to be used by e.g. swarp
                        # step.
                        output_coadd_path = get_output_coadd_path(
                            desdata, desrun, tilename, band, base_dir,
                            fz=False)
                        stash.set_filepaths(
                            "coadd_file", output_coadd_path, tilename,
                            band=band)
                        band_file_info["coadd_ext"] = 0

                        added_mask = False
                        if "badpix" in config["output"]:
                            if "hdu" in config["output"]["badpix"]:
                                stash.set_filepaths(
                                    "coadd_mask_file", output_coadd_path,
                                    tilename, band=band)
                                band_file_info["coadd_mask_ext"] \
                                    = config["output"]["badpix"]["hdu"]
                                added_mask = True
                        if not added_mask:
                            self.logger.error(
                                "not adding coadd_mask_file to tile_info..."
                                "this will likely cause problems downstream")

                        added_weight = False
                        if "weight" in config["output"]:
                            if "hdu" in config["output"]["weight"]:
                                stash.set_filepaths(
                                    "coadd_weight_file", output_coadd_path,
                                    tilename, band=band)
                                band_file_info["coadd_weight_ext"] \
                                    = config["output"]["weight"]["hdu"]
                                added_weight = True
                        if not added_weight:
                            self.logger.error(
                                "not adding coadd_weight_file to tile_info..."
                                "this will likely cause problems downstream")

                        # truth
                        truth_file = get_truth_from_image_file(
                            output_coadd_path, tilename)
                        stash.set_filepaths(
                            "truth_files", [truth_file], tilename, band=band)

        elif (image_type == "MultibandMEDS"):
            # set the meds and truth filenames in the stash
            stash["tile_info"] = {}
            # To do this, we can loop through file_num calling the builder
            # setup function.
            output = config["output"]
            bands = output["bands"]
            nbands = len(bands)
            file_num = 0
            logger = get_logger("", 0)

            # loop through tiles and bands, calling the setup function
            # for MultibandMEDS,
            # which should then allow access to the correct output filenames,
            # and also the tilenames
            tilenames = []
            builder = MultibandMEDSBuilder()
            for tile_num in range(ntiles):
                for band_num, band in enumerate(bands):
                    galsim.config.SetupConfigFileNum(
                        config, file_num, 0, 0, logger)
                    builder.setup(output, config, file_num, logger)
                    tilename = galsim.config.GetCurrentValue(
                        "tilename", config, str)
                    if tilename not in tilenames:
                        tilenames.append(tilename)
                    if tilename not in stash["tile_info"]:
                        stash["tile_info"][tilename] = {}
                    if band not in stash["tile_info"][tilename]:
                        stash["tile_info"][tilename][band] = {}

                    # get meds file and truth file
                    meds_filename = galsim.config.GetCurrentValue(
                        "file_name", output, str, config)
                    d = None
                    if "dir" in config["output"]:
                        d = galsim.config.GetCurrentValue(
                            "dir", output, str, config)
                        meds_filename = os.path.join(d, meds_filename)
                    stash.set_filepaths(
                        "meds_file", meds_filename, tilename, band=band)
                    if "truth" in output:
                        truth_filename = galsim.config.GetCurrentValue(
                            "truth.file_name", output, str, config)
                        if "dir" in output["truth"]:
                            truth_filename = os.path.join(
                                output["truth"]["dir"], truth_filename)
                        elif d is not None:
                            truth_filename = os.path.join(d, truth_filename)
                    stash.set_filepaths(
                        "truth_files", [truth_filename], tilename, band=band)
                    file_num += 1

            # Set the MEDS_DIR environment variable, assumed to be
            # self.base_dir/meds
            os.environ["MEDS_DIR"] = os.path.join(self.base_dir, "meds")
            # add tilenames to stash for later steps
            stash["tilenames"] = tilenames

    @classmethod
    def from_config_file(cls, config_file, logger=None):
        all_config = galsim.config.ReadConfig(config_file, None, logger)
        assert len(all_config) == 1
        return cls(all_config[0], logger=logger)

    def set_base_dir(self, base_dir):
        self.base_dir = base_dir
        # Update the output directory.
        self.config['output']['dir'] = base_dir


eastlake.register_pipeline_step("galsim_montara", MontaraGalSimRunner, is_galsim=True)
