from __future__ import print_function, absolute_import
import os
import pprint

import numpy as np
import galsim
import galsim.config
import fitsio
import eastlake

from eastlake.step import Step
from .utils import safe_mkdir, get_truth_from_image_file
from eastlake.rejectlist import RejectList
from eastlake.des_files import read_pizza_cutter_yaml


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
        if output["type"] in ["DESTile"]:
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
            if "catalog_sampler" in self.config.get("input", {}):
                add_to_truth["gal_catalog_row"] = {
                    "type": "Eval",
                    "str": "-1 if @current_obj_type=='star' else int(gal_catalog_row)",  # noqa
                    "fgal_catalog_row": {
                        "type": "catalog_sampler_value",
                        "col": "catalog_row"}
                }
            if "desstar" in self.config.get("input", {}):
                add_to_truth["star_catalog_row"] = {
                    "type": "Eval",
                    "str": "-1 if @current_obj_type=='gal' else int(star_catalog_row)",  # noqa
                    "fstar_catalog_row": {
                        "type": "DESStarValue",
                        "col": "catalog_row"}
                }

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
        bands = config["output"]["bands"]
        nbands = len(bands)
        tilenames = stash["tilenames"]
        tilename = tilenames[0]
        assert len(tilenames) == 1

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

        desrun = galsim.config.GetCurrentValue(
            "desrun", config["output"], str, config)
        try:
            imsim_data = galsim.config.GetCurrentValue(
                "imsim_data", config["output"], str, config)
        except KeyError:
            imsim_data = os.environ['IMSIM_DATA']
        mode = config["output"].get("mode", "single-epoch")
        stash["desrun"] = desrun
        stash["imsim_data"] = imsim_data
        base_dir = self.base_dir
        n_se_test = config["output"].get("n_se_test", None)

        # get source list files if running in single-epoch mode
        if mode == "single-epoch":
            for tilename in tilenames:
                _tfiles = []
                for band in bands:
                    stash.set_input_pizza_cutter_yaml(
                        read_pizza_cutter_yaml(
                            imsim_data, desrun, tilename, band, n_se_test=n_se_test,
                        ),
                        tilename,
                        band,
                    )

                    # truth
                    with stash.update_output_pizza_cutter_yaml(tilename, band) as pyml:
                        for i in range(len(pyml["src_info"])):
                            fname = pyml["src_info"][i]["image_path"]
                            if fname.endswith(".fz"):
                                fname = fname[:-3]

                            with fitsio.FITS(fname, "rw") as fits:
                                fits[0].write_key("EXTNAME", "sci")
                                fits[1].write_key("EXTNAME", "msk")
                                fits[2].write_key("EXTNAME", "wgt")

                            pyml["src_info"][i]["image_path"] = fname
                            pyml["src_info"][i]["image_ext"] = "sci"

                            pyml["src_info"][i]["bmask_path"] = fname
                            pyml["src_info"][i]["bmask_ext"] = "msk"

                            pyml["src_info"][i]["weight_path"] = fname
                            pyml["src_info"][i]["weight_ext"] = "wgt"

                        truth_files = [
                            get_truth_from_image_file(src["image_path"], tilename)
                            for src in pyml["src_info"]
                        ]
                    stash.set_filepaths("truth_files", truth_files, tilename, band=band)
                    _tfiles += truth_files

                # if doing gridded objects, save the true position data
                # to a fits file
                if config['output'].get('grid_objects', False):
                    self._write_truth(_tfiles, tilename, base_dir, stash)

        elif mode == "coadd":
            for tilename in tilenames:
                _tfiles = []
                for band in bands:
                    stash.set_input_pizza_cutter_yaml(
                        read_pizza_cutter_yaml(
                            imsim_data, desrun, tilename, band, n_se_test=n_se_test,
                        ),
                        tilename,
                        band,
                    )

                    with stash.update_output_pizza_cutter_yaml(tilename, band) as pyml:
                        fname = pyml["image_path"]
                        if fname.endswith(".fz"):
                            fname = fname[:-3]

                        with fitsio.FITS(fname, "rw") as fits:
                            fits[0].write_key("EXTNAME", "sci")

                            pyml["image_path"] = fname
                            pyml["image_ext"] = "sci"

                            if (
                                "badpix" in config["output"]
                                and "hdu" in config["output"]["badpix"]
                            ):
                                pyml["bmask_path"] = fname
                                pyml["bmask_ext"] = "msk"
                                fits[config["output"]["badpix"]["hdu"]].write_key("EXTNAME", "msk")
                            else:
                                self.logger.error(
                                    "not updating coadd bmask path and ext..."
                                    "this will likely cause problems downstream"
                                )

                            if (
                                "weight" in config["output"]
                                and "hdu" in config["output"]["weight"]
                            ):
                                pyml["weight_path"] = fname
                                pyml["weight_ext"] = "wgt"
                                fits[config["output"]["weight"]["hdu"]].write_key("EXTNAME", "wgt")
                            else:
                                self.logger.error(
                                    "not updating coadd weight path and ext..."
                                    "this will likely cause problems downstream")

                    # truth
                    truth_file = get_truth_from_image_file(fname, tilename)
                    stash.set_filepaths(
                        "truth_files", [truth_file], tilename, band=band)
                    _tfiles.append(truth_file)

                # if doing gridded objects, save the true position data
                # to a fits file
                if config['output'].get('grid_objects', False):
                    self._write_truth(_tfiles, tilename, base_dir, stash)

            # add tilenames to stash for later steps
            stash["tilenames"] = tilenames

    def _write_truth(self, fnames, tilename, base_dir, stash):
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
