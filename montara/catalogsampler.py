import galsim
import numpy as np
from galsim.config.input import InputLoader
import fitsio

from .utils import add_field


class CatalogSampler(object):
    """class for randomly sampling object properties from a catalog"""
    _req_params = {"file_name": str}
    # _opt_params = { "cuts" : dict, "replace" : bool }
    _opt_params = {"replace": bool, "verbose": bool}
    _single_params = []

    # It doesn't actually need an rng, but this marks it as "unsafe"
    # to the ProcessInput function, which avoids some multiprocessing
    # pickle problems.
    _takes_rng = True

    def __init__(self, file_name, cuts=None, rng=None, replace=True, verbose=False):
        # Read in data
        self.catalog_data = fitsio.read(file_name)
        catalog_row_inds = np.arange(len(self.catalog_data))
        self.catalog_data = add_field(
            self.catalog_data, [("catalog_row", int)], [catalog_row_inds]
        )
        self.colnames = self.catalog_data.dtype.names
        self.dtype = self.catalog_data.dtype
        if cuts is not None:
            self.apply_cuts(cuts, verbose=verbose)
        self.replace = replace
        self.verbose = verbose

    def apply_cuts(self, cuts, verbose=False):
        """cuts is a dictionary. A key should be the same as a column in
        the catalog. The value corresponding to that cut can either be
        a single integer, in which case that value will be retained
        e.g. "flags":0 would just keep objects with flags==0.
        Or it can be a range e.g. "hlr":[a,b] in which case only
        objects with a<=hlr<b will be retained."""
        use = np.ones(len(self.catalog_data), dtype=bool)
        for key, val in cuts.items():
            col_data = self.catalog_data[key]
            if len(val) == 1:
                mask = col_data == int(val[0])
                cut_string = "{0} == {1}".format(key, val[0])
            elif len(val) == 2:
                mask = (col_data >= val[0]) * (col_data < val[1])
                cut_string = "{0} <= {1} < {2}".format(val[0], key, val[1])
            else:
                raise ValueError("cut value should be length 1 or 2")
            use[~mask] = False
            if verbose:
                print('applying {0} leaves fraction {1}'.format(
                    cut_string, float(mask.sum())/len(mask))
                )
        if verbose:
            print("%d/%d objects remaining after cuts" % (use.sum(), len(use)))
        self.catalog_data = self.catalog_data[use]

    def sample(self, rng):
        rand = np.random.RandomState(rng.raw())
        index = rand.randint(0, len(self.catalog_data))
        if self.replace is False:
            self.catalog_data = np.delete(self.catalog_data, index)
        return self.catalog_data[index], self.dtype, index


class CatalogSamplerLoader(InputLoader):
    def getKwargs(self, config, base, logger):
        """Parse the config dict and return the kwargs needed to build the input object.

        The default implementation looks for special class attributes called:

            _req_params     A dict of required parameters and their types.
            _opt_params     A dict of optional parameters and their types.
            _single_params  A list of dicts of parameters such that one and only one of
                            parameter in each dict is required.
            _takes_rng      A bool value saying whether an rng object is required.

        See galsim.Catalog for an example of a class that sets these attributes.

        In addition to the kwargs, we also return a bool value, safe, that
        indicates whether the constructed object will be safe to keep around for
        multiple files (True) of if it will need to be rebuilt for each output
        file (False).

        @param config       The config dict for this input item
        @param base         The base config dict
        @param logger       If given, a logger object to log progress. [default: None]

        @returns kwargs, safe
        """
        req = self.init_func._req_params
        opt = self.init_func._opt_params
        single = self.init_func._single_params
        ignore = ["cuts"]
        kwargs, safe = galsim.config.GetAllParams(
            config, base, req=req, opt=opt, single=single, ignore=ignore,
        )
        if self.init_func._takes_rng:
            rng = galsim.config.GetRNG(
                config, base, logger, 'input '+self.init_func.__name__,
            )
            kwargs['rng'] = rng
            safe = False
        if "cuts" in config:
            kwargs["cuts"] = config["cuts"]
        return kwargs, safe


def CatalogRow(config, base, name):
    index, index_key = galsim.config.GetIndex(config, base)
    rng = galsim.config.GetRNG(config, base)
    if base.get('_catalog_sampler_index', None) != index:
        catalog_sampler = galsim.config.GetInputObj(
            'catalog_sampler', config, base, name,
        )
        catalog_row_data, dtype, catalog_row_index = catalog_sampler.sample(rng)
        colnames = dtype.names
        base['_catalog_row_data_catalog_row_ind'] = catalog_row_index
        base['_catalog_row_data'] = catalog_row_data
        base['_catalog_sampler_index'] = index
        base['_catalog_colnames'] = colnames
        base['_catalog_used_rngnum'] = config.get("rng_num", None)
        print(
            "\n\n\n sampling catalog row info:",
            index,
            index_key,
            base['_catalog_row_data_catalog_row_ind'],
            config.get("rng_num", None),
            "\n\n\n",
            flush=True,
        )
    else:
        if base['_catalog_used_rngnum'] != config.get("rng_num", None):
            raise ValueError("Catalog sampler rng num changed from %s to %s!" % (
                base['_catalog_used_rngnum'], config.get("rng_num", None)
            ))

    return base['_catalog_row_data'], base['_catalog_colnames']


def CatalogValue(config, base, value_type):
    row_data, colnames = CatalogRow(config, base, value_type)
    col = galsim.config.ParseValue(config, 'col', base, str)[0]
    try:
        return float(row_data[colnames.index(col)])
    except ValueError as e:
        print("%s not in colnames" % col)
        raise e


galsim.config.RegisterInputType('catalog_sampler', CatalogSamplerLoader(CatalogSampler))
galsim.config.RegisterValueType(
    'catalog_sampler_value', CatalogValue, [float], input_type='catalog_sampler')
