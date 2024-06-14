import logging

import fitsio
import galsim
import numpy as np
from galsim.config.input import InputLoader, RegisterInputType, GetInputObj
from galsim.config.value import RegisterValueType

from .utils import add_field

logger = logging.getLogger("pipeline")


class DESGalCatalog(object):
    _req_params = {'file_name': str}
    _opt_params = {'cuts': dict, 'verbose': bool}
    _single_params = []
    _takes_rng = False

    def __init__(self, file_name, cuts=None, _nobjects_only=False, verbose=False):
        self.gal_data = fitsio.read(file_name)
        self.gal_data = add_field(
            self.gal_data,
            [("catalog_row", int)],
            [np.arange(len(self.gal_data))],
        )
        self.cuts = cuts
        if cuts is not None:
            self.apply_cuts(cuts, verbose=verbose)
        self.nobjects = len(self.gal_data)

    def apply_cuts(self, cuts, verbose=False):
        """Apply some cuts.

         - `cuts` is a dictionary
         - A key should be the same as a column in
           the catalog.
         - The value corresponding to that cut can either be
           a single integer, in which case that value will be retained
           e.g. "flags":0 would just keep objects with flags==0.
           Or it can be a range e.g. "hlr":[a,b] in which case only
           objects with a<=hlr<b will be retained.
        """

        use = np.ones(len(self.gal_data), dtype=bool)

        for key, val in cuts.items():
            col_data = self.gal_data[key]

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

        self.gal_data = self.gal_data[use]

    def get(self, index, col):
        return self.gal_data[index][col]

    def getNObjects(self):
        return self.nobjects


def _GenerateFromDESGalCatalog(config, base, value_type):
    """@brief Return a value read from an input catalog
    """
    gal_input = GetInputObj('desgal', config, base, 'DESGalValue')

    # Setup the indexing sequence if it hasn't been specified.
    # The normal thing with a Catalog is to just use each object in order,
    # so we don't require the user to specify that by hand.  We can do it for them.
    galsim.config.SetDefaultIndex(config, gal_input.getNObjects())

    req = {'col': str, 'index': int}
    opt = {'num': int}
    kwargs, safe = galsim.config.GetAllParams(config, base, req=req, opt=opt)
    col = kwargs['col']
    index = kwargs['index']

    logger.log(
        logging.DEBUG,
        "sampling fixed gal catalog band|index|col: %s %s %s" % (
            base["eval_variables"]["sband"],
            index,
            col,
        ),
    )

    val = gal_input.get(index, col)
    return val, safe


RegisterInputType('desgal', InputLoader(DESGalCatalog, has_nobj=True))
RegisterValueType(
    'DESGalValue',
    _GenerateFromDESGalCatalog,
    [float],
    input_type='desgal',
)
