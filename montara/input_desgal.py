import galsim
import numpy as np
from galsim.config.input import InputLoader, RegisterInputType, GetInputObj
from galsim.config.value import RegisterValueType
import fitsio
from .utils import add_field


class DESGalCatalog(object):
    _req_params = {'file_name': str}
    _single_params = []
    _takes_rng = False

    def __init__(self, file_name, mag_i_col='mag_i', mag_i_max=None, mag_i_min=None,
                 _nobjects_only=False):
        self.gal_data = fitsio.read(file_name)
        self.gal_data = add_field(
            self.gal_data, [("catalog_row", int)],
            [np.arange(len(self.gal_data))],
        )
        self.nobjects = len(self.gal_data)

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
    val = gal_input.get(index, col)
    return val, safe


RegisterInputType('desgal', InputLoader(DESGalCatalog, has_nobj=True))
RegisterValueType(
    'DESGalValue',
    _GenerateFromDESGalCatalog,
    [float],
    input_type='desgal',
)
