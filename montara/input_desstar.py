import galsim
import numpy as np
from galsim.config.input import InputLoader, RegisterInputType, GetInputObj
from galsim.config.value import RegisterValueType
import fitsio
from .utils import add_field


class DESStarCatalog(object):
    _req_params = {'file_name': str}
    _opt_params = {'mag_i_col': str, 'mag_i_max': float, 'mag_i_min': float}
    _single_params = []
    _takes_rng = False

    def __init__(self, file_name, mag_i_col='mag_i', mag_i_max=None, mag_i_min=None,
                 _nobjects_only=False):
        # Read in fits star catalog and apply cuts if necessary
        self.star_data = fitsio.read(file_name)
        self.star_data = add_field(self.star_data, [("catalog_row", int)], [
                                   np.arange(len(self.star_data))])

        # optionally apply an i-band magnitude cut
        use = np.ones(len(self.star_data), dtype=bool)
        if mag_i_max is not None:
            use[self.star_data[mag_i_col] > mag_i_max] = False
        if mag_i_min is not None:
            use[self.star_data[mag_i_col] < mag_i_min] = False
        self.star_data = self.star_data[use]
        self.nobjects = len(self.star_data)

    def get(self, index, col):
        return self.star_data[index][col]

    def getNObjects(self): return self.nobjects

# Now define the value generators connected to the catalog and dict input types.


def _GenerateFromStarCatalog(config, base, value_type):
    """@brief Return a value read from an input catalog
    """
    star_input = GetInputObj('desstar', config, base, 'DESStarValue')

    # Setup the indexing sequence if it hasn't been specified.
    # The normal thing with a Catalog is to just use each object in order,
    # so we don't require the user to specify that by hand.  We can do it for them.
    galsim.config.SetDefaultIndex(config, star_input.getNObjects())

    req = {'col': str, 'index': int}
    opt = {'num': int}
    kwargs, safe = galsim.config.GetAllParams(config, base, req=req, opt=opt)
    col = kwargs['col']
    index = kwargs['index']
    val = star_input.get(index, col)
    return val, safe


RegisterInputType('desstar', InputLoader(DESStarCatalog, has_nobj=True))
RegisterValueType('DESStarValue', _GenerateFromStarCatalog,
                  [float], input_type='desstar')
