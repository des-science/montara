import galsim
import numpy as np


def test_build_gsobject_evalrepr():
    g = galsim.Gaussian(fwhm=1.0).drawImage(nx=25, ny=25, scale=0.263)
    ii = galsim.InterpolatedImage(g, x_interpolant='lanczos15')
    with np.printoptions(threshold=np.inf, precision=32):
        r = repr(ii)
    cfg = {
        "type": "Eval",
        "str": r.replace("array(", "np.array("),
    }
    rii = galsim.config.BuildGSObject({'blah': cfg}, 'blah')
    assert rii == ii
