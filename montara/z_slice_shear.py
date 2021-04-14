import galsim
from galsim.config.value import GetAllParams, RegisterValueType


def _GenerateFromG1G2_ZSlice(config, base, value_type):
    """Return a shear constructed from (g1_slice, g2_slice) when the redshift is within some
    range [zlow, zhigh], otherwise return (g1_other, g2_other) (which defaults to (0,0))
    """
    req = {
        "g1_slice": float, "g2_slice": float,
        "zlow": float, "zhigh": float,
        "z": float,
    }
    opt = {"g1_other": float, "g2_other": float}
    params, safe = GetAllParams(config, base, req=req, opt=opt)
    if "g1_other" not in params:
        params["g1_other"] = 0.
    if "g2_other" not in params:
        params["g2_other"] = 0.
    if ((params["z"] > params["zlow"]) and (params["z"] <= params["zhigh"])):
        return galsim.Shear(g1=params["g1_slice"], g2=params["g2_slice"])
    elif params["z"] <= 0.:
        return galsim.Shear(g1=0., g2=0.)
    else:
        return galsim.Shear(g1=params["g1_other"], g2=params["g2_other"])


RegisterValueType('G1G2_zslice', _GenerateFromG1G2_ZSlice, [galsim.Shear])
