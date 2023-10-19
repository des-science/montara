import galsim


def EvalGSObject(config, base, ignore, gsparams, logger):
    req = {'str': str}
    params, safe = galsim.config.GetAllParams(config, base, req=req, ignore=ignore)
    return galsim.utilities.math_eval(params['str']).withGSParams(**gsparams), safe


galsim.config.RegisterObjectType('Eval', EvalGSObject)
