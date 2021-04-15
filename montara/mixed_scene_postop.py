from galsim_extra.mixed_scene import MixedSceneBuilder
from galsim.config.gsobject import TransformObject
from galsim.config.stamp import RegisterStampType


class MixedScenePostOpStampBuilder(MixedSceneBuilder):
    def setup(self, config, base, xsize, ysize, ignore, logger):
        ignore = ignore + ['psf_postop']
        return super(MixedScenePostOpStampBuilder, self).setup(
            config, base, xsize, ysize, ignore, logger,
        )

    def buildProfile(self, config, base, psf, gsparams, logger):
        # Change the psf appropriately
        if 'psf_postop' in config:
            psf, safe = TransformObject(psf, config['psf_postop'], base, logger)
        # Then call the normal buildProfile with the new psf object.
        return super(MixedScenePostOpStampBuilder, self).buildProfile(
            config, base, psf, gsparams, logger,
        )


RegisterStampType('MixedPostOp', MixedScenePostOpStampBuilder())
