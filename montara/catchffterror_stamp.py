import galsim
from galsim_extra.mixed_scene import MixedSceneBuilder


class CatchFFTErrorStampBuilder(MixedSceneBuilder):
    def draw(self, prof, image, method, offset, config, base, logger):
        try:
            return super(CatchFFTErrorStampBuilder, self).draw(
                prof, image, method, offset, config, base, logger
            )
        except galsim.GalSimFFTSizeError as e:
            logger.error('Caught FFT size error: %s', repr(e))
            logger.error('when drawing object:\n %s', repr(prof))
            logger.error('trying to increase maximum_fft_size to %s', repr(e.size))
            # ... or not.  Maybe do something else here.  Like just re-raise
            p2 = prof.withGSParams(galsim.GSParams(maximum_fft_size=e.size))
            return super(CatchFFTErrorStampBuilder, self).draw(
                p2, image, method, offset, config, base, logger
            )


galsim.config.stamp.RegisterStampType('CatchFFTError', CatchFFTErrorStampBuilder())
