from galsim.config.extra import ExtraOutputBuilder, RegisterExtraOutput
from galsim.config.value import ParseValue
import galsim.fits


class BadPixFromFitsBuilder(ExtraOutputBuilder):
    # The function to call at the end of building each image
    def processImage(self, index, obj_nums, config, base, logger):
        mask_file = ParseValue(config, 'mask_file', base, str)[0]
        mask_hdu = ParseValue(config, 'mask_hdu', base, int)[0]

        mask_image = galsim.fits.read(mask_file, hdu=mask_hdu)

        if 'bits_to_null' in config:
            b2nlist = ParseValue(config, 'bits_to_null', base, list)[0]
            b2n = 0
            for tb2n in b2nlist:
                b2n |= tb2n

            mask_image.array[:, :] &= ~b2n

        self.data[index] = mask_image


# Register this as a valid extra output
RegisterExtraOutput('badpixfromfits', BadPixFromFitsBuilder())
