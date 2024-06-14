import galsim
import hpgeom
import numpy as np

from montara.make_input_desgal import project_to_tile, sample_from_pixel


COADD_WCS = galsim.GSFitsWCS(header=galsim.FitsHeader(header={
    'xtension': 'BINTABLE',
    'bitpix': 8,
    'naxis': 2,
    'naxis1': 24,
    'naxis2': 10000,
    'pcount': 83807884,
    'gcount': 1,
    'tfields': 3,
    'ttype1': 'COMPRESSED_DATA',
    'tform1': '1PB(8590)',
    'ttype2': 'ZSCALE  ',
    'tform2': '1D      ',
    'ttype3': 'ZZERO   ',
    'tform3': '1D      ',
    'zimage': True,
    'ztile1': 10000,
    'ztile2': 1,
    'zcmptype': 'RICE_ONE',
    'zname1': 'BLOCKSIZE',
    'zval1': 32,
    'zname2': 'BYTEPIX ',
    'zval2': 4,
    'zsimple': True,
    'zbitpix': -32,
    'znaxis': 2,
    'znaxis1': 10000,
    'znaxis2': 10000,
    'zextend': True,
    'extname': 'SCI     ',
    'equinox': 2000.0,
    'mjd-obs': 56545.15853046,
    'radesys': 'ICRS    ',
    'ctype1': 'RA---TAN',
    'cunit1': 'deg     ',
    'crval1': 320.688891,
    'crpix1': 5000.5,
    'cd1_1': -7.305555555556e-05,
    'cd1_2': 0.0,
    'ctype2': 'DEC--TAN',
    'cunit2': 'deg     ',
    'crval2': 0.016667,
    'crpix2': 5000.5,
    'cd2_1': 0.0,
    'cd2_2': 7.305555555556e-05,
    'exptime': 450.0,
    'gain': 19.93043199192,
    'saturate': 31274.22430892,
    'softname': 'SWarp   ',
    'softvers': '2.40.0  ',
    'softdate': '2016-09-19',
    'softauth': '2010-2012 IAP/CNRS/UPMC',
    'softinst': 'IAP  http://www.iap.fr',
    'author': 'unknown ',
    'origin': 'nid18189',
    'date': '2016-10-13T00:30:52',
    'combinet': 'WEIGHTED',
    'bunit': 'electrons',
    'filter': 'i DECam SDSS c0003 7835.0 1470.0',
    'band': 'i       ',
    'tilename': 'DES2122+0001',
    'tileid': 90126,
    'resampt1': 'LANCZOS3',
    'centert1': 'MANUAL  ',
    'pscalet1': 'MANUAL  ',
    'resampt2': 'LANCZOS3',
    'centert2': 'MANUAL  ',
    'pscalet2': 'MANUAL  ',
    'desfname': 'DES2122+0001_r2601p01_i.fits',
    'pipeline': 'multiepoch',
    'unitname': 'DES2122+0001',
    'attnum': 1,
    'eupsprod': 'MEPipeline',
    'eupsver': 'Y3A1+0  ',
    'reqnum': 2601,
    'des_ext': 'IMAGE   ',
    'fzalgor': 'RICE_1  ',
    'fzdthrsd': 'CHECKSUM',
    'fzqvalue': 16,
    'fzqmethd': 'SUBTRACTIVE_DITHER_2',
    'ra_cent': 320.688927527779,
    'dec_cent': 0.016630472222219,
    'rac1': 321.054126640963,
    'decc1': -0.348562220896466,
    'rac2': 320.323655359037,
    'decc2': -0.348562220896466,
    'rac3': 320.323654004521,
    'decc3': 0.381895543631536,
    'rac4': 321.054127995479,
    'decc4': 0.381895543631536,
    'racmin': 320.323654004521,
    'racmax': 321.054127995479,
    'deccmin': -0.348562220896466,
    'deccmax': 0.381895543631536,
    'crossra0': 'N       ',
    'magzero': 30.0,
    'history': "'SUBTRACTIVE_DITHER_2' / Pixel Quantization Algorithm",
    'zquantiz': 'SUBTRACTIVE_DITHER_2',
    'zdither0': 5591,
    'checksum': 'ZJHKaGGKUGGKZGGK',
    'datasum': 1452922543}))


def test_project_to_tile():
    tile_wcs = COADD_WCS

    cen_ra = 3.5467
    cen_dec = -55.231432

    new_ra, new_dec, new_x, new_y = project_to_tile(
        np.array([cen_ra]),
        np.array([cen_dec]),
        cen_ra,
        cen_dec,
        tile_wcs,
    )
    np.testing.assert_allclose(new_ra, tile_wcs.toWorld(new_x, new_y, units="degrees")[0])
    np.testing.assert_allclose(new_dec, tile_wcs.toWorld(new_x, new_y, units="degrees")[1])
    np.testing.assert_allclose(new_ra, tile_wcs.center.ra / galsim.degrees)
    np.testing.assert_allclose(new_dec, tile_wcs.center.dec / galsim.degrees)
    np.testing.assert_allclose(new_x, 5000.5)
    np.testing.assert_allclose(new_y, 5000.5)

    _new_ra, _new_dec, _new_x, _new_y = project_to_tile(
        np.array([cen_ra]),
        np.array([cen_dec + 0.263/3600.0]),
        cen_ra,
        cen_dec,
        tile_wcs,
    )
    np.testing.assert_allclose(new_x, _new_x)
    np.testing.assert_allclose(new_y + 1, _new_y)
    np.testing.assert_allclose(new_ra, _new_ra)
    np.testing.assert_allclose(new_dec + 0.263/3600.0, _new_dec)



def test_sample_from_pixel_seeding():
    nside = 8
    pix = 12

    rng = np.random.RandomState(1234)
    ra1, dec1 = sample_from_pixel(nside, pix, size=1000, nest=True, rng=rng)

    rng = np.random.RandomState(1234)
    ra2, dec2 = sample_from_pixel(nside, pix, size=1000, nest=True, rng=rng)
    ra3, dec3 = sample_from_pixel(nside, pix, size=1000, nest=True, rng=rng)

    np.testing.assert_allclose(ra1, ra2)
    np.testing.assert_allclose(dec1, dec2)
    assert not np.allclose(ra1, ra3)
    assert not np.allclose(dec1, dec3)


def test_sample_from_pixel_size():
    nside = 8
    pix = 12

    rng = np.random.RandomState(1234)
    ra1, dec1 = sample_from_pixel(nside, pix, size=1000, nest=True, rng=rng)
    assert ra1.shape == (1000,)
    assert dec1.shape == (1000,)

    ra2, dec2 = sample_from_pixel(nside, pix, size=1, nest=True, rng=rng)
    assert ra2.shape == (1,)
    assert dec2.shape == (1,)

    ra3, dec3 = sample_from_pixel(nside, pix, size=None, nest=True, rng=rng)
    assert isinstance(ra3, float)
    assert isinstance(dec3, float)


def test_sample_from_pixel_uniform():
    nside = 8
    pix = 12
    total = 1000000

    rng = np.random.RandomState(1234)
    ra1, dec1 = sample_from_pixel(nside, pix, size=total, nest=True, rng=rng)

    nside = 8
    _pix = hpgeom.angle_to_pixel(nside, ra1, dec1, nest=True, lonlat=True)
    assert np.all(_pix == pix)

    nside = 16
    _pix = hpgeom.angle_to_pixel(nside, ra1, dec1, nest=True, lonlat=True)
    vals = np.unique(_pix)
    assert len(vals) == 4
    nums = []
    for val in vals:
        assert np.sum(_pix == val) > 0
        nums.append(np.sum(_pix == val))
    np.testing.assert_allclose(nums, total / 4, atol=np.sqrt(total / 4) * 3, rtol=0)

    nside = 32
    _pix = hpgeom.angle_to_pixel(nside, ra1, dec1, nest=True, lonlat=True)
    vals = np.unique(_pix)
    assert len(vals) == 16
    nums = []
    for val in vals:
        assert np.sum(_pix == val) > 0
        nums.append(np.sum(_pix == val))
    np.testing.assert_allclose(nums, total / 16, atol=np.sqrt(total / 16) * 3, rtol=0)

    nside = 64
    _pix = hpgeom.angle_to_pixel(nside, ra1, dec1, nest=True, lonlat=True)
    vals = np.unique(_pix)
    assert len(vals) == 64
    nums = []
    for val in vals:
        assert np.sum(_pix == val) > 0
        nums.append(np.sum(_pix == val))
    np.testing.assert_allclose(nums, total / 64, atol=np.sqrt(total / 64) * 3, rtol=0)
