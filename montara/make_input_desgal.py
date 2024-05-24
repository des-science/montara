from contextlib import contextmanager
import time
import sys

import esutil
import numpy as np
import galsim as gs
import hpgeom

from des_y6utils.mdet import _read_hsp_file, _compute_dered_flux_fac

GLOBAL_START_TIME = time.time()
SILENT = True


@contextmanager
def timer(name, silent=SILENT):
    """A simple timer context manager.

    Exmaple usage:

    >>> with timer("sleeping"):
    ...     time.sleep(1)

    Parameters
    ----------
    name : str
        The name of the timer to print.
    silent : bool, optional
        If True, do not print to stderr/stdout.
    """
    t0 = time.time()
    if not silent:
        print(
            "[% 8ds] %s" % (t0 - GLOBAL_START_TIME, name),
            flush=True,
            file=sys.stderr,
        )
    yield
    t1 = time.time()
    if not silent:
        print(
            "[% 8ds] %s done (%f seconds)" % (t1 - GLOBAL_START_TIME, name, t1 - t0),
            flush=True,
            file=sys.stderr,
        )


def match_cosmos_to_cardinal(cosmos, cardinal, *, max_dz, rng):
    """Match a cosmos catalog to cardinal.

    The matching is done by breaking the catalog up into redshift bins of size `max_dz`.
    Within each redshift bin, we use an abundance matching-like scheme to find the best
    cardinal galaxy for each cosmos galaxy. We sort both catalogs by magnitude and match
    the sorted lists, element by element. Assuming the cardinal catalog has approximately
    the same density and area as the input cosmos catalog, this scheme is roughly
    equivalent to abundance matching.

    The cardinal simulation typically stops at redshift 2.3. Objects at a redshift higher than
    this are matched at random to objects at the faint end of the cardinal catalog (fainter
    than 25th magnitude).

    Parameters
    ----------
    cosmos : np.ndarray
        The cosmos catalog to match to cardinal. Needs to have columns `photoz` and
        `mag_i_dered`.
    cardinal : np.ndarray
        The cardinal catalog to match to. Needs to have columns `Z` and `TMAG`.
    max_dz : float
        The starting redshift bin size to use. Values around 0.1 are reasonable.
        The code will use bigger bins if it can't find matches.
    rng : np.random.RandomState
        The random number generator to use.

    Returns
    -------
    match_inds : np.ndarray
        An array of indices into the cardinal catalog for each cosmos galaxy.
    match_flags : np.ndarray
        An array of flags indicating how the match was made.

            - 2**30 means no match was made
            - 2**0 means a match was made in the redshift bin
            - 2**1 means a match was made at random to a faint object
    """
    with timer("making catalogs"):
        # we cut the cosmos redshift range to match cardinal
        match_inds = np.zeros(len(cosmos), dtype=int) - 1
        match_flags = np.zeros(len(cosmos), dtype=np.int32) + 2**30
        msk_cosmos_to_match = (
            cosmos["photoz"] < 2.3
        )
        inds_msk_cosmos_to_match = np.where(msk_cosmos_to_match)[0]
        mcosmos = cosmos[msk_cosmos_to_match]

        # we remove very bright things from cardinal
        msk_cardinal_to_match_to = (
            cardinal["TMAG"][:, 2] > 17.0
        )
        if not np.any(msk_cardinal_to_match_to):
            raise ValueError("No cardinal objects to match to!")
        inds_msk_cardinal_to_match_to = np.where(msk_cardinal_to_match_to)[0]
        cardinal = cardinal[msk_cardinal_to_match_to]

        # this set holds galaxies we have used
        used_cd_inds = set()

    # now we match to cardinal in redshift bins and use ~abundance matching within each one
    # we gradually expand the bin size until everything is matched
    for facind in range(10):
        if np.all(match_inds[inds_msk_cosmos_to_match] >= 0):
            break

        # scale down the photoz match radius so that it has roughly the same size
        # in 1d as it had in n_features dimensions
        fac = 1.5**facind
        dz = max_dz * fac
        zbins = np.linspace(0, 2.3, int(np.ceil(2.3 / dz)) + 1)
        with timer("getting redshift bin matches w/ dz = %0.2f" % dz):
            for zind in range(zbins.shape[0] - 1):
                zmin = zbins[zind]
                zmax = zbins[zind + 1]
                cs_msk = (
                    (mcosmos["photoz"] >= zmin)
                    & (mcosmos["photoz"] < zmax)
                    & (match_inds[inds_msk_cosmos_to_match] == -1)
                )
                if np.any(cs_msk):
                    cd_msk = (
                        (cardinal["Z"] >= zmin)
                        & (cardinal["Z"] < zmax)
                    )
                    if np.any(cd_msk):
                        # sort cosmos by i-band magnitude so brightest get matches first
                        cs_inds = np.where(cs_msk)[0]
                        binds = np.argsort(mcosmos["mag_i_dered"][cs_inds])
                        cs_inds = cs_inds[binds]

                        # remove used cardinal inds
                        cd_inds = set(_ind for _ind in np.where(cd_msk)[0]) - used_cd_inds
                        cd_inds = np.fromiter(cd_inds, dtype=int)

                        # can match at most this many
                        max_match = min(len(cs_inds), len(cd_inds))

                        # select at random and sort by magnitude
                        binds = np.argsort(cardinal["TMAG"][cd_inds, 2])
                        cd_inds = cd_inds[binds]

                        # now assign
                        match_inds[inds_msk_cosmos_to_match[cs_inds[:max_match]]] \
                            = inds_msk_cardinal_to_match_to[cd_inds[:max_match]]
                        match_flags[inds_msk_cosmos_to_match[cs_inds[:max_match]]] = 2**0
                        used_cd_inds.update(cd_inds[:max_match])

        if not SILENT:
            print(
                "found matches for %0.2f percent of cosmos w/ z < 2.3" % (
                    np.sum(match_inds >= 0)
                    / inds_msk_cosmos_to_match.shape[0]
                    * 100
                ),
                flush=True,
            )

    with timer("assigning anything else to something at faint magnitude at random"):
        cs_msk = match_inds == -1
        cs_inds = np.where(cs_msk)[0]

        cd_inds = np.where(
            (cardinal["TMAG"][:, 2] >= 25)
        )[0]
        cd_inds = set(_ind for _ind in cd_inds) - used_cd_inds
        cd_inds = np.fromiter(cd_inds, dtype=int)

        rng.shuffle(cd_inds)

        binds = np.argsort(cosmos["mag_i_dered"][cs_msk])
        cs_inds = cs_inds[binds]

        # now assign the most we can
        max_match = min(len(cs_inds), len(cd_inds))
        match_inds[cs_inds[:max_match]] = inds_msk_cardinal_to_match_to[cd_inds[:max_match]]
        match_flags[cs_inds[:max_match]] = 2**1

    if not SILENT:
        print(
            "found matches for %0.2f percent of cosmos" % (
                np.sum(match_inds >= 0)
                / match_inds.shape[0]
                * 100
            ),
            flush=True,
        )

    assert np.all(match_inds >= 0)
    assert np.unique(match_inds).shape[0] == match_inds.shape[0]

    return match_inds, match_flags


def project_to_tile(ra, dec, cen_ra, cen_dec, tile_wcs):
    """Project points at ra,dec about cen_ra,cen_dec to the same relative position
    in the tile_wcs.

    Parameters
    ----------
    ra : np.ndarray
        The right ascension of the points to project in decimal degrees.
    dec : np.ndarray
        The declination of the points to project in decimal degrees.
    cen_ra : float
        The right ascension of the center of the region to project in decimal degrees.
    cen_dec : float
        The declination of the center of the region to project in decimal degrees.
    tile_wcs : galsim.TanWCS
        The WCS of the DES coadd tile to project to.

    Returns
    -------
    new_ra : np.ndarray
        The right ascension of the projected points in decimal degrees.
    new_dec : np.ndarray
        The declination of the projected points in decimal degrees.
    new_x : np.ndarray
        The x pixel coordinate of the projected points in the tile_wcs.
    new_y : np.ndarray
        The y pixel coordinate of the projected points in the tile_wcs.
    """
    # build the DES-style coadd WCS at center ra,dec
    aft = gs.AffineTransform(
        -0.263,
        0.0,
        0.0,
        0.263,
        origin=gs.PositionD(5000.5, 5000.5),
    )
    cen = gs.CelestialCoord(ra=cen_ra * gs.degrees, dec=cen_dec * gs.degrees)
    cen_wcs = gs.TanWCS(aft, cen, units=gs.arcsec)

    # project to u,v about cen - needs radians as input
    u, v = cen_wcs.center.project_rad(
        np.radians(ra),
        np.radians(dec)
    )

    # now deproject about tile_wcs - comes out in radians
    new_ra, new_dec = tile_wcs.center.deproject_rad(u, v)
    new_ra, new_dec = np.degrees(new_ra), np.degrees(new_dec)
    x, y = tile_wcs.radecToxy(new_ra, new_dec, units="degrees")

    return new_ra, new_dec, x, y


def sample_from_pixel(nside, pix, size=None, nest=True, rng=None):
    """Sample a point on the sky randomly from a pixel.

    Parameters
    ----------
    nside : int
        The nside of the healpix grid.
    pix : int
        The pixel to sample from.
    size : tuple, optional
        The size of the output. If None, a single point is returned.
    nest : bool, optional
        Whether the healpix pixel index `pix` is in the NEST or RING scheme.
    rng : np.random.RandomState or int, optional
        The random number generator to use. If an integer is passed, a new
        random number generator is created with the seed set to this value.

    Returns
    -------
    ra : float or np.ndarray
        The right ascension of the sampled point(s) in decimal degrees.
    dec : float or np.ndarray
        The declination of the sampled point(s) in decimal degrees.
    """
    if not isinstance(rng, np.random.RandomState):
        rng = np.random.RandomState(seed=rng)

    if size is None:
        ntot = 1
        scalar = True
    else:
        ntot = np.prod(size)
        scalar = False

    ra, dec = hpgeom.pixel_to_angle(
        nside,
        pix,
        nest=nest,
    )
    ddec = 2.0 * hpgeom.nside_to_resolution(nside)
    dec_range = np.array([
        dec - ddec,
        dec + ddec,
    ])
    dec_range = np.clip(dec_range, -90, 90)
    cosdec = np.cos(np.radians(dec))
    dra = ddec / cosdec
    ra_range = np.array([
        ra - dra,
        ra + dra
    ])
    sin_dec_range = np.sin(np.radians(dec_range))

    ra = np.empty(ntot)
    dec = np.empty(ntot)
    nsofar = 0
    while nsofar < ntot:
        ngen = int(1.5 * min(ntot, ntot - nsofar))
        _ra = rng.uniform(low=ra_range[0], high=ra_range[1], size=ngen)
        _sindec = rng.uniform(low=sin_dec_range[0], high=sin_dec_range[1], size=ngen)
        _dec = np.degrees(np.arcsin(_sindec))
        inds = np.where(hpgeom.angle_to_pixel(nside, _ra, _dec, nest=nest) == pix)[0]
        _n = inds.shape[0]
        _n = min(_n, ntot - nsofar)
        if _n > 0:
            ra[nsofar:nsofar+_n] = _ra[inds[:_n]]
            dec[nsofar:nsofar+_n] = _dec[inds[:_n]]
            nsofar += _n

    if scalar:
        return ra[0], dec[0]
    else:
        return np.reshape(ra, size), np.reshape(dec, size)


def _get_tile_bounds_at_point(cen_ra, cen_dec, buff=0):
    # build the DES-style coadd WCS at center ra,dec
    aft = gs.AffineTransform(
        -0.263,
        0.0,
        0.0,
        0.263,
        origin=gs.PositionD(5000.5, 5000.5),
    )
    cen = gs.CelestialCoord(ra=cen_ra * gs.degrees, dec=cen_dec * gs.degrees)
    wcs = gs.TanWCS(aft, cen, units=gs.arcsec)

    xv = np.array([1 - buff, 1 - buff, 10000 + buff, 10000 + buff])
    yv = np.array([1 - buff, 10000 + buff, 10000 + buff, 1 - buff])

    rav, decv = wcs.xyToradec(xv, yv, units="degrees")
    return rav, decv


def _ratio_mag_v3(mag, *coeffs):
    if len(coeffs) == 0:
        coeffs = [
            -1.06490672e+02,
            1.74424846e+00,
            -1.17390019e-02,
            3.80657424e-05,
            -5.87562207e-08,
            3.47153346e-11,
        ]
    poly = np.zeros_like(mag)
    for i, c in enumerate(coeffs):
        poly += c * mag**(2*i)
    return 1.0 / (1.0 + np.exp(-poly))


def _ratio_mag_v45(mag, *coeffs):
    if len(coeffs) == 0:
        coeffs = [-3.05005992e+01,  1.29795854e-01, -1.39023858e-04,  4.33811478e-08]
    x = mag
    poly = np.zeros_like(x)
    for i, c in enumerate(coeffs):
        poly += c * x**(2*i)
    return 1.0 / (1.0 + np.exp(-poly))


def _ratio_mag_v6(mag, *coeffs):
    if len(coeffs) == 0:
        coeffs = [-4.81181605e+01,  2.49037882e-01, -3.96914217e-04,  2.16742025e-07]
    x = mag
    poly = np.zeros_like(x)
    for i, c in enumerate(coeffs):
        poly += c * x**(2*i)
    return 1.0 / (1.0 + np.exp(-poly))


ratio_mag = _ratio_mag_v6


def _get_cosmos_renorm(cosmos):
    bins = np.linspace(17, 27, 100)
    hcosmos = np.histogram(cosmos["mag_i_dered"], bins=bins, density=True)[0]
    _mag = (bins[:-1] + bins[1:]) / 2
    return np.trapz(hcosmos / ratio_mag(_mag), _mag) / np.trapz(hcosmos, _mag)


def make_input_cosmos_cat(
    *,
    cosmos,
    sim,
    nside,
    pix,
    seed,
    wcs,
    dz,
    dustmap_fname="SFD_dust_4096.hsp",
):
    """Make a cosmos catalog to input to the DES Y6 image simulations.

    Parameters
    ----------
    cosmos : np.ndarray
        The cosmos catalog to use. Needs to have columns `mag_i`, `mag_i_dered`,
        `isgal`, `mask_flags`, and `bdf_hlr`.
    sim : np.ndarray
        The cardinal catalog to use. Needs to have columns `TRA`, `TDEC`, `Z`, `TMAG`.
    nside : int
        The nside of the healpix grid for the sim catalog.
    pix : int
        The healpix pixel index of the sim catalog in the NEST scheme.
    seed : int
        The seed for the random number generator.
    wcs : galsim.TanWCS
        The WCS of the DES coadd tile for which the sim catalog is intended.
    dz : float
        The starting size of the redshift bins for matching between cosmos and the sim.
    dustmap_fname : str, optional
        The filename of the dustmap to use. This will be downloaded if it is not present locally.

    Returns
    -------
    final_tcat : np.ndarray
        The cosmos catalog with the following additional columns:

            - ra_sim : float
                The right ascension of the object in the sim catalog in decimal degrees.
            - dec_sim : float
                The declination of the object in the sim catalog in decimal degrees.
            - x_sim : float
                The x pixel coordinate of the object in the sim catalog in the tile WCS.
            - y_sim : float
                The y pixel coordinate of the object in the sim catalog in the tile WCS.
            - match_type_sim : int
                The type of match made to the sim catalog. 2**30 means no match was made,
                2**0 means a match was made in the redshift bin, and 2**1 means a match was
                made at random to a faint object.
            - mag_g_red_sim : float
                The reddened g-band magnitude of the object in the sim catalog.
            - mag_r_red_sim : float
                The reddened r-band magnitude of the object in the sim catalog.
            - mag_i_red_sim : float
                The reddened i-band magnitude of the object in the sim catalog.
            - mag_z_red_sim : float
                The reddened z-band magnitude of the object in the sim catalog.
            - hpix_sim : int
                The healpix pixel index of the sim catalog in the NEST scheme.
            - cat_index_sim : int
                The index of the object in the sim catalog.
    """
    rng = np.random.RandomState(seed=seed)

    #################################################
    # first we cut cosmos to what we want

    oversampling_factor = _get_cosmos_renorm(cosmos)

    # we scale the input # of objects to draw by the value for a DES coadd tile
    # at the default cuts for Y3
    cmsk = (
        (cosmos["mag_i"] > 15)
        & (cosmos["mag_i"] <= 25)
        & (cosmos["isgal"] == 1)
        & (cosmos["mask_flags"] == 0)
        & (cosmos["bdf_hlr"] > 0)
        & (cosmos["bdf_hlr"] <= 5)
    )
    n_cuts_orig = np.sum(cmsk)
    n_draw_orig = 170_000

    # for Y6 we are going a bit deeper in the cosmos catalog
    cmsk = (
        (cosmos["mag_i"] > 15)
        & (cosmos["mag_i"] <= 25.4)
        & (cosmos["isgal"] == 1)
        & (cosmos["mask_flags"] == 0)
        & (cosmos["bdf_hlr"] > 0)
        & (cosmos["bdf_hlr"] <= 5)
    )
    cosmos = cosmos[cmsk]
    n_draw = int(np.ceil(cosmos.shape[0] / n_cuts_orig * n_draw_orig * oversampling_factor))

    ndraw = rng.poisson(n_draw)
    prob = 1.0 / ratio_mag(cosmos["mag_i_dered"])
    prob /= np.sum(prob)
    cinds = rng.choice(cosmos.shape[0], size=ndraw, p=prob, replace=True)
    tcat = cosmos[cinds]

    # now we get a random point in the cardinal data that has a coadd tile contained
    # in the input healpixel
    while True:
        rac, decc = sample_from_pixel(nside, pix, rng=rng)
        rav, decv = _get_tile_bounds_at_point(rac, decc, buff=100)
        pixv = hpgeom.angle_to_pixel(nside, rav, decv)
        if np.all(pixv == pix):
            break

    # we project the sim at the random input point to the data tile WCS location
    new_ra, new_dec, new_x, new_y = project_to_tile(
        sim["TRA"],
        sim["TDEC"],
        rac,
        decc,
        wcs,
    )

    # match cosmos to sim using only stuff that falls in the tile
    dx_ccd = 4096
    dy_ccd = 2048
    tmsk = (
        (new_x >= 0.5 - dx_ccd)
        & (new_x <= 10000.5 + dx_ccd)
        & (new_y >= 0.5 - dy_ccd)
        & (new_y <= 10000.5 + dy_ccd)
    )
    inds_tmsk = np.where(tmsk)[0]

    match_inds, match_flags = match_cosmos_to_cardinal(tcat, sim[tmsk], max_dz=dz, rng=rng)
    match_inds = inds_tmsk[match_inds]

    # build final catalog with new positions and reddened fluxes
    final_tcat = esutil.numpy_util.add_fields(
        tcat,
        [
            ("ra_sim", "f8"),
            ("dec_sim", "f8"),
            ("x_sim", "f8"),
            ("y_sim", "f8"),
            ("match_type_sim", "i4"),
            ("mag_g_red_sim", "f8"),
            ("mag_r_red_sim", "f8"),
            ("mag_i_red_sim", "f8"),
            ("mag_z_red_sim", "f8"),
            ("hpix_sim", "i8"),
            ("cat_index_sim", "i8"),
        ]
    )
    final_tcat["ra_sim"] = new_ra[match_inds]
    final_tcat["dec_sim"] = new_dec[match_inds]
    final_tcat["x_sim"] = new_x[match_inds]
    final_tcat["y_sim"] = new_y[match_inds]
    final_tcat["match_type_sim"] = match_flags
    final_tcat["hpix_sim"] = pix
    final_tcat["cat_index_sim"] = match_inds

    # redden the fluxes
    bands = ['g', 'r', 'i', 'z']
    dustmap = _read_hsp_file("SFD_dust_4096.hsp")

    dered = dustmap.get_values_pos(final_tcat["ra_sim"], final_tcat["dec_sim"])
    for ii, b in enumerate(bands):
        dered_fac = _compute_dered_flux_fac(ii, dered)
        red_shift = 2.5 * np.log10(dered_fac)
        final_tcat["mag_" + b + "_red_sim"] = final_tcat["mag_" + b + "_dered"] + red_shift

    return final_tcat
