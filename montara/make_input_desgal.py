from contextlib import contextmanager
import glob
import time
import sys

import click
import esutil
import numpy as np
import fitsio
import galsim as gs
import hpgeom
from scipy.spatial import KDTree
from esutil.pbar import PBar

from des_y6utils.mdet import _read_hsp_file, _compute_dered_flux_fac

GLOBAL_START_TIME = time.time()


@contextmanager
def timer(name, silent=False):
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


def match_cosmos_to_cardinal(cosmos, cardinal, *, max_dz, max_di, max_dgmi, rng):
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
        inds_msk_cardinal_to_match_to = np.where(msk_cardinal_to_match_to)[0]
        cardinal = cardinal[msk_cardinal_to_match_to]

        # we match in i-band, g-i color, and photo-z
        # we scale the photo-z by 3 to match the typical range of magnitudes
        # this has the side effect of equating differences of 0.1 in photo-z to 0.3
        # in magnitudes or colors, which is about right
        n_features = 3
        di_scale_factor = max_di / max_dz
        dgmi_scale_factor = max_dgmi / max_dz
        cd_data = np.zeros((cardinal.shape[0], n_features))
        cd_data[:, 0] = cardinal["Z"]
        # i-band
        cd_data[:, 1] = cardinal["TMAG"][:, 2] / di_scale_factor
        # g-i color
        cd_data[:, 2] = (cardinal["TMAG"][:, 0] - cardinal["TMAG"][:, 2]) / dgmi_scale_factor

        cs_data = np.zeros((mcosmos.shape[0], n_features))
        cs_data[:, 0] = mcosmos["photoz"]
        cs_data[:, 1] = mcosmos["mag_i_dered"] / di_scale_factor
        cs_data[:, 2] = (mcosmos["mag_g_dered"] - mcosmos["mag_i_dered"]) / dgmi_scale_factor

    used_cd_inds = set()
    all_cd_inds = np.arange(cd_data.shape[0], dtype=int)

    # now find the closest nbrs in cardinal to each cosmos object
    for fac_ind, fac in enumerate([1]):
        with timer("doing radius search for fac %0.2f" % fac):
            cd_avail_msk = np.isin(all_cd_inds, used_cd_inds, invert=True)
            inds_cd_avail = np.where(cd_avail_msk)[0]

            tree = KDTree(cd_data[cd_avail_msk, :])

            fac_msk = match_inds[inds_msk_cosmos_to_match] == -1
            inds_fac_msk = np.where(fac_msk)[0]

        with timer("getting best matches for fac %0.2f" % fac):
            # do brightest things first
            binds = np.argsort(mcosmos["mag_i_dered"][fac_msk])

            chunk_size = 1000
            n_chunks = int(np.ceil(inds_fac_msk.shape[0] / chunk_size))
            for chunk_ind in PBar(range(n_chunks), desc="getting best matches for fac %0.2f" % fac):
                chunk_start = chunk_ind * chunk_size
                chunk_end = min((chunk_ind + 1) * chunk_size, inds_fac_msk.shape[0])
                nbr_inds = tree.query_ball_point(
                    cs_data[inds_fac_msk[binds[chunk_start:chunk_end]], :],
                    max_dz * fac,
                    eps=0.5,
                    return_sorted=True,
                    workers=-1,
                )

                # now we loop over the cosmos objects and find the closest cardinal object
                # that has not been matched yet
                for offset, bind in enumerate(binds[chunk_start:chunk_end]):
                    i = inds_fac_msk[bind]

                    if match_inds[inds_msk_cosmos_to_match[i]] >= 0:
                        continue

                    cd_nbr_inds = [inds_cd_avail[ind] for ind in nbr_inds[offset]]

                    allowed_inds = [ind for ind in cd_nbr_inds if ind not in used_cd_inds and ind < cd_data.shape[0]]
                    if allowed_inds:
                        min_ind = allowed_inds[0]
                        used_cd_inds.add(min_ind)
                        match_inds[inds_msk_cosmos_to_match[i]] = inds_msk_cardinal_to_match_to[min_ind]
                        match_flags[inds_msk_cosmos_to_match[i]] |= 2**0

        print(
            "found matches for %0.2f percent of cosmos w/ z < 2.3 at fac %0.2f" % (
                np.sum(match_inds >= 0)
                / inds_msk_cosmos_to_match.shape[0]
                * 100,
                fac,
            ),
            flush=True,
        )

    # anything that didn't match gets a position from the same redshift in slices
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
                        # remove used inds
                        cd_inds = set(_ind for _ind in np.where(cd_msk)[0]) - used_cd_inds
                        cd_inds = np.fromiter(cd_inds, dtype=int)
                        # rng.shuffle(cd_inds)
                        binds = np.argsort(cardinal["TMAG"][cd_inds, 2])
                        cd_inds = cd_inds[binds]

                        # sort cosmos by i-band magnitude so brightest get matches first
                        cs_inds = np.where(cs_msk)[0]
                        binds = np.argsort(mcosmos["mag_i_dered"][cs_inds])
                        cs_inds = cs_inds[binds]

                        # now assign the most we can
                        max_match = min(len(cs_inds), len(cd_inds))
                        match_inds[inds_msk_cosmos_to_match[cs_inds[:max_match]]] = inds_msk_cardinal_to_match_to[cd_inds[:max_match]]
                        match_flags[inds_msk_cosmos_to_match[cs_inds[:max_match]]] |= 2**1
                        used_cd_inds.update(cd_inds[:max_match])

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
        match_flags[cs_inds[:max_match]] |= 2**2

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


def make_input_cosmos_cat(
    *,
    cosmos,
    sim,
    nside,
    pix,
    seed,
    wcs,
    dz,
    di,
    dgmi,
    dustmap_fname="SFD_dust_4096.hsp",
):
    """
    """
    rng = np.random.RandomState(seed=seed)

    #################################################
    # first we cut cosmos to what we want

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
    n_draw = int(np.ceil(cosmos.shape[0] / n_cuts_orig * n_draw_orig))

    ndraw = rng.poisson(n_draw)
    cinds = rng.choice(cosmos.shape[0], size=ndraw, replace=True)
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

    match_inds, match_flags = match_cosmos_to_cardinal(tcat, sim[tmsk], max_dz=dz, max_di=di, max_dgmi=dgmi, rng=rng)
    match_inds = inds_tmsk[match_inds]

    # build final catalog with new positions and reddened fluxes
    final_tcat = esutil.numpy_util.add_fields(
        tcat,
        [
            ("ra_sim", "f8"),
            ("dec_sim", "f8"),
            ("x_sim", "f8"),
            ("y_sim", "f8"),
            ("sim_match_flags", "i4"),
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
    final_tcat["sim_match_type"] = match_flags
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


@click.command()
@click.option("--cosmos-cat", type=str, required=True, help="Path to the cosmos catalog.")
@click.option("--sim-glob", type=str, required=True, help="Glob pattern for the sim catalogs.")
@click.option("--seed", type=int, required=True, help="Seed for the RNG")
@click.option("--coadd-tile", type=str, required=True, help="Path to the coadd tile.")
@click.option("--dz", type=float, default=0.1, help="Redshift matching radius.")
@click.option("--di", type=float, default=0.5, help="i-band magnitude matching radius.")
@click.option("--dgmi", type=float, default=0.5, help="g-i color matching radius.")
@click.option("--output", type=str, required=True, help="Path to which to write the output catalog.")
def make_input_cosmos_cat_cli(cosmos_cat, sim_glob, seed, coadd_tile, dz, di, dgmi, output):
    """Make a cosmos catalog for input to a montara sim."""
    all_sim_files = glob.glob(sim_glob)
    rng = np.random.RandomState(seed=seed)
    findex = rng.choice(len(all_sim_files))
    sim = fitsio.read(all_sim_files[findex])
    nside = 8
    pix = int(all_sim_files[findex].split(".")[-2])

    output_catalog = make_input_cosmos_cat(
        cosmos=fitsio.read(cosmos_cat),
        sim=sim,
        nside=nside,
        pix=pix,
        seed=seed,
        wcs=gs.FitsWCS(coadd_tile, hdu=1),
        dz=dz,
        di=di,
        dgmi=dgmi,
        dustmap_fname="SFD_dust_4096.hsp",
    )

    with timer("writing output"):
        fitsio.write(output, output_catalog, clobber=True)
