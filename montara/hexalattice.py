# -----------------------------------------------------------
# hexalattice module creates and prints hexagonal lattices
#
# (C) 2020 Alex Kazakov,
# Released under MIT License
# email alex.kazakov@mail.huji.ac.il
# Full documentation: https://github.com/alexkaz2/hexalattice
# -----------------------------------------------------------
# copied by MRB from hexalattice under MIT

import numpy as np
from typing import List, Union, Any


def create_hex_grid(
    nx: int = 4,
    ny: int = 5,
    min_diam: float = 1.0,
    n: int = 0,
    align_to_origin: bool = True,
    face_color: Union[List[float], str] = None,
    edge_color: Union[List[float], str] = None,
    plotting_gap: float = 0.0,
    crop_circ: float = 0.0,
    do_plot: bool = False,
    rotate_deg: float = 0.0,
    keep_x_sym: bool = True,
    h_ax=None,
    line_width: float = 0.2,
    background_color: Union[List[float], str] = None,
) -> (np.ndarray, Any):
    """
    Creates and prints hexagonal lattices.
    :param nx: Number of horizontal hexagons in rectangular grid, [nx * ny]
    :param ny: Number of vertical hexagons in rectangular grid, [nx * ny]
    :param min_diam: Minimal diameter of each hexagon.
    :param n: Alternative way to create rectangular grid. The final grid might have less hexagons
    :param align_to_origin: Shift the grid s.t. the central tile will center at the origin
    :param face_color: Provide RGB triplet, valid abbreviation (e.g. 'k') or RGB+alpha
    :param edge_color: Provide RGB triplet, valid abbreviation (e.g. 'k') or RGB+alpha
    :param plotting_gap: Gap between the edges of adjacent tiles, in fraction of min_diam
    :param crop_circ: Disabled if 0. If >0 a circle of central tiles will be kept, with radius r=crop_circ
    :param do_plot: Add the hexagon to an axes. If h_ax not provided a new figure will be opened.
    :param rotate_deg: Rotate the grid around the center of the central tile, by rotate_deg degrees
    :param keep_x_sym: NOT YET IMPLEMENTED
    :param h_ax: Handle to axes. If provided the grid will be added to it, if not a new figure will be opened.
    :param line_width: The width of the hexagon lines
    :param background_color: The color of the axis background
    :return:
    """

    args_are_ok = check_inputs(
        nx,
        ny,
        min_diam,
        n,
        align_to_origin,
        face_color,
        edge_color,
        plotting_gap,
        crop_circ,
        do_plot,
        rotate_deg,
        keep_x_sym,
        background_color,
    )
    if not args_are_ok:
        print("Aborting hexagonal grid creation...")
        exit()
    coord_x, coord_y = make_grid(
        nx, ny, min_diam, n, crop_circ, rotate_deg, align_to_origin
    )

    if do_plot:
        raise RuntimeError("plotting has been removed!")

    return np.hstack([coord_x, coord_y]), h_ax


def check_inputs(
    nx,
    ny,
    min_diam,
    n,
    align_to_origin,
    face_color,
    edge_color,
    plotting_gap,
    crop_circ,
    do_plot,
    rotate_deg,
    keep_x_sym,
    background_color,
):
    """
    Validate input types, ranges and co-compatibility
    :return: bool - Assertion verdict
    """
    args_are_valid = True
    if (
        (not isinstance(nx, (int, float)))
        or (not isinstance(ny, (int, float)))
        or (not isinstance(n, (int, float)))
        or (nx < 0)
        or (nx < 0)
        or (nx < 0)
    ):
        print("Argument error in hex_grid: nx, ny and n are expected to be integers")
        args_are_valid = False

    if (
        (not isinstance(min_diam, (float, int)))
        or (not isinstance(crop_circ, (float, int)))
        or (min_diam < 0)
        or (crop_circ < 0)
    ):
        print(
            "Argument error in hex_grid: min_diam and crop_circ are expected to be floats"
        )
        args_are_valid = False

    if (not isinstance(align_to_origin, bool)) or (not isinstance(do_plot, bool)):
        print(
            "Argument error in hex_grid: align_to_origin and do_plot are expected to be booleans"
        )
        args_are_valid = False

    VALID_C_ABBR = {"b", "g", "r", "c", "m", "y", "k", "w"}
    if (
        (isinstance(face_color, str) and (face_color not in VALID_C_ABBR))
        or (
            isinstance(background_color, str) and (background_color not in VALID_C_ABBR)
        )
        or (isinstance(edge_color, str) and (edge_color not in VALID_C_ABBR))
    ):
        print(
            "Argument error in hex_grid: edge_color and face_color are expected to valid color abbrs, e.g. `k`"
        )
        args_are_valid = False

    if (
        (
            isinstance(face_color, List)
            and (
                (len(face_color) not in (3, 4))
                or (True in ((x < 0) or (x > 1) for x in face_color))
            )
        )
        or (
            isinstance(background_color, List)
            and (
                (len(background_color) not in (3, 4))
                or (True in ((x < 0) or (x > 1) for x in face_color))
            )
        )
        or (
            isinstance(edge_color, List)
            and (
                (len(edge_color) not in (3, 4))
                or (True in ((x < 0) or (x > 1) for x in edge_color))
            )
        )
    ):
        print(
            "Argument error in hex_grid: edge_color and face_color are expected to be valid RGB color triplets or "
            "color abbreviations, e.g. [0.1 0.3 0.95] or `k`"
        )
        args_are_valid = False

    if (
        (not isinstance(plotting_gap, float))
        or (plotting_gap < 0)
        or (plotting_gap >= 1)
    ):
        print(
            "Argument error in hex_grid: plotting_gap is expected to be a float in range [0, 1)"
        )
        args_are_valid = False

    if not isinstance(rotate_deg, (float, int)):
        print("Argument error in hex_grid: float is expected to be float or integer")
        args_are_valid = False

    if (n == 0) and ((nx == 0) or (ny == 0)):
        print("Argument error in hex_grid: Expected either n>0 or both [nx.ny]>0")
        args_are_valid = False

    if (
        (isinstance(min_diam, (float, int)) and isinstance(crop_circ, (float, int)))
        and (not np.isclose(crop_circ, 0))
        and (crop_circ < min_diam)
    ):
        print(
            "Argument error in hex_grid: Cropping radius is expected to be bigger than a single hexagon diameter"
        )
        args_are_valid = False

    if not isinstance(keep_x_sym, bool):
        print("Argument error in hex_grid: keep_x_sym is expected to be boolean")
        args_are_valid = False

    return args_are_valid


def make_grid(
    nx, ny, min_diam, n, crop_circ, rotate_deg, align_to_origin
) -> (np.ndarray, np.ndarray):
    """
    Computes the coordinates of the hexagon centers, given the size rotation and layout specifications
    :return:
    """
    ratio = np.sqrt(3) / 2
    if n > 0:  # n variable overwrites (nx, ny) in case all three were provided
        ny = int(np.sqrt(n / ratio))
        nx = n // ny

    coord_x, coord_y = np.meshgrid(
        np.arange(nx), np.arange(ny), sparse=False, indexing="xy"
    )
    coord_y = coord_y * ratio
    coord_x = coord_x.astype("float")
    coord_x[1::2, :] += 0.5
    coord_x = coord_x.reshape(-1, 1)
    coord_y = coord_y.reshape(-1, 1)

    coord_x *= min_diam  # Scale to requested size
    coord_y = coord_y.astype("float") * min_diam

    # Pick center of some hexagon as origin for rotation or crop...
    mid_x = (np.ceil(nx / 2) - 1) + 0.5 * (np.ceil(ny / 2) % 2 == 0)
    mid_y = (
        np.ceil(ny / 2) - 1
    ) * ratio  # np.median() averages center 2 values for even arrays :\
    mid_x *= min_diam
    mid_y *= min_diam

    # mid_x = (nx // 2 - (nx % 2 == 1)) * min_diam + 0.5 * (ny % 2 == 1)
    # mid_y = (ny // 2 - (ny % 2)) * min_diam * ratio

    if crop_circ > 0:
        rad = ((coord_x - mid_x) ** 2 + (coord_y - mid_y) ** 2) ** 0.5
        coord_x = coord_x[rad.flatten() <= crop_circ, :]
        coord_y = coord_y[rad.flatten() <= crop_circ, :]

    if not np.isclose(
        rotate_deg, 0
    ):  # Check if rotation is not 0, with tolerance due to float format
        # Clockwise, 2D rotation matrix
        RotMatrix = np.array(
            [
                [np.cos(np.deg2rad(rotate_deg)), np.sin(np.deg2rad(rotate_deg))],
                [-np.sin(np.deg2rad(rotate_deg)), np.cos(np.deg2rad(rotate_deg))],
            ]
        )
        rot_locs = np.hstack((coord_x - mid_x, coord_y - mid_y)) @ RotMatrix.T
        # rot_locs = np.hstack((coord_x - mid_x, coord_y - mid_y))
        coord_x, coord_y = np.hsplit(rot_locs + np.array([mid_x, mid_y]), 2)

    if align_to_origin:
        coord_x -= mid_x
        coord_y -= mid_y

    return coord_x, coord_y
