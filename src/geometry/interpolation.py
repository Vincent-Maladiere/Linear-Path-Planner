import numpy as np
from shapely.geometry import LineString, Point

from src.utils.config import DISTANCE_STEP_INTERPOLATE

from src.geometry import core


def interpolate_outer_obs(lr_perimeter, ls_tangent, list_front_left, list_front_right):
    """
    Run interpolation for left and right side on a outer big obstacle.
    """

    centroid_tangent = ls_tangent.centroid
    coords_tangent = list(ls_tangent.coords)

    ls_tangent_left = LineString([coords_tangent[0], centroid_tangent])
    ls_tangent_right = LineString([centroid_tangent, coords_tangent[1]])

    list_front_left.insert(0, ls_tangent_left)
    list_front_right.insert(0, ls_tangent_right)

    list_front_interpolated_left = interpolate_side(lr_perimeter, ls_tangent, list_front_left)
    list_front_interpolated_right = interpolate_side(lr_perimeter, ls_tangent, list_front_right)

    return list_front_interpolated_left, list_front_interpolated_right


def interpolate_side(lr_perimeter, ls_tangent, list_front):
    """
    Run interpolation for a given side of a outer big obstacle.

    Use patch fix to remove snap too close from ls_tangent
    """

    list_front_interpolate = []

    for jdx in range(len(list_front) - 1):
        ls_front_low = list_front[jdx]
        ls_front_up = list_front[jdx + 1]
        list_ls = get_interpolated_front(
            lr_perimeter, ls_front_low, ls_front_up
        )
        if jdx == 0:
            list_ls = patch_fix_remove_bad_snap(ls_tangent, list_ls)

        list_front_interpolate.extend(list_ls)

    return list_front_interpolate


def get_interpolated_front(lr_perimeter, ls_front_low, ls_front_up):
    """
    Generate all inter front between a low front and a up front.
    """

    centroid_distance = ls_front_low.centroid.distance(ls_front_up.centroid)
    if centroid_distance < DISTANCE_STEP_INTERPOLATE:
        return []

    list_ls_front_interpolated = []

    (
        (x_1, y_1, x_2, y_2),
        (x_steps_1, y_steps_1, x_steps_2, y_steps_2),
        n_steps,
    ) = get_interpolate_front_params(ls_front_low, ls_front_up)

    for _ in range(n_steps):

        x_1 += x_steps_1
        y_1 += y_steps_1

        x_2 += x_steps_2
        y_2 += y_steps_2

        ls_front = LineString([(x_1, y_1), (x_2, y_2)])
        ls_front_snap = core.get_snap_segment(lr_perimeter, ls_front, ls_front.centroid)
        list_ls_front_interpolated.append(ls_front_snap)

    return list_ls_front_interpolated


def get_interpolate_front_params(ls_front_low, ls_front_up):
    """
    Compute the steps distance and number of steps for interpolation
    """

    (x_1_low, y_1_low), (x_2_low, y_2_low) = ls_front_low.coords
    (x_1_up, y_1_up), (x_2_up, y_2_up) = ls_front_up.coords

    d_x_1 = x_1_up - x_1_low
    d_y_1 = y_1_up - y_1_low

    d_x_2 = x_2_up - x_2_low
    d_y_2 = y_2_up - y_2_low

    alpha_x_1 = d_x_1 / DISTANCE_STEP_INTERPOLATE
    alpha_y_1 = d_y_1 / DISTANCE_STEP_INTERPOLATE

    alpha_x_2 = d_x_2 / DISTANCE_STEP_INTERPOLATE
    alpha_y_2 = d_y_2 / DISTANCE_STEP_INTERPOLATE

    n_steps = int(max(np.abs([alpha_x_1, alpha_x_2, alpha_y_1, alpha_y_2])))

    x_steps_1 = d_x_1 / n_steps
    y_steps_1 = d_y_1 / n_steps

    x_steps_2 = d_x_2 / n_steps
    y_steps_2 = d_y_2 / n_steps

    return (
        (x_1_low, y_1_low, x_2_low, y_2_low),
        (x_steps_1, y_steps_1, x_steps_2, y_steps_2),
        n_steps,
    )


def patch_fix_remove_bad_snap(ls_tangent, list_ls_inter):

    for idx, ls_inter in enumerate(list_ls_inter):
        coords_tangent = sorted(list(ls_tangent.coords), key=lambda x: x[0])
        coords_inter = sorted(list(ls_inter.coords), key=lambda x: x[0])

        distance_left = Point(coords_tangent[0]).distance(Point(coords_inter[0]))
        distance_right = Point(coords_tangent[1]).distance(Point(coords_inter[1]))

        # we remove ls when distance between edges is too small compared to the ls length
        tolerance_length = 20
        if distance_left > ls_inter.length / tolerance_length \
            or distance_right > ls_inter.length / tolerance_length:
            return list_ls_inter[idx:]

    return list_ls_inter