import numpy as np
from copy import deepcopy

from shapely import affinity
from shapely.geometry import Point, LineString

from src.geometry import core


def get_tangent_points(lr_perimeter, ls_front, p_tangent, p_1_last, p_6_last):

    xoff_low = p_tangent.x - ls_front.centroid.x
    yoff_low = p_tangent.y - ls_front.centroid.y
    ls_tangent = affinity.translate(ls_front, xoff_low, yoff_low)
    ls_tangent_snap = core.get_snap_segment(lr_perimeter, ls_tangent, p_tangent)

    for coords in ls_tangent_snap.coords:
        p = Point(coords)
        if p.distance(p_1_last) < p.distance(p_6_last):
            ls_left = LineString([p, p_tangent])
        else:
            ls_right = LineString([p_tangent, p])

    coords_left = list(ls_left.coords)
    coords_right = list(ls_right.coords)

    return [
        Point(coords_left[0]),
        ls_left.centroid,
        Point(coords_left[1]),
        Point(coords_right[0]),
        ls_right.centroid,
        Point(coords_right[1]),
    ]


def generate_coords(
    lr_perimeter, list_coords_obstacle, list_p_tangent_low, list_p_ref, start_point
):
    """
    Generate the robot coordinates along the two sides of the small obstacles.

    Parameters
    ----------
    - list_coords_obstacle: List[List[Tuple]],
        the list of coordinate for the two side (left and right) of the obstacle.
    - list_p_tangent_low: List[Point],
        the tangent below the obstacle
    - list_p_ref: List[Point],
        p_ref_left and p_ref_right, representing reference to assess the left
        or right side of the obstacle.

    Return
    ------
    - matrix_p_left: List[List[Point]],
        the coordinates of left side robots
    - matrix_p_right: List[List[Point]],
        the coordinates of right side robots
    """
    p_ref_left, _ = list_p_ref

    # determine which obstacle side is left or right
    idx_left, idx_right = get_side_orientation(list_coords_obstacle, p_ref_left)

    coords_left_side = list_coords_obstacle[idx_left]
    coords_right_side = list(reversed(list_coords_obstacle[idx_right]))

    distance_low = Point(coords_left_side[0]).distance(start_point)
    distance_up = Point(coords_left_side[-1]).distance(start_point)

    if distance_up < distance_low:
        coords_left_side = list(reversed(coords_left_side))
        coords_right_side = list(reversed(coords_right_side))

    matrix_p_left = get_coords_left(lr_perimeter, list_p_tangent_low, coords_left_side)
    matrix_p_right = get_coords_right(
        lr_perimeter, list_p_tangent_low, coords_right_side
    )

    return matrix_p_left, matrix_p_right


def get_side_orientation(list_coords_obstacle, p_ref_left):
    """
    Get orientation of list_coords_obstacle by comparing
    distance from centroids to a left reference.
    """
    min_left_distance = np.inf
    idx_left = None
    for idx, coords_side in enumerate(list_coords_obstacle):
        centroid_side = LineString(coords_side).centroid
        distance = centroid_side.distance(p_ref_left)
        if distance < min_left_distance:
            min_left_distance = distance
            idx_left = idx
    idx_right = list({0, 1} - {idx_left})[0]

    return idx_left, idx_right


def get_coords_left(lr_perimeter, list_p_tangent_low, coords_left_side):

    matrix_p_left = []
    p_1_last = list_p_tangent_low[0]
    p_3_last = list_p_tangent_low[2]
    for coord in coords_left_side:
        p_3 = Point(coord)
        xoff = p_3.x - p_3_last.x
        yoff = p_3.y - p_3_last.y

        ls_last = LineString([p_1_last, p_3_last])
        ls = affinity.translate(ls_last, xoff=xoff, yoff=yoff)
        ls_snap = core.get_snap_segment(lr_perimeter, ls, ls.centroid)

        p_1 = Point(list(ls_snap.coords)[0])
        #p_2 = ls_snap.centroid
        p_2 = LineString([p_1, p_3]).centroid
        matrix_p_left.append([p_1, p_2, p_3])
        p_1_last = deepcopy(p_1)
        p_3_last = deepcopy(p_3)

    return matrix_p_left


def get_coords_right(lr_perimeter, list_p_tangent_low, coords_right_side):

    # attribute right side
    matrix_p_right = []
    p_4_last = list_p_tangent_low[3]
    p_6_last = list_p_tangent_low[5]
    for coord in coords_right_side:
        p_4 = Point(coord)
        xoff = p_4.x - p_4_last.x
        yoff = p_4.y - p_4_last.y

        ls_last = LineString([p_4_last, p_6_last])
        ls = affinity.translate(ls_last, xoff=xoff, yoff=yoff)
        ls_snap = core.get_snap_segment(lr_perimeter, ls, ls.centroid)

        p_6 = Point(list(ls_snap.coords[-1]))
        # p_5 = ls_snap.centroid
        p_5 = LineString([p_4, p_6]).centroid
        matrix_p_right.append([p_4, p_5, p_6])
        p_4_last = deepcopy(p_4)
        p_6_last = deepcopy(p_6)

    return matrix_p_right
