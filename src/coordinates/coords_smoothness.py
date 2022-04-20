import numpy as np
from shapely.geometry import Point, LineString
from shapely import affinity, ops

from src.utils.logs import logger

from src.coordinates import coords_core
from src.utils.config import DISTANCE_STEP_INTERPOLATE


class CoordsSmoothnessError(Exception):
    pass


def get_range_idx_smooth(coords_local, list_ls_centroid):
    """
    Find coords local range of idx to update.
    """
    
    ls_centroid_low, ls_centroid_up = list_ls_centroid

    p_inter_low = Point(list(ls_centroid_low.coords)[0])
    p_tangent_low = Point(list(ls_centroid_low.coords)[1])

    p_tangent_up = Point(list(ls_centroid_up.coords)[0])
    p_inter_up = Point(list(ls_centroid_up.coords)[1])

    idx_inter_low = get_nearest_ls_idx(coords_local, p_inter_low)
    idx_tangent_low = get_nearest_ls_idx(coords_local, p_tangent_low)

    idx_tangent_up = get_nearest_ls_idx(coords_local, p_tangent_up)
    idx_inter_up = get_nearest_ls_idx(coords_local, p_inter_up)

    range_idx_low = range(idx_inter_low, idx_tangent_low)
    range_idx_up = range(idx_tangent_up, idx_inter_up)
    range_idx_obs = range(idx_tangent_low, idx_tangent_up)

    return range_idx_low, range_idx_up, range_idx_obs


def get_nearest_ls_idx(coords_local, p, idx_left=None, idx_right=None):
    """
    Binary search to find idx of the nearest coords_local 
    """
    idx_left = idx_left or 0
    idx_right = idx_right or (coords_local.shape[0] - 1)
    idx_middle = (idx_left + idx_right) // 2

    distance_middle = get_distance(coords_local[idx_middle], p)
    distance_left = get_distance(coords_local[idx_middle-1], p)
    distance_right = get_distance(coords_local[idx_middle+1], p)

    if distance_middle < min(distance_left, distance_right):
        return idx_middle

    elif distance_left < distance_right:
        if idx_middle-1 == idx_left:
            return idx_left
        return get_nearest_ls_idx(coords_local, p, idx_left, idx_middle-1)

    else:
        if idx_middle+1 == idx_right:
            return idx_right
        return get_nearest_ls_idx(coords_local, p, idx_middle+1, idx_right)

    
def get_distance(coords_front, p):

    ls = LineString([coords_front[0], coords_front[-1]])

    return ls.distance(p)


def get_updated_coords(coords_front, ls_centroid):
    """
    Update coordinates row.

    Parameters
    ----------
    - coords_front: np.ndarray,
        the coordinates to update
    - ls_centroid: LineString
        the segment centroid to intersect
    """
    ls_front = LineString([coords_front[0], coords_front[-1]])
    centroid = ls_centroid.intersection(ls_front)

    if not isinstance(centroid, Point):
        if ls_centroid.distance(ls_front) < 1:
           centroid = _patch_intersection_issue(ls_centroid, ls_front)
        else:
            raise CoordsSmoothnessError(
                f"No single intersection between\n"
                f"ls_front: {list(ls_front.coords)}\n"
                f"and ls_centroid: {list(ls_centroid.coords)}"
            )
    
    p_1, p_6 = Point(list(ls_front.coords)[0]), Point(list(ls_front.coords)[1])
    p_3, p_4 = coords_core.get_centroid_offsets(centroid, p_1, p_6)

    p_2 = LineString([p_1, p_3]).centroid
    p_5 = LineString([p_4, p_6]).centroid

    row = coords_core.make_row([p_1, p_2, p_3, p_4, p_5, p_6])

    return row


def get_list_centroid_obs(coords_local, range_idx_obs):

    idx_obs_low = list(range_idx_obs)[0] - 1
    idx_obs_up = list(range_idx_obs)[-1] + 1

    coords_low = coords_local[idx_obs_low]
    coords_up = coords_local[idx_obs_up]

    list_ls_centroid_obs = [
        LineString([coords_low[1], coords_up[1]]),
        LineString([coords_low[4], coords_up[4]]),
    ]

    return list_ls_centroid_obs


def get_updated_coords_obs(coords_front, list_ls_centroid):
    """
    Upate coords obstacle to get p_2 and p_5.
    Might be useless if n_robot = 2 though.
    """

    ls_centroid_left, ls_centroid_right = list_ls_centroid    

    p_1 = Point(coords_front[0])
    p_3 = Point(coords_front[2])
    p_4 = Point(coords_front[3])
    p_6 = Point(coords_front[5])

    ls_front_left = LineString([p_1, p_3])
    ls_front_right = LineString([p_4, p_6])

    p_2 = ls_centroid_left.intersection(ls_front_left)
    if not isinstance(p_2, Point):
        if ls_centroid_left.distance(ls_front_left) < DISTANCE_STEP_INTERPOLATE:
            p_2 = _patch_intersection_issue(ls_centroid_left, ls_front_left)
        else:
            raise CoordsSmoothnessError(
                f"No single intersection between\n"
                f"ls_front_left: {list(ls_front_left.coords)}\n"
                f"and ls_centroid_left: {list(ls_centroid_left.coords)}"
            )
    p_2 = check_closeness(p_2, p_3, p_1)

    p_5 = ls_centroid_right.intersection(ls_front_right)
    if not isinstance(p_5, Point):
        if ls_centroid_right.distance(ls_front_right) < DISTANCE_STEP_INTERPOLATE:
            p_5 = _patch_intersection_issue(ls_centroid_right, ls_front_right)
        else:
            raise CoordsSmoothnessError(
                f"No single intersection between\n"
                f"ls_front_right: {list(ls_front_right.coords)}\n"
                f"and ls_centroid_right: {list(ls_centroid_right.coords)}"
            )
    p_5 = check_closeness(p_5, p_4, p_6)

    row = coords_core.make_row([p_1, p_2, p_3, p_4, p_5, p_6])

    return row


def check_closeness(p_middle, p_obs, p_perimeter):
    """
    Check if
    1. p is out of the segment [p_obs, p_perimeter]
    2. p is too close from p_obs
    """
    ls_front = LineString([p_obs, p_perimeter])
    if p_middle.distance(ls_front) > 1e-3 \
        or p_middle.distance(p_obs) < 5:
        p_middle = p_obs.buffer(5).exterior.intersection(ls_front)

    return p_middle


def _patch_intersection_issue(ls_centroid, ls_front):
    """
    Force ls_front to intersect with ls_centroid on the last iteration.
    """
    logger.warning(" # [Coords Smooth] Use of _patch_intersection_issue")

    ls_centroid_scaled = affinity.scale(ls_centroid, 1.1, 1.1)
    p = ls_centroid_scaled.intersection(ls_front)

    if not isinstance(p, Point):
        if ls_centroid.distance(ls_front) < DISTANCE_STEP_INTERPOLATE:
            _, p = ops.nearest_points(ls_centroid, ls_front)
        else: 
            raise CoordsSmoothnessError(
                f"No single intersection between\n"
                f"ls_front_left: {list(ls_front.coords)}\n"
                f"and ls_centroid_left: {list(ls_centroid.coords)}"
            )

    return p
