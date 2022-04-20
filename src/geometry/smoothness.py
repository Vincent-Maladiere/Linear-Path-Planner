import numpy as np
from shapely.geometry import Point, LineString
from shapely.geometry.multipoint import MultiPoint
from shapely import affinity, ops

from src.utils.coords import perp_dot_product


class SmoothnessError(Exception):
    pass


def get_angles(list_ls_front_interpolated, centroid_obs, list_idx_front, angle):
    """
    Determine the sign of the angle to use for both low and up front.

    First, find the angle of the low front by multiplying the vertical direction
    sign and the obstacle closeness to the edge side sign.
    - if we go upward, the sign is 1. Otherwise, the sign is -1.
    - if the obstacle is closer to p_6 than p_1, the sign is 1. O.w., the sign is -1.

    |                | closer to p_6 | closer to p_1 |
    |----------------|---------------|---------------|
    | up direction   | 1             | -1            |
    | down direction | -1            | 1             |
    |                |               |               |

    Second, angle up is the opposite of angle low
    """

    idx_front_low, idx_front_up = list_idx_front
    ls_front_up = list_ls_front_interpolated[idx_front_up]
    ls_front_low = list_ls_front_interpolated[idx_front_low]

    sign_direction = get_sign_direction(ls_front_low, ls_front_up)
    sign_closeness = get_sign_closeness(centroid_obs, ls_front_low)
    angle_low = angle * sign_direction * sign_closeness
    angle_up = -angle_low #get_angle(centroid_obs, ls_front_up)

    return [angle_low, angle_up]


def get_sign_direction(ls_front_low, ls_front_up):

    vec_1 = Point(0, 1)
    vec_2 = Point(
        *(np.array(ls_front_up.centroid.coords) - np.array(ls_front_low.centroid))
    )
    theta = perp_dot_product(vec_1, vec_2)
    if np.abs(theta) < 90:
        return 1
    else:
        return -1


def get_sign_closeness(centroid_obs, ls_front):

    p_1 = Point(list(ls_front.coords)[0])
    p_6 = Point(list(ls_front.coords)[-1])

    if p_1.distance(centroid_obs) < p_6.distance(centroid_obs):
        return -1
    else:
        return 1


def find_intersections(list_ls_front_interpolated, centroid_obstacle, list_idx_front, list_angle):
    """
    Find the intersection between the angular projection of the obstacle
    and the joint line.

    This intersection will mark the beginning of the smoothness operation.

    1. Get the index of the obstacle by taking 
    2. Create the joint LineString for both low and up within their index range
       low: [0, idx_obs], up: [idx_obs+1, N]
    3. Get the intersection between the joint line and the obstacle angular projection
    """

    idx_obstacle = np.mean(list_idx_front).astype(int)
    ls_obstacle = list_ls_front_interpolated[idx_obstacle]

    idx_front_low, idx_front_up = list_idx_front
    list_ls_low = list_ls_front_interpolated[:idx_front_low]
    list_ls_up = list_ls_front_interpolated[idx_front_up:]

    angle_low, angle_up = list_angle
    p_low = find_intersection(ls_obstacle, centroid_obstacle, list_ls_low, angle_low)
    p_up = find_intersection(ls_obstacle, centroid_obstacle, list_ls_up, angle_up)

    #list_idx_inter = convert_p_to_idx(list_ls_front_interpolated, p_low, p_up)
    
    #return list_idx_inter

    return [p_low, p_up]


def find_intersection(ls_obstacle, centroid_obstacle, list_ls, angle):
    """
    Rotate the ls_obstacle and create the obstacle
    """
    ls_joint = LineString([ls.centroid for ls in list_ls])

    ls_obstacle_rot = affinity.rotate(ls_obstacle, angle, origin=centroid_obstacle)
    ls_obstacle_scale = affinity.scale(ls_obstacle_rot, 5, 5)

    p_int = ls_obstacle_scale.intersection(ls_joint)
    if isinstance(p_int, Point):
        return p_int
    elif isinstance(p_int, MultiPoint):
        p_int = ops.nearest_points(p_int, centroid_obstacle)[0]
        return p_int
    else:
        raise SmoothnessError(
            f"{p_int} is not a Point.\n"
            f"ls_obstacle_scale: {list(ls_obstacle_scale.coords)}\n"
            f"ls_obstacle: {list(ls_obstacle.coords)}\n"
            f"ls_obstacle_rot: {list(ls_obstacle_rot.coords)}\n"
            f"ls_joint: {list(ls_joint.coords)}"
        )


def convert_ls_to_idx(list_ls_front_interpolated, list_ls):

    ls_low, ls_up = list_ls 
    smallest_distance_low, smallest_distance_up = np.inf, np.inf
    idx_low, idx_up = None, None
    for idx, ls_front in enumerate(list_ls_front_interpolated):
        if not isinstance(ls_front, list):
            distance_low = ls_low.distance(ls_front)
            distance_up = ls_up.distance(ls_front)
            if distance_low < smallest_distance_low:
                idx_low = idx
                smallest_distance_low = distance_low
            if distance_up < smallest_distance_up:
                idx_up = idx
                smallest_distance_up = distance_up
    
    return [idx_low, idx_up]


## OLDD

def convert_p_to_idx(list_ls_front_interpolated, p_low, p_up):
    """
    Get the indexes of the closest front for both p_low and p_up
    """

    smallest_distance_low, smallest_distance_up = np.inf, np.inf
    for idx, ls_front in enumerate(list_ls_front_interpolated):
        if not isinstance(ls_front, list):
            distance_low = ls_front.distance(p_low)
            distance_up = ls_front.distance(p_up)
            if distance_low < smallest_distance_low:
                idx_low = idx
                smallest_distance_low = distance_low
            if distance_up < smallest_distance_up:
                idx_up = idx
                smallest_distance_up = distance_up

    return [idx_low, idx_up]
    

def convert_p_tangent_to_ls(list_ls_front_interpolated, p_tangent):

    smallest_distance = np.inf
    ls_front_closest = None
    for ls_front in list_ls_front_interpolated:
        if not isinstance(ls_front, list):
            distance = ls_front.distance(p_tangent)
            if distance < smallest_distance:
                ls_front_closest = ls_front
                smallest_distance = distance
    
    return ls_front_closest
