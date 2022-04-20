from tqdm import tqdm
from collections import deque
from shapely import affinity, ops
from shapely.geometry import Point, LineString, Polygon

from src.utils.logs import logger
from src.utils.config import DISTANCE_STEP_INTERPOLATE, RATE_AREA_LIMIT

from src.geometry import core
from src.geometry import big_obstacles_outer as big_outer


def get_first_ls_front(lr_perimeter, ls_sweep_axis):
    """
    Define the first front using the sweep_axis.
    """
    ls_sweep_axis_snap = core.get_snap_segment(
        lr_perimeter, ls_sweep_axis, ls_sweep_axis.centroid
    )
    ls_sweep_axis_scaled = affinity.scale(ls_sweep_axis_snap, xfact=1.5, yfact=1.5)
    intersections = LineString(lr_perimeter.intersection(ls_sweep_axis_scaled))

    return intersections


def get_list_front(
    lr_perimeter,
    ls_front,
    centroid_front_next,
    polygon_direction,
    polygon_grazed,
    connectivity_last=1,
):

    list_ls_front = [ls_front]
    q_rate_area_grazed = deque([0], maxlen=1)

    pbar = tqdm(total=100) if connectivity_last == 1 else None

    while not check_end_area(
        lr_perimeter, ls_front, centroid_front_next, polygon_direction, polygon_grazed, q_rate_area_grazed, pbar
    ):

        last_centroid_front = ls_front.centroid
        connectivity, intersections = get_connectivity(
            lr_perimeter, ls_front, centroid_front_next
        )

        if connectivity == 1 or connectivity_last == 2:
            best_angle = core.find_optimal_angle(
                lr_perimeter, centroid_front_next, ls_front, polygon_grazed
            )
            ls_front = core.update_ls_front(
                lr_perimeter, centroid_front_next, ls_front, best_angle
            )
            list_ls_front.append(ls_front)

            polygon_direction, polygon_grazed = core.split_polygon(
                lr_perimeter, ls_front, last_centroid_front
            )
            centroid_front_next = core.forward_centroid_front(
                ls_front,
                polygon_direction.centroid,
            )

        elif connectivity == 2:

            ls_front_tangent = big_outer.get_outer_tangent_front(
                lr_perimeter, ls_front, intersections
            )
            list_ls_front.append(ls_front_tangent)

            ls_front_left, ls_front_right = big_outer.get_outer_split_front(
                intersections
            )

            polygon_direction_left, polygon_grazed_left = core.split_polygon(
                lr_perimeter, ls_front_left, last_centroid_front
            )
            centroid_front_next_left = core.forward_centroid_front(
                ls_front_left, polygon_direction_left.centroid
            )
            list_front_left = get_list_front(
                lr_perimeter,
                ls_front_left,
                centroid_front_next_left,
                polygon_direction_left,
                polygon_grazed_left,
                connectivity_last=2,
            )

            polygon_direction_right, polygon_grazed_right = core.split_polygon(
                lr_perimeter, ls_front_right, last_centroid_front
            )
            centroid_front_next_right = core.forward_centroid_front(
                ls_front_right, polygon_direction_right.centroid
            )
            list_front_right = get_list_front(
                lr_perimeter,
                ls_front_right,
                centroid_front_next_right,
                polygon_direction_right,
                polygon_grazed_right,
                connectivity_last=2,
            )

            list_ls_front.append([list_front_left, list_front_right])
            break

        else:
            logger.error(
                f" # [Sequence] connectivity = {connectivity}, not implemented"
            )
            break

    list_ls_front = patch_fix_reverse_bug(list_ls_front)

    return list_ls_front


def check_end_area(
    lr_perimeter,
    ls_front,
    centroid_front_next,
    polygon_direction,
    polygon_grazed,
    q_rate_area_grazed,
    pbar,
):
    area_direction = polygon_direction.area
    area_grazed = polygon_grazed.area
    rate_area_grazed = area_grazed / (area_grazed + area_direction)

    if not pbar is None:
        rate_update = int((rate_area_grazed - q_rate_area_grazed[0]) * 100)
        pbar.update(rate_update)
        q_rate_area_grazed.append(rate_area_grazed)

    return rate_area_grazed > RATE_AREA_LIMIT or is_centroid_outside_parcel(
        lr_perimeter, ls_front, centroid_front_next
    )


def is_centroid_outside_parcel(lr_perimeter, ls_front, centroid_front_next):

    connectivity, _ = get_connectivity(lr_perimeter, ls_front, centroid_front_next)

    distance_parcel = Polygon(lr_perimeter).distance(centroid_front_next)

    return distance_parcel > 1e-14 and connectivity <= 1


def get_connectivity(lr_perimeter, ls_front, centroid_front_next):
    """
    The connectivity of a front is the number of distinct segments within the parcel
    formed by that front.

    Hack: if a connectivity is > 2 (so there's 4 intersections), and the 3rd
    intersection does not belong to the tranlated previous ls_front, then
    we discard this connectivity as 1 (aka obstacle-free).
    """

    centroid_front = ls_front.centroid
    xoff = centroid_front_next.x - centroid_front.x
    yoff = centroid_front_next.y - centroid_front.y
    ls_front_next = affinity.translate(ls_front, xoff=xoff, yoff=yoff)

    ls_front_scaled = affinity.scale(ls_front_next, xfact=100, yfact=100)
    intersections = ls_front_scaled.intersection(lr_perimeter)
    # if LineString, there is no intersections
    if isinstance(intersections, LineString):
        intersections = []
    connectivity = len(intersections) / 2

    if len(intersections) > 2 and ls_front_next.distance(intersections[2]) < 1e-3:
        logger.warning(
            f" # [Sequence] get_connectivity - Connectivity is {connectivity}"
        )
    else:
        connectivity = 1

    return connectivity, intersections


def patch_fix_reverse_bug(list_ls_front):
    """
    Reverse points of line string when angle > 90°
    Only check for connectivity = 1, hence the need to have non list element.
    """

    for idx in range(1, len(list_ls_front)):

        if not isinstance(list_ls_front[idx - 1], list) and not isinstance(
            list_ls_front[idx], list
        ):

            coords_low = list_ls_front[idx - 1].coords
            coords_up = list_ls_front[idx].coords

            p_1_low, p_2_low = Point(coords_low[0]), Point(coords_low[1])
            p_1_up, p_2_up = Point(coords_up[0]), Point(coords_up[1])

            if p_1_up.distance(p_1_low) > p_1_up.distance(p_2_low):
                list_ls_front[idx] = LineString([p_2_up, p_1_up])

    return list_ls_front

