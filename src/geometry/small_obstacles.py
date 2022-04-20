import numpy as np

from shapely import affinity, ops
from shapely.geometry import Point, LineString, LinearRing, Polygon

from src.geometry.core import get_snap_segment


class SmallObstacleError(Exception):
    pass


def get_closest_fronts(lr_perimeter, list_ls_front_interpolated, small_obstacle, start_point):
    """
    [1/3] Identify the 2 closest fronts (non secant) to the obstacle
    """
    # get the closest front overall
    ls_front_closest = _get_closest_front(small_obstacle, list_ls_front_interpolated)

    # split the parcel in two polygons
    dict_poly_split = _split_parcel(lr_perimeter, ls_front_closest, start_point)

    # for each polygon, get the closest front
    idx_ls_low, idx_ls_up = _find_closest_tangent(list_ls_front_interpolated, dict_poly_split, small_obstacle)

    return idx_ls_low, idx_ls_up


def _get_closest_front(small_obstacle, list_ls_front_interpolated):

    smallest_distance = np.inf
    ls_front_closest = None
    centroid_obstacle = small_obstacle.centroid
    for ls_front in list_ls_front_interpolated:
        if isinstance(ls_front, list):
            continue
            # for list_ls_side in ls_front:
            #     for ls_side in list_ls_side:
            #         distance_centroid = ls_side.distance(centroid_obstacle)
            #         if distance_centroid < smallest_distance:
            #             smallest_distance = distance_centroid
            #             ls_front_closest = ls_side
        else:
            distance_centroid = ls_front.distance(centroid_obstacle)
            if distance_centroid < smallest_distance:
                smallest_distance = distance_centroid
                ls_front_closest = ls_front

    return ls_front_closest


def _split_parcel(lr_perimeter, ls_front_closest, start_point):

    ls_front_snap = get_snap_segment(
        lr_perimeter, ls_front_closest, ls_front_closest.centroid
    )
    ls_front_scaled = affinity.scale(ls_front_snap, xfact=1.5, yfact=1.5)
    list_poly_split = ops.split(Polygon(lr_perimeter), ls_front_scaled)
    dict_poly_split = dict()
    for poly_split in list_poly_split:
        if poly_split.contains(start_point):
            dict_poly_split["low"] = poly_split
        else:
            dict_poly_split["up"] = poly_split

    if len(dict_poly_split) != 2:
        raise SmallObstacleError("Polygon not split correctly")

    return dict_poly_split


def _find_closest_tangent(list_ls_front_interpolated, dict_poly_split, small_obstacle):

    dict_ls_closest = {
        "low": {
            "smallest_distance": np.inf,
            "idx": None,
        },
        "up": {
            "smallest_distance": np.inf,
            "idx": None,
        },
    }
    for idx, ls_front in enumerate(list_ls_front_interpolated):
        if isinstance(ls_front, list):
            continue
        else:
            for direction, poly_split in dict_poly_split.items():
                if (
                    poly_split.distance(ls_front) < 1e-10
                    and not small_obstacle.intersection(ls_front).coords
                ):
                    #distance_centroid = small_obstacle.distance(ls_front.centroid)
                    # do not use centroid, as they might be far off the obstacle on the same ls 
                    distance_centroid = small_obstacle.distance(ls_front)
                    if (
                        distance_centroid
                        < dict_ls_closest[direction]["smallest_distance"]
                    ):
                        dict_ls_closest[direction] = {
                            "smallest_distance": distance_centroid,
                            "idx": idx,
                        }
                        break

    return dict_ls_closest["low"]["idx"], dict_ls_closest["up"]["idx"]


def get_tangent_points(ls_low, ls_up, small_obstacle):
    """
    [2/3] Make ls_low and ls_up tangent to their respective closest point of the obstacle
    """
    p_tangent_low = ops.nearest_points(ls_low, small_obstacle)[1]
    p_tangent_up = ops.nearest_points(ls_up, small_obstacle)[1]

    return p_tangent_low, p_tangent_up


def split_obstacle(p_tangent_low, p_tangent_up, small_obstacle):
    """
    [3/3] Split the obstacles with left and right pair of points
    """
    ls_secant = LineString([p_tangent_low, p_tangent_up])
    ls_secant_scaled = affinity.scale(ls_secant, xfact=1.5, yfact=1.5)
    list_poly_split = ops.split(small_obstacle, ls_secant_scaled)

    if len(list_poly_split) != 2:
        raise SmallObstacleError(
            f"Wrong polygon split.\n"
            f"ls_secant_coords: ax.plot(*LineString({list(ls_secant_scaled.coords)}).xy)\n\n"
            f"small_obstacle: ax.plot(*LinearRing({list(small_obstacle.exterior.coords)}).xy)"
        )

    list_coords_obstacle = []
    # Iterate through the 2 polygons and reorder point to make a LineString
    # We look for the biggest shift, and break the list at this index
    for poly_split in list_poly_split:
        # coords_approx = approximate_polygon(
        #    np.array(poly_split.exterior.coords), tolerance=.5
        # )
        coords_approx = np.array(poly_split.exterior.coords)
        # turn a LinearRing to a LineString by sorting the coords
        coords_abs_diff = np.abs(np.diff(coords_approx, axis=0))
        coords_abs_diff = np.sum(coords_abs_diff, axis=1)
        idx_jump = np.argmax(coords_abs_diff)
        coords = [*coords_approx[idx_jump + 1 :], *coords_approx[: idx_jump + 1]]
        list_coords_obstacle.append(coords)

    return list_coords_obstacle
