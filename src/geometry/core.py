import numpy as np

from shapely import affinity, ops
from shapely.geometry import Point, LineString, LinearRing, Polygon
from shapely.geometry.multipoint import MultiPoint

from src.utils.config import DISTANCE_STEP


class CoreGeometryError(Exception):
    pass


def find_optimal_angle(
    lr_perimeter, centroid_front_next, ls_front, polygon_grazed, verbose=False
):
    """
    Find the angle that minimizes the new grazing area
    """
    centroid_front = ls_front.centroid
    xoff = centroid_front_next.x - centroid_front.x
    yoff = centroid_front_next.y - centroid_front.y
    ls_front_next = affinity.translate(ls_front, xoff=xoff, yoff=yoff)

    ls_front_next = get_snap_segment(lr_perimeter, ls_front_next, centroid_front_next)

    length_front_min = None
    best_angle = None

    all_angles = range(0, 180, 1)
    for angle in all_angles:

        ls_front_rot = affinity.rotate(
            ls_front_next, angle, origin=centroid_front_next
        )
        ls_front_snap = get_snap_segment(lr_perimeter, ls_front_rot, centroid_front_next)

        if check_backwards_contrainst(polygon_grazed, ls_front_snap):

            length_front = ls_front_snap.length

            if length_front_min is None or length_front < length_front_min:
                best_angle = angle
                length_front_min = length_front
                if verbose:
                    print("Best angle found!")
            if verbose:
                print(angle, length_front, "\n")

    return best_angle


def check_backwards_contrainst(polygon_grazed, ls_front_snap):

    buffer_size = 0.1
    for coords in ls_front_snap.coords:
        if polygon_grazed.buffer(buffer_size).contains(Point(coords)):
            return False

    return True


def get_snap_segment(lr_perimeter, ls_front, centroid_front_next, scale_factor=1.5, scale_limit=20
):
    """
    Get the smallest intersecting segment with the perimeter.
    We want intersection = 2.
    """

    if scale_factor > scale_limit:
        raise CoreGeometryError("ls_front can't reach lr_perimeter")

    ls_front_scaled = affinity.scale(
        ls_front, xfact=scale_factor, yfact=scale_factor
    )
    intersections = lr_perimeter.intersection(ls_front_scaled)

    # intersection < 2
    if isinstance(intersections, (Point, LineString)):
        scale_factor *= 2
        return get_snap_segment(
            lr_perimeter, ls_front_scaled, centroid_front_next, scale_factor
        )

    # intersections >= 2
    elif isinstance(intersections, MultiPoint):

        p_intersections = list(intersections)
        # if len(p_intersections) > 2:
        p_intersections = _get_closest_intersections(
            ls_front,
            centroid_front_next,
            p_intersections,
        )
        # several intersections on the same side, but not on both.
        if len(p_intersections) != 2:
            scale_factor *= 2
            p_intersections = get_snap_segment(
                lr_perimeter, ls_front, centroid_front_next, scale_factor
            )

        return LineString(p_intersections)

    else:
        raise CoreGeometryError(f"Object 'intersections' unknown: {intersections}")


def _get_closest_intersections(ls_front, centroid_front_next, p_intersections):
    """
    Find the single closest intersection to the centroid on the left and right side.

    Create left and right point, super close to the center.
    The goal is to verify that these points belong to the segment 
    p0-p_intersection (aka p_0-p)

    `centroid_front_next` is needed because it may be different from
    the centroid of ls_front.
    """

    coords = list(ls_front.coords)
    p_1, p_2 = Point(coords[0]), Point(coords[1])
    p_0 = centroid_front_next

    #alpha_1 = 1 / p_0.distance(p_1)
    alpha_1 = 1e-3
    coords_p_left = alpha_1 * np.array(p_1.coords) + (1 - alpha_1) * np.array(
        p_0.coords
    )
    p_left = Point(coords_p_left.reshape(-1, 1))

    #alpha_2 = 1 / p_0.distance(p_2)
    alpha_2 = 1e-3
    coords_p_right = alpha_2 * np.array(p_2.coords) + (1 - alpha_2) * np.array(
        p_0.coords
    )
    p_right = Point(coords_p_right.reshape(-1, 1))

    dict_p_closest = {
        "p_left": {
            "shortest_distance": np.inf,
            "point": None,
        },
        "p_right": {
            "shortest_distance": np.inf,
            "point": None,
        },
    }
    for p in p_intersections:
        ls = LineString([p_0, p])
        if ls.distance(p_left) < 1e-3:
            p_direction = "p_left"
        elif ls.distance(p_right) < 1e-3:
            p_direction = "p_right"
        else:
            raise CoreGeometryError("[p0-p] contains neither p_left nor p_right")

        distance_p0 = p_0.distance(p)
        if distance_p0 < dict_p_closest[p_direction]["shortest_distance"]:
            dict_p_closest[p_direction]["shortest_distance"] = distance_p0
            dict_p_closest[p_direction]["point"] = p

    p_intersections = []
    for dict_p_values in dict_p_closest.values():
        if not dict_p_values["point"] is None:
            p_intersections.append(dict_p_values["point"])

    return p_intersections


def split_polygon(lr_perimeter, ls_front, last_centroid_front, **kwargs):
    """
    Get polygon direction and polygon grazed.

    Return
    ------
    polygon_direction, polygon_grazed
    """
    ls_front_scaled = affinity.scale(ls_front, xfact=1.1, yfact=1.1)

    polygon_perimeter = Polygon(lr_perimeter, **kwargs)
    list_polygon_split = ops.split(polygon_perimeter, ls_front_scaled)
    if last_centroid_front is None:
        list_polygon_split = sorted(
            list_polygon_split, key=lambda x: x.area, reverse=True
        )
        polygon_direction = list_polygon_split[0]
        polygon_grazed = list_polygon_split[1]
    else:
        list_polygon_direction, list_polygon_grazed = [], []
        for poly in list_polygon_split:
            if poly.contains(last_centroid_front):
                list_polygon_grazed.append(poly)
            else:
                list_polygon_direction.append(poly)
        polygon_direction = sorted(
            list_polygon_direction, key=lambda x: x.area, reverse=True
        )[0]
        polygon_grazed = sorted(
            list_polygon_grazed, key=lambda x: x.area, reverse=True
        )[0]

    return polygon_direction, polygon_grazed


def forward_centroid_front(ls_front, point_direction):
    """
    Get the translated front centroid.
    """
    ls_front_ortho = affinity.scale(affinity.rotate(ls_front, 90), xfact=5, yfact=5)
    p_project_ortho = ls_front_ortho.interpolate(
        ls_front_ortho.project(point_direction)
    )

    centroid_front = ls_front.centroid
    distance_ahead = LineString([p_project_ortho, centroid_front]).length
    alpha = DISTANCE_STEP / distance_ahead

    centroid_front_next = Point(
        (alpha * p_project_ortho.x + (1 - alpha) * centroid_front.x),
        (alpha * p_project_ortho.y + (1 - alpha) * centroid_front.y),
    )

    return centroid_front_next


def update_ls_front(lr_perimeter, centroid_front_next, ls_front, best_angle):

    centroid_front = ls_front.centroid

    xoff = centroid_front_next.x - centroid_front.x
    yoff = centroid_front_next.y - centroid_front.y
    ls_front_next = affinity.translate(ls_front, xoff=xoff, yoff=yoff)
    ls_front_next = affinity.rotate(
        ls_front_next, best_angle, origin=centroid_front_next
    )
    ls_front_next = get_snap_segment(lr_perimeter, ls_front_next, centroid_front_next)

    return ls_front_next
