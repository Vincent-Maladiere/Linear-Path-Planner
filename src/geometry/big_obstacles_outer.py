from shapely import ops, affinity
from shapely.geometry import LineString

from src.utils.config import DISTANCE_STEP_INTERPOLATE


class BigObstacleInnerError(Exception):
    pass


def get_outer_split_front(intersections):
    """
    Get the two front surrounding a outter connectivity = 2
    """

    if len(intersections) != 4:
        raise BigObstacleInnerError(
            f"Connectivity = 2 but {len(intersections)} intersections\n"
            f"{[list(inter.coords) for inter in intersections]}"
        )

    ls_front_left = LineString([intersections[0], intersections[1]])
    ls_front_right = LineString([intersections[2], intersections[3]])

    return ls_front_left, ls_front_right


def get_outer_tangent_front(lr_perimeter, ls_front, intersections):
    """
    Find the last front before the outer obstacle.
    """

    centroid_front = ls_front.centroid
    centroid_intersections = LineString(
        [intersections[1], intersections[2]]
    ).centroid

    distance_centroid = centroid_front.distance(centroid_intersections)
    n_steps = int(distance_centroid // DISTANCE_STEP_INTERPOLATE) + 1

    xoff = (centroid_intersections.x - centroid_front.x) / n_steps
    yoff = (centroid_intersections.y - centroid_front.y) / n_steps

    ls_front_tangent = ls_front
    for _ in range(n_steps):

        ls_front = affinity.translate(ls_front, xoff=xoff, yoff=yoff)
        ls_front_scaled = affinity.scale(ls_front, 100, 100)
        intersections = lr_perimeter.intersection(ls_front_scaled)
        if len(intersections) > 2:
            break
        ls_front_tangent = ls_front

    return ls_front_tangent
