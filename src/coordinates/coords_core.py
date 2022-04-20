import numpy as np
from shapely.geometry import Point, LineString

from src.utils.config import INTER_FRONT_DISTANCE


def get_p_front(ls_front, centroid):

    coords = list(ls_front.coords)
    p_1, p_6 = Point(coords[0]), Point(coords[1])

    p_3, p_4 = get_centroid_offsets(centroid, p_1, p_6)
    front_left = LineString([p_1, p_3])
    front_right = LineString([p_4, p_6])

    p_2 = front_left.centroid
    p_5 = front_right.centroid

    return [p_1, p_2, p_3, p_4, p_5, p_6]


def get_centroid_offsets(centroid, p_1, p_6):

    ls_left = LineString([p_1, centroid])
    ls_right = LineString([centroid, p_6])

    circle_distance = centroid.buffer(INTER_FRONT_DISTANCE / 2).exterior
    p_3 = ls_left.intersection(circle_distance)
    p_4 = ls_right.intersection(circle_distance)

    return p_3, p_4


def add_coords(matrix_coords, points):

    row = make_row(points)
    matrix_coords = np.vstack([matrix_coords, row])

    return matrix_coords


def make_row(points):

    row  = np.array([list(p.coords) for p in points]).reshape(1, 6, 2)

    return row