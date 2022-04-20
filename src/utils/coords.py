import math
import numpy as np
from typing import Tuple

from shapely.geometry import Point


def get_local_coords(coords_global: np.ndarray, point_ref: Tuple[int, int]):
    """
    Convert global coords (longitude, latitude) into local ones (x, y),
    using a global reference point.

    Parameters
    ----------
    - coords: np.ndarray, (N, 2)
    - point_ref: Tuple[float, float]
    """
    coords_local = []
    lon_ref, lat_ref = point_ref

    for (lon, lat) in coords_global:
        dx = (
            (lon - lon_ref)
            * 40000
            * math.cos((lat_ref + lat) * math.pi / 360)
            / 360
            * 1000
        )
        dy = (lat - lat_ref) * 40000 / 360 * 1000
        coords_local.append((dx, dy))

    return coords_local


def get_global_coords(coords_local: np.ndarray, point_ref: Tuple[int, int]):
    """
    Convert local coords (x, y) into global ones (longitude, latitude),
    using a global reference point.

    Parameters
    ----------
    - coords: np.ndarray, (N, 2)
    - point_ref: Tuple[float, float]
    """
    coords_global = []
    lon_ref, lat_ref = point_ref

    for (dx, dy) in coords_local:
        lat = dy * 360 / (40000 * 1000) + lat_ref
        lon = (
            dx * 360 / (40000 * 1000 * math.cos((lat_ref + lat) * math.pi / 360))
            + lon_ref
        )
        coords_global.append((lon, lat))

    return coords_global


def perp_dot_product(vec_1: Point, vec_2: Point) -> float:
    """
    Compute perp dot product between vec_1 and vec_2, in degree.

    https://stackoverflow.com/questions/2150050/finding-signed-angle-between-vectors
    """

    if not isinstance(vec_1, Point) or not isinstance(vec_1, Point):
        raise ValueError(
            f"Perp dot product inputs must be of type shapely.Point, "
            f"got {vec_1} ({type(vec_1)}) and {vec_2} ({type(vec_2)}) instead"
        )

    theta = np.arctan2(
        vec_1.x * vec_2.y - vec_1.y * vec_2.x,
        vec_1.x * vec_2.x + vec_1.y * vec_2.y,
    ) * 180 / np.pi

    return theta