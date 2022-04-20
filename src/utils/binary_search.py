import numpy as np
from shapely.geometry import LineString


class BinarySearchError(Exception):
    pass


def get_nearest_ls_idx(list_coords, obj, idx_left=None, idx_right=None):
    """
    Binary search to find the index (idx) of the nearest list_coords item from obj

    Parameters
    ----------
    list_coords: iterable, np.ndarray or list
        list of coordinates to scan
    obj: shapely geometry, Point, LineString, LinearRing, Polygon
        object to find the nearest `list_coords` item from

    Return
    ------
    """
    idx_left = idx_left or 0
    idx_right = idx_right or (len(list_coords) - 1)
    idx_middle = (idx_left + idx_right) // 2

    distance_middle = get_distance(list_coords[idx_middle], obj)
    distance_left = get_distance(list_coords[idx_middle-1], obj)
    distance_right = get_distance(list_coords[idx_middle+1], obj)

    if distance_middle < min(distance_left, distance_right):
        return idx_middle

    elif distance_left < distance_right:
        if idx_middle-1 == idx_left:
            return idx_left
        return get_nearest_ls_idx(list_coords, obj, idx_left, idx_middle-1)

    else:
        if idx_middle+1 == idx_right:
            return idx_right
        return get_nearest_ls_idx(list_coords, obj, idx_middle+1, idx_right)

    
def get_distance(coords, obj):

    if isinstance(coords, LineString):
        return coords.distance(obj)

    elif isinstance(coords, (np.matrix, np.ndarray, list, tuple)):

        p_a, p_b = coords[0], coords[-1]

        if not isinstance(p_a[0], (int, np.integer, float, np.floating)) \
            or not isinstance(p_b[0], (int, np.integer, float, np.floating)):
            raise BinarySearchError(
                f"coords type of {p_a[0]} and {p_b[0]}" \
                f"are {repr(type(p_a[0]))} and {repr(type(p_b[0]))}"
            )

        ls = LineString([coords[0], coords[-1]])
        return ls.distance(obj)

    else:
        raise BinarySearchError(f"coords type of {coords} is {repr(type(coords))}")