import numpy as np
import pandas as pd

from shapely.geometry import Point, LineString
from shapely.geometry.polygon import LinearRing, Polygon

from src.utils.config import ROBOT_BETA_ANGLE, INTER_FRONT_DISTANCE, AREAL_SPEED
from src.utils.coords import perp_dot_product
from src.utils.logs import logger

from src.geometry import core, small_obstacles as geo_small
from src.coordinates import coords_small_obstacles as coords_small


class ExportError(Exception):
    pass


def write_coords_table(filename, parcel, coords_local):
    """
    1. Turn coords matrix into a pd.DataFrame
    2. Compute angles
    3. Compute m2
    4. Add orders
    5. Export

    Front structure: robot_a, robot_c, robot_b
    Variable are structure as: x_{fence}_{robot}
    """

    df = init_df(coords_local)

    # compute angle
    df = get_angle(df)

    # compute area differences
    df = get_area(df, parcel, coords_local)

    # add wait orders at the end of a connectivity > 1
    df = get_wait_order(df)

    df = get_speed(df)

    # write
    df.to_csv(filename, index=False)

    logger.info(f" # [Fleet] wrote {filename}")

def init_df(coords_local):
    """
    Generate initial dataframe
    """
    cols = [
        "x_1_a",
        "y_1_a",
        "x_1_c",
        "y_1_c",
        "x_1_b",
        "y_1_b",
        "x_2_a",
        "y_2_a",
        "x_2_c",
        "y_2_c",
        "x_2_b",
        "y_2_b",
    ]
    n_rows = coords_local.shape[0]
    df = pd.DataFrame(coords_local.reshape(n_rows, -1), columns=cols)
    cols_to_remove = [col for col in cols if "c" in col]
    df.drop(cols_to_remove, axis=1, inplace=True)

    return df


def get_wait_order(df):

    df.loc[df.connectivity.diff() == -1, "wait_order"] = 1
    df["wait_order"] = df.wait_order.shift(-1).fillna(0).astype(int)

    return df


def get_angle(df, beta=ROBOT_BETA_ANGLE):
    """
    Get angles of the four robots.
    """

    for idx in [1, 2]:

        delta_x = df[f"x_{idx}_b"] - df[f"x_{idx}_a"]
        delta_y = df[f"y_{idx}_b"] - df[f"y_{idx}_a"]

        alpha_a = 2 * np.pi - np.arctan(delta_x / delta_y)
        alpha_b = np.pi - alpha_a

        df[f"theta_{idx}_a"] = alpha_a - beta
        df[f"theta_{idx}_b"] = alpha_b + beta

    return df


def get_area(df, parcel, coords_local):

    area_grazed_last = 0
    centroid_front_last = None

    list_area_left, list_area_right, list_connectivity = [], [], []

    idx = 0
    while idx < coords_local.shape[0]:

        row = coords_local[idx]
        connectivity = get_connectivity(row)
        list_connectivity.append(connectivity)

        if connectivity == 1:

            area, area_diff, centroid_front_last, area_grazed_last = get_area_simple(
                row, parcel, centroid_front_last, area_grazed_last,
            )
            list_area_left.append(area_diff)
            list_area_right.append(area_diff)
            idx += 1

        elif connectivity == 2:

            jdx, is_inner = is_inner_obstacle(coords_local, idx)

            if is_inner:
                sub_list_area_left = get_area_inner_obs(
                    idx,
                    coords_local,
                    parcel,
                    side="left",
                    area_grazed_last=area,
                )
                sub_list_area_right = get_area_inner_obs(
                    idx,
                    coords_local,
                    parcel,
                    side="right",
                    area_grazed_last=area,
                )
                idx += jdx
                jdx -= 1

            else:
                sub_list_area_left = get_area_outer_obs(
                    idx,
                    coords_local,
                    parcel,
                    side="left",
                )
                sub_list_area_right = get_area_outer_obs(
                    idx,
                    coords_local,
                    parcel,
                    side="right",
                )
                idx += jdx

            list_area_left.extend(sub_list_area_left)
            list_area_right.extend(sub_list_area_right)
            list_connectivity += [2] * jdx
            if not is_inner:
                break

        else:
            raise NotImplementedError(f"connectivity is {connectivity}")

    df["area_left"] = list_area_left
    df["area_right"] = list_area_right
    df["connectivity"] = list_connectivity

    return df


def get_area_simple(row, parcel, centroid_front_last, area_grazed_last):

    if len(row) != 6:
        raise ExportError(f"row length is not 6: {row}")

    ls_front = LineString([row[0], row[-1]])
    holes = [obs.exterior for obs in parcel.small_obstacles]
    _, polygon_grazed = core.split_polygon(
        parcel.lr_local_perimeter,
        ls_front,
        centroid_front_last,
        holes=holes,
    )
    area_grazed = polygon_grazed.area
    area_diff = area_grazed - area_grazed_last

    centroid_front_last = ls_front.centroid
    area_grazed_last = polygon_grazed.area

    return area_grazed, area_diff, centroid_front_last, area_grazed_last


def get_area_inner_obs(idx, coords_local, parcel, side, area_grazed_last):
    """
    Get sub list of area difference for either left or right,
    upon an inner obstacle.

    1. Combine the last front_1, the obstacle and the first front_2 into a single LineString
    2. Split the polygon with this LineString

    Parameter
    --------
    idx: int,
        index of the current front to compute area difference from
    coords_local: np.ndarray,
        local coordinates table of the robots
    lr_perimeter: shapely.LinearRing,
        parcel perimeter
    side: str, "left" or "right",
        the side of the obstacle fronts to move in order to get the list of area differences
    holes: List[LinearRing],
        the inner obstacles of the parcel to remove to compute the grazed area
    area_diff_last: float,
        last computed difference of area

    Return
    ------
    list_area: List[float],
        list of area differences, matching each front of the side of the inner obstacle.
    """
    list_area = []
    ls_2_low_ref = None

    holes = [obs.exterior for obs in parcel.small_obstacles]
    obstacle = find_obstacle(coords_local[idx], holes)

    for jdx in range(idx, len(coords_local)):

        row = coords_local[jdx]
        row_last = coords_local[jdx - 1]

        if get_connectivity(row) == 1:
            break

        _, list_ls = extract_fronts(side, row, row_last)
        ls_1_low, _, ls_2_low, _ = list_ls

        if ls_2_low_ref is None:
            ls_2_low_ref = ls_2_low

        # Hack: split on ls_left and ls_right (that we call here ls_1 and ls_2)
        # instead of ls_low and ls_up.
        p_tangent_low, p_tangent_up = geo_small.get_tangent_points(ls_1_low, ls_2_low_ref, obstacle)
        
        # Which item we choose from `list_coords_obstacle` doesn't matter,
        # since we use `holes` to compute area. So let's choose the first one.
        try:
            coords_obstacle = geo_small.split_obstacle(
                p_tangent_low, p_tangent_up, obstacle
            )[0]
        except geo_small.SmallObstacleError:
            logger.warning(
                " # [CoordsExport] inner obstacle yields SmallObstacleError, use tangent points instead"
            )
            coords_obstacle = [list(p_tangent_low.coords)[0], list(p_tangent_up.coords)[0]]

        coords_1 = list(ls_1_low.coords)
        coords_2 = list(ls_2_low_ref.coords)
        ls_front = merge_coords(coords_1, coords_obstacle, coords_2, side)

        _, polygon_grazed = core.split_polygon(
            parcel.lr_local_perimeter,
            ls_front,
            last_centroid_front=parcel.start_point,
            holes=holes,
        )

        area_grazed = polygon_grazed.area
        area_diff = area_grazed - area_grazed_last
        list_area.append(area_diff)
        area_grazed_last = area_grazed

    return list_area


def get_area_outer_obs(idx, coords_local, parcel, side):
    """
    Get sub list of area difference for either left or right,
    upon an outer obstacle.

    front_1 can be either front_left or front_right.
    front_1 is the one moving at each iteration, while front_2 is fixed.
    """

    list_area = []
    row = coords_local[idx]
    row_last = coords_local[idx - 1]
    holes = [obs.exterior for obs in parcel.small_obstacles]

    list_idx, list_ls = extract_fronts(side, row, row_last)
    idx_1_left, idx_1_right, _, _ = list_idx
    ls_1, _, ls_2, _ = list_ls

    _, polygon_grazed_full = core.split_polygon(
        parcel.lr_local_perimeter,
        ls_2,
        parcel.start_point,
        holes=holes,
    )
    _, polygon_grazed_ref = core.split_polygon(
        polygon_grazed_full.exterior,
        ls_1,
        parcel.start_point,
        holes=holes,
    )
    area_grazed_last = polygon_grazed_ref.area

    for jdx in range(idx, coords_local.shape[0]):

        row = coords_local[jdx]
        if get_connectivity(row) == 1:
            break

        ls_1 = LineString([row[idx_1_left], row[idx_1_right]])

        _, polygon_grazed = core.split_polygon(
            polygon_grazed_full.exterior,
            ls_1,
            parcel.start_point,
            holes=holes,
        )
        area_grazed = polygon_grazed.area
        area_diff = area_grazed - area_grazed_last
        list_area.append(area_diff)
        area_grazed_last = area_grazed

    return list_area


def find_obstacle(row: np.ndarray, holes: list) -> Polygon:

    if not holes:
        ExportError("No obstacle but connectivity > 1 and inner obstacle detected")

    distance_min = np.inf
    obstacle = None
    ls = LineString([row[0], row[-1]])

    for hole in holes:
        distance = hole.distance(ls)
        if distance < distance_min:
            distance_min = distance
            obstacle = hole

    return Polygon(obstacle)


def get_connectivity(row):
    """
    Do not work w/ connectivity > 2

    Return
    ------
    - connectivity: 1 or 2
    """

    p_3, p_4 = Point(row[2]), Point(row[3])
    connectivity = int(p_3.distance(p_4) > INTER_FRONT_DISTANCE) + 1

    return connectivity


def extract_fronts(side, row, row_last):
    """
    Determine which points are at the left position of the front
    given the direction, and return corresponding LineString.

    We check the angle between a reference vector and the aledged
    left and right side, because some parcel geometry can cause these
    sides to inverse.

    If the angle (ref -> left) is positive and the angle (ref -> right)
    is negative (from the trigonometry standpoint), then we can
    consider that the left and right coordinates are effectively at
    the left and right position of the front.

    Parameters
    ----------
    - side: str, "left" or "right",
        side to take reference from. `ls_front_1` means that we will
        vary this front to get difference areas, while `ls_front_2`
        will stay static.
    - row: np.ndarray, shape: (,6)
        current vector of coordinates
    - row_last: np.ndarray, shape: (,6)
        previous vector of coordinates

    Return
    ------
    - idx_left: int,
        index of `ls_front_1` left point
    - idx_right: int,
        index of `ls_front_1` right point
    - Other variables are self-explainatory
    """

    if len(row) != 6:
        ExportError(f"row length is not 6: {row}")

    # check direction using one position as reference vector
    # and two others for left and right
    vec_ref = Point(np.array(row[2]) - np.array(row_last[2]))
    vec_left = Point(np.array(row[0]) - np.array(row_last[0]))
    vec_right = Point(np.array(row[-1]) - np.array(row_last[-1]))

    angle_left = perp_dot_product(vec_ref, vec_left)
    angle_right = perp_dot_product(vec_ref, vec_right)

    if angle_left > 0 and angle_right < 0:
        pass
    elif angle_left < 0 and angle_right > 0:
        side = {"left": "right", "right": "left"}[side]
    else:
        ExportError(
            f"Left and right element of row misplaced\n"
            f"row: {row}\n"
            f"row_last: {row_last}"
        )

    if side == "left":
        idx_1_left, idx_1_right = 0, 2
        idx_2_left, idx_2_right = 3, 5
        ls_1 = LineString([row[0], row[2]])
        ls_1_last = LineString([row_last[0], row_last[2]])
        ls_2 = LineString([row[3], row[5]])
        ls_2_last = LineString([row_last[3], row_last[5]])
    else:
        idx_1_left, idx_1_right = 3, 5
        idx_2_left, idx_2_right = 0, 2
        ls_2 = LineString([row[0], row[2]])
        ls_2_last = LineString([row_last[0], row_last[2]])
        ls_1 = LineString([row[3], row[5]])
        ls_1_last = LineString([row_last[3], row_last[5]])

    list_idx = [idx_1_left, idx_1_right, idx_2_left, idx_2_right]
    list_ls = [ls_1, ls_1_last, ls_2, ls_2_last]

    return list_idx, list_ls


def is_inner_obstacle(coords_local, idx):
    """
    Compute connectivity for all following rows.
    If there's a connectivity = 1 at least once, the obstacle is a inner one
    (either big or small). Otherwise, this is a outer obstacle.
    """
    for jdx, row in enumerate(coords_local[idx:, :]):
        if get_connectivity(row) == 1:
            return jdx, True
    return jdx, False


def get_ls_2_up(idx, coords_local, list_idx_2):

    idx_left, idx_right = list_idx_2
    for jdx in range(idx, len(coords_local)):
        row = coords_local[jdx]
        if get_connectivity(row) == 1:
            return LineString([row[idx_left], row[idx_right]])
    raise ExportError(
        f"Outer obstacle processed as an inner one\n"
        f"row: {row}\n"
        f"list_idx_2: {list_idx_2}"
    )


def merge_coords(coords_1, coords_obstacle, coords_2, side):
    """
    ls_front = ls_1 + obstacle + ls_2

    Parameters
    ----------
    - coords_1: list,
    - coords_obstacle: list,
    - coords_2: list,
    - side: str, "left" or "right"

    Return
    ------
    - ls_front: LineString
    """
    if side == "left":
        p_left, p_right = Point(coords_1[-1]), Point(coords_2[0])
        coords_obstacle = check_coords_obs(coords_obstacle, p_left, p_right)
        coords = coords_1 + coords_obstacle + coords_2
    else:
        p_left, p_right = Point(coords_2[-1]), Point(coords_1[0])
        coords_obstacle = check_coords_obs(coords_obstacle, p_left, p_right)
        coords = coords_2 + coords_obstacle + coords_1

    return LineString(coords)


def check_coords_obs(coords_obstacle, p_left, p_right):

    p_obs_first = Point(coords_obstacle[0])
    p_obs_last = Point(coords_obstacle[-1])

    d_1_left = LineString([p_left, p_obs_first]).length
    d_1_right = LineString([p_left, p_obs_last]).length

    d_2_left = LineString([p_right, p_obs_first]).length
    d_2_right = LineString([p_right, p_obs_last]).length

    if d_1_left < d_1_right and d_2_right < d_2_left:
        pass

    elif d_1_left > d_1_right and d_2_right > d_2_left:
        coords_obstacle = coords_obstacle[::-1]
    
    else:
        ExportError(
            f"Misplaced obstacle coordinates\n"
            f"coords_1: {p_left}\n"
            f"coords_obs: {coords_obstacle}\n"
            f"coords_2: {p_right}"
        )

    return coords_obstacle


def get_speed(df):
    """
    Compute each robot speed in cm/s
    """
    
    cols = df.columns
    col2idx = dict(zip(cols, range(cols.size)))
    v = df.values

    list_v_1_a = [0]
    list_v_1_b = [0]
    list_v_2_a = [0]
    list_v_2_b = [0]

    for idx in range(1, v.shape[0]):

        row = v[idx, :]
        row_last = v[idx-1, :]

        area_left = row[col2idx["area_left"]]
        area_right = row[col2idx["area_right"]]

        v_1_a, v_1_b = None, None
        if area_left == 0:
            v_1_a = 0
            v_1_b = 0

        v_2_a, v_2_b = None, None
        if area_right == 0:
            v_2_a = 0
            v_2_b = 0

        delta_t_left = area_left / AREAL_SPEED
        delta_t_right = area_right / AREAL_SPEED

        d_1_a, d_1_b, d_2_a, d_2_b  = compute_distances(row, row_last, col2idx)

        # mult by 100 to convert m into cm
        if v_1_a is None:
            v_1_a = 100 * d_1_a / delta_t_left
        if v_1_b is None:
            v_1_b = 100 * d_1_b / delta_t_left
        if v_2_a is None:
            v_2_a = 100 * d_2_a / delta_t_right
        if v_2_b is None:
            v_2_b = 100 * d_2_b / delta_t_right

        list_v_1_a.append(v_1_a)
        list_v_1_b.append(v_1_b)
        list_v_2_a.append(v_2_a)
        list_v_2_b.append(v_2_b)

    df["speed_1_a"] = list_v_1_a
    df["speed_1_b"] = list_v_1_b
    df["speed_2_a"] = list_v_2_a
    df["speed_2_b"] = list_v_2_b

    return df


def compute_distances(row, row_last, col2idx):
    """
    Euclidean distance of all robots between two rows.
    """

    list_distance = []
    for robot_id in ["1_a", "1_b", "2_a", "2_b"]:
        distance = np.sqrt(
            (row[col2idx[f"x_{robot_id}"]] - row_last[col2idx[f"x_{robot_id}"]])**2
            + (row[col2idx[f"y_{robot_id}"]] - row_last[col2idx[f"y_{robot_id}"]])**2
        )
        list_distance.append(distance)

    return list_distance



