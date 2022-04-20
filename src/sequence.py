from shapely.geometry.linestring import LineString
from tqdm import tqdm
from matplotlib import pyplot as plt

from src.geometry import (
    small_obstacles as small_obs,
    interpolation as inter,
    list_front,
    core,
    smoothness as smooth,
)

from src.utils.logs import logger
from src.utils.config import SMOOTH_ANGLE_MAX, SMOOTH_ANGLE_MIN


class SequenceError(Exception):
    pass


class Sequence:
    def __init__(self, parcel):
        self.parcel = parcel
        self.start_point = parcel.water_tanks[0].centroid
        self.lr_perimeter = parcel.lr_local_perimeter
        self.ls_sweep_axis = parcel.ls_sweep_axis
        self.list_ls_front = []
        self.list_ls_front_interpolated = []
        self.list_centroid = []
        self.last_rate_area_grazed = 0

    def get_list_front(self):

        logger.info(f" # [Sequence] scan of {self.parcel.name}")

        ls_front = list_front.get_first_ls_front(self.lr_perimeter, self.ls_sweep_axis)
        last_centroid_front = None

        polygon_direction, polygon_grazed = core.split_polygon(
            self.lr_perimeter, ls_front, last_centroid_front
        )

        centroid_front_next = core.forward_centroid_front(
            ls_front,
            point_direction=polygon_direction.centroid,
        )

        self.list_ls_front = list_front.get_list_front(
            self.lr_perimeter,
            ls_front,
            centroid_front_next,
            polygon_direction,
            polygon_grazed,
        )
        return self.list_ls_front

    def get_interpolated_front(self):

        logger.info(f" # [Sequence] interpolate front of {self.parcel.name}")

        if not self.list_ls_front:
            raise SequenceError(
                "list_ls_front is empty, you need to run `get_list_front` first"
            )

        n_front = len(self.list_ls_front)

        for idx in tqdm(range(n_front - 1)):

            ls_front_low = self.list_ls_front[idx]
            ls_front_up = self.list_ls_front[idx + 1]

            if isinstance(self.list_ls_front[idx + 1], list):

                ls_tangent = self.list_ls_front[idx]
                list_front_left, list_front_right = self.list_ls_front[idx + 1]
                (
                    list_interpolated_left,
                    list_interpolated_right,
                ) = inter.interpolate_outer_obs(
                    self.lr_perimeter, ls_tangent, list_front_left, list_front_right
                )

                self.list_ls_front_interpolated.append(
                    [list_interpolated_left, list_interpolated_right]
                )

            else:
                list_front_interpolated_current = inter.get_interpolated_front(
                    self.lr_perimeter, ls_front_low, ls_front_up
                )
                self.list_ls_front_interpolated.extend(list_front_interpolated_current)

        return self.list_ls_front_interpolated

    def dodge_big_obstacle(self, big_obstacle):
        # TODO
        pass

    def dodge_small_obstacle(self, small_obstacle):
        """
        Update coords to dodge a small obstacle.

        Important: dodging a small obstacle while connectivity > 1 is not implemented yet.

        1. Identify the 2 closest fronts (non secant) to the obstacle
            - get the closest front
            - divide the parcel in two polygons
            - for each polygon, get the closest front to the centroid
        2. Make them tangent to their respective closest point of the obstacle
        3. Split the obstacles with left and right pair of points
        """
        idx_front_low, idx_front_up = small_obs.get_closest_fronts(
            self.lr_perimeter,
            self.list_ls_front_interpolated,
            small_obstacle,
            self.start_point,
        )

        ls_low = self.list_ls_front_interpolated[idx_front_low]
        ls_up = self.list_ls_front_interpolated[idx_front_up]
        p_tangent_low, p_tangent_up = small_obs.get_tangent_points(
            ls_low, ls_up, small_obstacle
        )

        list_coords_obstacle = small_obs.split_obstacle(
            p_tangent_low, p_tangent_up, small_obstacle
        )

        return (
            list_coords_obstacle,
            [idx_front_low, idx_front_up],
            [ls_low, ls_up],
            [p_tangent_low, p_tangent_up],
        )

    def linear_smooth_small_obstacle(
        self, centroid_obs, list_ls_tangent, list_p_tangent
    ):
        """
        Smooth the fast moving coordinates by anticipating obstacles, hence limiting
        the speed needed by the robot.

        1. Detect the side of the low centroid w.r.t the obstacle (both low and up front)
           and the angle
        2. Find the intersection with joint line (both low and up)
        3. Bring the joint centroid closer to the obstacle centroid iteratively, while
           updating the coordinates (for both low and up)
        """

        logger.info(" # [Fleet] Linear smoothing of small obstacle")

        list_idx_front = smooth.convert_ls_to_idx(
            self.list_ls_front_interpolated, list_ls_tangent
        )
        success = False
        for angle in range(SMOOTH_ANGLE_MAX, SMOOTH_ANGLE_MIN, -5):
            list_angle = smooth.get_angles(
                self.list_ls_front_interpolated, centroid_obs, list_idx_front, angle
            )
            try:
                list_p_inter = smooth.find_intersections(
                    self.list_ls_front_interpolated,
                    centroid_obs,
                    list_idx_front,
                    list_angle,
                )
                success = True
                logger.info(f" # [Sequence] Smoothing angle found: {angle}")
                break
            except smooth.SmoothnessError:
                continue

        if not success:
            logger.warning(
                f" # [Sequence] No smoothing angle between {SMOOTH_ANGLE_MIN} and {SMOOTH_ANGLE_MAX} found"
            )
            return None

        list_ls_centroid = [
            LineString([list_p_inter[0], list_p_tangent[0]]),
            LineString([list_p_tangent[1], list_p_inter[1]]),
        ]

        return list_ls_centroid

    def plot(self, show_interpolation=True):

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)

        ax.plot(*self.lr_perimeter.xy)

        if show_interpolation:
            list_ls_front = self.list_ls_front_interpolated
        else:
            list_ls_front = self.list_ls_front

        for ls_front in list_ls_front:

            if isinstance(ls_front, list):
                for sub_list_ls_front in ls_front:
                    for ls_front_current in sub_list_ls_front:
                        self._plot(ax, ls_front_current)

            else:
                self._plot(ax, ls_front)

        ax.set_title(self.parcel.name)
        plt.show()
        ax.remove()

    def _plot(self, ax, ls_front):
        centroid_front = ls_front.centroid
        ax.plot(centroid_front.x, centroid_front.y, "o")
        ax.plot(*ls_front.xy)
