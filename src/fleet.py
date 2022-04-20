import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from slugify import slugify

from shapely.geometry import Point, LineString, Polygon

from src.utils.config import DISTANCE_STEP_INTERPOLATE
from src.utils.logs import logger
from src.utils import binary_search

from src.sequence import Sequence
from src.coordinates import (
    coords_smoothness as coords_smooth,
    coords_small_obstacles as coords_small,
    coords_core,
    coords_export,
    make_gif,
    profil,
)

plt.style.use("dark_background")


class Fleet:
    def __init__(self, imgdir, name="beta-fleet", n_front=2, n_robot_per_front=3):
        self.imgdir = imgdir
        self.n_front = n_front
        self.n_robot_per_front = n_robot_per_front
        self.name = name
        self.coords_global = np.zeros((0, self.n_front * self.n_robot_per_front, 2))

    def forward(self, parcel):
        """
        Bring the fleet forward, by filling `coords_local` with sequence results.
        """
        logger.info(" # [Fleet] forward")

        self.coords_local = np.zeros((0, self.n_front * self.n_robot_per_front, 2))

        self.parcel = parcel
        self.lr_perimeter = parcel.lr_local_perimeter
        self.start_point = parcel.start_point

        self.sequence = Sequence(parcel)
        self.list_ls_front = self.sequence.get_list_front()
        self.list_ls_front_interpolated = self.sequence.get_interpolated_front()

        self.update_coords()
        self.dodge_obstacles()

    def update_coords(self):

        for ls_front in self.list_ls_front_interpolated:

            if isinstance(ls_front, list):
                list_ls_left, list_ls_right = ls_front
                length = max(len(list_ls_left), len(list_ls_right))
                # repeat the last item for the shortest list
                if len(list_ls_left) < len(list_ls_right):
                    list_ls_left += list_ls_left[-1:] * (length - len(list_ls_left))
                else:
                    list_ls_right += list_ls_right[-1:] * (length - len(list_ls_right))

                for ls_left, ls_right in zip(list_ls_left, list_ls_right):

                    coords_left = list(ls_left.coords)
                    p_1 = Point(coords_left[0])
                    p_2 = ls_left.centroid
                    p_3 = Point(coords_left[1])

                    coords_right = list(ls_right.coords)
                    p_4 = Point(coords_right[0])
                    p_5 = ls_right.centroid
                    p_6 = Point(coords_right[1])

                    points = [p_1, p_2, p_3, p_4, p_5, p_6]
                    self.coords_local = coords_core.add_coords(
                        self.coords_local, points
                    )
            else:
                points = coords_core.get_p_front(ls_front, ls_front.centroid)
                self.coords_local = coords_core.add_coords(self.coords_local, points)

    def dodge_obstacles(self):

        logger.info(
            f" # [Fleet] Got {len(self.parcel.obstacles)} obstacle(s)"
        )
        obstacles = sorted(self.parcel.obstacles, key=lambda x: x.distance(self.start_point))
        for obstacle in obstacles:
            (
                list_obstacle_coords,
                list_idx_tangent,
                list_ls_tangent,
                list_p_tangent,
            ) = self.sequence.dodge_obstacle(obstacle)
            self.update_coords_obstacle(
                obstacle, list_obstacle_coords, list_idx_tangent, list_p_tangent
            )
            list_ls_centroid_smooth = self.sequence.linear_smooth_obstacle(
                obstacle.centroid, list_ls_tangent, list_p_tangent
            )
            if list_ls_centroid_smooth is None:
                continue
            self.update_coords_smooth(list_ls_centroid_smooth)

    def update_coords_obstacle(
        self, small_obstacle, list_coords_obstacle, list_idx_front, list_p_tangent
    ):
        """
        - Find tangent points for low front and up front
        - Generate coords using the left and right side of the small obstacle
        - Merge all the local coords
        """

        idx_front_low, idx_front_up = list_idx_front
        ls_front_low = self.list_ls_front_interpolated[idx_front_low]
        ls_front_up = self.list_ls_front_interpolated[idx_front_up]

        # use func to find new idx front 
        p_tangent_low, p_tangent_up = list_p_tangent

        idx_coords_low = binary_search.get_nearest_ls_idx(self.coords_local, p_tangent_low)
        idx_coords_up = binary_search.get_nearest_ls_idx(self.coords_local, p_tangent_up)
        array_coords_low = self.coords_local[idx_coords_low]
        array_coords_up = self.coords_local[idx_coords_up]

        p_1_low, p_6_low = Point(array_coords_low[0]), Point(array_coords_low[5])
        p_1_up, p_6_up = Point(array_coords_up[0]), Point(array_coords_up[5])

        list_p_tangent_low = coords_small.get_tangent_points(
            self.lr_perimeter, ls_front_low, p_tangent_low, p_1_low, p_6_low
        )
        list_p_tangent_up = coords_small.get_tangent_points(
            self.lr_perimeter, ls_front_up, p_tangent_up, p_1_up, p_6_up
        )

        list_p_ref = [p_1_low, p_6_low]
        matrix_p_left, matrix_p_right = coords_small.generate_coords(
            self.lr_perimeter,
            list_coords_obstacle,
            list_p_tangent_low,
            list_p_ref,
            self.start_point,
        )
        
        self.merge_obstacle_coords(
            small_obstacle,
            matrix_p_left,
            matrix_p_right,
            p_tangent_low,
            list_p_tangent_up,
        )

    def merge_obstacle_coords(
        self,
        small_obstacle,
        matrix_p_left,
        matrix_p_right,
        p_tangent_low,
        list_p_tangent_up,
    ):
        """
        Merging order:

        coords_low
        ---
        coords_update
        coords_tangent_up
        ---
        coords_up
        """
        # remove duplicate coords
        idx_to_remove = []
        for idx in range(len(self.coords_local)):
            coords = self.coords_local[idx]
            ls_front = LineString([coords[0], coords[-1]])
            if small_obstacle.distance(ls_front) < DISTANCE_STEP_INTERPOLATE:
                idx_to_remove.append(idx)
        self.coords_local = np.delete(self.coords_local, idx_to_remove, axis=0)

        idx_coords_low = binary_search.get_nearest_ls_idx(self.coords_local, p_tangent_low)

        # split the coords using the lowest front
        coords_low = self.coords_local[:idx_coords_low, :]
        coords_up = self.coords_local[idx_coords_low + 1 :, :]

        # merge the obstacle left and right side into a single coords row
        for list_left_p, list_right_p in zip(matrix_p_left, matrix_p_right):
            list_p = list_left_p + list_right_p
            coords_update = np.array([list(p.coords) for p in list_p]).reshape(1, 6, 2)
            coords_low = np.vstack([coords_low, coords_update])

        # add the upside tangent front to close the dodging move
        coords_tangent_up = np.array(
            [list(p.coords) for p in list_p_tangent_up]
        ).reshape(1, 6, 2)
        coords_low = np.vstack([coords_low, coords_tangent_up])

        # add the upside coords
        self.coords_local = np.vstack([coords_low, coords_up])

    def update_coords_smooth(self, list_ls_centroid):
        """
        Apply the smooth centroids to `local_coords`.

        Update p_2, p_3, p_4, p_5 based on the intersections with the centroid list.

        Parameter
        ---------
        - ls_centroid_smooth: List[LineString],
            The low and up linestring of centroids.

        """
        range_idx_low, range_idx_up, range_idx_obs = coords_smooth.get_range_idx_smooth(
            self.coords_local, list_ls_centroid
        )

        ls_centroid_low, ls_centroid_up = list_ls_centroid
        for idx in range_idx_low:
            self.coords_local[idx] = coords_smooth.get_updated_coords(
                self.coords_local[idx], ls_centroid_low
            )

        for jdx in range_idx_up:
            self.coords_local[jdx] = coords_smooth.get_updated_coords(
                self.coords_local[jdx], ls_centroid_up
            )

        list_ls_centroid_obs = coords_smooth.get_list_centroid_obs(
            self.coords_local, range_idx_obs
        )

        for kdx in range_idx_obs:
            try:
                self.coords_local[kdx] = coords_smooth.get_updated_coords_obs(
                    self.coords_local[kdx], list_ls_centroid_obs
                )
            except coords_smooth.CoordsSmoothnessError as e:
                logger.error(e)

    def plot(self):

        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(111)
        self.sequence.parcel._plot(ax, step="original")
        colors = ["red", "blue", "green", "yellow", "orange", "pink"]
        for list_coords in self.coords_local:
            for coords, color in zip(list_coords, colors):
                ax.plot(coords[0], coords[1], "o", color=color)
            ls_left = LineString([list_coords[0], list_coords[2]])
            ls_right = LineString([list_coords[3], list_coords[5]])
            ax.plot(*ls_left.xy, color="gray")
            ax.plot(*ls_right.xy, color="gray")

        # for small_obstacle in self.parcel.small_obstacles:
        #    ax.fill(*small_obstacle.exterior.xy, color="black")

        plt.show()
        ax.remove()

    def make_gif(self):

        logger.info(" # [Fleet] start making gif")
        make_gif.run(self)

    def write_coords(self):

        logger.info(" # [Fleet] start writing coords table")
        filename = os.path.join(self.imgdir, f"{slugify(self.parcel.name)}.coords_table.csv")
        coords_export.write_coords_table(filename, self.parcel, self.coords_local)
        
    def plot_profil(self):

        filename = os.path.join(self.imgdir, f"{slugify(self.parcel.name)}.coords_table.csv")
        profil.plot_profil(filename)
