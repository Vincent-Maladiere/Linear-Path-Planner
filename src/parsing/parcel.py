import os
from enum import Enum
from matplotlib import pyplot as plt
import dill as pickle
from slugify import slugify
import numpy as np

import networkx as nx
from shapely.geometry import LinearRing, LineString, Polygon
from shapely.ops import cascaded_union, snap
from shapely.affinity import rotate
from geopy.distance import geodesic
from skimage.draw import polygon2mask

from src.utils.logs import logger
from src.utils.config import (
    MIN_OBSTACLE_WIDTH,
    SNAP_ACTIVATE,
    SNAP_TOLERANCE,
    SNAP_USE_CONVEX_HULL,
)
from src.utils.shapely_figures import (
    SIZE,
    plot_coords,
    plot_line,
    plot_poly,
    BLACK,
    DARKGRAY,
    YELLOW,
    GREEN,
    RED,
    BLUE,
    PURPLE,
)
from src.utils.smallest_enclosing_circle import make_circle
from src.utils.coords import get_local_coords

COLOR_WATER_TANK = BLUE
COLOR_OBSTACLE = RED
COLOR_SMALL_OBSTACLE = PURPLE
COLOR_GATE = YELLOW
COLOR_PERIMETER = DARKGRAY
COLOR_SWEEP_AXIS = BLUE
COLOR_IGNORED = GREEN


class ParcelItem(str, Enum):

    perimeter = "perimeter"
    obstacle = "obstacle"
    water_tank = "watertank"
    gate = "gate"
    sweep_axis = "sweep axis"


class ParcelGroup:
    def __init__(self, workdir, name, dict_parcel, graph_parcel) -> None:
        self.workdir = workdir
        self.name = name
        self.dict_parcel = dict_parcel
        self.graph_parcel = graph_parcel

    def split_small_obstacles(self):
        """
        Separate small obstacles to regular obstacles.

        1. Merging obstacles and watertanks altogether
        2. Snapping (projecting obstacle to the parcel perimeter)
        """
        for parcel in self.dict_parcel.values():
            parcel.merge_obstacles()
            if SNAP_ACTIVATE:
                parcel.snap_obstacles()
            parcel.split_small_obstacles()

    def rotate_sweep_axis(self):
        for parcel in self.dict_parcel.values():
            parcel.rotate_sweep_axis()

    def convert_binary(self):
        for parcel in self.dict_parcel.values():
            parcel.convert_binary()

    def plot_parcels(self):
        """
        Plot each parcel together using plot method individually.
        """
        fig = plt.figure(1, figsize=SIZE, dpi=90)
        ax = fig.add_subplot(111)
        for parcel in self.dict_parcel.values():
            parcel._plot(ax, step="original")
        ax.set_title(self.name)
        plt.axis("scaled")
        plt.show()
        ax.remove()
    
    def plot_rot(self):
        for parcel in self.dict_parcel.values():
            parcel.plot(step="rot")
    
    def plot_binary(self):
        for parcel in self.dict_parcel.values():
            parcel.plot(step="binary")

    def plot_graph(self):
        """
        Plot graph of linked parcel.
        """
        ax = plt.subplot(111)
        pos = nx.kamada_kawai_layout(self.graph_parcel)
        nx.draw_networkx_edges(self.graph_parcel, pos, alpha=0.3, edge_color="m")
        nx.draw_networkx_nodes(self.graph_parcel, pos, node_color="#210070", alpha=0.9)
        label_options = {"ec": "k", "fc": "white", "alpha": 0.7}
        nx.draw_networkx_labels(
            self.graph_parcel, pos, font_size=14, bbox=label_options
        )
        plt.show()
        ax.remove()

    def save_figs(self):
        """
        Save each fig and dump the `ParcelGroup` instance as a pickle file.
        """
        for parcel in self.dict_parcel.values():
            parcel.save_figs(self.workdir)
        filename_parcel_group = os.path.join(self.workdir, f"parcel_group_{self.slug_name}.pkl")
        pickle.dump(self, open(filename_parcel_group, "wb+"))

    @property
    def slug_name(self):
        return slugify(self.name)


class Parcel:
    def __init__(self, name, **kwargs):

        self.name = name
        self.inner_offset = kwargs.get("inner_offset", 1)
        self.outer_offset = kwargs.get("outer_offset", 1)
        self.lr_local_perimeter = None
        self.gates = list()
        self.obstacles = list()
        self.small_obstacles = list()
        self.water_tanks = list()
        self.ls_sweep_axis = None
        self.point_ref = None
        self.angle_alpha_deg = None
        self.start_point = None

    def add_item(self, item_type, coords_global, point_ref):
        """
        Add any item from `ParcelItem` to the parcel

        Parameters
        ----------
        - item_type: str,
            item from Enumerator `ParcelItem`
        - coords_global: List[Tuple[float, float]],
            longitude and Latitude coordinates of the item
        - point_ref: Tuple[float, float],
            parser reference point to use for global to local coordinate conversion.
        """

        if self.point_ref is None:
            self.point_ref = point_ref

        coords_local = get_local_coords(coords_global, self.point_ref)
        item_type = item_type.strip()

        if item_type == ParcelItem.perimeter:
            self.add_perimeter(coords_local)

        elif item_type == ParcelItem.gate:
            self.add_gate(coords_local)

        elif item_type == ParcelItem.obstacle:
            self.add_obstacle(coords_local)

        elif item_type == ParcelItem.water_tank:
            # water tank are also obstacles
            self.add_obstacle(coords_local)
            self.add_water_tank(coords_local)

        elif item_type == ParcelItem.sweep_axis:
            self.add_sweep_axis(coords_local)
        else:
            raise ValueError(f" # [Parcel] {item_type} is an unknown type")

    def add_perimeter(self, coords_local):
        self.lr_local_perimeter = LinearRing(coords_local)
        self.lr_local_perimeter_offset = self.lr_local_perimeter.parallel_offset(
            self.inner_offset,
            "left",
            resolution=16,
            join_style=1,
            mitre_limit=5.0,
        )

    def add_gate(self, coords_local):
        ls_gate = LineString(coords_local)
        self.gates.append(ls_gate)

    def add_sweep_axis(self, coords_local):
        if self.ls_sweep_axis is not None:
            logger.error("Multiple sweep axis for the same parcel detected! Overwrite.")
        self.ls_sweep_axis = LineString(coords_local)

    def add_obstacle(self, coords_local):
        p_obstacle = Polygon(coords_local)
        self.obstacles.append(p_obstacle)

    def add_water_tank(self, coords_local):
        lr_water_tank = Polygon(coords_local)
        self.water_tanks.append(lr_water_tank)
        if self.start_point is None:
            self.start_point = lr_water_tank.centroid

    def merge_obstacles(self):
        """
        Merge 'close' obstacles together.

        1. Add external buffer to Polygon exterior
        2. Merge overlapping polygons
        3. Remove external buffer of every merged and unmerged polygon
        """
        obstacles_offset = []
        for p_obstacle in self.obstacles:
            p_obstacle_offset = Polygon(
                p_obstacle.exterior.buffer(MIN_OBSTACLE_WIDTH / 2).exterior
            )
            obstacles_offset.append(p_obstacle_offset)

        obstacles_merged = cascaded_union(obstacles_offset)
        # We want `obstacles_merged` to be iterable: a MultiPolygon or a list of a single Polygon
        if isinstance(obstacles_merged, Polygon):
            obstacles_merged = [obstacles_merged]

        obstacles_processed = []
        for p_obstacle_offset in obstacles_merged:
            p_obstacle_processed = Polygon(
                p_obstacle_offset.exterior.buffer(MIN_OBSTACLE_WIDTH / 2).interiors[0]
            )
            obstacles_processed.append(p_obstacle_processed)

        self.obstacles = obstacles_processed

    def snap_obstacles(self):
        """
        Snapping is projecting a geometry onto another.
        We project the obstacle onto the Parcel perimeter.

        Tolerance represent the snapping reach.

        1. For a given LinearRing, add points to its perimeter
           to increase snapping reach
        2. Snap every obstacles with the parcel perimeter
        """
        lr_perimeter_augmented = self._get_perimeter_augmented()

        obstacles_snap = []
        for p_obstacle in self.obstacles:
            p_obstacle_snap = snap(
                p_obstacle, lr_perimeter_augmented, tolerance=SNAP_TOLERANCE
            )
            if SNAP_USE_CONVEX_HULL:
                p_obstacle_snap = p_obstacle_snap.convex_hull
            obstacles_snap.append(p_obstacle_snap)

        self.obstacles = obstacles_snap

    def _get_perimeter_augmented(self):
        """
        Add points to the perimeter every `SNAP_TOLERANCE` meters.
        """
        coords_perimeter_augmented = []
        coords_perimeter = list(self.lr_local_perimeter.coords)

        for idx in range(len(coords_perimeter) - 1):
            coord_1, coord_2 = coords_perimeter[idx], coords_perimeter[idx + 1]
            length_line = LineString((coord_1, coord_2)).length
            n_segments = length_line // SNAP_TOLERANCE
            coords_perimeter_augmented.append(coord_1)
            for idx in range(1, int(n_segments)):
                alpha = idx / n_segments
                x = alpha * coord_1[0] + (1 - alpha) * coord_2[0]
                y = alpha * coord_1[1] + (1 - alpha) * coord_2[1]
                coords_perimeter_augmented.append((x, y))
            coords_perimeter_augmented.append(coord_2)
        lr_perimeter_augmented = LinearRing(coords_perimeter_augmented)

        return lr_perimeter_augmented
        
    def split_small_obstacles(self):
        """
        Split each obstacles between 'regular' obstacles and 'small' ones.
        `small_obstacle` will be ignored during path computation, but
        considered by the robots upon executing movements.

        For each obstacles, compute the smallest enclosing circle and
        compare its diameter to the classification threshold `MIN_OBSTACLE_WIDTH`.
        """
        regular_obstacles = []
        small_obstacles = []
        for p_obstacle in self.obstacles:
            coords_obstacle = list(p_obstacle.exterior.coords)
            _, _, radius = make_circle(coords_obstacle)
            if 2 * radius < MIN_OBSTACLE_WIDTH:
                small_obstacles.append(p_obstacle)
            else:
                regular_obstacles.append(p_obstacle)
        self.obstacles = regular_obstacles
        self.small_obstacles = small_obstacles

    def rotate_sweep_axis(self):
        """
        1. get the alpha angle in degree
        2. apply the angle to all the elements
        """
        angle = self.get_angle_alpha()
        p_centroid = self.lr_local_perimeter.convex_hull.centroid

        self.lr_local_perimeter_rot = rotate(
            self.lr_local_perimeter, angle, origin=p_centroid
        )
        self.lr_local_perimeter_offset_rot = rotate(
            self.lr_local_perimeter_offset, angle, origin=p_centroid
        )
        self.obstacles_rot = [
            rotate(obs, angle, origin=p_centroid) for obs in self.obstacles
        ]
        self.small_obstacles_rot = [
            rotate(obs, angle, origin=p_centroid) for obs in self.small_obstacles
        ]
        self.water_tanks_rot = [
            rotate(wt, angle, origin=p_centroid) for wt in self.water_tanks
        ]
        self.gates_rot = [rotate(gate, angle, origin=p_centroid) for gate in self.gates]
        self.ls_sweep_axis_rot = rotate(self.ls_sweep_axis, angle, origin=p_centroid)

        self.angle_alpha_deg = angle

    def get_angle_alpha(self):
        """
        Compute the sweep axis vertical angle

        - Alpha is the angle to apply to rotate the object vertically
        - A and B are the points of the sweep axis, C is the orthogonal
        projection of both points.
        - Theta is the angle (AB, BC)

        A +     + C

                + B

        Steps
        -----
        1. sort sweep axis coordinates by x increasing
        2. choose the triangle configuration, depending on when AB is ascendant, descendant or flat
        3. find theta by computing the arccos(BC/AB) and convert it to degree
        4. find alpha
        """

        coords = sorted(self.ls_sweep_axis.coords, key=lambda x: x[0])
        (x_a, y_a), (x_b, y_b) = coords

        if y_a == y_b:
            return 90

        sign_alpha = -1 if y_a > y_b else 1
        x_c, y_c = x_b, y_a
        segment_ab = LineString([(x_a, y_a), (x_b, y_b)]).length
        segment_bc = LineString([(x_b, y_b), (x_c, y_c)]).length
        angle_theta_rad = np.arccos(segment_bc / segment_ab)
        angle_theta_deg = angle_theta_rad * 180 / np.pi
        angle_alpha_deg = angle_theta_deg * sign_alpha

        return angle_alpha_deg

    def convert_binary(self):

        coords_perimeter = np.array(self.lr_local_perimeter_rot.coords)

        # define min values and floor the coords to center the figure
        # and avoid negative index values.
        x_min_perimeter = coords_perimeter[:, 0].min()
        y_min_perimeter = coords_perimeter[:, 1].min()
        coords_perimeter[:, 0] -= x_min_perimeter
        coords_perimeter[:, 1] -= y_min_perimeter

        # get max values of the centered perimeter
        # these max define the image shape
        x_max = int(np.ceil(coords_perimeter[:, 0].max()))
        y_max = int(np.ceil(coords_perimeter[:, 1].max()))
        image_shape = (y_max, x_max)

        # image represention space and matrix representation
        # space invert x and y axis.
        coords_perimeter_swap = np.array(
            [coords_perimeter[:, 1], coords_perimeter[:, 0]]
        ).T
        mask = polygon2mask(image_shape, coords_perimeter_swap)
        mask = np.uint8(mask)

        # obstacles rot, using image shape
        obstacles = self.obstacles_rot
        for obstacle in obstacles:
            coords_obstacle = np.array(obstacle.exterior.coords)
            coords_obstacle[:, 0] -= x_min_perimeter
            coords_obstacle[:, 1] -= y_min_perimeter
            coords_obstacle_swap = np.array(
                [coords_obstacle[:, 1], coords_obstacle[:, 0]]
            ).T
            mask_obstacle = polygon2mask(image_shape, coords_obstacle_swap)
            mask_obstacle = np.uint8(mask_obstacle)
            mask[np.where(mask_obstacle)] = 0

        self.mask = mask
        self.image_shape = image_shape
        self.x_min_perimeter = x_min_perimeter
        self.y_min_perimeter = y_min_perimeter

    def plot(self, step="original"):
        fig = plt.figure(1, figsize=SIZE, dpi=90)
        ax = fig.add_subplot(111)
        self._plot(ax, step)
        ax.set_title(self.name)
        plt.axis("scaled")
        plt.show()

    def save_figs(self, workdir):

        self._save_fig(workdir, step="original")
        self._save_fig(workdir, step="rot")
        self._save_fig(workdir, step="binary")

    def _save_fig(self, workdir, step):

        filename = os.path.join(workdir, f"{self.slug_name}_{step}.jpg")
        if step in ["original", "rot"]:
            fig = plt.figure(1, figsize=SIZE, dpi=90)
            ax = fig.add_subplot(111)
            # ax = plt.axes([0,0,1,1])
            self._plot(ax, step)
            # plt.axis("off")
            plt.axis("scaled")
            plt.savefig(filename)  # , bbox_inches='tight', pad_inches=0)
            ax.remove()
        elif step == "binary":
            plt.imsave(filename, self.mask, cmap="gray")
            filename_matrix = os.path.join(workdir, f"{self.slug_name}_input_matrix.pkl")
            pickle.dump(self.mask, open(filename_matrix, "wb+"))
        else:
            raise ValueError(f" # [Parcel] _save_fig - {step} is not a valid option.")

    def _plot(self, ax, step):

        if step == "rot":
            if getattr(self, "obstacles_rot", None) is None:
                raise ValueError(
                    "Parcel needs to be rotated first. Use _rotate_sweep_axis."
                )
            obstacles = self.obstacles_rot
            small_obstacles = self.small_obstacles_rot
            water_tanks = self.water_tanks_rot
            gates = self.gates_rot
            ls_sweep_axis = self.ls_sweep_axis_rot
            lr_local_perimeter = self.lr_local_perimeter_rot
            lr_local_perimeter_offset = self.lr_local_perimeter_offset_rot
        elif step == "original":
            obstacles = self.obstacles
            small_obstacles = self.small_obstacles
            water_tanks = self.water_tanks
            gates = self.gates
            ls_sweep_axis = self.ls_sweep_axis
            lr_local_perimeter = self.lr_local_perimeter
            lr_local_perimeter_offset = self.lr_local_perimeter_offset
        elif step == "binary":
            plt.imshow(self.mask, cmap="gray")
            return
        else:
            raise ValueError(f" # [Parcel] _plot - {step} is not a valid option")

        for lr_obstacle in obstacles:
            # plot_coords(ax, lr_obstacle, color=COLOR_OBSTACLE)
            plot_poly(ax, lr_obstacle, color=COLOR_OBSTACLE)

        for lr_small_obstacle in small_obstacles:
            plot_poly(ax, lr_small_obstacle, color=COLOR_SMALL_OBSTACLE)

        for lr_water_tank in water_tanks:
            # plot_coords(ax, lr_water_tank, color=COLOR_WATER_TANK)
            plot_poly(ax, lr_water_tank, color=COLOR_WATER_TANK)

        for ls_gate in gates:
            # plot_coords(ax, gate.ls, color=COLOR_GATE)
            plot_line(ax, ls_gate, linewidth=6, color=COLOR_GATE)

        if ls_sweep_axis is None:
            logger.warning(" # [Parcel] plot - Sweep axis is not defined")
        else:
            plot_coords(ax, ls_sweep_axis, color=COLOR_SWEEP_AXIS)
            plot_line(ax, ls_sweep_axis, color=COLOR_SWEEP_AXIS)

        plot_coords(ax, lr_local_perimeter, color=COLOR_PERIMETER)
        plot_line(ax, lr_local_perimeter, color=COLOR_PERIMETER)

        plot_coords(ax, lr_local_perimeter_offset)
        plot_line(ax, lr_local_perimeter_offset)

    @property
    def slug_name(self):
        return slugify(self.name)

    @staticmethod
    def get_item_width(coords_global):

        point_left = coords_global[0]
        point_right = coords_global[-1]
        return geodesic(point_left, point_right).m

    def __str__(self):
        return self.name

    def __setattr__(self, name, value):
        """
        When water_tanks is updated, start_point is updated as well
        """

        if name == "water_tanks":

            if not isinstance(value, (tuple, list)):
                raise ValueError("'water_tanks' must be a list of shapely.Polygon")
            if len(value) > 0:
                if not isinstance(value[0], Polygon):
                    raise ValueError("'water_tanks' must be a list of shapely.Polygon")
                else:
                    self.start_point = value[0].centroid

        super().__setattr__(name, value)
        