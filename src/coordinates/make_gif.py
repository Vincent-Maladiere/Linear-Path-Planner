import os
import glob
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image

from shapely import ops, affinity
from shapely.geometry import LineString, Polygon


def run(fleet):

    write_imgs(fleet)

    fp_in = f"{os.path.join(fleet.imgdir, fleet.parcel.name)}/*.jpg"
    fp_out = os.path.join(fleet.imgdir, f"{fleet.parcel.name}.gif")

    img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
    img.save(fp=fp_out, format='GIF', append_images=imgs,
            save_all=True, duration=10, loop=0, dpi=(150, 150))


def write_imgs(fleet):

    for idx in tqdm(range(fleet.coords_local.shape[0])):

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111)

        ax.plot(*fleet.parcel.lr_local_perimeter.xy, color="white", linewidth=2)
        for small_obstacle in fleet.parcel.small_obstacles:
            ax.fill(*small_obstacle.exterior.xy, color="#F20409", zorder=200)

        row = fleet.coords_local[idx, :, :]
        p_1, p_2, p_3, p_4, p_5, p_6 = row

        ax.plot(*p_1, "o", color="#1207FC", markersize=8.92, zorder=100)
        ax.plot(*p_2, "^", color="#1207FC", markersize=19.12, zorder=100)
        ax.plot(*p_3, "o", color="#1207FC", markersize=8.92, zorder=100)
        ax.plot(*p_4, "o", color="#1207FC", markersize=8.92, zorder=100)
        ax.plot(*p_5, "^", color="#1207FC", markersize=19.12, zorder=100)
        ax.plot(*p_6, "o", color="#1207FC", markersize=8.92, zorder=100)

        ls_left = LineString([p_1, p_3])
        ls_right = LineString([p_4, p_6])

        for jdx in np.arange(1, 30, 1):

            centroid_side_left = ls_left.parallel_offset(jdx, "left").centroid
            centroid_side_right = ls_left.parallel_offset(jdx, "right").centroid

            distance_left = centroid_side_left.distance(fleet.start_point)
            distance_right = centroid_side_right.distance(fleet.start_point)
            side = "left" if distance_left < distance_right else "right"

            ls_left_offset = ls_left.parallel_offset(jdx, side)
            ls_right_offset = ls_right.parallel_offset(jdx, side)

            poly_left = Polygon([*ls_left.coords, *ls_left_offset.coords])
            poly_right = Polygon([*ls_right.coords, *ls_right_offset.coords])

            list_poly_split = ops.split(
                Polygon(fleet.sequence.lr_perimeter),
                affinity.scale(LineString([p_1, p_6]), 1.5, 1.5),
            )

            for poly_split in list_poly_split:
                if poly_split.contains(fleet.start_point):
                    poly_grazed = poly_split

            ax.fill(*poly_left.exterior.xy, color="white", alpha=0.4 / jdx)
            ax.fill(*poly_right.exterior.xy, color="white", alpha=0.4 / jdx)
            ax.fill(*poly_grazed.exterior.xy, color="white", alpha=0.009, zorder=0)

        ax.plot(*ls_left.xy, color="#1207FC", linewidth=3)
        ax.plot(*ls_right.xy, color="#1207FC", linewidth=3)

        ax.set_title(fleet.parcel.name)
        plt.axis("off")

        path = os.path.join(fleet.imgdir, fleet.parcel.name)
        Path(path).mkdir(parents=True, exist_ok=True)
        plt.savefig(f"{path}/{idx:04}.jpg")
        plt.show()

        ax.remove()
