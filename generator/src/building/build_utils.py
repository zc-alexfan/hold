import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os.path as op
import cv2
import sys

sys.path = [".."] + sys.path


def get_center_point(num_cams, cameras):
    A = np.zeros((3 * num_cams, 3 + num_cams))
    b = np.zeros((3 * num_cams, 1))
    camera_centers = np.zeros((3, num_cams))
    for i in range(num_cams):
        if "cam_%d" % i in cameras:
            P0 = cameras["cam_%d" % i][:3, :]
        else:
            P0 = cameras[i]
        K = cv2.decomposeProjectionMatrix(P0)[0]
        R = cv2.decomposeProjectionMatrix(P0)[1]
        c = cv2.decomposeProjectionMatrix(P0)[2]
        c = c / c[3]
        camera_centers[:, i] = c[:3].flatten()

        v = R[2, :]
        A[3 * i : (3 * i + 3), :3] = np.eye(3)
        A[3 * i : (3 * i + 3), 3 + i] = -v
        b[3 * i : (3 * i + 3)] = c[:3]

    return camera_centers


def normalize_cameras(cameras, scene_bounding_sphere, max_radius_ratio):
    all_files = cameras.keys()
    maximal_ind = 0
    for field in all_files:
        maximal_ind = np.maximum(maximal_ind, int(field.split("_")[-1]))
    num_of_cameras = maximal_ind + 1

    camera_centers = get_center_point(num_of_cameras, cameras)
    center = np.array([0, 0, 0])
    max_radius = (
        np.linalg.norm((center[:, np.newaxis] - camera_centers), axis=0).max()
        * max_radius_ratio
    )

    normalization = np.eye(4).astype(np.float32)
    normalization[0, 0] = max_radius / scene_bounding_sphere
    normalization[1, 1] = max_radius / scene_bounding_sphere
    normalization[2, 2] = max_radius / scene_bounding_sphere

    # normalization = np.eye(4).astype(np.float32)

    cameras_new = {}
    for i in range(num_of_cameras):
        cameras_new["scale_mat_%d" % i] = normalization

        # world -> cam -> pixel
        # -> MANO 3D verts -> 2D pixels
        cameras_new["world_mat_%d" % i] = cameras[
            "cam_%d" % i
        ].copy()  # original P matrix
    return cameras_new


def save_visualizations(rgb_ps, mask_ps, pts_2d, mano_fit, num_frames, out_dir, flag):
    print("Visualizing dataset..")
    for idx in tqdm(range(num_frames)):
        rgb_p = rgb_ps[idx]
        mask_p = mask_ps[idx]
        im = Image.open(rgb_p)
        mask = Image.open(mask_p)

        vis_dir = op.join(out_dir, "vis")

        # Directories for saving images
        colmap_dir = op.join(vis_dir, "colmap")
        mask_dir = op.join(vis_dir, "mask")
        j2d_dir = op.join(vis_dir, f"{flag}_fit_j2d")
        v2d_dir = op.join(vis_dir, f"{flag}_fit_v2d")

        os.makedirs(colmap_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
        os.makedirs(j2d_dir, exist_ok=True)
        os.makedirs(v2d_dir, exist_ok=True)

        # First Figure: pts_2d over im
        plt.scatter(pts_2d[idx, :, 0], pts_2d[idx, :, 1], s=2)
        plt.imshow(im)
        plt.savefig(op.join(colmap_dir, f"{idx:04}.png"))
        plt.close()  # Clear the current figure

        # Second Figure: mask over im
        plt.imshow(im)
        plt.imshow(mask, alpha=0.8)
        plt.savefig(op.join(mask_dir, f"{idx:04}.png"))
        plt.close()  # Clear the current figure

        j2d = mano_fit["j2d"]

        # Third Figure: j2d over im
        plt.scatter(j2d[idx, :, 0], j2d[idx, :, 1], s=2, color="w")
        plt.imshow(im)
        plt.savefig(op.join(j2d_dir, f"{idx:04}.png"))
        plt.close()  # Clear the current figure

        v2d = mano_fit["v2d"]

        # Fourth Figure: v2d over im
        plt.scatter(v2d[idx, :, 0], v2d[idx, :, 1], s=0.5, color="w")
        plt.imshow(im)
        plt.savefig(op.join(v2d_dir, f"{idx:04}.png"))
        plt.close()  # Clear the current figure

    print(f"Done! See results at {vis_dir}")
