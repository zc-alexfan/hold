import numpy as np
import matplotlib.pyplot as plt
import trimesh
from glob import glob
import os
from PIL import Image
from tqdm import tqdm
import sys

sys.path = ["."] + sys.path
import src.hand_pose.slerp as slerp


def plot_2d_keypoints(fnames, outliers, j2d_all):
    for idx, fname in tqdm(enumerate(fnames), total=len(fnames)):
        # Load the image
        im_full = Image.open(fname.replace("/processed/crop_image/", "/images/"))

        # Prepare the output path
        out_p = fname.replace("/crop_image/", "/2d_keypoints/")
        os.makedirs(os.path.dirname(out_p), exist_ok=True)

        # Extract the transformed keypoints for the current frame
        j2d_current = j2d_all[idx, :, :2]

        # Determine the color based on whether the current index is an outlier
        color = "r" if idx in outliers else "b"

        # Plotting
        plt.figure(figsize=(15, 15))
        plt.imshow(im_full)
        plt.scatter(j2d_current[:, 0], j2d_current[:, 1], s=20, color=color)
        plt.axis("off")  # Optional: to hide axes for visual clarity
        plt.savefig(out_p)
        plt.close()
    return out_p


def crop2full_keypoints(num_frames, boxes, j2d_list):
    # map 2d keypoints from crop space to full image space
    img_size = 224

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scale_x = (x2 - x1) / img_size
    scale_y = (y2 - y1) / img_size
    affine_matrices = np.zeros((num_frames, 3, 3))
    affine_matrices[:, 0, 0] = scale_x
    affine_matrices[:, 1, 1] = scale_y
    affine_matrices[:, 0, 2] = x1
    affine_matrices[:, 1, 2] = y1
    affine_matrices[:, 2, 2] = 1

    # denormalize 2d keypoints
    j2d_homogeneous = np.concatenate(
        [j2d_list, np.ones((j2d_list.shape[0], j2d_list.shape[1], 1))], axis=2
    )
    j2d_all = j2d_homogeneous @ affine_matrices.swapaxes(-1, -2)
    return j2d_all


def validate_poses(seq_name):
    print(f"Validating {seq_name}")
    fnames = sorted(glob(f"./data/{seq_name}/processed/mesh_fit_vis/*_right_fine.ply"))
    assert len(fnames) > 0

    volumes = compute_mesh_volumes(fnames)
    outliers = slerp.identify_outliers(volumes)
    print(f"Outlier indices: {outliers}")

    # SLERP
    num_frames = len(fnames)
    key_frames = np.array([f for f in list(range(num_frames)) if f not in outliers])

    data = np.load(
        f"./data/{seq_name}/processed/hold_fit.init.npy", allow_pickle=True
    ).item()["right"]

    out = slerp.slerp_mano_params(outliers, num_frames, key_frames, data)
    out_slerp_p = f"./data/{seq_name}/processed/hold_fit.slerp.npy"
    print(f"Saving to {out_slerp_p}")
    np.save(out_slerp_p, {"right": out})

    fnames = sorted(glob(f"./data/{seq_name}/processed/crop_image/*"))
    assert len(fnames) > 0

    # denormalize 2d keypoints
    boxes = np.load(f"./data/{seq_name}/processed/boxes.npy")
    j2d_list = np.load(f"./data/{seq_name}/processed/j2d.crop.npy")
    j2d_all = crop2full_keypoints(num_frames, boxes, j2d_list)

    out_p = plot_2d_keypoints(fnames, outliers, j2d_all)

    # metro to mano ordering
    mano2metro = np.array(
        [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
    )
    metro2mano = np.argsort(mano2metro)
    j2d_all[outliers] *= np.nan
    j2d_all = j2d_all[:, :, :2]
    j2d_all = j2d_all[:, metro2mano]
    out_p = f"./data/{seq_name}/processed/j2d.full.npy"
    np.save(out_p, {"j2d.right": j2d_all})
    print(f"Saved to {out_p}")


def compute_mesh_volumes(fnames):
    metrics = []
    for fname in fnames:
        mesh = trimesh.load(fname, process=False)
        metrics.append(mesh.volume)
    volumes = np.array(metrics)
    return volumes


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_name", type=str, default=None)
    args = parser.parse_args()
    validate_poses(args.seq_name)
