import numpy as np
import os.path as op
from easydict import EasyDict as edict
from aitviewer.utils.so3 import aa2rot_numpy
from aitviewer.renderables.point_clouds import PointClouds

import sys

sys.path = [".."] + sys.path
from common.viewer import ViewerData, HOLDViewer
from common.viewer import materials


def model_output_to_meshes(init_out, f3d_r):
    v3d = init_out["v3d"]  # (time, 778, 3)
    from aitviewer.renderables.meshes import Meshes

    meshes_r = Meshes(v3d, f3d_r)
    return meshes_r


def main(args):
    from aitviewer.renderables.meshes import Meshes

    MANO_DIR_R = "../code/body_models/MANO_RIGHT.pkl"
    MANO_DIR_L = "../code/body_models/MANO_LEFT.pkl"
    import smplx

    f3d_r = mano_layer = smplx.create(
        model_path=MANO_DIR_R, model_type="mano", use_pca=False, is_rhand=True
    ).faces
    f3d_l = mano_layer = smplx.create(
        model_path=MANO_DIR_L, model_type="mano", use_pca=False, is_rhand=False
    ).faces
    sealed_vertices_sem_idx = np.load(
        "../code/body_models/sealed_vertices_sem_idx.npy", allow_pickle=True
    )
    tip_sem_idx = [12, 11, 4, 5, 6]
    vidx = np.concatenate([sealed_vertices_sem_idx[sem_idx] for sem_idx in tip_sem_idx])
    meshes_all = {}
    rotation_flip = aa2rot_numpy(np.array([1, 0, 0]) * np.pi)

    vis_p = f"./data/{args.seq_name}/processed/hold_fit.aligned.npy"
    data = np.load(vis_p, allow_pickle=True).item()

    if "right" in data:
        v3d_r = data["right"]["v3d"]
        im_paths = data["right"]["im_paths"]
        K = data["right"]["K"]

        meshes_all["right"] = Meshes(
            v3d_r,
            f3d_r,
            rotation=rotation_flip,
            flat_shading=True,
            material=materials["red"],
        )

    if "left" in data:
        v3d_l = data["left"]["v3d"]
        meshes_all["left"] = Meshes(
            v3d_l,
            f3d_l,
            rotation=rotation_flip,
            flat_shading=True,
            material=materials["blue"],
        )

    # PointClouds
    pts_w = data["object"]["j3d"]
    assert op.exists(im_paths[0])
    meshes_all["object"] = PointClouds(pts_w, point_size=5.0, rotation=rotation_flip)

    num_frames = len(v3d_r)
    Rt = np.eye(4)[None, :, :]
    Rt = np.repeat(Rt, num_frames, axis=0)
    Rt = Rt[:, :3, :]
    Rt[:, 1:3, :3] *= -1.0

    from PIL import Image

    images = [Image.open(im_p) for im_p in im_paths]

    width, height = images[0].size

    viewer_data = ViewerData(Rt, K, width, height, imgnames=images)

    viewer = HOLDViewer(size=(2000, 2000))
    viewer.render_seq([meshes_all, viewer_data])
    viewer.view_interactive()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_name", type=str, default="")
    args = parser.parse_args()
    args = edict(vars(args))

    main(args)
