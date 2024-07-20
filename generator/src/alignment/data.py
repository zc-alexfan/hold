from glob import glob
import torch
import numpy as np
from PIL import Image
import sys

sys.path = [".."] + sys.path
from common.xdict import xdict
import trimesh
from src.hand_pose.slerp import slerp_xyz


def read_data(seq_name, K):
    # load data
    im_ps = sorted(glob(f"./data/{seq_name}/images/*"))
    j2d_p = f"./data/{seq_name}/processed/j2d.full.npy"
    colmap_pts = trimesh.load(
        f"./data/{seq_name}/processed/colmap/sparse_points_normalized.obj",
        process=False,
    ).vertices

    o2w_all = torch.FloatTensor(np.load(f"./data/{seq_name}/processed/colmap/o2w.npy"))

    # move in front of camera to avoid stucking at the back
    o2w_all[:, 2, 3] += 1.0

    normalize_mat = torch.FloatTensor(
        np.load(f"./data/{seq_name}/processed/colmap/normalization_mat.npy")
    )

    if K is None:
        input_img = np.array(Image.open(im_ps[0]))
        focal_length = max(input_img.shape[0], input_img.shape[1])
        K = torch.FloatTensor(
            np.array(
                [
                    [focal_length, 0.0, input_img.shape[1] // 2],
                    [0.0, focal_length, input_img.shape[0] // 2],
                    [0.0, 0.0, 1.0],
                ]
            )
        )

    data = np.load(
        f"./data/{seq_name}/processed/hold_fit.slerp.npy", allow_pickle=True
    ).item()

    def read_hand_data(data):
        mydata = {}
        for k, v in data.items():
            mydata[k] = torch.FloatTensor(v)
        return mydata

    j2d_data = np.load(j2d_p, allow_pickle=True).item()

    meta = {}
    meta["K"] = K
    meta["im_paths"] = im_ps

    data_o = {}
    data_o["j2d.gt"] = torch.FloatTensor(
        np.load(f"./data/{seq_name}/processed/colmap_2d/keypoints.npy")
    )
    data_o["object_cano"] = torch.FloatTensor(colmap_pts)
    data_o["o2w_all"] = o2w_all
    data_o["normalize_mat"] = normalize_mat

    mydata = xdict()
    entities = {}
    if "right" in data:
        data_r = read_hand_data(data["right"])
        j2d_right = j2d_data["j2d.right"]
        j2d_right = slerp_xyz(j2d_right)

        ## CVPR version:
        # right_valid = (~np.isnan(j2d_right.reshape(-1, 21*2).mean(axis=1))).astype(np.float32) # num_frames
        # right_valid = np.repeat(right_valid[:, np.newaxis], 21, axis=1)
        # right_valid = np.ones_like(right_valid) ### no invalid

        ## New version as it is more robust
        right_valid = np.ones((j2d_right.shape[0], 21))
        j2d_right_pad = torch.FloatTensor(
            np.concatenate([j2d_right, right_valid[:, :, None]], axis=2)
        )
        data_r["j2d.gt"] = j2d_right_pad
        entities["right"] = data_r

    if "left" in data:
        data_l = read_hand_data(data["left"])

        j2d_left = j2d_data["j2d.left"]
        j2d_left = slerp_xyz(j2d_left)

        # left_valid = (~np.isnan(j2d_left.reshape(-1, 21*2).mean(axis=1))).astype(np.float32)
        # left_valid = np.repeat(left_valid[:, np.newaxis], 21, axis=1)
        # left_valid = np.ones_like(left_valid) ### no invalid

        left_valid = np.ones((j2d_left.shape[0], 21))
        j2d_left_pad = torch.FloatTensor(
            np.concatenate([j2d_left, left_valid[:, :, None]], axis=2)
        )

        data_l["j2d.gt"] = j2d_left_pad
        entities["left"] = data_l

    # mydata['object'] = data_o
    entities["object"] = data_o
    mydata["entities"] = entities
    mydata["meta"] = meta
    return mydata


class FakeDataset(torch.utils.data.Dataset):
    def __init__(self, num_iter):
        self.num_iter = num_iter

    def __len__(self):
        return self.num_iter

    def __getitem__(self, idx):
        return idx
