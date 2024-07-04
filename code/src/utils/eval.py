import numpy as np
import torch
import os.path as op
import sys

sys.path = ["../", "../generator"] + sys.path
from src.utils.eval_modules import compute_bounding_box_centers, convert_to_tensors
import torch.nn.functional as F


def load_eval_space_data(sd_p):
    # Load prediction data

    data_pred = load_ckpt_data_for_eval(sd_p)
    exp_id = sd_p.split("/")[1]
    v3d_h_c_pred = data_pred["v3d_h_c"]
    v3d_o_c_pred = data_pred["v3d_o_c"]
    fnames_pred = data_pred["fnames"]
    full_seq_name = fnames_pred[0].split("/")[2]
    seq_name = full_seq_name.split("_")[1]

    # Load ground truth data
    data_gt = load_gt_ho3d_data(seq_name, full_seq_name)
    assert np.linalg.norm(data_pred["K"] - data_gt["K"]) < 1e-9
    v3d_h_c_gt = data_gt["v3d_h_c"]
    v3d_o_c_gt = data_gt["v3d_o_c"]
    is_valid = data_gt["is_valid"]

    # scale pred mask to gt mask size
    masks_gt = data_gt["masks_gt"]

    # Get sequence full name
    seq_full_name = fnames_pred[0].split("/")[-5]

    # Select filenames from directory
    with open(f"./data/{seq_full_name}/build/corres.txt", "r") as f:
        selected_fnames = sorted([line.strip() for line in f])
    assert len(selected_fnames) > 0

    # Get selected file IDs
    selected_fids = np.array(
        [int(op.basename(fname).split(".")[0]) for fname in selected_fnames]
    )
    assert len(selected_fids) > 0

    # Select ground truth data based on selected file IDs
    v3d_h_c_gt = v3d_h_c_gt[selected_fids]
    v3d_o_c_gt = v3d_o_c_gt[selected_fids]
    j3d_h_c_gt = data_gt["j3d_h_c"][selected_fids]
    fnames_gt = np.array(data_gt["fnames"])[selected_fids]
    is_valid = is_valid[selected_fids]

    assert len(fnames_gt) == len(fnames_pred)

    # Adjust joint positions relative to the first joint
    j3d_h_c_gt_ra = j3d_h_c_gt - j3d_h_c_gt[:, :1]
    j3d_h_c_pred_ra = data_pred["j3d_h_c"] - data_pred["j3d_h_c"][:, :1]

    # Compute object root positions
    root_o_gt = compute_bounding_box_centers(v3d_o_c_gt)
    root_o_pred = compute_bounding_box_centers(v3d_o_c_pred)

    # Set validity flag
    is_valid = torch.FloatTensor(is_valid)
    # Update data dictionaries with new info
    data_pred.update(
        {
            "j3d_h_c": data_pred["j3d_h_c"],
            "j3d_h_c_ra": j3d_h_c_pred_ra,
            "v3d_h_c": v3d_h_c_pred,  # this is in camera space
            "v3d_o_c_pred": v3d_o_c_pred,
            "v3d_o_c_ra": v3d_o_c_pred - root_o_pred[:, None, :],
            "v3d_o_c_rh": v3d_o_c_pred - data_pred["j3d_h_c"][:, :1, :],
            "root_o": root_o_pred,
            "faces_o": data_pred["faces_o"],
            "faces_h": data_pred["faces_h"],
        }
    )
    data_gt.update(
        {
            "j3d_h_c": j3d_h_c_gt,
            "j3d_h_c_ra": j3d_h_c_gt_ra,
            "v3d_h_c": v3d_h_c_gt,
            "v3d_o_c": v3d_o_c_gt,
            "v3d_o_c_ra": v3d_o_c_gt - root_o_gt[:, None, :],
            "v3d_o_c_rh": v3d_o_c_gt - j3d_h_c_gt[:, :1, :],
            "root_o": root_o_gt,
            "masks_gt": masks_gt,
            "faces_o": data_gt["faces_o"],
            "faces_h": data_gt["faces_h"],
        }
    )

    if "insta_map" in data_pred:
        masks_pred = data_pred["insta_map"]
        masks_pred = (
            F.interpolate(
                masks_pred[:, None, :, :], size=masks_gt.shape[1:], mode="nearest"
            )
            .squeeze(1)
            .numpy()
        )
        data_pred.update({"masks_pred": masks_pred})

    # Prepare metadata
    meta = {
        "fnames": fnames_pred,
        "is_valid": is_valid,
        "seq_name": seq_name,
        "seq_full_name": seq_full_name,
        "K": data_pred["K"],
        "exp_id": exp_id,
    }

    data_pred = convert_to_tensors(data_pred)
    data_gt = convert_to_tensors(data_gt)
    meta = convert_to_tensors(meta)
    return data_pred, data_gt, meta
