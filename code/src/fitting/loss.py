import sys
import numpy as np
import torch

# Standard Libraries
import sys


# Third-party Libraries
import numpy as np
import torch


sys.path = [".."] + sys.path
from common.transforms import project2d_batch
import pickle as pkl

from pytorch3d.ops import knn_points

# from src.fitting.generic import Generic
from src.fitting.utils import (
    l1_loss,
)

import torch.nn.functional as F

with open("./body_models/contact_zones.pkl", "rb") as f:
    contact_zones = pkl.load(f)
contact_zones = contact_zones["contact_zones"]
contact_idx = np.array([item for sublist in contact_zones.values() for item in sublist])


def generate_bbox_mask(mask_pred, margin_gap):
    def get_bounding_boxes(mask_tensor, margin_gap):
        num_masks, height, width = mask_tensor.shape

        rows = torch.any(mask_tensor, dim=2).int()
        cols = torch.any(mask_tensor, dim=1).int()

        rmin = torch.argmax(rows, dim=1)
        rmax = height - 1 - torch.argmax(rows.flip(dims=[1]), dim=1)

        cmin = torch.argmax(cols, dim=1)
        cmax = width - 1 - torch.argmax(cols.flip(dims=[1]), dim=1)

        # Extend the bounding box by margin_gap pixels, ensuring they remain within bounds
        rmin = torch.clamp(rmin - margin_gap, min=0)
        rmax = torch.clamp(rmax + margin_gap + 1, max=height)
        cmin = torch.clamp(cmin - margin_gap, min=0)
        cmax = torch.clamp(cmax + margin_gap + 1, max=width)

        return rmin, rmax, cmin, cmax

    with torch.no_grad():
        # Assuming mask_pred is a 3D tensor with shape (num_masks, height, width)
        mask_pred = torch.tensor(
            mask_pred.cpu()
        )  # Ensure the input is a PyTorch tensor
        num_masks, height, width = mask_pred.shape
        mask_box = torch.zeros_like(mask_pred)

        rmin, rmax, cmin, cmax = get_bounding_boxes(mask_pred, margin_gap=margin_gap)

        # Creating a grid of indices
        grid_r, grid_c = torch.meshgrid(torch.arange(height), torch.arange(width))

        # Adjust the dimensions for broadcasting
        grid_r = grid_r.unsqueeze(0).expand(num_masks, -1, -1)
        grid_c = grid_c.unsqueeze(0).expand(num_masks, -1, -1)

        # Create a mask for each bounding box
        inside_box = (
            (grid_r >= rmin[:, None, None])
            & (grid_r < rmax[:, None, None])
            & (grid_c >= cmin[:, None, None])
            & (grid_c < cmax[:, None, None])
        )

        mask_box[inside_box] = 1

    return mask_box


def loss_fn_h(out, targets, flag):
    v3d_h_c = out[f"{flag}.v3d_c"]
    v3d_o_c = out["object.v3d_c"]
    v3d_tips = v3d_h_c[:, contact_idx]

    # contact
    loss_fine_ho = knn_points(v3d_tips, v3d_o_c, K=1, return_nn=False)[0].mean()

    mask_h = out[f"{flag}.mask"]
    mask_o = out["object.mask"]

    valid_pix = 1 - targets[flag]
    err = l1_loss(mask_o, targets["object"]) * valid_pix
    loss_mask_o = torch.sum(err) / torch.sum(valid_pix)

    valid_pix = 1 - targets["object"]
    err = l1_loss(mask_h, targets[flag]) * valid_pix
    loss_mask_h = torch.sum(err) / torch.sum(valid_pix)

    loss_dict = {}
    loss_dict["mask_o"] = loss_mask_o * 1000
    loss_dict["mask_h"] = loss_mask_h * 1000
    loss_dict["fine_ho"] = loss_fine_ho * 100.0

    loss = sum(loss_dict.values())
    loss_dict["loss"] = loss
    return loss_dict


def loss_fn_rh(out, targets):
    return loss_fn_h(out, targets, "right")


def loss_fn_lh(out, targets):
    return loss_fn_h(out, targets, "left")


def loss_fn_ih(out, targets):
    valid_pix = (1 - targets["right"]) * (1 - targets["left"])
    err = l1_loss(out["object.mask"], targets["object"]) * valid_pix
    loss_mask_o = torch.sum(err) / torch.sum(valid_pix)

    v3d_r_tips = out[f"right.v3d_c"][:, contact_idx]
    v3d_l_tips = out[f"left.v3d_c"][:, contact_idx]
    v3d_o = out["object.v3d_c"]

    dist_thres = 2.0**2  # meter
    loss_contact_ro = knn_points(v3d_r_tips, v3d_o, K=1, return_nn=False)[0][
        :, :, 0
    ].mean(dim=1)
    loss_contact_lo = knn_points(v3d_l_tips, v3d_o, K=1, return_nn=False)[0][
        :, :, 0
    ].mean(dim=1)
    loss_contact_ro[loss_contact_ro < dist_thres] = 0
    loss_contact_lo[loss_contact_lo < dist_thres] = 0
    loss_contact_ro = loss_contact_ro.mean()
    loss_contact_lo = loss_contact_lo.mean()

    K = out["K"][:, :3, :3]
    j2d_r = project2d_batch(K, out["right.v3d_c"])
    j2d_l = project2d_batch(K, out["left.v3d_c"])

    if "j2d_r_target" not in targets:
        targets["j2d_r_target"] = j2d_r.detach().clone()
        targets["j2d_l_target"] = j2d_l.detach().clone()

    j2d_r_target = targets["j2d_r_target"]
    j2d_l_target = targets["j2d_l_target"]

    loss_2d_r = F.mse_loss(j2d_r, j2d_r_target)
    loss_2d_l = F.mse_loss(j2d_l, j2d_l_target)

    loss_dict = {}
    loss_dict["mask_o"] = loss_mask_o * 1000
    loss_dict["v2d_r"] = loss_2d_r * 1.0
    loss_dict["v2d_l"] = loss_2d_l * 1.0
    loss_dict["contact_ro"] = loss_contact_ro * 0.05
    loss_dict["contact_lo"] = loss_contact_lo * 0.05

    loss = sum(loss_dict.values())
    loss_dict["loss"] = loss
    return loss_dict
