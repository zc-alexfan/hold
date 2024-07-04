import torch.nn as nn
import torch

import torch
import torch.nn as nn
import sys
from src.alignment.loss_terms import gmof

sys.path = [".."] + sys.path

mse_loss = nn.MSELoss(reduction="none")
l1_loss = nn.L1Loss(reduction="none")


def loss_fn_h(preds, targets, conf):
    # op2d
    loss = 0.0
    device = preds["right.j2d"].device
    targets_j2d = targets["right.j2d.gt"].to(device)
    # weights = targets_j2d[:, :, 2] + 0.1
    is_valid = ~torch.isnan(targets_j2d[:, 0, 0])

    loss_2d = gmof(
        preds["right.j2d"][is_valid] - targets_j2d[is_valid, :, :2],
        sigma=conf.j2d_sigma,
    ).sum(dim=-1)

    loss_2d = loss_2d.mean() * conf.j2d
    loss += loss_2d
    return loss


def loss_fn_o(preds, targets, conf):
    targets_o2d = targets["object.j2d.gt"]
    v3d = targets["right.j3d"]
    o3d = preds["object.j3d"]

    # coarse contact
    centroid_h = v3d.mean(dim=1)
    centroid_o = o3d.mean(dim=1)
    loss = l1_loss(centroid_h, centroid_o).mean() * conf.contact

    # 2d reprojection
    loss += (
        gmof(preds["object.j2d"] - targets_o2d, sigma=conf.o2d_sigma).sum(dim=-1).mean()
        * conf.o2d
    )

    # encourage: in front of camera
    z_min = torch.clamp(-o3d[:, :, 2].mean(dim=1), min=0.0)
    if z_min.sum() > 0:
        loss_z = z_min.sum() / torch.nonzero(z_min).shape[0]
        loss += loss_z * conf.z_min
    return loss


def loss_fn_ho(preds, targets, conf):
    v3d_h = preds["right.v3d"]
    v3d_o = preds["object.j3d"]

    centroid_h = v3d_h.mean(dim=1)
    centroid_o = v3d_o.mean(dim=1)
    diff_h = centroid_h[:-1] - centroid_h[1:]
    diff_o = centroid_o[:-1] - centroid_o[1:]
    loss_smooth_h = mse_loss(diff_h, torch.zeros_like(diff_h).detach()).mean()
    loss_smooth_o = mse_loss(diff_o, torch.zeros_like(diff_o).detach()).mean()
    loss = loss_smooth_h + loss_smooth_o
    loss = loss * 100.0
    return loss


from generator.src.alignment.pl_module.generic_module import PLModule


class HOModule(PLModule):
    def __init__(self, data, args, conf):
        super().__init__(data, args, conf, loss_fn_h, loss_fn_o, loss_fn_ho)
