import torch.nn as nn
import torch

import torch
import torch.nn as nn
import sys
from src.alignment.loss_terms import gmof

sys.path = [".."] + sys.path
from pytorch3d.ops import knn_points

mse_loss = nn.MSELoss(reduction="none")
l1_loss = nn.L1Loss(reduction="none")


import torch


def esimate_contact(j2d_r, j2d_l, j2d_o, thres):
    from pytorch3d.ops import knn_points

    bbox_r = get_bbox_from_kp2d_torch(j2d_r)
    bbox_l = get_bbox_from_kp2d_torch(j2d_l)
    bbox_o = get_bbox_from_kp2d_torch(j2d_o)

    knn_dists, knn_idx, _ = knn_points(j2d_o, j2d_r, None, None, K=1, return_nn=True)
    knn_dists = knn_dists.sqrt()[:, :, 0]

    dist_r = knn_dists.min(dim=1).values

    knn_dists, knn_idx, _ = knn_points(j2d_o, j2d_l, None, None, K=1, return_nn=True)
    knn_dists = knn_dists.sqrt()[:, :, 0]

    dist_l = knn_dists.min(dim=1).values

    contact_r = dist_r < thres * bbox_r[:, 3]
    contact_l = dist_l < thres * bbox_l[:, 3]
    return contact_r, contact_l


def get_bbox_from_kp2d_torch(kp_2d):
    if kp_2d.dim() > 2:
        ul = torch.stack(
            [
                kp_2d[:, :, 0].min(dim=1)[0],  # upper left x
                kp_2d[:, :, 1].min(dim=1)[0],  # upper left y
            ],
            dim=1,
        )
        lr = torch.stack(
            [
                kp_2d[:, :, 0].max(dim=1)[0],  # lower right x
                kp_2d[:, :, 1].max(dim=1)[0],  # lower right y
            ],
            dim=1,
        )
    else:
        ul = torch.tensor([kp_2d[:, 0].min(), kp_2d[:, 1].min()])  # upper left
        lr = torch.tensor([kp_2d[:, 0].max(), kp_2d[:, 1].max()])  # lower right

    w = lr[:, 0] - ul[:, 0]
    h = lr[:, 1] - ul[:, 1]
    c_x, c_y = ul[:, 0] + w / 2, ul[:, 1] + h / 2

    # To keep the aspect ratio
    max_side = torch.maximum(w, h)
    w = h = max_side * 1.0

    bbox = torch.stack([c_x, c_y, w, h], dim=1)  # shape = (N, 4)
    return bbox


def bbox_iou_hand(bbox1, bbox2):
    # bbox1: hand bbox
    # this IoU compute intersection(bbox1, bbox2)/area(bbox1)

    # Expand dimensions to allow broadcasting
    box1 = bbox1.unsqueeze(1)  # Shape: (N, 1, 4)
    box2 = bbox2.unsqueeze(0)  # Shape: (1, M, 4)

    # Calculate the coordinates of the intersections
    inter_x1 = torch.max(
        box1[:, :, 0] - box1[:, :, 2] / 2, box2[:, :, 0] - box2[:, :, 2] / 2
    )
    inter_y1 = torch.max(
        box1[:, :, 1] - box1[:, :, 3] / 2, box2[:, :, 1] - box2[:, :, 3] / 2
    )
    inter_x2 = torch.min(
        box1[:, :, 0] + box1[:, :, 2] / 2, box2[:, :, 0] + box2[:, :, 2] / 2
    )
    inter_y2 = torch.min(
        box1[:, :, 1] + box1[:, :, 3] / 2, box2[:, :, 1] + box2[:, :, 3] / 2
    )

    # Calculate intersection area
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(
        inter_y2 - inter_y1, min=0
    )

    # Calculate union area
    box1_area = box1[:, :, 2] * box1[:, :, 3]
    box2_area = box2[:, :, 2] * box2[:, :, 3]
    #     union_area = box1_area + box2_area - inter_area
    union_area = box1_area
    # Compute the IoU
    iou = inter_area / union_area
    return iou


def loss_fn_h(preds, targets, conf):
    # op2d
    loss = 0.0
    device = preds["right.j2d"].device

    targets_j2d = targets["right.j2d.gt"].to(device)
    is_valid = ~torch.isnan(targets_j2d[:, 0, 0])
    loss_2d = gmof(
        preds["right.j2d"][is_valid] - targets_j2d[is_valid, :, :2],
        sigma=conf.j2d_sigma,
    ).sum(dim=-1)
    loss_2d = loss_2d.mean() * conf.j2d

    loss += loss_2d

    targets_j2d = targets["left.j2d.gt"].to(device)
    is_valid = ~torch.isnan(targets_j2d[:, 0, 0])
    loss_2d = gmof(
        preds["left.j2d"][is_valid] - targets_j2d[is_valid, :, :2], sigma=conf.j2d_sigma
    ).sum(dim=-1)
    loss_2d = loss_2d.mean() * conf.j2d
    loss += loss_2d
    loss /= 2.0
    return loss


def huber_loss(y_true, y_pred, delta=5.0):
    error = y_true - y_pred
    is_small_error = torch.abs(error) <= delta
    squared_loss = 0.5 * error**2
    linear_loss = delta * (torch.abs(error) - 0.5 * delta)
    loss = torch.where(is_small_error, squared_loss, linear_loss)
    return loss.mean()


def loss_fn_o(preds, targets, conf):
    contact_r, contact_l = esimate_contact(
        targets["right.j2d.gt"][:, :, :2],
        targets["left.j2d.gt"][:, :, :2],
        targets["object.j2d.gt"],
        thres=0.01,
    )

    targets_o2d = targets["object.j2d.gt"]
    r3d = targets["right.v3d"]
    l3d = targets["left.v3d"]
    o3d = preds["object.j3d"]

    dist_r = knn_points(r3d, o3d, K=1, return_nn=False)[0]
    dist_l = knn_points(l3d, o3d, K=1, return_nn=False)[0]

    dist_r = dist_r.min(dim=1)[0][:, 0]
    dist_l = dist_l.min(dim=1)[0][:, 0]
    loss_r = dist_r[contact_r].mean() * conf.contact
    loss_l = dist_l[contact_l].mean() * conf.contact
    loss_h = (loss_r + loss_l) / 2.0

    loss = 0.0
    loss += loss_h

    # 2d reprojection
    loss_2d = (
        gmof(preds["object.j2d"] - targets_o2d, sigma=conf.o2d_sigma).sum(dim=-1).mean()
        * conf.o2d
    )
    loss += loss_2d

    # encourage: in front of camera
    z_min = torch.clamp(-o3d[:, :, 2].mean(dim=1), min=0.0)
    loss_z = 0.0
    if z_min.sum() > 0:
        loss_z = z_min.sum() / torch.nonzero(z_min).shape[0] * conf.z_min
        loss += loss_z

    z_r = r3d[:, :, 2].mean(dim=1)
    z_l = l3d[:, :, 2].mean(dim=1)
    z_o = o3d[:, :, 2].mean(dim=1)
    loss_ro = huber_loss(z_r - z_o, torch.zeros_like(z_r).detach()).mean()
    loss_lo = huber_loss(z_l - z_o, torch.zeros_like(z_l).detach()).mean()
    loss += (loss_ro + loss_lo) / 2.0 * 1000.0
    # print(f"loss: {loss} loss_h: {loss_h}, loss_2d: {loss_2d}, loss_z: {loss_z}")
    return loss


def loss_fn_ho(preds, targets, conf):
    v3d_r = preds["right.v3d"]
    v3d_l = preds["left.v3d"]
    v3d_o = preds["object.j3d"]

    centroid_r = v3d_r.mean(dim=1)
    centroid_l = v3d_l.mean(dim=1)
    centroid_o = v3d_o.mean(dim=1)
    diff_r = centroid_r[:-1] - centroid_r[1:]
    diff_l = centroid_l[:-1] - centroid_l[1:]
    diff_o = centroid_o[:-1] - centroid_o[1:]
    loss_smooth_r = mse_loss(diff_r, torch.zeros_like(diff_r).detach()).mean()
    loss_smooth_l = mse_loss(diff_l, torch.zeros_like(diff_l).detach()).mean()
    loss_smooth_o = mse_loss(diff_o, torch.zeros_like(diff_o).detach()).mean()
    loss = 0.0

    loss += (loss_smooth_r + loss_smooth_l + loss_smooth_o) / 3.0 * 100.0
    return loss


from generator.src.alignment.pl_module.generic_module import PLModule


class ARCTICModule(PLModule):
    def __init__(self, data, args, conf):
        super().__init__(data, args, conf, loss_fn_h, loss_fn_o, loss_fn_ho)
