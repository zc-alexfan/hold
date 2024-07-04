import torch
import torch.nn as nn
from common.xdict import xdict
from common.transforms import project2d_batch
from src.alignment.loss_terms import gmof
import numpy as np
from pytorch3d.transforms import matrix_to_axis_angle
from pytorch3d.transforms import axis_angle_to_matrix


l1_loss = nn.L1Loss(reduction="none")


def loss_fn_o2(preds, targets_r, targets_l, targets_o, conf):
    targets_o2d = targets_o["o2d.gt"]
    r3d = targets_r["v3d"]
    l3d = targets_l["v3d"]
    h3d = torch.cat([r3d, l3d], dim=1)
    o3d = preds["o3d"]

    # coarse contact
    centroid_h = h3d.mean(dim=1)
    centroid_o = o3d.mean(dim=1)
    loss = l1_loss(centroid_h, centroid_o).mean() * conf.contact

    # 2d reprojection
    loss += (
        gmof(preds["o2d"] - targets_o2d, sigma=conf.o2d_sigma).sum(dim=-1).mean()
        * conf.o2d
    )

    # encourage: in front of camera
    z_min = torch.clamp(-o3d[:, :, 2].mean(dim=1), min=0.0)
    if z_min.sum() > 0:
        loss_z = z_min.sum() / torch.nonzero(z_min).shape[0]
        loss += loss_z * conf.z_min
    return loss


class ObjectParameters(nn.Module):
    def __init__(self, data, meta):
        super().__init__()
        # unpacking
        K = meta["K"]
        o2w_all = data["o2w_all"]
        obj_rot = matrix_to_axis_angle(o2w_all[:, :3, :3])
        obj_transl = o2w_all[:, :3, 3]

        denorm_mat = data["normalize_mat"].inverse()
        device = data["normalize_mat"].device
        obj_cano_pad = torch.cat(
            [
                data["object_cano"],
                torch.ones(len(data["object_cano"]), 1, device=device),
            ],
            dim=-1,
        )
        obj_cano_denorm = (denorm_mat @ obj_cano_pad.T).T

        # object parameters
        obj_scale = torch.FloatTensor(np.array([1.0]))

        self.register_parameter("obj_scale", nn.Parameter(obj_scale))
        self.register_parameter("obj_rot", nn.Parameter(obj_rot))
        self.register_parameter("obj_transl", nn.Parameter(obj_transl))

        self.register_buffer("obj_cano", obj_cano_denorm)

        self.K = K

        targets = xdict()
        self.targets = targets
        self.im_paths = meta["im_paths"]

    def forward(self):
        num_frames = len(self.obj_rot)
        device = self.obj_rot.device

        rot_mat = axis_angle_to_matrix(self.obj_rot)

        K = self.K[None, :, :].repeat(num_frames, 1, 1).to(device)

        obj_cano = self.obj_cano[:, :3].clone() * self.obj_scale
        obj_cano = obj_cano.T[None, :, :].repeat(num_frames, 1, 1)
        # rotate
        pts_w = torch.bmm(rot_mat, obj_cano)
        pts_w = pts_w + self.obj_transl[:, :, None]
        pts_w = pts_w.permute(0, 2, 1)

        # divided by zero
        pts_w_results = pts_w.clone()
        pts_w_results[pts_w[:, :, 2] == 0.0] = 1e-8
        pts_w = pts_w_results

        out = xdict()
        o2d = project2d_batch(K, pts_w)

        out["j3d"] = pts_w
        out["j2d"] = o2d
        out["im_paths"] = self.im_paths
        out["K"] = self.K
        out["obj_scale"] = self.obj_scale

        o2w_all = torch.eye(4, device=device).unsqueeze(0).repeat(num_frames, 1, 1)
        o2w_all[:, :3, :3] = rot_mat
        o2w_all[:, :3, 3] = self.obj_transl
        out["o2w_all"] = o2w_all

        return out
