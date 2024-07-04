import torch.nn as nn
import torch
import torch.nn.functional as F
import common.torch_utils as torch_utils


eps = 1e-6
l1_loss = nn.L1Loss(reduction="none")
l2_loss = nn.MSELoss(reduction="none")
from src.utils.const import SEGM_IDS


# L1 reconstruction loss for RGB values
def get_rgb_loss(rgb_values, rgb_gt, valid_pix, scores):
    rgb_loss = l1_loss(rgb_values, rgb_gt) * valid_pix[:, None]
    num_pix = rgb_loss.shape[0] // scores.shape[0]
    scores = scores[:, None].repeat(1, num_pix).view(-1, 1)
    rgb_loss = rgb_loss * scores
    rgb_loss = rgb_loss.sum() / (valid_pix.sum() + 1e-6)
    return rgb_loss


# Eikonal loss introduced in IGR
def get_eikonal_loss(grad_theta):
    eikonal_loss = ((grad_theta.norm(2, dim=-1) - 1) ** 2).mean()
    return eikonal_loss


# BCE loss for clear boundary
def get_bce_loss(acc_map, scores):
    bce_loss = (
        acc_map * (acc_map + eps).log() + (1 - acc_map) * (1 - acc_map + eps).log()
    )
    num_pix = bce_loss.shape[0] // scores.shape[0]
    scores = scores[:, None].repeat(1, num_pix).view(-1, 1)
    bce_loss = bce_loss * scores

    binary_loss = -1 * (bce_loss).mean() * 2
    return binary_loss


# Global opacity sparseness regularization
def get_opacity_sparse_loss(acc_map, index_off_surface, scores):
    opacity_sparse_loss = l1_loss(
        acc_map[index_off_surface], torch.zeros_like(acc_map[index_off_surface])
    )

    num_pix = acc_map.shape[0] // scores.shape[0]
    scores = scores[:, None].repeat(1, num_pix).view(-1, 1)
    scores = scores[index_off_surface]
    opacity_sparse_loss = opacity_sparse_loss * scores

    opacity_sparse_loss = opacity_sparse_loss.mean()
    return opacity_sparse_loss


def get_mask_loss(mask_prob, mask_gt, valid_pix):
    assert torch.all((mask_gt == 0) | (mask_gt == 1))
    # bce loss on mask
    mask_loss = F.binary_cross_entropy(
        mask_prob, mask_gt[:, None].float(), reduction="none"
    )
    mask_loss = mask_loss * valid_pix[:, None]
    mask_loss = mask_loss.sum() / (valid_pix.sum() + 1e-6)
    return mask_loss


def get_sem_loss(sem_pred, mask_gt, valid_pix, scores):
    semantic_gt = mask_gt.clone()

    bnd_bg = semantic_gt < 25
    bnd_o = torch.logical_and(25 <= semantic_gt, semantic_gt < 100)
    bnd_r = torch.logical_and(100 <= semantic_gt, semantic_gt < 200)
    bnd_l = 200 <= semantic_gt

    # bandaid fix for aliasinng
    semantic_gt[bnd_bg] = SEGM_IDS["bg"]
    semantic_gt[bnd_o] = SEGM_IDS["object"]
    semantic_gt[bnd_r] = SEGM_IDS["right"]
    semantic_gt[bnd_l] = SEGM_IDS["left"]

    # remap to 0,1,2
    semantic_gt[semantic_gt == SEGM_IDS["bg"]] = 0
    semantic_gt[semantic_gt == SEGM_IDS["object"]] = 1
    semantic_gt[semantic_gt == SEGM_IDS["right"]] = 2
    semantic_gt[semantic_gt == SEGM_IDS["left"]] = 3

    semantic_gt_onehot = torch_utils.one_hot_embedding(semantic_gt, len(SEGM_IDS)).to(
        mask_gt.device
    )
    sem_loss = l2_loss(sem_pred, semantic_gt_onehot) * valid_pix[:, None]

    num_pix = sem_loss.shape[0] // scores.shape[0]
    scores = scores[:, None].repeat(1, num_pix).view(-1, 1)
    sem_loss = sem_loss * scores

    sem_loss = sem_loss.sum() / valid_pix.sum()
    return sem_loss


def get_mano_cano_loss(pred_sdf, gt_sdf, limit, scores):
    pred_sdf = torch.clamp(pred_sdf, -limit, limit)
    gt_sdf = torch.clamp(gt_sdf, -limit, limit)
    mano_cano_loss = l1_loss(pred_sdf, gt_sdf)

    scores = scores[:, None]

    mano_cano_loss = mano_cano_loss * scores

    mano_cano_loss = mano_cano_loss.mean()
    return mano_cano_loss
