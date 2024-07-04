# Standard Libraries
import sys

# Third-party Libraries
import numpy as np
import torch
import torch.nn as nn
import pymeshlab as ml

# PyTorch3D imports
from pytorch3d.renderer import (
    BlendParams,
    MeshRasterizer,
    MeshRenderer,
    PerspectiveCameras,
    RasterizationSettings,
    SoftSilhouetteShader,
    TexturesVertex,
)
from PIL import Image
from pytorch3d.structures import Meshes

# Internal modules
sys.path = [".."] + sys.path
from common.xdict import xdict
import torch.nn.functional as F
from src.utils.const import SEGM_IDS

l1_loss = nn.L1Loss(reduction="none")
mse_loss = nn.MSELoss(reduction="none")


def crop_masks(masks, boxes, hand_id, obj_id, scale):
    masks_hand = (masks == hand_id).astype(np.float32)
    masks_object = (masks == obj_id).astype(np.float32)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    dim_x = x2 - x1
    dim_y = y2 - y1

    dim_max = np.maximum(dim_y, dim_x)

    dim_max *= scale

    boxes[:, 0] = cx - dim_max / 2.0
    boxes[:, 1] = cy - dim_max / 2.0
    boxes[:, 2] = cx + dim_max / 2.0
    boxes[:, 3] = cy + dim_max / 2.0

    h, w = masks.shape[1:]

    boxes[:, 0] = np.clip(boxes[:, 0], 1, w - 1)
    boxes[:, 1] = np.clip(boxes[:, 1], 1, h - 1)
    boxes[:, 2] = np.clip(boxes[:, 2], 1, w - 1)
    boxes[:, 3] = np.clip(boxes[:, 3], 1, h - 1)

    mask_boxes = np.zeros_like((masks))
    for idx in range(mask_boxes.shape[0]):
        x1, y1, x2, y2 = boxes[idx].astype(np.int64)
        mask_boxes[idx, y1:y2, x1:x2] = 1
    masks_hand_crop = masks_hand * mask_boxes
    masks_crop = np.zeros_like(masks)
    masks_crop[masks_object > 0] = obj_id
    masks_crop[masks_hand_crop > 0] = hand_id
    return masks_crop


def remesh_and_clean_mesh(input_path, out_path, target_face_count=5000):
    ms = ml.MeshSet()
    ms.load_new_mesh(input_path)

    # Attempt to fix non-manifold edges if they exist
    ms.apply_filter("meshing_repair_non_manifold_edges")

    # 1. Remesh for target face count
    ms.apply_filter(
        "meshing_decimation_quadric_edge_collapse",
        targetfacenum=target_face_count,
        qualitythr=1.0,
        preserveboundary=True,
        preservenormal=True,
    )

    # 2. Collapse near vertices
    ms.meshing_merge_close_vertices()

    # 3. Remove identical vertices and faces
    ms.meshing_remove_duplicate_vertices()
    ms.meshing_remove_duplicate_faces()

    ms.save_current_mesh(out_path)


def create_silhouette_renderer(K, device, imsize):
    cameras = create_camera(K.clone(), imsize, device)
    blend_params = BlendParams(sigma=1e-6, gamma=1e-4)
    raster_settings = RasterizationSettings(
        image_size=imsize,
        blur_radius=np.log(1.0 / 1e-4 - 1.0) * blend_params.sigma,
        faces_per_pixel=100,
    )
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    shader = SoftSilhouetteShader(blend_params=blend_params)
    # shader = HardFlatShader(blend_params=blend_params, cameras=cameras, device='cuda')

    silhouette_renderer = MeshRenderer(
        rasterizer=rasterizer,
        shader=shader,
    )

    return rasterizer, shader, silhouette_renderer


def create_camera(K, imsize, device):
    K_3d = K
    K_3d[:, [2, 3], :] = K_3d[:, [3, 2], :]
    K_3d = K_3d[:1]
    cam_R = (
        torch.from_numpy(
            np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
        )
        .to(device)
        .float()
        .unsqueeze(0)
    )

    cam_T = torch.zeros((1, 3)).to(device).float()

    cameras = PerspectiveCameras(
        K=K,
        R=cam_R,
        T=cam_T,
        in_ndc=False,
        image_size=torch.tensor(np.array(imsize)).unsqueeze(0),
        device=device,
    )

    return cameras


def create_meshes(v3d, faces, device):
    verts = v3d
    verts_rgb = torch.ones_like(verts)
    textures = TexturesVertex(verts_features=verts_rgb.to(device))
    batch_size = verts.shape[0]
    meshes = Meshes(
        verts=[vv for vv in verts],
        faces=[faces for _ in range(batch_size)],
        textures=textures,
    )
    return meshes


def construct_targets(target_masks):
    targets = {}
    targets["object"] = (target_masks == SEGM_IDS["object"]).float()
    targets["right"] = (target_masks == SEGM_IDS["right"]).float()
    targets["left"] = (target_masks == SEGM_IDS["left"]).float()
    return targets


def create_color_map(target_masks):
    color_map = (
        torch.zeros_like(target_masks, dtype=torch.float32)[:, :, :, None]
        .repeat(1, 1, 1, 3)
        .to("cpu")
    )
    color_map[target_masks == SEGM_IDS["bg"]] = torch.tensor(
        [0, 0, 0], dtype=torch.float32
    )
    color_map[target_masks == SEGM_IDS["object"]] = torch.tensor(
        [255, 0, 0], dtype=torch.float32
    )
    color_map[target_masks == SEGM_IDS["right"]] = torch.tensor(
        [0, 0, 255], dtype=torch.float32
    )
    color_map /= 255.0
    return color_map


def scaling_masks_K(masks, K, target_dim):
    device = K.device
    batch_size = masks.shape[0]

    im_h = masks.shape[1]
    im_w = masks.shape[2]
    curr_dim = max(im_h, im_w)
    k = target_dim / curr_dim

    new_h = int(im_h * k)
    new_w = int(im_w * k)
    masks = masks[:, None, :, :]
    masks = F.interpolate(masks, size=(new_h, new_w), mode="nearest").squeeze(dim=1)

    # 3x3 to 4x4
    K_4x4 = torch.eye(4).to(device)
    K_4x4[:3, :3] = K[0]
    K_4x4 = K_4x4.unsqueeze(0)

    scaling_matrix = torch.diag(torch.tensor([k, k, 1.0, 1.0]).to(device)).unsqueeze(0)
    K_scaled = torch.bmm(scaling_matrix, K_4x4)

    K_scaled = K_scaled.repeat(batch_size, 1, 1)
    return masks, K_scaled


def vis_fn_ih(out, targets):
    out = out.to_np()
    targets = xdict(targets).to_np()

    base_colors = np.array(
        [
            [0, 0, 0, 255],
            [100, 0, 0, 255],
            [0, 100, 0, 255],
            [0, 0, 100, 255],
        ]
    )
    base_colors_pred = np.array(
        [
            [0, 0, 0, 255],
            [255, 255, 255, 255],
            [255, 255, 255, 255],
            [255, 255, 255, 255],
        ]
    )

    pred_segm = np.zeros_like(out["object.mask"].astype(np.int64))
    pred_segm[out["object.mask"] > 0] = 1
    if "right.mask" in out:
        pred_segm[out["right.mask"] > 0] = 2
    if "left.mask" in out:
        pred_segm[out["left.mask"] > 0] = 3
    pred_segm_rgba = base_colors_pred[pred_segm]

    target_segm = np.zeros_like(pred_segm)
    target_segm[targets["object"] > 0] = 1
    if "right" in targets:
        target_segm[targets["right"] > 0] = 2
    if "left" in targets:
        target_segm[targets["left"] > 0] = 3
    target_segm_rgba = base_colors[target_segm]

    # Normalize the alpha channels to [0, 1]
    alpha_pred = pred_segm_rgba[..., 3:] / 255.0
    alpha_target = target_segm_rgba[..., 3:] / 255.0

    # Compute the blended image
    blended_rgb = (
        pred_segm_rgba[..., :3] * alpha_pred + target_segm_rgba[..., :3] * alpha_target
    ) / (alpha_pred + alpha_target)
    # Avoid division by zero for pixels where both alpha values are 0
    blended_rgb[np.isnan(blended_rgb)] = 0
    blended_alpha = (
        np.maximum(alpha_pred, alpha_target) * 255
    )  # For alpha, taking the maximum for demonstration

    # Combine RGB and alpha back into RGBA
    blended_rgba = np.concatenate([blended_rgb, blended_alpha], axis=-1).astype(
        np.uint8
    )
    return blended_rgba


class MyParameterDict(nn.ParameterDict):
    def search(self, keyword):
        sub_dict = MyParameterDict()
        for key, value in self.items():
            if keyword in key:
                sub_dict[key] = value
        return sub_dict

    def fuzzy_get(self, keyword):
        for k, v in self.items():
            if keyword in k:
                return v
        return None


def extract_batch_data(batch_idx, out, mask_ps, device, itw):
    batch_masks = np.stack(
        [np.array(Image.open(mask_ps[idx])) for idx in batch_idx], axis=0
    )
    boxes = out["boxes"]
    param_dict = out["param_dict"]

    if boxes is not None:
        crop_scale = 1.0 if itw else 0.6
        hand_id = SEGM_IDS["right"]
        obj_id = SEGM_IDS["object"]
        assert hand_id in np.unique(batch_masks)
        batch_masks = crop_masks(
            batch_masks, boxes[np.array(batch_idx)], hand_id, obj_id, scale=crop_scale
        )

    masks_batch = torch.stack(
        [torch.FloatTensor(np.array(Image.fromarray(mask))) for mask in batch_masks],
        dim=0,
    ).to(device)
    with torch.no_grad():
        scene_scale = out["scene_scale"].to(device)
        param_dict = {
            k: v[batch_idx].to(device) if ".betas" not in k else v.to(device)
            for k, v in param_dict.items()
        }
    fnames_batch = [out["fnames"][i] for i in batch_idx]
    w2c_batch = out["w2c"].repeat(len(batch_idx), 1, 1).to(device)

    return masks_batch, scene_scale, param_dict, fnames_batch, w2c_batch
