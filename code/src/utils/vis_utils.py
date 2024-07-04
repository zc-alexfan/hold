import os
import os.path as op

import numpy as np
import torch
from PIL import Image


def aggregate_reshape(outputs, img_size):
    image = torch.cat(outputs, dim=0)
    image = image.reshape(*img_size, -1)
    return image


def scale_to_image(image):
    if image is None:
        return None
    image = (image * 255).cpu().numpy().astype(np.uint8)
    image = Image.fromarray(image)
    return image


def segm_pred_to_cmap(segm_pred):
    # [r, g, b]
    class2color = torch.FloatTensor(
        np.array(
            [
                [0, 0, 0],  # background, black
                [255, 0, 0],  # object, red
                [100, 100, 100],  # right, grey,
                [0, 0, 255],  # left, blue
            ]
        )
    )
    segm_map = class2color[segm_pred] / 255.0
    return segm_map


def make_normal_transparent(imap_raw, normal_map):
    bg_idx = imap_raw == 0
    normal_map = np.array(normal_map)
    alpha = np.zeros_like(normal_map)[:, :, :1] + 255
    normal_map_alpha = np.concatenate((normal_map, alpha), axis=2)
    normal_map_alpha[bg_idx, 3] = 0
    normal_map_alpha = Image.fromarray(normal_map_alpha)
    return normal_map_alpha


def output2images(outputs, img_size):
    from common.xdict import xdict
    from common.ld_utils import ld2dl

    out = xdict()
    outputs = xdict(ld2dl(outputs))
    # process normals
    for key in outputs.search("normal").keys():
        pred_normal = aggregate_reshape(outputs[key], img_size)
        # transform to colormap in [0, 1]
        pred_normal = (pred_normal + 1) / 2
        # scale to image [0, 255]
        pred_normal = scale_to_image(pred_normal)
        out[key] = pred_normal

    # process mask
    for key in outputs.search("mask_prob").keys():
        pred_mask = aggregate_reshape(outputs[key], img_size).repeat(1, 1, 3)
        pred_mask = (pred_mask > 0.5).float()
        pred_mask = scale_to_image(pred_mask)
        out[key] = pred_mask

    # fg_rgb
    for key in outputs.search("fg_rgb.vis").keys():
        pred_rgb = aggregate_reshape(outputs[key], img_size)
        pred_rgb = scale_to_image(pred_rgb)
        out[key] = pred_rgb

    # composite rendering
    pred_imap = aggregate_reshape(outputs["instance_map"], img_size)
    pred_rgb = aggregate_reshape(outputs["rgb"], img_size)
    gt_rgb = aggregate_reshape(outputs["gt.rgb"], img_size)
    bg_rgb = aggregate_reshape(outputs["bg_rgb_only"], img_size)
    out.overwrite("normal", make_normal_transparent(pred_imap.squeeze(), out["normal"]))

    pred_imap = segm_pred_to_cmap(pred_imap.squeeze())

    pred_rgb = scale_to_image(pred_rgb)
    gt_rgb = scale_to_image(gt_rgb)
    bg_rgb = scale_to_image(bg_rgb)

    # concat PIL images horizontally
    rgb = Image.new("RGB", (gt_rgb.width + pred_rgb.width, gt_rgb.height))
    rgb.paste(gt_rgb, (0, 0))
    rgb.paste(pred_rgb, (gt_rgb.width, 0))

    out["rgb"] = rgb
    out["imap"] = scale_to_image(pred_imap)
    out["bg_rgb"] = bg_rgb
    return out


def create_transparent_image(normal, fg_mask_t):
    """
    Creates a transparent PNG image from the 'normal' RGB image (np.uint8) based on the 'fg_mask_t' (np.float32).
    When fg_mask_t[i, j] is 0, the pixel at the same position in 'normal' becomes transparent.

    :param normal: np.ndarray of shape (H, W, 3), representing an RGB image.
    :param fg_mask_t: np.ndarray of shape (H, W), where 0 values indicate transparent pixels.
    :return: PIL Image with transparency applied.
    """
    # Ensure the mask is in the correct range [0, 1]
    fg_mask_t = np.clip(fg_mask_t, 0, 1)

    # Convert the mask to an alpha channel (255 for visible, 0 for transparent)
    alpha_channel = (fg_mask_t * 255).astype(np.uint8)

    # Add the alpha channel to the original image
    rgba_image = np.dstack((normal, alpha_channel))

    # Convert to PIL Image for easier handling and saving
    return Image.fromarray(rgba_image, "RGBA")


def record_vis(idx, current_step, log_dir, experiment, vis_dict):
    idx = int(idx)
    filenames = [f"{key}/step_{current_step:09}_id_{idx:04}" for key in vis_dict.keys()]
    out_ps = [op.join(log_dir, "visuals", f"{fn}.png") for fn in filenames]
    for out_p, im in zip(out_ps, vis_dict.values()):
        os.makedirs(op.dirname(out_p), exist_ok=True)
        im.save(out_p)

        key = "/".join(out_p.split("/")[-2:])
        experiment.log_image(out_p, key, step=current_step)
