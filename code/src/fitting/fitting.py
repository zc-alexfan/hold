import sys
import numpy as np
import sys
import numpy as np


sys.path = [".."] + sys.path
import pickle as pkl

with open("./body_models/contact_zones.pkl", "rb") as f:
    contact_zones = pkl.load(f)
contact_zones = contact_zones["contact_zones"]
contact_idx = np.array([item for sublist in contact_zones.values() for item in sublist])

import os


from src.fitting.utils import scaling_masks_K, extract_batch_data
from src.fitting.model import Model


def optimize_batch(
    batch_idx,
    args,
    pbar,
    out,
    device,
    obj_scale=None,
    freeze_scale=False,
    freeze_shape=False,
):
    print("Optimizing batch with idx:", batch_idx)
    mask_ps = [fname.replace("/image/", "/mask/") for fname in out["fnames"]]
    # import pdb; pdb.set_trace()
    masks_batch, scene_scale, param_batch, fnames_batch, w2c_batch = extract_batch_data(
        batch_idx, out, mask_ps, device, args.itw
    )

    # Prepare output paths
    out_paths = [f"./vis/{idx:05d}.gif" for idx in batch_idx]
    os.makedirs(os.path.dirname(out_paths[0]), exist_ok=True)

    masks_batch, K_scaled = scaling_masks_K(masks_batch, out["K"], target_dim=300)
    model = Model(
        out["servers"],
        scene_scale,
        obj_scale,
        param_batch,
        device,
        masks_batch,
        w2c_batch,
        K_scaled,
        fnames_batch,
        out["faces"],
    )
    model.pbar = pbar
    model.defrost_all()
    model.obj_scale.requires_grad = not freeze_scale
    for k in model.param_dict.keys():
        if "betas" in k and freeze_shape:
            model.param_dict[k].requires_grad = False
        if "pose" in k:
            model.param_dict[k].requires_grad = False
        if "global_orient" in k and "object" not in k:
            model.param_dict[k].requires_grad = False
        if "scene_scale" in k:
            model.param_dict[k].requires_grad = False
    model.print_requires_grad()
    model.setup_optimizer()
    model.fit(
        num_iterations=args.iters,
        vis_every=args.vis_every,
        write_gif=args.write_gif,
        out_ps=out_paths,
    )
    return model
