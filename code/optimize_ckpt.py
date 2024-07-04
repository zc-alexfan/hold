import numpy as np
import torch
from tqdm import tqdm

from src.fitting.fitting import optimize_batch
from src.utils.io.optim import load_data
from common.xdict import xdict


def fit_ckpt(args):
    device = "cuda"
    out, ckpt = load_data(args.ckpt_p)
    node_ids = out["node_ids"]
    num_frames = out["num_frames"]
    batch_size = args.batch_size
    obj_scale = out["servers"]["object"].object_model.obj_scale.cpu().detach().numpy()
    hand_shapes = xdict()
    for node_id in node_ids:
        if "right" in node_id or "left" in node_id:
            shape_key = f"model.nodes.{node_id}.params.betas.weight"
            initial_shape = ckpt["state_dict"][shape_key].clone().cpu().detach().numpy()
            hand_shapes[node_id] = initial_shape

    batch_idx = (
        torch.linspace(0, num_frames - 1, steps=batch_size).floor().long().tolist()
    )

    print("Stage [1/2]: Optimizing object scale and hand shape")
    pbar = None
    model = optimize_batch(
        batch_idx,
        args,
        pbar,
        out,
        device,
        obj_scale=obj_scale,
        freeze_scale=False,
        freeze_shape=False,
    )
    final_obj_scale = model.obj_scale.data.clone().cpu().detach().numpy()
    print()
    print("Stage [1/2]: Done")
    print("Changes in obj_scale:", final_obj_scale - obj_scale)
    print("Changes in hand shape:")
    final_shapes = model.param_dict.search("__betas")
    for node_id in node_ids:
        if "object" in node_id:
            continue
        norm_shape = np.linalg.norm(
            hand_shapes.fuzzy_get(node_id)
            - final_shapes.fuzzy_get(node_id).data.cpu().numpy()
        )
        print(f"\t{node_id}: {norm_shape}")
    print("Stage [2/2]: Optimizing entire sequence")
    out, ckpt = load_data(args.ckpt_p)
    out_param_dict = dict(out["param_dict"])
    pbar = tqdm(range(0, num_frames, batch_size))
    for batch_start in pbar:
        batch_end = min(batch_start + batch_size, num_frames)
        batch_idx = torch.arange(batch_start, batch_end, dtype=torch.long)

        # Skip the batch if inspect_idx is out of range
        if obj_scale is not None:
            if args.inspect_idx is not None and args.inspect_idx not in batch_idx:
                continue

        model = optimize_batch(
            batch_idx,
            args,
            pbar,
            out,
            device,
            obj_scale=final_obj_scale,
            freeze_scale=True,
            freeze_shape=True,
            # freeze_scale=False,
            # freeze_shape=False,
        )

        # Write back results
        for k, v in model.param_dict.items():
            if "scene_scale" in k:
                continue
            k_new = "model.nodes." + k + ".weight"
            k_new = k_new.replace("__", ".params.")
            assert k_new in out_param_dict, f"{k_new} not in out_param_dict"
            out_param_dict[k_new] = out_param_dict[k_new].cpu().detach()
            if "betas" in k_new:
                out_param_dict[k_new] = v.cpu().detach()
            else:
                out_param_dict[k_new][batch_idx] = v.cpu().detach()

    out_p = args.out_p
    if args.inspect_idx is not None:
        out_p = out_p + ".inspect"

    # write back results
    sd = ckpt["state_dict"]
    for key, val in out_param_dict.items():
        assert key in sd
        assert val.shape == sd[key].shape
        sd[key] = val
    sd["model.nodes.object.server.object_model.obj_scale"] = torch.tensor(
        final_obj_scale
    )

    ckpt["state_dict"] = sd
    print("Saving to", out_p)
    torch.save(ckpt, out_p)


def main(args):
    fit_ckpt(args)


def fetch_parser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect_idx", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--ckpt_p", type=str, default=None)
    parser.add_argument("--write_gif", action="store_true")
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--vis_every", type=int, default=5)
    parser.add_argument("--itw", action="store_true")
    args = parser.parse_args()
    from easydict import EasyDict as edict

    args = edict(vars(args))

    out_p = args.ckpt_p.replace(".ckpt", ".pose_ref")
    args.out_p = out_p
    return args


if __name__ == "__main__":
    args = fetch_parser()

    main(args)
