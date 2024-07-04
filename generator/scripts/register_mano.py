import os
import numpy as np
import torch
import smplx
from tqdm import tqdm
import sys

sys.path = ["."] + sys.path
from src.hand_pose.registration import optimize_mano_shape

sys.path = [".."] + sys.path
from common.ld_utils import ld2dl
from common.xdict import xdict

MANO_DIR_L = "../code/body_models/MANO_LEFT.pkl"
MANO_DIR_R = "../code/body_models/MANO_RIGHT.pkl"

mano_layers = {
    "right": smplx.create(
        model_path=MANO_DIR_R, model_type="mano", use_pca=False, is_rhand=True
    ),
    "left": smplx.create(
        model_path=MANO_DIR_L, model_type="mano", use_pca=False, is_rhand=False
    ),
}


def fit_frame(
    seq_name,
    target_v3d,
    save_mesh,
    iteration,
    is_right,
    init_params=None,
    first_frame=False,
    use_beta_loss=False,
):
    target_v3d = torch.FloatTensor(target_v3d.reshape(1, -1, 3)).cuda()

    vis_dir = f"./data/{seq_name}/processed/mesh_fit_vis/"

    tip_sem_idx = [12, 11, 4, 5, 6]

    if first_frame:
        optim_specs = {
            "epoch_coarse": 10000,
            "epoch_fine": 10000,
            "is_right": is_right,
            "save_mesh": save_mesh,
            "criterion": torch.nn.MSELoss(reduction="none"),
            "seed": 0,
            "vis_dir": vis_dir,
            "sem_idx": tip_sem_idx,
        }
    else:
        optim_specs = {
            "epoch_coarse": 2000,
            "epoch_fine": 2000,
            "is_right": is_right,
            "save_mesh": save_mesh,
            "criterion": torch.nn.MSELoss(reduction="none"),
            "seed": 0,
            "vis_dir": vis_dir,
            "sem_idx": tip_sem_idx,
        }

    os.makedirs(optim_specs["vis_dir"], exist_ok=True)
    params = optimize_mano_shape(
        target_v3d,
        mano_layers,
        optim_specs,
        iteration,
        init_params=init_params,
        use_beta_loss=use_beta_loss,
    )
    return params


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_name", type=str, default="")
    parser.add_argument("--save_mesh", action="store_true")
    parser.add_argument("--use_beta_loss", action="store_true")
    parser.add_argument("--hand_type", type=str, default=None)
    args = parser.parse_args()
    return args


def fit_single_hand(v3d_ra_list, seq_name, args, is_right):
    import copy

    pbar = tqdm(enumerate(v3d_ra_list), total=len(v3d_ra_list))
    prev_out = None
    out_list = []
    for iteration, v3d_ra in pbar:
        pbar.set_description(
            "Processing %s [%d/%d]" % (seq_name, iteration + 1, len(v3d_ra_list))
        )
        is_valid = np.isnan(v3d_ra).sum() == 0
        if not is_valid:
            out = {}
            out["global_orient"] = torch.zeros(1, 3).cuda() * np.nan
            out["hand_pose"] = torch.zeros(1, 45).cuda() * np.nan
            out["betas"] = torch.zeros(1, 10).cuda() * np.nan
            out["transl"] = torch.zeros(1, 3).cuda() * np.nan
        else:
            out = fit_frame(
                seq_name,
                v3d_ra,
                save_mesh=args.save_mesh,
                init_params=prev_out,
                iteration=iteration,
                is_right=is_right,
                first_frame=prev_out is None,
                use_beta_loss=args.use_beta_loss,
            )
            prev_out = copy.deepcopy(out)
        out_list.append(out)

    out_dict = ld2dl(out_list)
    out_dict = dict(
        xdict({key: torch.cat(val, axis=0) for key, val in out_dict.items()}).to_np()
    )
    return out_dict


def main():
    args = parse_args()
    seq_name = args.seq_name
    data = np.load(f"./data/{seq_name}/processed/v3d.npy", allow_pickle=True).item()
    data = xdict(data).search("v3d.")

    if args.hand_type is not None:
        data = data.search(args.hand_type)

    out_dict = {}
    for key, val in data.items():
        print("Processing " + key)
        flag = key.split(".")[1]
        is_right = flag == "right"

        mydict = fit_single_hand(val, seq_name, args, is_right=is_right)
        out_dict[flag] = mydict
    out_p = f"./data/{seq_name}/processed/hold_fit.init.npy"
    os.makedirs(os.path.dirname(out_p), exist_ok=True)
    np.save(out_p, out_dict)
    print(f"Saved to {out_p}")


if __name__ == "__main__":
    main()
