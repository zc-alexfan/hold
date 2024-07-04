import os

import torch
import pytorch_lightning as pl
import os.path as op
import numpy as np
from easydict import EasyDict as edict
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
import sys

sys.path = [".", ".."] + sys.path

from src.alignment.data import read_data

from common.xdict import xdict
from generator.src.alignment.data import FakeDataset


def main(args):
    device = "cuda"
    k_path_colmap = op.join(f"./data/{args.seq_name}/processed/colmap/intrinsic.npy")
    ho3d_seq = args.seq_name.split("_")[1]
    k_path_ho3d = f"./assets/datasets/HO3D_v3/processed/{ho3d_seq}.pt"
    if op.exists(k_path_ho3d):
        K = torch.load(k_path_ho3d)["K"][0]
    else:
        K = torch.FloatTensor(np.load(k_path_colmap))
    data = read_data(args.seq_name, K=K).to(device)

    out_p = op.join(f"./data/{args.seq_name}/processed/hold_fit.aligned.npy")
    checkpoint_callback = ModelCheckpoint(
        dirpath=op.join(f"./data/{args.seq_name}/processed/mano_fit_ckpt/{args.mode}"),
        save_last=True,
    )
    os.makedirs(op.dirname(out_p), exist_ok=True)

    conf = load_conf(args)
    if args.is_arctic:
        from generator.src.alignment.pl_module.arctic import ARCTICModule as PLModule

        print("Using ARCTIC module..")
    elif len(data["entities"]) == 3:
        from generator.src.alignment.pl_module.h2o import H2OModule as PLModule

        print("Using H2O module..")
    else:
        from generator.src.alignment.pl_module.ho import HOModule as PLModule

        print("Using HO module..")
    pl_model = PLModule(data, args, conf)
    trainer = pl.Trainer(
        logger=False,
        gpus=1,
        accelerator="gpu",
        gradient_clip_val=0.5,
        max_epochs=1,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        callbacks=[checkpoint_callback],
    )

    dataset = FakeDataset(conf.num_iters)
    trainset = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=1
    )

    if args.mode == "h":
        load_ckpt = None
    elif args.mode == "o":
        load_ckpt = f"data/{args.seq_name}/processed/mano_fit_ckpt/h/last.ckpt"
    elif args.mode == "ho":
        load_ckpt = f"data/{args.seq_name}/processed/mano_fit_ckpt/o/last.ckpt"
    else:
        assert False, f"Invalid args.mode {args.mode}"

    if load_ckpt is not None:
        sd = torch.load(load_ckpt)["state_dict"]
        pl_model.load_state_dict(sd)
        print(f"Loaded hand model from {load_ckpt}")

    trainer.fit(pl_model, trainset)
    pl_model.to("cpu")
    out = xdict()
    for key in pl_model.models.keys():
        out[key] = pl_model.models[key]()
    out = out.to("cpu").to_np()
    np.save(out_p, out)
    print(f"Saved to {out_p}")


def load_conf(args):
    conf_generic = OmegaConf.load(args.config)
    conf_path = f"./confs/{args.seq_name}.yaml"
    if op.exists(conf_path):
        conf_curr = OmegaConf.load(conf_path)
        config = OmegaConf.merge(conf_generic, conf_curr)
    else:
        config = conf_generic
    config = edict(OmegaConf.to_container(config, resolve=True))

    if args.mode == "h":
        conf = config["optim_configs"]["hand_optim"]
    elif args.mode == "o":
        conf = config["optim_configs"]["object_optim"]
    elif args.mode == "ho":
        conf = config["optim_configs"]["hand_object_optim"]
    else:
        raise NotImplementedError

    conf.update(config.weights)
    return conf


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_name", type=str, default="")
    parser.add_argument("--colmap_k", action="store_true")
    parser.add_argument("--mode", type=str, default="")
    parser.add_argument("--config", type=str, default="confs/generic.yaml")
    parser.add_argument("--is_arctic", action="store_true")
    args = parser.parse_args()
    args = edict(vars(args))
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
