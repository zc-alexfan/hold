import comet_ml
from pytorch_lightning.loggers import CometLogger
from tqdm import tqdm
import sys
import torch
import os
import numpy as np
import torch
import time
from loguru import logger
import os.path as op
import json
import common.sys_utils as sys_utils

# folder used for debugging
DUMMY_EXP = "xxxxxxxxx"


def add_paths(args):
    exp_key = args.exp_key
    args_p = f"./logs/{exp_key}/args.json"
    ckpt_p = f"./logs/{exp_key}/checkpoints/last.ckpt"
    if not op.exists(ckpt_p) or DUMMY_EXP in ckpt_p:
        ckpt_p = ""
    if args.load_ckpt != "":
        ckpt_p = args.load_ckpt
    args.ckpt_p = ckpt_p
    args.log_dir = f"./logs/{exp_key}"

    if args.infer_ckpt != "":
        basedir = "/".join(args.infer_ckpt.split("/")[:2])
        basename = op.basename(args.infer_ckpt).replace(".ckpt", ".params.pt")
        args.interface_p = op.join(basedir, basename)
    args.args_p = args_p
    return args


def save_args(args, save_keys):
    args_save = {}
    for key, val in args.items():
        if key in save_keys:
            args_save[key] = val
    with open(args.args_p, "w") as f:
        json.dump(args_save, f, indent=4)
    logger.info(f"Saved args at {args.args_p}")


def create_files(args):
    os.makedirs(args.log_dir, exist_ok=True)



def log_exp_meta(args):
    tags = [args.hostname, args.git_branch]
    logger.info(f"Experiment tags: {tags}")
    args.experiment.set_name(args.exp_key)
    args.experiment.add_tags(tags)
    args.python = sys.executable
    args.experiment.log_parameters(args)


def init_experiment(args, api_key, workspace):
    if args.fast_dev_run:
        args.exp_key = DUMMY_EXP
    if args.exp_key == "":
        args.exp_key = generate_exp_key()

    args = add_paths(args)
    if op.exists(args.args_p) and args.exp_key not in [DUMMY_EXP]:
        with open(args.args_p, "r") as f:
            args_disk = json.load(f)
            args.git_commit = args_disk["git_commit"]
            args.git_branch = args_disk["git_branch"]
            if "comet_key" in args_disk.keys():
                args.comet_key = args_disk["comet_key"]
    else:
        args.git_commit = sys_utils.get_commit_hash()
        args.git_branch = sys_utils.get_branch()

    create_files(args)

    args.hostname = sys_utils.get_host_name()
    project_name = args.project
    disabled = args.mute
    comet_url = args["comet_key"] if "comet_key" in args.keys() else None
    if args.load_ckpt != "":
        comet_url = None

    if comet_url is None:
        experiment = comet_ml.Experiment(
            api_key=api_key,
            workspace=workspace,
            project_name=project_name,
            disabled=disabled,
            display_summary_level=0,
        )
        args.comet_key = experiment.get_key()
    else:
        experiment = comet_ml.ExistingExperiment(
            previous_experiment=comet_url,
            api_key=api_key,
            project_name=project_name,
            workspace=workspace,
            disabled=disabled,
            display_summary_level=0,
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.add(
        os.path.join(args.log_dir, "train.log"),
        level="INFO",
        colorize=True,
    )
    logger.info(torch.cuda.get_device_properties(device))
    args.gpu = torch.cuda.get_device_properties(device).name

    args.experiment = experiment
    return experiment, args


def log_dict(experiment, metric_dict, step=None, epoch=None, postfix=None):
    if experiment is None:
        return
    for key, value in metric_dict.items():
        if postfix is not None:
            key = key + postfix
        if isinstance(value, tuple):
            value = value[0]
        if isinstance(value, torch.Tensor) and len(value.view(-1)) == 1:
            value = value.item()

        if isinstance(value, (int, float, np.float32)):
            experiment.log_metric(key, value, step=step, epoch=epoch)


def generate_exp_key():
    import random

    hash = random.getrandbits(128)
    key = "%032x" % (hash)
    key = key[:9]
    return key


def fetch_key_from_experiment(experiment):
    if experiment is not None:
        key = str(experiment.get_key())
        key = key[:9]
        experiment.set_name(key)
    else:
        import random

        hash = random.getrandbits(128)
        key = "%032x" % (hash)
        key = key[:9]
    return key


def push_images(experiment, all_im_list, global_step=None, no_tqdm=False, verbose=True):
    if verbose:
        print("Pushing PIL images")
        tic = time.time()
    iterator = all_im_list if no_tqdm else tqdm(all_im_list)
    for im in iterator:
        im_np = np.array(im["im"])
        if "fig_name" in im.keys():
            experiment.log_image(im_np, im["fig_name"], step=global_step)
        else:
            experiment.log_image(im_np, "unnamed", step=global_step)
    if verbose:
        toc = time.time()
        print("Done pushing PIL images (%.1fs)" % (toc - tic))
