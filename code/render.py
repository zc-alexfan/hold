import torch
import os
from src.hold.hold import HOLD
from src.utils.parser import parser_args
import os.path as op
from common.torch_utils import reset_all_seeds
import numpy as np
from pprint import pprint

import sys

sys.path = [".."] + sys.path
from src.datasets.utils import create_dataset
import common.thing as thing


def main():
    device = "cuda:0"
    args, opt = parser_args()

    print("Working dir:", os.getcwd())
    exp_key = args.load_ckpt.split("/")[1]
    args.log_dir = op.join("logs", exp_key, "test")

    pprint(args)

    model = HOLD(opt, args)
    testset = create_dataset(opt.dataset.test, args)

    print("img_paths: ")
    img_paths = np.array(testset.dataset.dataset.img_paths)
    print(img_paths[:3])
    print("...")
    print(img_paths[-3:])
    reset_all_seeds(1)
    ckpt_path = None if args.ckpt_p == "" else args.ckpt_p
    sd = torch.load(ckpt_path)["state_dict"]
    model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()

    # disable barf masks
    nodes = model.model.nodes
    for node in nodes.values():
        node.implicit_network.embedder_obj.eval()
    model.model.background.bg_implicit_network.embedder_obj.eval()
    model.model.background.bg_rendering_network.embedder_obj.eval()
    for batch in testset:
        with torch.no_grad():
            batch = thing.thing2dev(batch, device)
            out = model.inference_step(batch)
            model.validation_epoch_end([out])


if __name__ == "__main__":
    main()
