from pprint import pprint
import os
import os.path as op

import numpy as np
import pytorch_lightning as pl

from src.hold.hold import HOLD
from src.datasets.utils import create_dataset
from src.utils.parser import parser_args
from common.torch_utils import reset_all_seeds


def main():
    args, opt = parser_args()
    print("Working dir:", os.getcwd())

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=op.join(args.log_dir, "checkpoints/"),
        filename="{epoch:04d}-{loss}",
        save_last=True,
        save_top_k=-1,
        every_n_epochs=args.eval_every_epoch,
        verbose=True,
    )

    trainer = pl.Trainer(
        gpus=1,
        accelerator="gpu",
        gradient_clip_val=0.5,
        callbacks=[checkpoint_callback],
        max_epochs=args.num_epoch,
        check_val_every_n_epoch=args.eval_every_epoch,
        log_every_n_steps=args.log_every,
        num_sanity_val_steps=0,
        logger=False,
    )

    pprint(args)

    trainset = create_dataset(opt.dataset.train, args)
    validset = create_dataset(opt.dataset.valid, args)
    model = HOLD(opt, args)
    model.trainset = trainset

    print("img_paths: ")
    img_paths = np.array(trainset.dataset.img_paths)
    print(img_paths[:3])
    print("...")
    print(img_paths[-3:])
    reset_all_seeds(1)
    ckpt_path = None if args.ckpt_p == "" else args.ckpt_p
    if args.load_ckpt != "":
        import torch

        sd = torch.load(args.load_ckpt)["state_dict"]
        model.load_state_dict(sd)
        ckpt_path = None

    if args.load_pose != "":
        import torch

        sd = torch.load(args.load_pose)["state_dict"]
        mysd = model.state_dict()
        print("Loading pose from: ", args.load_pose)
        print("Keys in loaded state dict:")
        for k, v in sd.items():
            if ".params." in k or "object_model.obj_scale" in k:
                assert k in mysd, f"{k} not in mysd"
                print("\t" + k)
                mysd[k] = v
        print("End of keys")
        model.load_state_dict(mysd, strict=True)
        ckpt_path = None
    trainer.fit(model, trainset, validset, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
