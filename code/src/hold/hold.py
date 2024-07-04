import os
import os.path as op
import sys

import pytorch_lightning as pl
import torch
import torch.optim as optim
from tqdm import tqdm

import src.hold.hold_utils as hold_utils
from src.hold.loss import Loss
from src.hold.hold_net import HOLDNet
from src.utils.metrics import Metrics

sys.path = [".."] + sys.path
from common.xdict import xdict

import src.utils.debug as debug
import src.utils.vis_utils as vis_utils
import common.comet_utils as comet_utils
import numpy as np
from loguru import logger
from src.datasets.utils import split_input, merge_output


class HOLD(pl.LightningModule):
    def __init__(self, opt, args) -> None:
        super().__init__()

        self.opt = opt
        self.args = args
        num_frames = args.n_images

        data_path = os.path.join("./data", args.case, f"build/data.npy")
        entities = np.load(data_path, allow_pickle=True).item()["entities"]

        betas_r = entities["right"]["mean_shape"] if "right" in entities else None
        betas_l = entities["left"]["mean_shape"] if "left" in entities else None

        self.model = HOLDNet(
            opt.model,
            betas_r,
            betas_l,
            num_frames,
            args,
        )

        for node in self.model.nodes.values():
            if self.args.freeze_pose:
                node.params.freeze()
            else:
                node.params.defrost()

        self.loss = Loss(args)
        self.metrics = Metrics(args.experiment)

    def save_misc(self):
        out = {}

        dataset = self.trainset.dataset
        K = dataset.intrinsics_all[0]
        w2c = dataset.extrinsics_all[0]

        for node in self.model.nodes.values():
            if "object" in node.node_id:
                out[f"{node.node_id}.obj_scale"] = node.server.object_model.obj_scale

        out["img_paths"] = dataset.img_paths
        out["K"] = K
        out["w2c"] = w2c
        out["scale"] = dataset.scale
        mesh_dict = self.meshing_cano("misc")
        out.update(mesh_dict)
        out_p = f"{self.args.log_dir}/misc/{self.global_step:09d}.npy"
        os.makedirs(op.dirname(out_p), exist_ok=True)
        np.save(out_p, out)
        print(f"Exported misc to {out_p}")

    def configure_optimizers(self):
        base_lr = self.args.lr
        node_params = set()
        params = []
        # collect pose parameters for each node
        for node in self.model.nodes.values():
            node_parameters = set(node.params.parameters())
            node_params.update(node_parameters)
            params.append(
                {
                    "params": list(node_parameters),
                    "lr": base_lr * 0.1,  # Adjust the learning rate for node parameters
                }
            )

        # neural network parameters
        main_params = [p for p in self.model.parameters() if p not in node_params]
        if main_params:  # Check if there are any main parameters
            params.append({"params": main_params, "lr": base_lr})

        self.optimizer = optim.Adam(params, lr=base_lr, eps=1e-8)

        return [self.optimizer], []

    def condition_training(self):
        import common.torch_utils as torch_utils

        if self.global_step in []:
            logger.info(f"Decaying learning rate at step {self.global_step}")
            torch_utils.decay_lr(self.optimizer, gamma=0.5)

    def training_step(self, batch):
        self.condition_training()

        batch["idx"] = torch.stack(batch["idx"], dim=1)
        batch = hold_utils.wubba_lubba_dub_dub(batch)
        batch = xdict(batch)
        batch["current_epoch"] = self.current_epoch
        batch["global_step"] = self.global_step

        for node in self.model.nodes.values():
            params = node.params(batch["idx"])
            batch.update(params)

        debug.debug_params(self)
        model_outputs = self.model(batch)
        loss_output = self.loss(batch, model_outputs)
        if self.global_step % self.args.log_every == 0:
            self.metrics(model_outputs, batch, self.global_step, self.current_epoch)
            comet_utils.log_dict(
                self.args.experiment,
                loss_output,
                step=self.global_step,
                epoch=self.current_epoch,
            )

        loss = loss_output["loss"]
        self.log("loss", loss)
        return loss

    def training_epoch_end(self, outputs) -> None:
        current_step = self.global_step
        current_epoch = self.current_epoch
        # Canonical mesh update every 20 epochs
        if (
            current_epoch > 0 and current_epoch % 3 == 0 and not self.args.no_meshing
        ) or (current_step > 0 and self.args.fast_dev_run and not self.args.no_meshing):
            self.meshing_cano(current_step)
            self.save_misc()

        return super().training_epoch_end(outputs)

    def meshing_cano(self, current_step):
        mesh_dict = {}
        for node in self.model.nodes.values():
            try:
                mesh_c = node.meshing_cano()
                out_p = op.join(
                    self.args.log_dir,
                    "mesh_cano",
                    f"mesh_cano_{node.node_id}_step_{current_step}.obj",
                )
                os.makedirs(op.dirname(out_p), exist_ok=True)
                mesh_c.export(out_p)
                print(f"Exported canonical to {out_p}")
                mesh_dict[f"{node.node_id}_cano"] = mesh_c
            except:
                logger.error(f"Failed to mesh out {node.node_id}")
        return mesh_dict

    def inference_step(self, batch, *args, **kwargs):
        batch = xdict(batch).to("cuda")
        self.model.eval()
        batch = xdict(batch)
        batch["current_epoch"] = self.current_epoch
        batch["global_step"] = self.global_step

        for node in self.model.nodes.values():
            params = node.params(batch["idx"])
            batch.update(params)

        output = xdict()
        if not self.args.no_vis:
            batch = hold_utils.downsample_rendering(batch, self.args.render_downsample)
            split = split_input(
                batch,
                batch["total_pixels"][0],
                n_pixels=batch["pixel_per_batch"],
            )
            out_list = []
            pbar = tqdm(split)
            for s in pbar:
                pbar.set_description("Rendering")
                out = self.model(s).detach().to("cpu")
                vis_dict = {}
                vis_dict["rgb"] = out["rgb"]
                vis_dict["instance_map"] = out["instance_map"]
                vis_dict["bg_rgb_only"] = out["bg_rgb_only"]
                vis_dict.update(out.search("fg_rgb.vis"))
                vis_dict.update(out.search("mask_prob"))
                vis_dict.update(out.search("normal"))

                out_list.append(vis_dict)

            batch_size = batch["gt.rgb"].shape[0]
            model_outputs = merge_output(out_list, batch["total_pixels"][0], batch_size)
            output.update(model_outputs)

        output.update(batch)
        return output

    def inference_step_end(self, batch_parts):
        return batch_parts

    def validation_step(self, batch, *args, **kwargs):
        return self.inference_step(batch, *args, **kwargs)

    def test_step(self, batch, *args, **kwargs):
        out = self.inference_step(batch, *args, **kwargs)
        img_size = out["img_size"]
        normal = out["normal"]
        normal = normal.view(img_size[0], img_size[1], -1)
        normal_np = normal.numpy().astype(np.float16)

        exp_key = self.args.exp_key
        out_p = f"./exports/{exp_key}/normal/{out['idx']:04}.npy"
        os.makedirs(op.dirname(out_p), exist_ok=True)
        np.save(out_p, normal_np)
        print(f"Exported normal to {out_p}")
        return out

    def validation_step_end(self, batch_parts):
        return self.inference_step_end(batch_parts)

    def test_step_end(self, batch_parts):
        return self.inference_step_end(batch_parts)

    def validation_epoch_end(self, outputs) -> None:
        if not self.args.no_vis:
            img_size = outputs[0]["img_size"]
            idx = outputs[0]["idx"]
            vis_dict = vis_utils.output2images(outputs, img_size)
            vis_utils.record_vis(
                idx, self.global_step, self.args.log_dir, self.args.experiment, vis_dict
            )
