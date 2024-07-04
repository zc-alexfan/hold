import torch.nn as nn
import torch

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import sys

sys.path = [".."] + sys.path
import common.torch_utils as torch_utils
from common.xdict import xdict


mse_loss = nn.MSELoss(reduction="none")


class PLModule(pl.LightningModule):
    def __init__(self, data, args, conf, loss_fn_h, loss_fn_o, loss_fn_ho):
        super().__init__()
        self.args = args
        self.conf = conf
        from src.alignment.params.hand import MANOParameters
        from src.alignment.params.object import ObjectParameters

        models = nn.ModuleDict()
        entities = data["entities"]
        for key in entities.keys():
            if key == "object":
                models[key] = ObjectParameters(entities[key], data["meta"])
            else:
                models[key] = MANOParameters(
                    entities[key], data["meta"], is_right=key == "right"
                )
        self.loss_fn_h = loss_fn_h
        self.loss_fn_o = loss_fn_o
        self.loss_fn_ho = loss_fn_ho
        self.entities = data["entities"]
        self.meta = data["meta"]
        self.models = models

    def training_step(self, batch, batch_idx):
        device = self.device
        self.condition_training()
        preds = xdict()
        for key in self.entities.keys():
            preds.merge(self.models[key]().prefix(key + "."))

        if self.global_step == 0:
            targets = preds.detach().to(device)
            for key in self.entities.keys():
                targets[f"{key}.j2d.gt"] = self.entities[key]["j2d.gt"]
            device = self.device
            self.targets = targets

        loss = 0.0
        if self.args.mode == "h":
            loss += self.loss_fn_h(preds, self.targets, self.conf)
        elif self.args.mode == "o":
            loss += self.loss_fn_o(preds, self.targets, self.conf)
        elif self.args.mode == "ho":
            loss += self.loss_fn_h(preds, self.targets, self.conf)
            loss += self.loss_fn_o(preds, self.targets, self.conf)
            loss += self.loss_fn_ho(preds, self.targets, self.conf)
        else:
            raise NotImplementedError

        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.conf.lr)
        self.optimizer = optimizer
        return optimizer

    def condition_training(self):
        step = self.global_step
        if step == 0:
            for val in self.models.values():
                torch_utils.toggle_parameters(val, requires_grad=False)

        # freeze the other model
        if self.args.mode == "h":
            # hand model schedule
            if step == 0:
                print("Hand: stage 0")
                for key, val in self.models.items():
                    if key in ["right", "left"]:
                        val.hand_transl.requires_grad = True

            if step == 5000:
                print("Hand: stage 2")
                for key, val in self.models.items():
                    if key in ["right", "left"]:
                        val.hand_beta.requires_grad = True
            if step % self.conf.decay_every == 0:
                torch_utils.decay_lr(self.optimizer, self.conf.decay_factor)

        elif self.args.mode == "o":
            if step == 0:
                print("Object: stage 0")
                self.models["object"].obj_transl.requires_grad = True

            if step == 1:
                print("Object: stage 1")
                self.models["object"].obj_scale[:] = self.conf.obj_scale

            if step == 2000:
                print("Object: stage 2")
                self.models["object"].obj_scale.requires_grad = True

            if step % self.conf.decay_every == 0:
                torch_utils.decay_lr(self.optimizer, self.conf.decay_factor)

        elif self.args.mode == "ho":
            for key, val in self.models.items():
                if key in ["right", "left"]:
                    val.hand_transl.requires_grad = True
                else:
                    val.obj_scale.requires_grad = True
                    val.obj_transl.requires_grad = True

            if step % self.conf.decay_every == 0:
                print("Decay")
                torch_utils.decay_lr(self.optimizer, self.conf.decay_factor)
        else:
            raise NotImplementedError
