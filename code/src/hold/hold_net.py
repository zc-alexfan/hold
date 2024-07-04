import sys

import torch
import torch.nn as nn
from loguru import logger


from src.model.renderables.background import Background
from src.model.renderables.object_node import ObjectNode
from src.model.renderables.mano_node import MANONode

sys.path = [".."] + sys.path
from common.xdict import xdict

import src.hold.hold_utils as hold_utils


from src.hold.hold_utils import prepare_loss_targets_hand
from src.hold.hold_utils import prepare_loss_targets_object
from src.hold.hold_utils import volumetric_render


class HOLDNet(nn.Module):
    def __init__(
        self,
        opt,
        betas_r,
        betas_l,
        num_frames,
        args,
    ):
        super().__init__()
        self.args = args
        self.opt = opt
        self.sdf_bounding_sphere = opt.scene_bounding_sphere
        self.threshold = 0.05
        node_dict = {}
        if betas_r is not None:
            right_node = MANONode(args, opt, betas_r, self.sdf_bounding_sphere, "right")
            node_dict["right"] = right_node

        if betas_l is not None:
            left_node = MANONode(args, opt, betas_l, self.sdf_bounding_sphere, "left")
            node_dict["left"] = left_node

        object_node = ObjectNode(args, opt, self.sdf_bounding_sphere, "object")
        node_dict["object"] = object_node
        self.nodes = nn.ModuleDict(node_dict)
        self.background = Background(opt, args, num_frames, self.sdf_bounding_sphere)

        self.init_network()

    def forward_fg(self, input):
        input = xdict(input)
        out_dict = xdict()
        if self.training:
            out_dict["epoch"] = input["current_epoch"]
            out_dict["step"] = input["global_step"]

        torch.set_grad_enabled(True)
        sample_dict = None
        factors_dicts = {}
        sample_dicts = {}
        for node in self.nodes.values():
            factors, sample_dict = node(input)
            factors_dicts[node.node_id] = factors
            sample_dicts[node.node_id] = sample_dict

        import src.utils.debug as debug

        debug.debug_deformer(sample_dicts, self)

        # compute canonical SDF and features
        out_dict = self.prepare_loss_targets(out_dict, sample_dicts)

        factors_list = list(factors_dicts.values())
        factors = hold_utils.merge_factors(factors_list, check=True)

        for myfactors in factors_dicts.values():
            myfactors["z_max"] = myfactors["z_vals"][:, -1]

        # volumetric rendering: all
        comp_out_dict = volumetric_render(factors, self.training)
        out_dict.merge(comp_out_dict)

        for node_id, myfactors in factors_dicts.items():
            my_out_dict = volumetric_render(myfactors, self.training)
            out_dict.merge(my_out_dict.prefix(f"{node_id}."))

        # background
        bg_z_vals = self.background.inverse_sphere_sampler.inverse_sample(
            sample_dict["ray_dirs"],
            sample_dict["cam_loc"],
            self.training,
            self.sdf_bounding_sphere,
        )
        out_dict["bg_z_vals"] = bg_z_vals
        out_dict["ray_dirs"] = sample_dict["ray_dirs"]
        out_dict["cam_loc"] = sample_dict["cam_loc"]
        out_dict["index"] = input["idx"]

        return out_dict

    def step_embedding(self):
        # step on BARF counter
        for node in self.nodes.values():
            node.step_embedding()
        self.background.step_embedding()

    def forward(self, input):
        fg_dict = self.forward_fg(input)
        bg_dict = self.background(
            fg_dict["bg_weights"],
            fg_dict["ray_dirs"],
            fg_dict["cam_loc"],
            fg_dict["bg_z_vals"],
            fg_dict["index"],
        )
        out_dict = self.composite(fg_dict, bg_dict)

        if self.training:
            self.step_embedding()
        return out_dict

    def composite(self, fg_dict, bg_dict):
        out_dict = fg_dict
        # Composite foreground and background
        out_dict["rgb"] = fg_dict["fg_rgb"] + bg_dict["bg_rgb"]
        out_dict["semantics"] = fg_dict["fg_semantics"] + bg_dict["bg_semantics"]

        if not self.training:
            out_dict["bg_rgb_only"] = bg_dict["bg_rgb_only"]
            out_dict["instance_map"] = torch.argmax(out_dict["semantics"], dim=1)
        return out_dict

    def init_network(self):
        if self.args.shape_init != "":
            model_state = torch.load(
                f"./saved_models/{self.args.shape_init}/checkpoints/last.ckpt"
            )
            sd = model_state["state_dict"]
            sd = {
                k.replace("model.", ""): v
                for k, v in sd.items()
                if "implicit_network" in k
                and "bg_implicit_network." not in k
                and ".embedder_obj." not in k
            }
            logger.warning("Using MANO init that is for h2o, not the one in CVPR.")
            self.load_state_dict(sd, strict=False)
        else:
            logger.warning("Skipping INIT human models!")

    def prepare_loss_targets(self, out_dict, sample_dicts):
        if not self.training:
            return out_dict

        step = out_dict["step"]
        assert [node.node_id for node in self.nodes.values()] == [
            key for key in sample_dicts.keys()
        ]

        if step % 200 == 0 and step > 0:
            # if True:
            for node, node_id in zip(self.nodes.values(), sample_dicts):
                if node.node_id in ["right", "left"]:
                    node.spawn_cano_mano(sample_dicts[node_id])

        for node in self.nodes.values():
            node_id = node.node_id
            sample_dict = sample_dicts[node_id]
            if "right" in node_id or "left" in node_id:
                prepare_loss_targets_hand(out_dict, sample_dict, node)
            elif "object" in node_id:
                prepare_loss_targets_object(out_dict, sample_dict, node)
            else:
                raise ValueError(f"Unknown node_id: {node_id}")

        return out_dict
