import sys
import numpy as np
import torch
import torch.nn as nn

# Standard Libraries
import sys

import imageio

# Third-party Libraries
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


sys.path = [".."] + sys.path

import common.transforms as transforms
from common.body_models import seal_mano_mesh
from src.fitting.utils import (
    create_meshes,
    create_silhouette_renderer,
)
from common.xdict import xdict


class Model(nn.Module):
    def __init__(
        self,
        servers,
        scene_scale,
        obj_scale,
        param_dict,
        device,
        target_masks,
        w2c,
        K,
        fnames,
        faces,
    ):
        super().__init__()
        self.w2c = w2c
        self.imsize = (target_masks.shape[1], target_masks.shape[2])
        self.servers = servers
        self.faces = faces
        self.fnames = fnames
        self.node_ids = list(self.servers.keys())
        self.rasterizer, self.shader, self.renderer = create_silhouette_renderer(
            K.clone(), device, self.imsize
        )
        self.scene_scale = scene_scale.clone().to(device)
        self.obj_scale = nn.Parameter(
            torch.FloatTensor(np.array(obj_scale)).clone().to(device)
        )

        self.reform_param_dict(param_dict)
        from src.fitting.utils import construct_targets, create_color_map

        self.targets = construct_targets(target_masks)
        self.color_map = create_color_map(target_masks.long()).to(device)

        self.setup_callbacks()
        self.K = K.clone()

    def setup_callbacks(self):
        from src.fitting.loss import loss_fn_ih, loss_fn_lh, loss_fn_rh
        from src.fitting.utils import vis_fn_ih

        self.vis_fn = vis_fn_ih
        if "left" in self.node_ids and "right" in self.node_ids:
            self.loss_fn = loss_fn_ih
        elif "left" in self.node_ids:
            self.loss_fn = loss_fn_lh
        elif "right" in self.node_ids:
            self.loss_fn = loss_fn_rh
        else:
            assert False, f"Unknown node ids: {self.node_ids}"
        print("Loss function:", self.loss_fn.__name__)

    def reform_param_dict(self, param_dict):
        param_dict_new = xdict()
        for key, val in param_dict.items():
            entitiy = key.split(".")[2]
            param_name = key.split(".")[4]
            new_key = f"{entitiy}__{param_name}"
            param_dict_new[new_key] = nn.Parameter(val)
        for node_id in self.servers.keys():
            param_dict_new[f"{node_id}__scene_scale"] = nn.Parameter(self.scene_scale)

        from src.fitting.utils import MyParameterDict

        self.param_dict = MyParameterDict(param_dict_new)

    def freeze_all(self):
        for param in self.parameters():
            param.requires_grad = False

    def defrost_all(self):
        for param in self.parameters():
            param.requires_grad = True

    def print_requires_grad(self):
        print("requires_grad status:")
        for param_name, param in self.named_parameters():
            print(f"\t{param_name}: {param.requires_grad}")

    def fwd_params(self):
        param_dict = self.param_dict
        w2c = self.w2c
        device = "cuda"
        out_dict = xdict()
        self.servers["object"].object_model.obj_scale = self.obj_scale
        # for each node
        for node_id in self.node_ids:
            out = self.servers[node_id].forward_param(param_dict.search(node_id))

            # Transform vertices
            v3d_c = transforms.rigid_tf_torch_batch(
                out["verts"], w2c[:, :3, :3], w2c[:, :3, 3:]
            )
            out["v3d_c"] = v3d_c

            # Seal mesh (only if 'right', 'left', or 'object')
            if node_id in ["right", "left"]:
                v3d_sealed, faces_sealed = seal_mano_mesh(
                    v3d_c, self.faces[node_id], is_rhand=(node_id == "right")
                )
            elif node_id == "object":
                v3d_sealed = v3d_c
                faces_sealed = self.faces[node_id]

            # Render mask
            meshes = create_meshes(v3d_sealed, faces_sealed, device)
            out["mask"] = self.renderer(
                meshes_world=meshes.clone(), image_size=self.imsize, bin_size=-1
            )[..., 3]

            # Prefix and merge output
            out = xdict(out).prefix(node_id + ".")
            out_dict.merge(out)
        out_dict["K"] = self.K.clone()
        return out_dict

    def forward(self):
        out = self.fwd_params()
        loss_dict = self.loss_fn(out, self.targets)
        segm_vis = self.vis_fn(out, self.targets)

        self.servers["object"].object_model.obj_scale = self.obj_scale
        return (loss_dict, segm_vis)

    def setup_optimizer(self):
        lr = 1e-2
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=30, verbose=True
        )

    def fit(
        self,
        num_iterations=200,
        vis_every=50,
        write_gif=True,
        out_ps=None,
    ):
        tol_lr = 1e-5
        writers = []

        if write_gif:
            for out_p in out_ps:
                writer = imageio.get_writer(out_p, mode="I", duration=0.3)
                writers.append(writer)

        loop = tqdm(range(num_iterations))
        for i in loop:
            self.optimizer.zero_grad()
            loss_dict, segm_vis = self()
            loss = loss_dict["loss"]
            if torch.isnan(loss) > 0:
                break
            loss.backward()
            self.optimizer.step()
            self.scheduler.step(loss)
            curr_lr = self.optimizer.param_groups[0]["lr"]
            if curr_lr < tol_lr:
                break

            desc = ""
            for k, v in loss_dict.items():
                desc += f"{k}: {v.item():.4f}, "

            loop.set_description(desc)
            if i % vis_every == 0:
                self.visualize_fitting(write_gif, segm_vis, writers)

        if write_gif:
            for writer in writers:
                writer.close()

    def visualize_fitting(self, write_gif, segm_vis, writers):
        if write_gif:
            for idx, writer in enumerate(writers):
                image_vis = segm_vis[idx]
                writers[idx].append_data(image_vis)
