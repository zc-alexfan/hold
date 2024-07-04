import sys

import torch

sys.path = ["."] + sys.path


def construct_da_object_pose():
    from src.model.obj.specs import object_specs

    param_canonical = torch.zeros(
        (1, object_specs.total_dim), dtype=torch.float32
    ).cuda()
    param_canonical[0, 0] = 1.0  # scale
    # identity is the canonical
    return param_canonical


class ObjectServer(torch.nn.Module):
    def __init__(self, seq_name, template=None):
        super().__init__()

        from src.model.obj.object_model import ObjectModel

        model = ObjectModel(seq_name, template)
        self.object_model = model.cuda()

        self.param_canonical = construct_da_object_pose()

        # scale, rot, trans
        scene_scale = self.param_canonical[:, 0]
        rot = self.param_canonical[:, 1:4]
        trans = self.param_canonical[:, 4 : 4 + 3]

        output = self.forward(scene_scale, rot, trans)
        self.verts_c = output["verts"]
        self.verts_c = self.object_model.v3d_cano[None, :, :]

    def forward(self, scene_scale, transl, thetas, absolute=False):
        output = {}

        obj_output = self.object_model.forward(
            rot=thetas, trans=transl, scene_scale=scene_scale
        )
        output["verts"] = obj_output["vertices"]
        output["obj_tfs"] = obj_output["T"][:, None, :, :]
        return output

    def forward_param(self, param_dict):
        global_orient = param_dict.fuzzy_get("global_orient")
        transl = param_dict.fuzzy_get("transl")
        scene_scale = param_dict.fuzzy_get("scene_scale")
        batch_size = global_orient.shape[0]
        scene_scale = scene_scale.view(-1).repeat(batch_size)
        out = self.forward(scene_scale, transl, global_orient)
        return out
