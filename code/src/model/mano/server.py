import sys

import torch

sys.path = ["."] + sys.path


import src.utils.external.body_models as body_models


def construct_da_mano_pose(hand_mean):
    from src.model.mano.specs import mano_specs as body_specs

    param_canonical = torch.zeros((1, body_specs.total_dim), dtype=torch.float32).cuda()
    param_canonical[0, 0] = 1  # scale
    param_canonical[0, 7:52] = -hand_mean.unsqueeze(0)
    return param_canonical


class GenericServer(torch.nn.Module):
    def __init__(
        self,
        body_specs,
        betas=None,
        human_layer=None,
    ):
        super().__init__()
        assert human_layer is not None
        self.human_layer = human_layer.cuda()
        self.bone_parents = self.human_layer.bone_parents.astype(int)
        self.bone_parents[0] = -1
        self.bone_ids = []
        self.faces = self.human_layer.faces
        for i in range(body_specs.num_full_tfs):
            self.bone_ids.append([self.bone_parents[i], i])

        self.v_template = None

        if betas is not None:
            self.betas = torch.tensor(betas).float().cuda()
        else:
            self.betas = None

        # define the canonical pose
        param_canonical = construct_da_mano_pose(self.human_layer.hand_mean)
        if self.betas is not None:
            param_canonical[0, -body_specs.shape_dim :] = self.betas  # shape
        self.param_canonical = param_canonical

        self.cano_params = torch.split(
            self.param_canonical,
            [1, 3, body_specs.full_pose_dim, body_specs.shape_dim],
            dim=1,
        )

        # forward to get verts and joints
        output = self.forward(*self.cano_params, absolute=True)
        self.verts_c = output["verts"]
        self.joints_c = output["jnts"]
        self.tfs_c_inv = output["tfs"].squeeze(0).inverse()

    def forward(self, scene_scale, transl, thetas, betas, absolute=False):
        out = {}

        # ignore betas if v_template is provided
        if self.v_template is not None:
            betas = torch.zeros_like(betas)

        outputs = body_models.forward_layer(
            self.human_layer,
            betas=betas,
            transl=torch.zeros_like(transl),
            pose=thetas[:, 3:],
            global_orient=thetas[:, :3],
            return_verts=True,
            return_full_pose=True,
            v_template=None,
        )
        verts = outputs.vertices.clone()

        scene_scale = scene_scale.view(-1, 1, 1)
        transl = transl.view(-1, 1, 3)

        out["verts"] = verts * scene_scale + transl * scene_scale

        joints = outputs.joints.clone()
        out["jnts"] = joints * scene_scale + transl * scene_scale
        tf_mats = outputs.T.clone()
        tf_mats[:, :, :3, :] = tf_mats[:, :, :3, :] * scene_scale.view(-1, 1, 1, 1)
        tf_mats[:, :, :3, 3] = tf_mats[:, :, :3, 3] + transl * scene_scale

        # adjust current pose relative to canonical pose
        if not absolute:
            tf_mats = torch.einsum("bnij,njk->bnik", tf_mats, self.tfs_c_inv)

        out["tfs"] = tf_mats
        out["skin_weights"] = outputs.weights
        out["v_posed"] = outputs.v_posed
        return out

    def forward_param(self, param_dict):
        global_orient = param_dict.fuzzy_get("global_orient")
        pose = param_dict.fuzzy_get("pose")
        transl = param_dict.fuzzy_get("transl")
        full_pose = torch.cat((global_orient, pose), dim=1)
        shape = param_dict.fuzzy_get("betas")
        scene_scale = param_dict.fuzzy_get("scene_scale")

        batch_size = full_pose.shape[0]
        scene_scale = scene_scale.view(-1).repeat(batch_size)
        shape = shape.repeat(batch_size, 1)
        out = self.forward(scene_scale, transl, full_pose, shape)
        return out


class MANOServer(GenericServer):
    def __init__(self, betas, is_rhand):
        from src.model.mano.specs import mano_specs
        from src.utils.external.body_models import MANO

        mano_layer = MANO(
            model_path="./body_models",
            is_rhand=is_rhand,
            batch_size=1,
            flat_hand_mean=False,
            dtype=torch.float32,
            use_pca=False,
        )
        super().__init__(
            body_specs=mano_specs,
            betas=betas,
            human_layer=mano_layer,
        )
