import torch
import torch.nn as nn
import smplx
from common.xdict import xdict
from common.transforms import project2d_batch
from src.alignment.loss_terms import gmof

MANO_DIR_L = "../code/body_models/MANO_LEFT.pkl"
MANO_DIR_R = "../code/body_models/MANO_RIGHT.pkl"


class MANOParameters(nn.Module):
    def __init__(self, data, meta, is_right):
        super().__init__()

        K = meta["K"]
        num_frames = len(data["global_orient"])

        # register parameters
        betas = nn.Parameter(data["betas"].mean(dim=0))
        global_orient = nn.Parameter(data["global_orient"])
        pose = nn.Parameter(data["hand_pose"])

        transl = torch.zeros(num_frames, 3)
        transl[:, 2] = 1.0  # init in front of camera (opencv)
        transl = nn.Parameter(transl)
        self.register_parameter("hand_beta", betas)
        self.register_parameter("hand_rot", global_orient)
        self.register_parameter("hand_pose", pose)
        self.register_parameter("hand_transl", transl)

        MANO_DIR = MANO_DIR_R if is_right else MANO_DIR_L
        self.mano_layer = smplx.create(
            model_path=MANO_DIR, model_type="mano", use_pca=False, is_rhand=is_right
        )

        self.K = K

        self.im_paths = meta["im_paths"]

    def to(self, device):
        self.mano_layer.to(device)
        return super().to(device)

    def forward(self):
        num_frames = len(self.hand_rot)
        betas = self.hand_beta[None, :].repeat(num_frames, 1)
        output = self.mano_layer(
            betas=betas,
            global_orient=self.hand_rot,
            hand_pose=self.hand_pose,
            transl=self.hand_transl,
        )

        j3d = output.joints
        j3d_ra = j3d - j3d[:, 0:1, :]

        K = self.K[None, :, :].repeat(num_frames, 1, 1).to(j3d.device)

        out = xdict()
        v3d = output.vertices
        v2d = project2d_batch(K, v3d)
        j2d = project2d_batch(K, j3d)

        out["j3d"] = output.joints
        out["j3d.ra"] = j3d_ra
        out["v3d"] = output.vertices
        out["v2d"] = v2d
        out["j2d"] = j2d
        out["hand_rot"] = self.hand_rot
        out["hand_pose"] = self.hand_pose
        out["hand_beta"] = self.hand_beta
        out["hand_transl"] = self.hand_transl
        out["im_paths"] = self.im_paths
        out["K"] = self.K
        return out
