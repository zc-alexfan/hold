import torch
from src.model.generic.params import GenericParams


class MANOParams(GenericParams):
    def forward(self, frame_ids):
        params = super().forward(frame_ids)
        node_id = self.node_id
        params[f"{node_id}.full_pose"] = torch.cat(
            (params[f"{node_id}.global_orient"], params[f"{node_id}.pose"]), dim=1
        )
        return params

    def load_params(self, case):
        import os

        import numpy as np

        # load parameter from preprocessing
        params_h = {param_name: [] for param_name in self.param_names}
        data_root = os.path.join("./data", case, f"build/data.npy")
        data = np.load(data_root, allow_pickle=True).item()["entities"][self.node_id]

        mean_shape = data["mean_shape"]
        params_h["betas"] = torch.tensor(
            mean_shape[None],
            dtype=torch.float32,
        )

        poses = data["hand_poses"]
        trans = data["hand_trans"]

        params_h["global_orient"] = torch.tensor(
            poses[:, :3],
            dtype=torch.float32,
        )
        params_h["pose"] = torch.tensor(
            poses[:, 3:],
            dtype=torch.float32,
        )
        params_h["transl"] = torch.tensor(
            trans,
            dtype=torch.float32,
        )
        for param_name in params_h.keys():
            self.init_parameters(param_name, params_h[param_name], requires_grad=False)
