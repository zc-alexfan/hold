from src.model.generic.params import GenericParams


class ObjectParams(GenericParams):
    def forward(self, frame_ids):
        params = super().forward(frame_ids)
        return params

    def load_params(self, case):
        import os

        import numpy as np
        import torch

        # load parameter from preprocessing
        params_o = {param_name: [] for param_name in self.param_names}
        data_root = os.path.join("./data", case, f"build/data.npy")
        data = np.load(data_root, allow_pickle=True).item()["entities"]["object"]

        obj_poses = data["object_poses"]
        params_o["global_orient"] = torch.tensor(
            obj_poses[:, :3],
            dtype=torch.float32,
        )
        params_o["transl"] = torch.tensor(
            obj_poses[:, 3:],
            dtype=torch.float32,
        )
        for param_name in params_o.keys():
            self.init_parameters(param_name, params_o[param_name], requires_grad=False)
