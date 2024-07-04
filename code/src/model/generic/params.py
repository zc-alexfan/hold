import torch
import torch.nn as nn
from common.xdict import xdict


class GenericParams(nn.Module):
    def __init__(self, num_frames, params_dim, node_id):
        super(GenericParams, self).__init__()
        self.num_frames = num_frames

        # parameter dims to keep track
        self.params_dim = params_dim

        self.node_id = node_id

        self.param_names = self.params_dim.keys()

        # init variables based on dim specs
        for param_name in self.param_names:
            if param_name == "betas":
                param = nn.Embedding(1, self.params_dim[param_name])
            else:
                param = nn.Embedding(num_frames, self.params_dim[param_name])
            param.weight.data.fill_(0)
            param.weight.requires_grad = False
            setattr(self, param_name, param)

    def init_parameters(self, param_name, data, requires_grad=False):
        getattr(self, param_name).weight.data = data[..., : self.params_dim[param_name]]
        getattr(self, param_name).weight.requires_grad = requires_grad

    def set_requires_grad(self, param_name, requires_grad=True):
        getattr(self, param_name).weight.requires_grad = requires_grad

    def forward(self, frame_ids):
        # given frame_ids, return parameters for the frames
        params = xdict()
        for param_name in self.param_names:
            if param_name == "betas":
                params[param_name] = getattr(self, param_name)(
                    torch.zeros_like(frame_ids)
                )
            else:
                params[param_name] = getattr(self, param_name)(frame_ids)

        params = params.prefix(self.node_id + ".")
        return params

    def load_params(self, case):
        raise NotImplementedError

    def defrost(self, keys=None):
        if keys is None:
            keys = self.param_names
        for param_name in keys:
            self.set_requires_grad(param_name, requires_grad=True)

    def freeze(self, keys=None):
        if keys is None:
            keys = self.param_names
        for param_name in keys:
            self.set_requires_grad(param_name, requires_grad=False)
