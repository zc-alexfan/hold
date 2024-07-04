import torch
import torch.nn as nn

from ..engine.embedders import get_embedder


class RenderingNet(nn.Module):
    def __init__(self, opt, args, body_specs):
        super().__init__()

        self.mode = opt.mode
        dims = [opt.d_in + opt.feature_vector_size] + list(opt.dims) + [opt.d_out]

        self.body_specs = body_specs

        self.embedder_obj = None
        if opt.multires_view > 0:
            embedder_obj, input_ch = get_embedder(
                opt.multires_view,
                mode=body_specs.embedding,
                barf_s=args.barf_s,
                barf_e=args.barf_e,
                no_barf=args.no_barf,
            )
            self.embedder_obj = embedder_obj
            dims[0] += input_ch - 3
        if self.mode == "nerf_frame_encoding":
            dims[0] += opt.dim_frame_encoding
        if self.mode == "pose":
            self.dim_cond_embed = 8
            self.cond_dim = (
                self.body_specs.pose_dim
            )  # dimension of the body pose, global orientation excluded.
            # lower the condition dimension
            self.lin_pose = torch.nn.Linear(self.cond_dim, self.dim_cond_embed)
        self.num_layers = len(dims)
        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)
            if opt.weight_norm:
                lin = nn.utils.weight_norm(lin)
            setattr(self, "lin" + str(l), lin)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        points,
        normals,
        view_dirs,
        body_pose,
        feature_vectors,
        frame_latent_code=None,
    ):
        if self.embedder_obj is not None:
            if self.mode == "nerf_frame_encoding":
                view_dirs = self.embedder_obj.embed(view_dirs)

        if self.mode == "nerf_frame_encoding":
            # frame_latent_code = frame_latent_code.expand(view_dirs.shape[1], -1)
            frame_latent_code = frame_latent_code[:, None, :].repeat(
                1, view_dirs.shape[1], 1
            )
            rendering_input = torch.cat(
                [view_dirs, frame_latent_code, feature_vectors], dim=-1
            )

            rendering_input = rendering_input.view(-1, rendering_input.shape[2])
        elif self.mode == "pose":
            num_images = body_pose.shape[0]
            points = points.view(num_images, -1, 3)

            num_points = points.shape[1]
            points = points.reshape(num_images * num_points, -1)
            body_pose = (
                body_pose[:, None, :]
                .repeat(1, num_points, 1)
                .reshape(num_images * num_points, -1)
            )
            num_dim = body_pose.shape[1]
            if num_dim > 0:
                body_pose = self.lin_pose(body_pose)
            else:
                # when no pose parameters
                body_pose = torch.zeros(points.shape[0], self.dim_cond_embed).to(
                    points.device
                )
            rendering_input = torch.cat(
                [points, normals, body_pose, feature_vectors], dim=-1
            )
        else:
            raise NotImplementedError

        x = rendering_input
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.relu(x)
        x = self.sigmoid(x)
        return x
