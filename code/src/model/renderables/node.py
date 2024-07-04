import torch.nn as nn

import src.engine.volsdf_utils as volsdf_utils
from src.engine.rendering import render_color

from ...engine.density import LaplaceDensity
from ...engine.ray_sampler import ErrorBoundSampler
from ...networks.shape_net import ImplicitNet
from ...networks.texture_net import RenderingNet


class Node(nn.Module):
    def __init__(
        self,
        args,
        opt,
        specs,
        sdf_bounding_sphere,
        implicit_network_opt,
        rendering_network_opt,
        deformer,
        server,
        class_id,
        node_id,
        params,
    ):
        super(Node, self).__init__()
        self.args = args
        self.specs = specs
        self.sdf_bounding_sphere = sdf_bounding_sphere
        self.implicit_network = ImplicitNet(implicit_network_opt, args, specs)
        self.rendering_network = RenderingNet(rendering_network_opt, args, specs)
        self.ray_sampler = ErrorBoundSampler(
            self.sdf_bounding_sphere, inverse_sphere_bg=True, **opt.ray_sampler
        )
        self.density = LaplaceDensity(**opt.density)
        self.deformer = deformer
        self.server = server
        self.class_id = class_id
        self.node_id = node_id
        self.params = params

    def meshing_cano(self, pose=None):
        return None

    def sample_points(self, input):
        raise NotImplementedError("Derived classes should implement this method.")

    def forward(self, input):
        if "time_code" in input:
            time_code = input["time_code"]
        else:
            time_code = None
        sample_dict = self.sample_points(input)

        # compute canonical SDF and features
        (
            sdf_output,
            canonical_points,
            feature_vectors,
        ) = volsdf_utils.sdf_func_with_deformer(
            self.deformer,
            self.implicit_network,
            self.training,
            sample_dict["points"].reshape(-1, 3),
            sample_dict["deform_info"],
        )
        num_samples = sample_dict["z_vals"].shape[1]
        color, normal, semantics = self.render(
            sample_dict, num_samples, canonical_points, feature_vectors, time_code
        )
        self.device = color.device

        num_samples = color.shape[1]
        density = self.density(sdf_output).view(-1, num_samples, 1)
        sample_dict["canonical_pts"] = canonical_points.view(
            sample_dict["batch_size"], sample_dict["num_pixels"], num_samples, 3
        )
        # color, normal, density, semantics
        factors = {
            "color": color,
            "normal": normal,
            "density": density,
            "semantics": semantics,
            "z_vals": sample_dict["z_vals"],
        }
        return factors, sample_dict

    def render(
        self, sample_dict, num_samples, canonical_points, feature_vectors, time_code
    ):
        color, normal, semantics = render_color(
            self.deformer,
            self.implicit_network,
            self.rendering_network,
            sample_dict["ray_dirs"],
            sample_dict["cond"],
            sample_dict["tfs"],
            canonical_points,
            feature_vectors,
            self.training,
            num_samples,
            self.class_id,
            time_code,
        )
        return color, normal, semantics

    def step_embedding(self):
        self.implicit_network.embedder_obj.step()
