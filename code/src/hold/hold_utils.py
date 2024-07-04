import sys

import torch

from src.engine.rendering import sort_tensor
import numpy as np

sys.path = [".."] + sys.path
from common.xdict import xdict

from kaolin.ops.mesh import index_vertices_by_faces
import trimesh
import src.engine.volsdf_utils as volsdf_utils

from src.engine.rendering import integrate


import torch
import numpy as np


class PointInSpace:
    def __init__(self, global_sigma=0.5, global_sigma_xyz=None, local_sigma=0.01):
        # self.global_sigma = global_sigma
        if global_sigma_xyz is None:
            self.global_sigma_xyz = torch.ones(3) * global_sigma
        else:
            self.global_sigma_xyz = torch.FloatTensor(np.array(global_sigma_xyz))
        self.local_sigma = local_sigma

    def get_points(self, pc_input=None, local_sigma=None, global_ratio=0.125):
        """Sample one point near each of the given point + 1/8 uniformly.
        Args:
            pc_input (tensor): sampling centers. shape: [B, N, D]
        Returns:
            samples (tensor): sampled points. shape: [B, N + N / 8, D]
        """
        if self.global_sigma_xyz.device != pc_input.device:
            self.global_sigma_xyz = self.global_sigma_xyz.to(pc_input.device)

        batch_size, sample_size, dim = pc_input.shape
        if local_sigma is None:
            sample_local = pc_input + (torch.randn_like(pc_input) * self.local_sigma)
        else:
            sample_local = pc_input + (torch.randn_like(pc_input) * local_sigma)
        sample_global = (
            torch.rand(
                batch_size, int(sample_size * global_ratio), dim, device=pc_input.device
            )
            * (self.global_sigma_xyz * 2)
        ) - self.global_sigma_xyz

        sample = torch.cat([sample_local, sample_global], dim=1)

        return sample


pt_in_space_sampler_h = PointInSpace(global_sigma_xyz=[0.15, 0.06, 0.12])


def query_oc(implicit_fn, x, cond):
    # query canonical occupancy
    mnfld_pred = implicit_fn(x, cond)
    mnfld_pred = mnfld_pred[:, :, 0]
    return {"sdf": mnfld_pred}


def prepare_batch(inputs, params_h, params_o, current_epoch, global_step):
    inputs.merge(params_h)
    inputs.merge(params_o)
    inputs["current_epoch"] = current_epoch
    inputs["global_step"] = global_step
    return inputs


def merge_factors(factors_list, check=True):
    # given factors from different nodes in the scene
    # merge them into one by concatenating them along the ray dimension and z-dimension order
    factors_ref = factors_list[0]
    # assert dimensions
    batch_size = factors_ref["color"].shape[0]
    num_rays = factors_ref["color"].shape[1]
    if check:
        for factors in factors_list:
            for k in factors.keys():
                if k != "z_vals":
                    assert len(factors[k].shape) == 3, f"{k}: factors should be 3D"
                    assert factors[k].shape[0] == batch_size, "batch size not equal"
                    assert factors[k].shape[1] == num_rays, "num rays not equal"
            assert len(factors["z_vals"].shape) == 2, "z_vals should be 2D"
            assert factors["z_vals"].shape[0] == batch_size, "batch size not equal"
            assert factors["z_vals"].shape[1] == num_rays, "num rays not equal"

    # concat each factor for all nodes
    factors_comp = xdict()
    for k in factors_ref.keys():
        factors_comp[k] = torch.cat([factors[k] for factors in factors_list], dim=1)

    # sort z
    z_vals = factors_comp["z_vals"]
    z_vals_sorted, indices = torch.sort(z_vals, dim=1)
    z_vals = sort_tensor(z_vals[:, :, None], indices)[:, :, 0]
    assert torch.abs(z_vals - z_vals_sorted).max() < 1e-6, "z_vals not sorted"
    factors_comp.overwrite("z_vals", z_vals)

    # sort others
    for k in factors_comp.keys():
        if k == "z_vals":
            continue
        factors_comp.overwrite(k, sort_tensor(factors_comp[k], indices))

    # remove repeated
    num_nodes = len(factors_list)
    for k in factors_comp.keys():
        ### WARNING: I think it should be
        # factors_comp.overwrite(k, factors_comp[k][:, (num_nodes-1):-(num_nodes - 1)])
        ### but keep this line the same as it is in CVPR code
        factors_comp.overwrite(k, factors_comp[k][:, (num_nodes - 1) : -num_nodes])

    factors_comp["z_max"] = z_vals[:, -num_nodes]
    return factors_comp


def wubba_lubba_dub_dub(batch):
    batch_size = batch["uv"].shape[0]
    num_images = batch["uv"].shape[1]

    # Iterate over each key, val pair in the dictionary
    for key, val in batch.items():
        if key in ["im_path", "total_pixels", "pixel_per_batch", "img_size"]:
            continue

        # Reshape each tensor while keeping the other dimensions unchanged
        batch[key] = val.reshape(batch_size * num_images, *val.shape[2:])
    return batch


def subdivide_cano(mesh_vh_cano, mesh_fh_cano):
    cano_v = mesh_vh_cano[0].cpu().detach().numpy()
    mano_f = mesh_fh_cano.cpu().detach().numpy()
    cano_v_div, mano_f_div = trimesh.remesh.subdivide_loop(cano_v, mano_f, iterations=1)

    mesh_v_cano_div = torch.FloatTensor(cano_v_div).cuda()
    mesh_f_cano_div = torch.LongTensor(mano_f_div).cuda()

    return mesh_v_cano_div, mesh_f_cano_div


def prepare_loss_targets_object(out_dict, sample_dict, node):
    num_pixels_total = sample_dict["batch_size"] * sample_dict["num_pixels"]
    node_id = node.node_id
    if node.mesh_o is not None:
        batch_size = sample_dict["batch_size"]
        mesh_vo_cano = node.mesh_vo_cano.repeat(batch_size, 1, 1)
        mesh_o = node.mesh_o.repeat(batch_size, 1, 1, 1)
        (
            index_off_obj_surface,
            _,
        ) = volsdf_utils.check_off_in_surface_points_cano_mesh(
            mesh_vo_cano,
            node.mesh_fo_cano,
            mesh_o,
            sample_dict["canonical_pts"].reshape(batch_size, -1, 3),
            num_pixels_total,
            threshold=0.05,
        )
        out_dict[f"{node_id}.index_off_surface"] = index_off_obj_surface

        # sample points around mesh
        num_samples = 256
        xyz = np.abs(node.mesh_vo_cano[0].cpu().detach().numpy()).max(axis=0) * 1.1
        pt_in_space_sampler_o = PointInSpace(global_sigma_xyz=xyz)
        out_dict[f"{node_id}.grad_theta"] = volsdf_utils.compute_gradient_samples(
            pt_in_space_sampler_o,
            node.implicit_network,
            sample_dict["cond"],
            num_samples,
            node.mesh_vo_cano.repeat(batch_size, 1, 1),
            local_sigma=0.03,
            global_ratio=0.20,
        )


def prepare_loss_targets_hand(out_dict, sample_dict, node):
    num_pixels_total = sample_dict["batch_size"] * sample_dict["num_pixels"]
    node_id = node.node_id
    if node.mesh_v_cano_div is not None:
        mesh_vh_cano = node.mesh_v_cano_div[None, :, :].repeat(
            sample_dict["batch_size"], 1, 1
        )
        mesh_fh_cano = node.mesh_f_cano_div
        mesh_h = index_vertices_by_faces(mesh_vh_cano, mesh_fh_cano)

        # compute SDF from cano samples to MANO cano mesh (pose dependent)
        mano_cano_samples = sample_on_barycentric_mesh(
            mesh_vh_cano, mesh_fh_cano, num_samples=256
        )

        mano_cano_samples = pt_in_space_sampler_h.get_points(
            mano_cano_samples, local_sigma=0.008, global_ratio=0.20
        )
        out_dict[f"{node_id}.pts2mano_sdf_cano"] = volsdf_utils.compute_mano_cano_sdf(
            mesh_vh_cano,
            mesh_fh_cano,
            mesh_h,
            mano_cano_samples,
        )

        # pose dependent canonical sdf
        cond = sample_dict["deform_info"]["cond"]
        batch_size = sample_dict["batch_size"]
        cano_pts = sample_dict["canonical_pts"].view(batch_size, -1, 3)
        pred_sdf = query_oc(node.implicit_network, mano_cano_samples, cond)["sdf"]
        (
            index_off_mano_surface,
            _,
        ) = volsdf_utils.check_off_in_surface_points_cano_mesh(
            mesh_vh_cano,
            mesh_fh_cano,
            mesh_h,
            cano_pts,
            num_pixels_total,
            threshold=0.01,
        )
        out_dict[f"{node_id}.pred_sdf"] = pred_sdf
        out_dict[f"{node_id}.index_off_surface"] = index_off_mano_surface

        # sample points around each  vertex
        # compute its gradient for eikonal later
        num_samples = 256
        verts_c = node.server.verts_c.repeat(sample_dict["batch_size"], 1, 1)
        out_dict[f"{node_id}.grad_theta"] = volsdf_utils.compute_gradient_samples(
            pt_in_space_sampler_h,
            node.implicit_network,
            sample_dict["cond"],
            num_samples,
            verts_c,
            local_sigma=0.008,
            global_ratio=0.20,
        )


def volumetric_render(factors, is_training):
    # density to weights
    fg_weights, bg_weights = volsdf_utils.density2weight(
        factors["density"], factors["z_vals"], factors["z_max"]
    )
    color = factors["color"]
    normal = factors["normal"]
    semantics = factors["semantics"]
    depth = factors["z_vals"][:, :, None]
    fg_rgb = integrate(color, fg_weights)
    fg_mask = integrate(torch.ones_like(color)[:, :, :1], fg_weights)
    fg_normal = integrate(normal, fg_weights)
    fg_depth = integrate(depth, fg_weights)
    fg_semantics = integrate(semantics, fg_weights)

    # output for hand + object
    out_dict = xdict()
    out_dict["fg_rgb"] = fg_rgb
    out_dict["fg_weights"] = fg_weights
    out_dict["mask_prob"] = torch.clamp(fg_mask, 0, 1)
    out_dict["normal"] = fg_normal
    out_dict["depth"] = fg_depth
    out_dict["fg_semantics"] = fg_semantics
    out_dict["bg_weights"] = bg_weights

    if not is_training:
        bg_rgb = torch.ones_like(fg_rgb, device=fg_rgb.device)
        out_dict["fg_rgb.vis"] = fg_rgb + bg_weights[:, None] * bg_rgb
    return out_dict


def sample_on_barycentric_mesh(verts, faces, num_samples):
    # Ensure that the random tensors are created on the same device as the input tensors
    device = verts.device

    batch_size, num_verts, _ = verts.shape
    num_faces = faces.shape[0]

    # Randomly select faces
    face_indices = torch.randint(0, num_faces, (batch_size, num_samples), device=device)

    # Gather the vertices corresponding to the faces
    sampled_faces = faces[face_indices]

    # Gather the coordinates of the vertices of the sampled faces
    v0 = torch.gather(verts, 1, sampled_faces[..., 0].unsqueeze(-1).expand(-1, -1, 3))
    v1 = torch.gather(verts, 1, sampled_faces[..., 1].unsqueeze(-1).expand(-1, -1, 3))
    v2 = torch.gather(verts, 1, sampled_faces[..., 2].unsqueeze(-1).expand(-1, -1, 3))

    # Sample random barycentric coordinates
    u = torch.rand((batch_size, num_samples, 1), device=device)
    v = torch.rand((batch_size, num_samples, 1), device=device)

    # If the sum of u, v exceeds 1, we flip the coordinates to ensure the points lie within the triangle
    mask = u + v > 1
    u[mask], v[mask] = 1 - u[mask], 1 - v[mask]

    # Compute the sampled points using the barycentric coordinates
    samples = u * v0 + v * v1 + (1 - u - v) * v2

    return samples


def downsample_rendering(batch, k):
    im_h = batch["img_size"][0].item()
    im_w = batch["img_size"][1].item()
    im_h_new = im_h
    im_w_new = im_w
    num_pixels = im_h * im_w
    for key, val in batch.items():
        if (
            isinstance(val, torch.Tensor)
            and len(val.shape) >= 2
            and val.shape[1] == num_pixels
        ):
            if len(val.shape) == 2:
                val = val.view(val.shape[0], im_h, im_w)[:, ::k, ::k]
                im_h_new, im_w_new = val.shape[1:]
                val = val.reshape(val.shape[0], -1)
            else:
                val = val.view(val.shape[0], im_h, im_w, val.shape[-1])[
                    :, ::k, ::k, :
                ].reshape(val.shape[0], -1, val.shape[-1])
            batch.overwrite(key, val)
    im_h_new = torch.LongTensor(np.array([im_h_new])).cuda()
    im_w_new = torch.LongTensor(np.array([im_w_new])).cuda()
    batch.overwrite("img_size", [im_h_new, im_w_new])
    batch.overwrite("total_pixels", im_h_new * im_w_new)
    return batch
