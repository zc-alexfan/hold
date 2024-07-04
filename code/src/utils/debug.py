import os
import os.path as op
import sys

import matplotlib.pyplot as plt
import torch
from PIL import Image

sys.path = [".."] + sys.path
from common.transforms import project2d, rigid_tf_torch_batch


def debug_params(self):
    return


def debug_deformer_mano(args, sample_dict, node):
    import trimesh

    verts_c = node.server.verts_c
    faces = node.server.faces
    idx = sample_dict["idx"][0]

    output = sample_dict["output"]
    # save canonical meshes first
    mesh = trimesh.Trimesh(
        verts_c[0].cpu().numpy().reshape(-1, 3), faces, process=False
    )
    out_p = op.join(args.log_dir, "debug", f"mesh_{node.node_id}_cano", f"0_cano.obj")
    os.makedirs(op.dirname(out_p), exist_ok=True)
    mesh.export(out_p)

    batch_size = sample_dict["batch_size"]

    # results from deformer
    verts = output["verts"].view(batch_size, -1, 3)[:1]
    x_c, outlier_mask = node.deformer.forward(
        verts.view(1, -1, 3),
        output["tfs"].view(batch_size, -1, 4, 4)[:1],
        return_weights=False,
        inverse=True,
        verts=verts,
    )
    mesh = trimesh.Trimesh(
        vertices=x_c.view(-1, 3).detach().cpu().numpy(),
        faces=faces,
        process=False,
    )
    out_p = op.join(args.log_dir, "debug", f"mesh_{node.node_id}_cano", f"{idx}.obj")
    os.makedirs(op.dirname(out_p), exist_ok=True)
    mesh.export(out_p)

    # results of   deformed space
    mesh = trimesh.Trimesh(
        vertices=verts.view(-1, 3).detach().cpu().numpy(),
        faces=faces,
        process=False,
    )
    out_p = op.join(
        args.log_dir, "debug", f"mesh_{node.node_id}_deform", f"{idx}_deform.obj"
    )
    os.makedirs(op.dirname(out_p), exist_ok=True)
    mesh.export(out_p)


def debug_deformer_obj(args, sample_dict, node):
    obj_verts_c = node.server.verts_c
    idx = sample_dict["idx"][0]
    batch_size = sample_dict["batch_size"]

    # canonical mesh
    obj_output = sample_dict["obj_output"]
    mesh_obj = pts2mesh(obj_verts_c[0].cpu().numpy(), radius=0.01, num_samples=100)
    out_p = op.join(args.log_dir, "debug", "mesh_obj_cano", f"0_cano.obj")
    os.makedirs(op.dirname(out_p), exist_ok=True)
    mesh_obj.export(out_p)

    # results from OBJ deformer
    obj_x = obj_output["verts"].view(batch_size, -1, 3)
    obj_x_c, _ = node.deformer.forward(obj_x, sample_dict["tfs"], inverse=True)
    mesh_obj = pts2mesh(
        obj_x_c[0].view(-1, 3).detach().cpu().numpy(), radius=0.01, num_samples=100
    )
    out_p = op.join(args.log_dir, "debug", "mesh_obj_cano", f"{idx}.obj")
    os.makedirs(op.dirname(out_p), exist_ok=True)
    mesh_obj.export(out_p)

    # mesh in deform space
    obj_x = obj_output["verts"].view(batch_size, -1, 3)
    mesh_obj = pts2mesh(obj_x[0].detach().cpu().numpy(), radius=0.01, num_samples=100)
    out_p = op.join(args.log_dir, "debug", "mesh_obj_deform", f"{idx}_deform.obj")
    os.makedirs(op.dirname(out_p), exist_ok=True)
    mesh_obj.export(out_p)

    # samples in deform space
    pts = sample_dict["points"].view(batch_size, -1, 3)
    mesh_obj = pts2mesh(
        pts[0].view(-1, 3).detach().cpu().numpy(), radius=0.01, num_samples=100
    )
    out_p = op.join(args.log_dir, "debug", "samples_obj_deform", f"{idx}.obj")
    os.makedirs(op.dirname(out_p), exist_ok=True)
    mesh_obj.export(out_p)


def debug_deformer(sample_dicts, self):
    if not self.args.debug:
        return

    args = self.args

    for node in self.nodes.values():
        sample_dict = sample_dicts[node.node_id]
        if node.node_id in ["right", "left"]:
            debug_deformer_mano(args, sample_dict, node)
        elif node.node_id in ["object"]:
            debug_deformer_obj(args, sample_dict, node)
        else:
            assert False


def debug_world2pix(args, output, input, node_id):
    if not args.debug:
        return

    # Load data
    data = torch.load(op.join(args.log_dir, "dataset_info.pth"))
    intrinsics_all = data["intrinsics_all"]
    extrinsics_all = data["extrinsics_all"]
    img_paths = data["img_paths"]
    idx = int(input["idx"][0])

    # Load image
    im = Image.open(img_paths[idx])
    plt.imshow(im)

    # Perform transformations
    w2c = extrinsics_all[idx].clone().inverse()
    if "verts" in output.keys():
        verts = output["verts"].cpu()  # world coordinate
        v3d_cam = rigid_tf_torch_batch(
            verts[:1], w2c[:3, :3][None, :, :], w2c[:3, 3:4][None, :, :]
        )[0]
        v2d_obj = project2d(intrinsics_all[idx][:3, :3], v3d_cam).detach().numpy()
        plt.scatter(v2d_obj[:, 0], v2d_obj[:, 1], s=1, color="r", alpha=0.5)

    out_p = op.join(args.log_dir, "debug", "world2pix", node_id, f"world2pix_{idx}.png")
    os.makedirs(op.dirname(out_p), exist_ok=True)
    plt.savefig(out_p)
    plt.close()


def pts2mesh(pts, radius, num_samples):
    import random

    import trimesh

    sampled_points_indices = random.sample(range(pts.shape[0]), num_samples)
    pts = pts[sampled_points_indices]

    # Initialize an empty scene
    scene = trimesh.Scene()

    # For each point, create a sphere and add it to the scene
    for pt in pts:
        # Create a sphere at the given point
        sphere = trimesh.creation.icosphere(subdivisions=1, radius=radius, color=None)

        # Translate the sphere to the right position
        sphere.apply_translation(pt)

        # Add the sphere to the scene
        scene.add_geometry(sphere)

    # Combine all spheres into a single mesh
    combined = trimesh.util.concatenate(scene.dump())

    return combined
