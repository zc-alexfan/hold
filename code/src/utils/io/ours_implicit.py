import os.path as op
from glob import glob

import common.viewer as viewer_utils
import numpy as np
import src.utils.factory as factory
import torch
from common.viewer import ViewerData
from PIL import Image
from common.xdict import xdict
from src.utils.eval_modules import compute_bounding_box_centers


def map_deform2eval(verts, scale):
    conversion_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    normalize_shift = np.array(
        [0.0085238, -0.01372686, 0.42570806]
    )  ### dumpy vec at all time for HO3D

    # data['cameras']['scale_mat_0'][0, 0]
    # scale = 0.2130073

    src_verts = np.copy(verts)

    src_verts = np.dot(src_verts, conversion_matrix)
    src_verts *= scale
    src_verts += normalize_shift
    return src_verts


def load_data(sd_p):
    device = "cuda:0"
    print("Loading data")

    data = torch.load(sd_p, map_location="cpu")

    sd = data["state_dict"]
    shape_h = sd["params_h.betas.weight"].to(device)
    rot_h = sd["params_h.global_orient.weight"].to(device)
    transl_h = sd["params_h.transl.weight"].to(device)
    pose_h = sd["params_h.pose.weight"].to(device)
    rot_o = sd["params_o.global_orient.weight"].to(device)
    transl_o = sd["params_o.transl.weight"].to(device)
    scale_key = "model.object_node.server.object_model.obj_scale"
    if scale_key in sd.keys():
        obj_scale = sd[scale_key]
    else:
        obj_scale = None
    exp_key = sd_p.split("/")[1]
    misc_ps = sorted(glob(op.join("logs", exp_key, "misc", "*")))
    misc = np.load(misc_ps[-1], allow_pickle=True).item()

    fnames = misc["img_paths"]
    K = torch.FloatTensor(misc["K"]).to(device).view(1, 4, 4)[:, :3, :3]
    scale = misc["scale"]
    mesh_c_o = misc["mesh_c_o"]

    assert len(fnames) == rot_o.shape[0]
    seq_name = fnames[0].split("/")[2]

    server_h = factory.get_server("mano", None, None)
    server_o = factory.get_server(seq_name, None, None, template=mesh_c_o)
    if obj_scale is not None:
        server_o.object_model.obj_scale = obj_scale.to(device)

    device = "cuda"

    scale = torch.tensor([scale]).float()
    num_frames = rot_o.shape[0]
    scale = scale.repeat(num_frames).view(-1, 1).to(device)

    full_pose_h = torch.cat((rot_h, pose_h), dim=1)
    out_o = server_o.forward(scale, transl_o, rot_o)
    v3d_o_c = out_o["obj_verts"]  # world is camera

    shape_h = shape_h.repeat(num_frames, 1)
    out_h = server_h.forward(scale.view(-1), transl_h, full_pose_h, shape_h)

    v3d_h_c = out_h["verts"]
    print("Loading MANO meshes")
    mesh_ps = sorted(glob(f"logs/{exp_key}/test/mesh_deform/*_mesh_deform_h_pose.obj"))
    assert len(mesh_ps) > 0
    import trimesh
    from tqdm import tqdm

    meshes_h = [trimesh.load(mesh_p, process=False) for mesh_p in tqdm(mesh_ps)]

    v3d_h_c = [np.array(mesh_h.vertices) for mesh_h in meshes_h]
    faces_h = [np.array(mesh_h.faces) for mesh_h in meshes_h]
    j3d_h_c = out_h["jnts"]
    assert len(v3d_h_c) == len(fnames)
    assert len(v3d_o_c) == len(fnames)

    # mapping to evaluation camera coordinate
    inverse_scale = float(1.0 / scale[0])
    hand_verts_c = [map_deform2eval(verts, inverse_scale) for verts in v3d_h_c]
    hand_joints_c = np.array(
        [
            map_deform2eval(verts, inverse_scale)
            for verts in j3d_h_c.cpu().detach().numpy()
        ]
    )
    object_verts_c = np.array(
        [
            map_deform2eval(verts, inverse_scale)
            for verts in v3d_o_c.cpu().detach().numpy()
        ]
    )

    out = {}
    out["fnames"] = fnames
    out["v3d_h_c"] = hand_verts_c
    out["j3d_h_c"] = hand_joints_c
    out["v3d_o_c"] = object_verts_c
    out["faces_o"] = np.array(mesh_c_o.faces)
    out["faces_h"] = faces_h

    v3d_h_rh_list = []
    for verts, rj3d in zip(hand_verts_c, hand_joints_c):
        v3d_h_rh_list.append(verts - rj3d[:1])
    out["v3d_h_rh"] = v3d_h_rh_list

    out["v3d_o_rh"] = np.stack(
        [verts - rj3d[:1, :] for verts, rj3d in zip(object_verts_c, hand_joints_c)],
        axis=0,
    )

    # stuff that I need for optimization
    out["server_h"] = server_h
    out["server_o"] = server_o
    out["shape_h"] = shape_h[0].cpu()  # .detach().numpy()
    out["full_pose_h"] = full_pose_h.cpu()  # .detach().numpy()
    out["transl_h"] = transl_h.cpu()  # .detach().numpy()
    out["scale"] = scale.cpu()  # .detach().numpy()
    out["transl_o"] = transl_o.cpu()  # .detach().numpy()
    out["rot_o"] = rot_o.cpu()  # .detach().numpy()
    # out['w2c'] = w2c.cpu()#.detach().numpy()

    out["K"] = K.cpu().numpy()
    out["full_seq_name"] = fnames[0].split("/")[2]

    insta_p = sd_p + ".insta_map.npy"
    if op.exists(insta_p):
        insta_map = torch.FloatTensor(np.load(sd_p + ".insta_map.npy"))
        out["insta_map"] = insta_map
    root_o = compute_bounding_box_centers(object_verts_c)[:, None]

    out["j3d_h_ra"] = hand_joints_c - hand_joints_c[:, :1]
    out["v3d_o_ra"] = object_verts_c - root_o
    out["root_o"] = root_o[:, 0]
    print("Done loading data")
    out = xdict(out).to_torch()
    return out


def load_viewer_data(args):
    data = load_data(args.ckpt_p)
    if args.rh:
        v3d_h_c = [verts.numpy() for verts in data["v3d_h_rh"]]
        v3d_o_c = data["v3d_o_rh"].numpy()
    elif args.ra:
        v3d_h_c = [verts.numpy() for verts in data["v3d_h_rh"]]
        v3d_o_c = data["v3d_o_ra"].numpy()
    else:
        v3d_h_c = [verts.numpy() for verts in data["v3d_h_c"]]
        v3d_o_c = data["v3d_o_c"].numpy()
    faces_o = data["faces_o"].numpy()
    faces_h = [faces.numpy() for faces in data["faces_h"]]

    K = data["K"].numpy().reshape(3, 3)
    fnames = data["fnames"]

    vis_dict = {}
    vis_dict["right-ours"] = {
        "v3d": v3d_h_c,
        "f3d": faces_h,
        "vc": None,
        "name": "right-ours",
        "color": "white",
        "vary_topo": True,
        "flat_shading": False,
    }

    vis_dict["obj-ours"] = {
        "v3d": v3d_o_c,
        "f3d": faces_o,
        "vc": None,
        "name": "object-ours",
        "color": "light-blue",
        "flat_shading": False,
    }

    meshes = viewer_utils.construct_viewer_meshes(
        vis_dict, draw_edges=False, flat_shading=False
    )
    # import pdb; pdb.set_trace()
    num_frames = len(fnames)
    Rt = np.zeros((num_frames, 3, 4))
    Rt[:, :3, :3] = np.eye(3)
    Rt[:, 1:3, :3] *= -1.0

    im = Image.open(fnames[0])
    cols, rows = im.size
    # fnames = None

    images = [Image.open(im_p) for im_p in fnames]
    data = ViewerData(Rt, K, cols, rows, images)
    return meshes, data
