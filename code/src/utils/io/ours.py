import os.path as op
from glob import glob


import numpy as np
import torch

from PIL import Image
from common.xdict import xdict
from src.model.mano.server import MANOServer
from src.model.obj.server import ObjectServer
from src.utils.eval_modules import compute_bounding_box_centers


def map_deform2eval(verts, scale, _normalize_shift):
    conversion_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    # normalize_shift = np.array(
    #     [0.0085238, -0.01372686, 0.42570806]
    # )  # dummy vec at all time for HO3D

    normalize_shift = _normalize_shift.copy()
    normalize_shift[0] *= -1

    src_verts = np.copy(verts)

    src_verts = np.dot(src_verts, conversion_matrix)
    src_verts *= scale
    src_verts += normalize_shift
    return src_verts


def load_data(sd_p):
    device = "cuda:0"
    print("Loading data")
    data = torch.load(sd_p, map_location="cpu")
    sd = xdict(data["state_dict"])
    misc_ps = sorted(glob(op.join("logs", sd_p.split("/")[1], "misc", "*")))
    misc = np.load(misc_ps[-1], allow_pickle=True).item()

    fnames = misc["img_paths"]
    K = torch.FloatTensor(misc["K"]).to(device).view(1, 4, 4)[:, :3, :3]
    scale = misc["scale"]
    scale = torch.tensor([scale]).float().to(device)
    mesh_c_o = misc["mesh_c_o"] if "mesh_c_o" in misc else misc["object_cano"]

    node_ids = []
    for key in sd.keys():
        if ".nodes." not in key:
            continue
        node_id = key.split(".")[2]
        node_ids.append(node_id)
    node_ids = list(set(node_ids))

    params = {}
    for node_id in node_ids:
        params[node_id] = sd.search(".params.").search(node_id)
        params[node_id]["scene_scale"] = scale
        params[node_id] = params[node_id].to(device)

    scale_key = "model.nodes.object.server.object_model.obj_scale"
    obj_scale = sd[scale_key] if scale_key in sd.keys() else None

    seq_name = fnames[0].split("/")[2]

    servers = {}
    faces = {}
    for node_id in node_ids:
        if "right" in node_id or "left" in node_id:
            hand = "right" if "right" in node_id else "left"
            is_right = hand == "right"
            server = MANOServer(betas=None, is_rhand=is_right).to(device)
            myfaces = torch.LongTensor(server.faces.astype(np.int64)).to(device)
        elif "object" in node_id:
            server = ObjectServer(seq_name, template=mesh_c_o)
            server.object_model.obj_scale = obj_scale
            server.to(device)
            myfaces = torch.LongTensor(mesh_c_o.faces).to(device)
        else:
            assert False, f"Unknown node id: {node_id}"

        servers[node_id] = server
        faces[node_id] = myfaces

    if obj_scale is not None:
        servers["object"].object_model.obj_scale = obj_scale.to(device)

    out = xdict()
    for node_id in node_ids:
        out.merge(
            xdict(servers[node_id].forward_param(params[node_id])).postfix(
                f".{node_id}"
            )
        )

    # mapping to evaluation camera coordinate
    def map_deform2eval_batch(verts, inverse_scale, normalize_shift):
        return np.array(
            [
                map_deform2eval(verts, inverse_scale, normalize_shift)
                for verts in verts.cpu().detach().numpy()
            ]
        )

    dataset = np.load(f"data/{seq_name}/build/data.npy", allow_pickle=True).item()
    normalize_shift = dataset["normalize_shift"]
    # normalize_shift = np.array([-0.0085238, -0.01372686, 0.42570806])

    # map predictions to evaluation space
    inverse_scale = float(1.0 / scale[0])
    for key, val in out.search("verts.").items():
        out[key.replace("verts.", "v3d_c.")] = map_deform2eval_batch(
            val, inverse_scale, normalize_shift
        )
    for key, val in out.search("jnts.").items():
        out[key.replace("jnts.", "j3d_c.")] = map_deform2eval_batch(
            val, inverse_scale, normalize_shift
        )

    # hand root relative
    for key, val in out.search("j3d_c.").items():
        # root
        out[key.replace("j3d_c.", "root.")] = val[:, :1].squeeze(1)
        # root relative
        out[key.replace("j3d_c.", "j3d_ra.")] = val - val[:, :1]
    out["root.object"] = compute_bounding_box_centers(out["v3d_c.object"])
    out["v3d_ra.object"] = out["v3d_c.object"] - out["root.object"][:, None, :]

    # object: relative to right hand
    out["v3d_right.object"] = out["v3d_c.object"] - out["root.right"][:, None, :]
    if "root.left" in out.keys():
        out["v3d_left.object"] = out["v3d_c.object"] - out["root.left"][:, None, :]
    out_dict = xdict()
    out_dict["fnames"] = fnames
    out_dict.merge(out)
    out_dict["faces"] = faces

    out_dict["servers"] = servers
    out_dict["K"] = K.cpu().numpy()
    out_dict["full_seq_name"] = fnames[0].split("/")[2]

    insta_p = sd_p + ".insta_map.npy"
    if op.exists(insta_p):
        insta_map = torch.FloatTensor(np.load(sd_p + ".insta_map.npy"))
        out_dict["insta_map"] = insta_map

    print("Done loading data")
    out_dict = out_dict.to_torch()
    return out_dict


def load_viewer_data(args):
    data = load_data(args.ckpt_p)
    faces = xdict(data["faces"]).to_np()
    K = data["K"].numpy().reshape(3, 3)
    fnames = data["fnames"]

    color_dict = {"right": "white", "left": "white", "object": "light-blue"}
    vis_dict = {}
    pred = data.search("v3d_c.").to_np()
    for v3d_key in pred.keys():
        node_id = v3d_key.split(".")[1]
        vis_dict[node_id] = {
            "v3d": pred[v3d_key],
            "f3d": faces[node_id],
            "vc": None,
            "name": node_id,
            "color": color_dict[node_id],
        }
    import common.viewer as viewer_utils

    meshes = viewer_utils.construct_viewer_meshes(
        vis_dict, draw_edges=False, flat_shading=False
    )
    num_frames = len(fnames)
    Rt = np.zeros((num_frames, 3, 4))
    Rt[:, :3, :3] = np.eye(3)
    Rt[:, 1:3, :3] *= -1.0

    im = Image.open(fnames[0])
    cols, rows = im.size

    images = [Image.open(im_p) for im_p in fnames]
    from common.viewer import ViewerData

    data = ViewerData(Rt, K, cols, rows, images)
    return meshes, data
