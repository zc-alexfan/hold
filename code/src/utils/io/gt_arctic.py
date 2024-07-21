import os.path as op
from glob import glob

import numpy as np
import torch
import trimesh
from common.body_models import build_mano_aa
from common.transforms import project2d_batch, rigid_tf_torch_batch
from PIL import Image

# from src_data.preprocessing_utils import tf.cv2gl_mano
import common.transforms as tf

# from src_data.smplx import MANO
from common.xdict import xdict
from src.utils.eval_modules import compute_bounding_box_centers
from src.utils.const import SEGM_IDS
import json
import os.path as op


def load_data(full_seq_name):
    out = {}
    _, sid, obj_name = full_seq_name.split("_")[:3]
    seq_name = full_seq_name.split(f"_{sid}_")[1]
    view_idx = int(seq_name.split("_")[-1])
    seq_name = "_".join(seq_name.split("_")[:-1])

    with open("./arctic_data/arctic/meta/misc.json", "r") as f:
        misc = json.load(f)

    data = np.load(
        f"./arctic_data/processed/{sid}/{seq_name}.npy", allow_pickle=True
    ).item()
    fnames = sorted(glob(f"./arctic_data/arctic/images/{sid}/{seq_name}/{view_idx}/*"))

    v3d_r = torch.FloatTensor(data["cam_coord"]["verts.right"][:, view_idx])
    v3d_l = torch.FloatTensor(data["cam_coord"]["verts.left"][:, view_idx])
    v3d_o = torch.FloatTensor(data["cam_coord"]["verts.object"][:, view_idx])
    j3d_r = torch.FloatTensor(data["cam_coord"]["joints.right"][:, view_idx])
    j3d_l = torch.FloatTensor(data["cam_coord"]["joints.left"][:, view_idx])
    K = torch.FloatTensor(np.array(misc[sid]["intris_mat"]))[view_idx - 1]
    ioi_offset = misc[sid]["ioi_offset"]
    faces_r = data["faces"]["right"]
    faces_l = data["faces"]["left"]
    faces_o = data["faces"]["object"]

    # Select filenames from directory
    with open(f"./data/{full_seq_name}/build/corres.txt", "r") as f:
        selected_fnames = sorted([line.strip() for line in f])
    assert len(selected_fnames) > 0

    # Get selected file IDs
    selected_fids = np.array(
        [int(op.basename(fname).split(".")[0]) for fname in selected_fnames]
    )
    selected_fids = selected_fids - ioi_offset
    assert len(selected_fids) > 0

    # Select ground truth data based on selected file IDs
    v3d_r = v3d_r[selected_fids]
    v3d_l = v3d_l[selected_fids]
    v3d_o = v3d_o[selected_fids]
    j3d_r = j3d_r[selected_fids]
    j3d_l = j3d_l[selected_fids]

    roots_o = compute_bounding_box_centers(v3d_o.numpy())
    v3d_o_ra = v3d_o.clone().numpy() - roots_o[:, None, :]

    root_right = j3d_r[:, :1].clone()
    root_left = j3d_l[:, :1].clone()
    out["v3d_right.object"] = v3d_o.clone() - root_right
    out["v3d_left.object"] = v3d_o.clone() - root_left

    j3d_r_ra = j3d_r.clone() - j3d_r[:, :1]
    j3d_l_ra = j3d_l.clone() - j3d_l[:, :1]

    is_valid = torch.ones(len(v3d_r)).float()

    v3d_o_cam = v3d_o.clone()
    fnames = [
        op.join(f"./arctic_data/arctic/images/{sid}/{seq_name}/{view_idx}/", basename)
        for basename in selected_fnames
    ]

    # object relative
    z_depth = 3.0  # 3 meters in front of camera
    camera_offset = torch.FloatTensor([0.0, 0.0, z_depth])
    v3d_r_center = v3d_r.clone() - roots_o[:, None, :] + camera_offset
    v3d_l_center = v3d_l.clone() - roots_o[:, None, :] + camera_offset

    v3d_o_center = v3d_o.clone() - roots_o[:, None, :] + camera_offset

    out["fnames"] = fnames
    out["v3d_object.right"] = v3d_r_center.detach().numpy()
    out["v3d_object.left"] = v3d_l_center.detach().numpy()
    out["v3d_object.object"] = v3d_o_center.detach().numpy()
    out["v3d_c.right"] = v3d_r.detach().numpy()
    out["v3d_c.left"] = v3d_l.detach().numpy()
    out["v3d_c.object"] = v3d_o_cam.detach().numpy()
    out["v3d_ra.object"] = v3d_o_ra
    out["j3d_c.right"] = j3d_r.detach().numpy()
    out["j3d_c.left"] = j3d_l.detach().numpy()
    out["j3d_ra.right"] = j3d_r_ra.detach().numpy()
    out["j3d_ra.left"] = j3d_l_ra.detach().numpy()
    out["faces_o"] = np.array(faces_o)
    out["faces_r"] = np.array(faces_r)
    out["faces_l"] = np.array(faces_l)
    out["K"] = K.numpy()
    out["is_valid"] = is_valid

    # rh
    out["root"] = out["j3d_c.right"][:, :1]
    out = xdict(out).to_torch()
    return out


def load_viewer_data(args):
    full_seq_name = args.seq_name
    data = load_data(full_seq_name)

    # object center at origin
    v3d_r_c = data["v3d_object.right"].numpy()
    v3d_l_c = data["v3d_object.left"].numpy()
    v3d_o_c = data["v3d_object.object"].numpy()

    faces_o = data["faces_o"].numpy()
    faces_r = data["faces_r"].numpy()
    faces_l = data["faces_l"].numpy()
    K = data["K"].numpy().reshape(3, 3)
    fnames = data["fnames"]
    from common.body_models import seal_mano_mesh_np

    v3d_r_c, faces_r = seal_mano_mesh_np(v3d_r_c, faces_r, is_rhand=True)
    v3d_l_c, faces_l = seal_mano_mesh_np(v3d_l_c, faces_l, is_rhand=False)

    vis_dict = {}
    vis_dict["left-gt"] = {
        "v3d": v3d_l_c,
        "f3d": faces_l,
        "vc": None,
        "name": "left-gt",
        "color": "cyan",
        "flat_shading": True,
    }

    vis_dict["right-gt"] = {
        "v3d": v3d_r_c,
        "f3d": faces_r,
        "vc": None,
        "name": "right-gt",
        "color": "cyan",
        "flat_shading": True,
    }

    vis_dict["obj-gt"] = {
        "v3d": v3d_o_c,
        "f3d": faces_o,
        "vc": None,
        "name": "object-gt",
        "color": "red",
        "flat_shading": False,
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
