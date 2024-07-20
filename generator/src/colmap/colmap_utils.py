import os
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pycolmap
import torch
import trimesh
from hloc import (
    extract_features,
    match_features,
    pairs_from_retrieval,
    reconstruction,
    visualization,
)
from PIL import Image
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from tqdm import tqdm

import src.colmap.colmap_readmodel as read_model


def plot_2d_projection(pts_cano, denormalize_mat, o2w_all, intrinsic, seq_name):
    print("Projecting 3d points to 2d")
    pts_cano_homogeneous = np.hstack([pts_cano, np.ones((pts_cano.shape[0], 1))])
    pts_cano_denorm = np.dot(pts_cano_homogeneous, denormalize_mat.T)

    pts_w = np.array(
        [np.dot(pts_cano_denorm, o2w_all[i].T) for i in range(o2w_all.shape[0])]
    )
    pts_w = pts_w[:, :, :3] / pts_w[:, :, 3:]

    projected_pts = np.dot(pts_w, intrinsic.T)
    projected_2d = projected_pts[:, :, :2] / projected_pts[:, :, 2:]

    fnames = sorted(glob(f"./data/{seq_name}/images/*"))

    for idx in tqdm(range(len(fnames))):
        fname = fnames[idx]
        v2d = projected_2d[idx]

        out_p = fname.replace("/images/", f"/processed/colmap_2d/")

        # Load image
        img = Image.open(fname)

        # Get the 2D points for the current frame
        points = v2d[:300]
        points_a = points[:150]
        points_b = points[150:]

        os.makedirs(os.path.dirname(out_p), exist_ok=True)
        # Create scatter plot on the image
        plt.figure(figsize=(8, 8))

        plt.scatter(points_a[:, 0], points_a[:, 1], c="red", s=5)
        plt.scatter(points_b[:, 0], points_b[:, 1], c="blue", s=5)
        plt.imshow(img)
        plt.savefig(out_p)
        plt.close()

    out_p = os.path.join(os.path.dirname(out_p), "keypoints.npy")

    np.save(out_p, projected_2d)
    print("Saving keypoints to", out_p)
    print("----------------------------------------------")


def slerp_o2w(o2w_all, key_frames, num_frames):
    # Assertions to check the input dimensions
    assert o2w_all.ndim == 3, "o2w_all should be a 3D array"
    assert o2w_all.shape[1:] == (4, 4), "Each element of o2w_all should be a 4x4 matrix"
    assert (
        len(key_frames) == o2w_all.shape[0]
    ), "Number of key frames should match the first dimension of o2w_all"
    assert (
        isinstance(num_frames, int) and num_frames > 0
    ), "num_frames should be a positive integer"

    expected_frames = np.arange(num_frames)

    # handle edge cases when key_frames are outside of expected_frames
    # check first keyframe
    if not (key_frames[0] <= expected_frames[0]):
        key_frames = np.concatenate([[expected_frames[0]], key_frames])
        o2w_all = np.concatenate([o2w_all[:1], o2w_all])

    # check last keyframe
    if not (key_frames[-1] >= expected_frames[-1]):
        key_frames = np.concatenate([key_frames, [expected_frames[-1]]])
        o2w_all = np.concatenate([o2w_all, o2w_all[-1:]])

    rots = o2w_all[:, :3, :3]
    key_rots = R.from_matrix(rots)
    key_trans = o2w_all[:, :3, 3]
    slerp = Slerp(key_frames, key_rots)
    interp_rots = slerp(expected_frames).as_matrix()

    # Create an interpolation object for translations, interpolating each dimension separately
    interp_trans_x = np.interp(expected_frames, key_frames, key_trans[:, 0])
    interp_trans_y = np.interp(expected_frames, key_frames, key_trans[:, 1])
    interp_trans_z = np.interp(expected_frames, key_frames, key_trans[:, 2])
    interp_trans = np.vstack([interp_trans_x, interp_trans_y, interp_trans_z]).T

    # Create the interpolated o2w_all matrix
    interp_o2w_all = np.zeros((num_frames, 4, 4))
    interp_o2w_all[:, :3, :3] = interp_rots
    interp_o2w_all[:, :3, 3] = interp_trans
    interp_o2w_all[:, 3, 3] = 1

    return interp_o2w_all


def load_colmap_data(realdir):
    camerasfile = os.path.join(realdir, "cameras.bin")
    camdata = read_model.read_cameras_binary(camerasfile)

    # cam = camdata[camdata.keys()[0]]
    list_of_keys = list(camdata.keys())
    cam = camdata[list_of_keys[0]]
    print("Cameras", len(cam))

    h, w, f = cam.height, cam.width, cam.params[0]
    # w, h, f = factor * w, factor * h, factor * f
    hwf = np.array([h, w, f]).reshape([3, 1])

    imagesfile = os.path.join(realdir, "images.bin")
    imdata = read_model.read_images_binary(imagesfile)

    w2c_mats = []
    bottom = np.array([0, 0, 0, 1.0]).reshape([1, 4])

    names = [imdata[k].name for k in imdata]
    print("Images #", len(names))
    perm = np.argsort(names)
    for k in imdata:
        im = imdata[k]
        R = im.qvec2rotmat()
        t = im.tvec.reshape([3, 1])
        m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        w2c_mats.append(m)

    w2c_mats = np.stack(w2c_mats, 0)
    c2w_mats = np.linalg.inv(w2c_mats)

    poses = c2w_mats[:, :3, :4].transpose([1, 2, 0])
    poses = np.concatenate(
        [poses, np.tile(hwf[..., np.newaxis], [1, 1, poses.shape[-1]])], 1
    )

    points3dfile = os.path.join(realdir, "points3D.bin")
    pts3d = read_model.read_points3d_binary(points3dfile)

    # must switch to [-u, r, -t] from [r, -u, t], NOT [r, u, -t]
    poses = np.concatenate(
        [
            poses[:, 1:2, :],
            poses[:, 0:1, :],
            -poses[:, 2:3, :],
            poses[:, 3:4, :],
            poses[:, 4:5, :],
        ],
        1,
    )

    return poses, pts3d, perm


def export_colmap_results(basedir, poses, pts3d, perm):
    # point cloud export
    pts = np.stack([pts3d[k].xyz for k in pts3d], axis=0)
    pcd = trimesh.PointCloud(pts)
    pcd.export(os.path.join(basedir, "sparse_points.ply"))

    # Adjust poses dimensions and apply permutation
    poses = np.moveaxis(poses, -1, 0)
    poses = poses[perm]

    # Save the adjusted poses as a .npy file
    np.save(os.path.join(basedir, "poses.npy"), poses)


def format_poses(seq_name):
    import torch

    poses_hwf = np.load(f"./data/{seq_name}/processed/colmap/poses.npy")

    poses_hwf = torch.FloatTensor(poses_hwf)

    poses_raw = poses_hwf[:, :, :4]
    hwf = poses_hwf[:, :, 4]

    h, w, f = hwf[0]

    intrinsic = np.diag([f, f, 1.0, 1.0]).astype(np.float32)
    intrinsic[0, 2] = (w - 1) * 0.5
    intrinsic[1, 2] = (h - 1) * 0.5
    intrinsic = intrinsic[:3, :3]

    num_frames = poses_raw.shape[0]

    convert_mat = torch.zeros([4, 4])
    convert_mat[0, 1] = 1.0
    convert_mat[1, 0] = 1.0
    convert_mat[2, 2] = -1.0
    convert_mat[3, 3] = 1.0

    # w2o mat to opencv format
    w2o_all = torch.eye(4)[None, :, :].repeat(num_frames, 1, 1)
    w2o_all[:, :3] = poses_raw
    w2o_all = torch.bmm(w2o_all, convert_mat[None, :, :].repeat(num_frames, 1, 1))
    o2w_all = w2o_all.inverse().numpy()

    # SLERP
    with open(
        f"./data/{seq_name}/processed/colmap/sfm_superpoint+superglue/converged_frames.txt",
        "r",
    ) as file:
        lines = file.readlines()
        integer_lines = [int(line.strip()) for line in lines if line.strip()]
        valid_frames = np.array(integer_lines)
    assert len(valid_frames) == len(o2w_all)
    assert valid_frames.min() > 0  ## assume 1-based in valid frames
    key_frames = valid_frames - 1
    num_frames = len(glob(f"./data/{seq_name}/images/*"))

    sort_idx = np.argsort(key_frames)
    key_frames = key_frames[sort_idx]
    o2w_all = o2w_all[sort_idx]

    interp_o2w_all = slerp_o2w(o2w_all, key_frames, num_frames)
    o2w_all = interp_o2w_all

    # colmap verts (not normalized)
    mesh = trimesh.load(
        f"./data/{seq_name}/processed/colmap/sparse_points_trim.ply", process=False
    )
    vertices = np.array(mesh.vertices)

    # construct normalization matrix
    bbox_max = np.max(vertices, axis=0)
    bbox_min = np.min(vertices, axis=0)

    # bbox center
    center = (bbox_max + bbox_min) * 0.5

    radius = np.linalg.norm(vertices - center, ord=2, axis=-1).max()
    # radius = 1.0
    # from scaled & center to unscaled and not centered (original in COLMAP)
    denormalize_mat = np.diag([radius, radius, radius, 1.0]).astype(np.float32)
    denormalize_mat[:3, 3] = center

    # center and scale COLMAP point cloud
    normalize_mat = np.linalg.inv(denormalize_mat)

    # normalize colmap points
    pts_cano = np.ones((vertices.shape[0], 4))
    pts_cano[:, :3] = vertices

    pts_cano = (normalize_mat @ pts_cano.T).T
    pts_cano = pts_cano[:, :3] / pts_cano[:, 3:]
    pc_p = f"./data/{seq_name}/processed/colmap/sparse_points_normalized.obj"
    mesh.vertices = pts_cano
    mesh.export(pc_p)

    norm_mat_p = f"./data/{seq_name}/processed/colmap/normalization_mat.npy"
    intrinsic_p = f"./data/{seq_name}/processed/colmap/intrinsic.npy"
    pose_p = f"./data/{seq_name}/processed/colmap/o2w.npy"

    np.save(norm_mat_p, normalize_mat)
    np.save(intrinsic_p, intrinsic)
    np.save(pose_p, o2w_all)

    print("Saving normalized point cloud to", pc_p)
    print("Saving normalization matrix to", norm_mat_p)
    print("Saving intrinsic matrix to", intrinsic_p)
    print("Saving pose matrix to", pose_p)


def colmap_pose_est(seq_name, num_keypoints):
    image_path = f"./data/{seq_name}/processed/images_object"
    output_path = f"./data/{seq_name}/processed/colmap"
    camera_path = f"./data/{seq_name}/processed/colmap/sfm_superpoint+superglue"

    images = Path(image_path)
    outputs = Path(output_path)
    num_images = len(glob(f"{image_path}/*"))
    assert (
        num_keypoints <= num_images
    ), f"{num_keypoints} should be less or equal to {num_images}"

    sfm_pairs = outputs / "pairs-netvlad.txt"
    sfm_dir = outputs / "sfm_superpoint+superglue"
    features = outputs / "features.h5"
    retrieval_conf = extract_features.confs["netvlad"]
    feature_conf = extract_features.confs["superpoint_aachen"]
    matcher_conf = match_features.confs["superglue"]
    references = [p.relative_to(images).as_posix() for p in (images).iterdir()]

    retrieval_path = extract_features.main(
        retrieval_conf, images, image_list=references, feature_path=features
    )
    pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=num_keypoints)

    feature_path = extract_features.main(feature_conf, images, outputs)
    match_path = match_features.main(
        matcher_conf, sfm_pairs, feature_conf["output"], outputs
    )

    model = reconstruction.main(
        sfm_dir,
        images,
        sfm_pairs,
        feature_path,
        match_path,
        camera_mode=pycolmap.CameraMode.PER_FOLDER,
    )

    visualization.visualize_sfm_2d(model, images, color_by="visibility", n=5)

    images_ret = read_model.read_images_binary(sfm_dir / "images.bin")
    file_name = "converged_frames.txt"
    frames_conv = images_ret.keys()
    with open(os.path.join(sfm_dir, file_name), "w") as file:
        # Write each word to a separate line
        for frame_number in frames_conv:
            file.write(str(frame_number) + "\n")

    poses, pts3d, perm = load_colmap_data(camera_path)
    export_colmap_results(outputs, poses, pts3d, perm)


def trim_point_cloud(sp_p, percentile=80, scale_factor=1.5):
    # this function trim point cloud by first computing its median
    # then find its 80 percentile for a threshold and pad with 1.5*thres to create a boundary
    # then we use the boundary to decide which points we include

    out_p = sp_p.replace("/sparse_points.ply", "/sparse_points_trim.ply")

    sp = trimesh.load(sp_p, process=False)
    verts = np.array(sp.vertices)
    center = np.median(verts, axis=0)
    dist = np.linalg.norm(verts - center[None, :], axis=1)

    thres = np.percentile(dist, percentile)
    thres = scale_factor * thres

    verts_trim = verts[dist < thres]

    pc = trimesh.Trimesh(vertices=verts_trim)
    pc.export(out_p)

    print("Saved trimmed point cloud to", out_p)
    return pc


def slerp_o2w(o2w_all, key_frames, num_frames):
    # Assertions to check the input dimensions
    assert o2w_all.ndim == 3, "o2w_all should be a 3D array"
    assert o2w_all.shape[1:] == (4, 4), "Each element of o2w_all should be a 4x4 matrix"
    assert (
        len(key_frames) == o2w_all.shape[0]
    ), "Number of key frames should match the first dimension of o2w_all"
    assert (
        isinstance(num_frames, int) and num_frames > 0
    ), "num_frames should be a positive integer"

    expected_frames = np.arange(num_frames)

    start_time = key_frames[0]
    end_time = key_frames[-1]

    start_o2w = o2w_all[:1]
    end_o2w = o2w_all[-1:]

    start_time_query = expected_frames[0]
    end_time_query = expected_frames[-1]

    if start_time_query < start_time:
        o2w_all = np.concatenate((start_o2w, o2w_all), axis=0)
        key_frames = np.concatenate(([start_time_query], key_frames), axis=0)

    if end_time < end_time_query:
        o2w_all = np.concatenate((o2w_all, end_o2w), axis=0)
        key_frames = np.concatenate((key_frames, [end_time_query]), axis=0)

    # interpolate rotation
    rots = o2w_all[:, :3, :3]
    key_rots = R.from_matrix(rots)
    slerp = Slerp(key_frames, key_rots)
    interp_rots = slerp(expected_frames).as_matrix()

    # interpolate translation
    key_trans = o2w_all[:, :3, 3]

    # Create an interpolation object for translations, interpolating each dimension separately
    interp_trans_x = np.interp(expected_frames, key_frames, key_trans[:, 0])
    interp_trans_y = np.interp(expected_frames, key_frames, key_trans[:, 1])
    interp_trans_z = np.interp(expected_frames, key_frames, key_trans[:, 2])
    interp_trans = np.vstack([interp_trans_x, interp_trans_y, interp_trans_z]).T

    # Create the interpolated o2w_all matrix
    interp_o2w_all = np.zeros((num_frames, 4, 4))
    interp_o2w_all[:, :3, :3] = interp_rots
    interp_o2w_all[:, :3, 3] = interp_trans
    interp_o2w_all[:, 3, 3] = 1

    return interp_o2w_all


def read_hwf_poses(hwf_p):
    poses_hwf = np.load(hwf_p)
    poses_hwf = torch.FloatTensor(poses_hwf)

    poses_raw = poses_hwf[:, :, :4]
    hwf = poses_hwf[:, :, 4]

    h, w, f = hwf[0]

    intrinsic = np.diag([f, f, 1.0, 1.0]).astype(np.float32)
    intrinsic[0, 2] = (w - 1) * 0.5
    intrinsic[1, 2] = (h - 1) * 0.5
    intrinsic = intrinsic[:3, :3]

    num_frames = poses_raw.shape[0]

    convert_mat = torch.zeros([4, 4])
    convert_mat[0, 1] = 1.0
    convert_mat[1, 0] = 1.0
    convert_mat[2, 2] = -1.0
    convert_mat[3, 3] = 1.0

    # w2o mat to opencv format
    w2o_all = torch.eye(4)[None, :, :].repeat(num_frames, 1, 1)
    w2o_all[:, :3] = poses_raw
    w2o_all = torch.bmm(w2o_all, convert_mat[None, :, :].repeat(num_frames, 1, 1))
    o2w_all = w2o_all.inverse().numpy()

    return intrinsic, o2w_all


def canonical_normalization(vertices):
    # construct normalization matrix
    bbox_max = np.max(vertices, axis=0)
    bbox_min = np.min(vertices, axis=0)

    # bbox center
    center = (bbox_max + bbox_min) * 0.5

    radius = np.linalg.norm(vertices - center, ord=2, axis=-1).max()
    # radius = 1.0
    # from scaled & center to unscaled and not centered (original in COLMAP)
    denormalize_mat = np.diag([radius, radius, radius, 1.0]).astype(np.float32)
    denormalize_mat[:3, 3] = center

    # center and scale COLMAP point cloud
    normalize_mat = np.linalg.inv(denormalize_mat)

    # normalize colmap points
    pts_cano = np.ones((vertices.shape[0], 4))
    pts_cano[:, :3] = vertices

    pts_cano = (normalize_mat @ pts_cano.T).T
    pts_cano = pts_cano[:, :3] / pts_cano[:, 3:]

    return pts_cano, denormalize_mat, normalize_mat


def read_valid_frames(seq_name):
    # Reading the key frames
    with open(
        f"./data/{seq_name}/processed/colmap/sfm_superpoint+superglue/converged_frames.txt",
        "r",
    ) as file:
        lines = file.readlines()
        integer_lines = [int(line.strip()) for line in lines if line.strip()]
        valid_frames = np.array(integer_lines)

    assert valid_frames.min() > 0  # assume 1-based in valid frames
    return valid_frames


def validate_colmap(seq_name, no_vis):
    # read object poses and intrinsics
    hwf_p = f"./data/{seq_name}/processed/colmap/poses.npy"
    intrinsic, o2w_all = read_hwf_poses(hwf_p)

    # check converged frames in colmap
    valid_frames = read_valid_frames(seq_name)
    key_frames = valid_frames - 1
    assert len(valid_frames) == len(o2w_all)

    # SLERP to interpolate failed poses
    sort_idx = np.argsort(key_frames)
    key_frames = key_frames[sort_idx]
    o2w_all = o2w_all[sort_idx]
    num_frames = len(glob(f"./data/{seq_name}/images/*"))
    interp_o2w_all = slerp_o2w(o2w_all, key_frames, num_frames)

    all_frames = np.arange(num_frames)
    missing_frames = [frame for frame in all_frames if frame not in key_frames]
    print("Missing frames", missing_frames)
    print("Number of missing frames", len(missing_frames))

    # remove outlier SfM points
    pc_trim = trim_point_cloud(
        f"./data/{seq_name}/processed/colmap/sparse_points.ply",
        percentile=80,
        scale_factor=1.5,
    )

    # zero-center and normalize the canonical space
    pts_cano, denormalize_mat, normalize_mat = canonical_normalization(
        np.array(pc_trim.vertices)
    )

    # save processed results
    pc_p = f"./data/{seq_name}/processed/colmap/sparse_points_normalized.obj"
    pc_trim.vertices = pts_cano
    pc_trim.export(pc_p)
    norm_mat_p = f"./data/{seq_name}/processed/colmap/normalization_mat.npy"
    intrinsic_p = f"./data/{seq_name}/processed/colmap/intrinsic.npy"
    pose_p = f"./data/{seq_name}/processed/colmap/o2w.npy"
    np.save(norm_mat_p, normalize_mat)
    np.save(intrinsic_p, intrinsic)
    np.save(pose_p, interp_o2w_all)

    # 2d projection
    if not no_vis:
        plot_2d_projection(
            pts_cano, denormalize_mat, interp_o2w_all, intrinsic, seq_name
        )
