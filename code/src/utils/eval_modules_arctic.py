import numpy as np
import torch
from scipy.spatial import cKDTree as KDTree
from pytorch3d.loss import chamfer_distance
import sys

sys.path = [".."] + sys.path
from tqdm import tqdm
import common.metrics as metrics


def compute_bounding_box_centers(vertices):
    """
    Compute the centers of the tight bounding box for a moving point cloud.

    Parameters:
    - vertices: A numpy array of shape (frames, num_verts, 3) representing the vertices of the object over time.

    Returns:
    - A numpy array of shape (frames, 3) where each row represents the center of the bounding box for each frame.
    """

    if isinstance(vertices, list):
        bbox_centers = []
        for verts in vertices:
            assert verts.shape[1] == 3
            bmin = np.min(verts, axis=0)
            bmax = np.max(verts, axis=0)
            bbox_center = (bmin + bmax) / 2
            bbox_centers.append(bbox_center)
        bbox_centers = np.stack(bbox_centers, axis=0)
    else:
        bbox_min = np.min(vertices, axis=1)
        bbox_max = np.max(vertices, axis=1)
        bbox_centers = (bbox_min + bbox_max) / 2
    return bbox_centers


def convert_to_tensors(data):
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            if value.dtype == np.uint32:
                data[key] = torch.from_numpy(value.astype(np.int64))
            elif value.dtype.kind == "f":  # Check if it's a floating-point type
                data[key] = torch.from_numpy(
                    value.astype(np.float32)
                )  # Convert to Float32
            else:
                data[key] = torch.from_numpy(value)
    return data


def eval_icp_first_frame_arctic(data_pred, data_gt, metric_dict):
    """
    Warning:
    In the CVPR HOLD paper, we used HO3D and followed the evaluation protocol from IHOR.
    The protocol uses squared chamfer distance (dist**2) version for evaluation.
    However, this metric is not in the metric space. Therefore, for ARCTIC here, we use squared-root CD (dist) for evaluation.
    """
    faces = data_pred["faces"]["object"]
    from src.utils.icp import compute_icp_metrics
    from open3d.geometry import TriangleMesh
    from open3d.utility import Vector3dVector, Vector3iVector

    v3d_o_ra = Vector3dVector(data_pred["v3d_ra.object"][0].numpy())
    faces_o = Vector3iVector(faces.cpu().numpy())
    v3d_o_ra_gt = Vector3dVector(data_gt["v3d_ra.object"][0].numpy())
    faces_o_gt = Vector3iVector(data_gt["faces_o"].numpy())
    source_mesh = TriangleMesh(v3d_o_ra, faces_o)
    target_mesh = TriangleMesh(v3d_o_ra_gt, faces_o_gt)
    best_cd, best_f5, best_f10 = compute_icp_metrics(
        # target_mesh, source_mesh, num_iters=60, is_sqrt=True
        target_mesh,
        source_mesh,
        num_iters=600,
        is_sqrt=True,
    )
    metric_dict["cd_icp"] = best_cd
    metric_dict["f5_icp"] = best_f5 * 100.0
    metric_dict["f10_icp"] = best_f10 * 100.0
    return metric_dict


def eval_icp_every_frame(data_pred, data_gt, metric_dict):
    from src.utils.icp import compute_icp_metrics
    from open3d.geometry import TriangleMesh
    from open3d.utility import Vector3dVector, Vector3iVector

    is_valid = data_gt["is_valid"]

    num_frames = len(data_pred["v3d_o_ra"])
    cd_list = []
    f5_list = []
    f10_list = []
    assert num_frames == len(data_gt["v3d_o_ra"])
    assert num_frames == len(is_valid)

    for idx in tqdm(range(num_frames)):
        if is_valid[idx]:
            v3d_o_ra = Vector3dVector(data_pred["v3d_o_ra"][idx].numpy())
            faces_o = Vector3iVector(data_pred["faces_o"][idx].numpy())
            v3d_o_ra_gt = Vector3dVector(data_gt["v3d_o_ra"][idx].numpy())
            faces_o_gt = Vector3iVector(data_gt["faces_o"].numpy())
            source_mesh = TriangleMesh(v3d_o_ra, faces_o)
            target_mesh = TriangleMesh(v3d_o_ra_gt, faces_o_gt)

            cd, f5, f10 = compute_icp_metrics(
                target_mesh, source_mesh, num_iters=10, no_tqdm=True
            )
        else:
            cd = float("nan")
            f5 = float("nan")
            f10 = float("nan")
        cd_list.append(cd)
        f5_list.append(f5)
        f10_list.append(f10)

    cd_list = np.array(cd_list)
    f5_list = np.array(f5_list)
    f10_list = np.array(f10_list)
    mean_cd = np.nanmean(cd_list)
    mean_f5 = np.nanmean(f5_list)
    mean_f10 = np.nanmean(f10_list)

    metric_dict["cd_icp"] = mean_cd
    metric_dict["f5_icp"] = mean_f5 * 100.0
    metric_dict["f10_icp"] = mean_f10 * 100.0
    return metric_dict


def eval_mrrpe_ho_right(data_pred, data_gt, metric_dict):
    j3d_h_c_pred = data_pred["j3d_c.right"]
    root_o_pred = data_pred["root.object"]

    j3d_h_c_gt = data_gt["j3d_c.right"]
    root_o_gt = data_gt["root.object"]
    is_valid = data_gt["is_valid"]

    root_h_gt = j3d_h_c_gt[:, 0]
    root_h_pred = j3d_h_c_pred[:, 0]
    mrrpe_ho = (
        metrics.compute_mrrpe(
            root_h_gt,
            root_o_gt,
            root_h_pred,
            root_o_pred,
            is_valid,
        )
        * 1000
    )
    not_valid = (1 - is_valid).numpy().astype(bool)
    mrrpe_ho[not_valid] = np.nan

    metric_dict["mrrpe_ho"] = mrrpe_ho
    return metric_dict


def calculate_chamfer_f_scores(vertices_source, vertices_target, is_sqrt=False):
    vertices_source = vertices_source * 100
    vertices_target = vertices_target * 100

    gen_points_kd_tree = KDTree(vertices_source)
    one_distances, one_vertex_ids = gen_points_kd_tree.query(vertices_target)
    if is_sqrt:
        gt_to_gen_chamfer = np.mean(one_distances)
    else:
        gt_to_gen_chamfer = np.mean(np.square(one_distances))
    # other direction
    gt_points_kd_tree = KDTree(vertices_target)
    two_distances, two_vertex_ids = gt_points_kd_tree.query(vertices_source)
    if is_sqrt:
        gen_to_gt_chamfer = np.mean(two_distances)
    else:
        gen_to_gt_chamfer = np.mean(np.square(two_distances))
    chamfer_obj = gt_to_gen_chamfer + gen_to_gt_chamfer
    threshold = 0.5  # 5 mm
    precision_1 = np.mean(one_distances < threshold).astype(np.float32)
    precision_2 = np.mean(two_distances < threshold).astype(np.float32)
    fscore_obj_5 = 2 * precision_1 * precision_2 / (precision_1 + precision_2 + 1e-7)

    threshold = 1.0  # 10 mm
    precision_1 = np.mean(one_distances < threshold).astype(np.float32)
    precision_2 = np.mean(two_distances < threshold).astype(np.float32)
    fscore_obj_10 = 2 * precision_1 * precision_2 / (precision_1 + precision_2 + 1e-7)
    return chamfer_obj, fscore_obj_5, fscore_obj_10


def compute_iou_per_frame(insta_map_pred, insta_map_gt):
    classes = [0, 100, 200]
    ious = []

    for frame_idx in range(insta_map_pred.shape[0]):
        iou_per_class = []
        for cls in classes:
            pred_mask = insta_map_pred[frame_idx] == cls
            gt_mask = insta_map_gt[frame_idx] == cls
            intersection = np.logical_and(pred_mask, gt_mask).sum()
            union = np.logical_or(pred_mask, gt_mask).sum()
            iou = intersection / union if union != 0 else 0
            iou_per_class.append(iou)
        ious.append(
            np.mean(iou_per_class)
        )  # Assuming you want the mean IoU for all classes per frame

    return np.array(ious)


def eval_cd_ra(data_pred, data_gt, metric_dict):
    v3d_o_c_pred_ra = data_pred["v3d_o_c_ra"]
    v3d_o_c_gt_ra = data_gt["v3d_o_c_ra"]

    torch.manual_seed(1)
    rand_gt_idx = torch.randperm(v3d_o_c_gt_ra.shape[1])[:3000]
    rand_pred_idx = torch.randperm(v3d_o_c_pred_ra.shape[1])[:3000]

    cd_ra = (
        chamfer_distance(
            v3d_o_c_pred_ra[:, rand_pred_idx],
            v3d_o_c_gt_ra[:, rand_gt_idx],
            batch_reduction=None,
        )[0]
        * 1000
    )

    metric_dict["cd_ra"] = cd_ra.numpy()  # Assuming cd_ra is a 1-element tensor
    return metric_dict


def eval_cd_f(data_pred, data_gt, metric_dict):
    v3d_o_c_pred_ra = data_pred["v3d_o_c"]
    v3d_o_c_gt_ra = data_gt["v3d_o_c"]
    is_valid = data_gt["is_valid"]

    torch.manual_seed(1)
    rand_gt_idx = torch.randperm(v3d_o_c_gt_ra.shape[1])[:3000]
    rand_pred_idx = torch.randperm(v3d_o_c_pred_ra.shape[1])[:3000]

    cd_list = []
    f5_list = []
    f10_list = []
    for idx in range(v3d_o_c_pred_ra.shape[0]):
        cd_error, f5, f10 = calculate_chamfer_f_scores(
            v3d_o_c_pred_ra[idx, rand_pred_idx].numpy(),
            v3d_o_c_gt_ra[idx, rand_gt_idx].numpy(),
        )
        cd_list.append(cd_error)
        f5_list.append(f5)
        f10_list.append(f10)
    cd_list = np.array(cd_list)
    f5_list = np.array(f5_list)
    f10_list = np.array(f10_list)

    not_valid = (1 - is_valid).numpy().astype(bool)
    cd_list[not_valid] = np.nan
    f5_list[not_valid] = np.nan
    f10_list[not_valid] = np.nan

    # metric_dict["cd_rh"] = cd_ra.numpy()  # Assuming cd_ra is a 1-element tensor
    metric_dict["cd"] = cd_list
    metric_dict["f5"] = f5_list * 100.0
    metric_dict["f10"] = f10_list * 100.0
    return metric_dict


def eval_cd_f_right_arctic(data_pred, data_gt, metric_dict):
    return eval_cd_f_arctic(data_pred, data_gt, metric_dict, "right")


def eval_cd_f_left_arctic(data_pred, data_gt, metric_dict):
    return eval_cd_f_arctic(data_pred, data_gt, metric_dict, "left")


def eval_cd_f_hand_arctic(data_pred, data_gt, metric_dict):
    eval_cd_f_left_arctic(data_pred, data_gt, metric_dict)
    eval_cd_f_right_arctic(data_pred, data_gt, metric_dict)
    cd_hand = np.stack([metric_dict["cd_r"], metric_dict["cd_l"]], axis=1)
    metric_dict["cd_h"] = cd_hand.mean(axis=1)
    return metric_dict


def eval_cd_f_arctic(data_pred, data_gt, metric_dict, flag):
    v3d_o_c_pred_ra = data_pred[f"v3d_{flag}.object"]
    v3d_o_c_gt_ra = data_gt[f"v3d_{flag}.object"]
    is_valid = data_gt["is_valid"]

    torch.manual_seed(1)

    cd_list = []
    f5_list = []
    f10_list = []
    for idx in range(len(v3d_o_c_pred_ra)):
        v3d_pred = v3d_o_c_pred_ra[idx]
        v3d_gt = v3d_o_c_gt_ra[idx]

        if torch.isnan(v3d_pred.mean()):
            cd_error = float("nan")
            f5 = float("nan")
            f10 = float("nan")
        else:
            rand_pred_idx = torch.randperm(v3d_pred.shape[0])[:3000]
            rand_gt_idx = torch.randperm(v3d_gt.shape[0])[:3000]
            cd_error, f5, f10 = calculate_chamfer_f_scores(
                v3d_pred[rand_pred_idx].numpy(),
                v3d_gt[rand_gt_idx].numpy(),
                is_sqrt=True,
            )
        cd_list.append(cd_error)
        f5_list.append(f5)
        f10_list.append(f10)
    cd_list = np.array(cd_list)
    f5_list = np.array(f5_list)
    f10_list = np.array(f10_list)

    not_valid = (1 - is_valid).numpy().astype(bool)
    cd_list[not_valid] = np.nan
    f5_list[not_valid] = np.nan
    f10_list[not_valid] = np.nan

    # metric_dict["cd_rh"] = cd_ra.numpy()  # Assuming cd_ra is a 1-element tensor
    indicator = "r" if flag == "right" else "l"
    metric_dict[f"cd_{indicator}"] = cd_list
    # metric_dict[f"f5_{indicator}"] = f5_list * 100.0
    # metric_dict[f"f10_{indicator}"] = f10_list * 100.0
    return metric_dict


def eval_cd_f_ra(data_pred, data_gt, metric_dict):
    v3d_o_c_pred_ra = data_pred["v3d_ra.object"]
    v3d_o_c_gt_ra = data_gt["v3d_ra.object"]
    is_valid = data_gt["is_valid"]

    torch.manual_seed(1)
    # rand_gt_idx = torch.randperm(v3d_o_c_gt_ra.shape[1])[:3000]
    # rand_pred_idx = torch.randperm(v3d_o_c_pred_ra.shape[1])[:3000]

    cd_list = []
    f5_list = []
    f10_list = []
    for idx in range(len(v3d_o_c_pred_ra)):
        v3d_pred = v3d_o_c_pred_ra[idx]

        if torch.isnan(v3d_pred.mean()):
            cd_error = float("nan")
            f5 = float("nan")
            f10 = float("nan")
        else:
            v3d_gt = v3d_o_c_gt_ra[idx]

            num_pts = min(3000, v3d_pred.shape[0])
            rand_pred_idx = torch.randperm(v3d_pred.shape[0])[:num_pts]
            rand_gt_idx = torch.randperm(v3d_gt.shape[0])[:3000]
            cd_error, f5, f10 = calculate_chamfer_f_scores(
                v3d_pred[rand_pred_idx].numpy(), v3d_gt[rand_gt_idx].numpy()
            )
        cd_list.append(cd_error)
        f5_list.append(f5)
        f10_list.append(f10)
    cd_list = np.array(cd_list)
    f5_list = np.array(f5_list)
    f10_list = np.array(f10_list)

    not_valid = (1 - is_valid).numpy().astype(bool)
    cd_list[not_valid] = np.nan
    f5_list[not_valid] = np.nan
    f10_list[not_valid] = np.nan

    # metric_dict["cd_rh"] = cd_ra.numpy()  # Assuming cd_ra is a 1-element tensor
    metric_dict["cd_ra"] = cd_list
    metric_dict["f5_ra"] = f5_list * 100.0
    metric_dict["f10_ra"] = f10_list * 100.0
    return metric_dict


def eval_mpjpe_right(data_pred, data_gt, metric_dict):
    return eval_mpjpe(data_pred, data_gt, metric_dict, "right")


def eval_mpjpe_left(data_pred, data_gt, metric_dict):
    return eval_mpjpe(data_pred, data_gt, metric_dict, "left")


def eval_mpjpe_hand(data_pred, data_gt, metric_dict):
    eval_mpjpe(data_pred, data_gt, metric_dict, "left")
    eval_mpjpe(data_pred, data_gt, metric_dict, "right")
    mpjpe_hand = np.stack(
        [metric_dict["mpjpe_ra_l"], metric_dict["mpjpe_ra_r"]], axis=1
    )
    metric_dict["mpjpe_ra_h"] = mpjpe_hand.mean(axis=1)
    return metric_dict


def eval_mpjpe(data_pred, data_gt, metric_dict, flag):
    j3d_h_c_pred_ra = data_pred[f"j3d_ra.{flag}"]
    j3d_h_c_gt_ra = data_gt[f"j3d_ra.{flag}"]
    is_valid = data_gt["is_valid"]

    mpjpe_ra = metrics.compute_joint3d_error(j3d_h_c_gt_ra, j3d_h_c_pred_ra, is_valid)
    mpjpe_ra = mpjpe_ra.mean(axis=1) * 1000  # Use dim instead of axis for PyTorch

    indicator = "r" if flag == "right" else "l"
    metric_dict[f"mpjpe_ra_{indicator}"] = mpjpe_ra
    return metric_dict


def eval_ious(data_pred, data_gt, metric_dict):
    masks_pred = data_pred["masks_pred"].long().numpy()
    masks_gt = data_gt["masks_gt"].long().numpy()
    is_valid = data_gt["is_valid"]
    ious = compute_iou_per_frame(masks_pred, masks_gt)
    not_valid = (1 - is_valid).numpy().astype(bool)
    ious[not_valid] = np.nan
    metric_dict["ious"] = ious * 100.0
    return metric_dict
