import copy

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree as KDTree
import copy

from tqdm import tqdm


def calculate_metrics(aligned_mesh, target_mesh, is_sqrt=False):
    vertices_source = np.asarray(aligned_mesh.vertices) * 100
    vertices_target = np.asarray(target_mesh.vertices) * 100

    # dist_bidirectional = chamferDist(vertices_source, vertices_target, bidirectional=True,point_reduction = "mean") #* 0.001
    gen_points_kd_tree = KDTree(vertices_source)
    one_distances, one_vertex_ids = gen_points_kd_tree.query(vertices_target)

    if is_sqrt:  # square-root chamfer
        gt_to_gen_chamfer = np.mean(one_distances)
    else:  # squared chamfer
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


def preprocess_point_cloud(pcd, voxel_size):
    # print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    # print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )

    radius_feature = voxel_size * 5
    # print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )
    return pcd_down, pcd_fpfh


def execute_global_registration(
    source_down, target_down, source_fpfh, target_fpfh, voxel_size
):
    distance_threshold = 0.01  # 1#voxel_size * 1.5
    # print(":: RANSAC registration on downsampled point clouds.")
    # print("   Since the downsampling voxel size is %.3f," % voxel_size)
    # print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold
            ),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )
    return result


def refine_registration(
    source, target, source_fpfh, target_fpfh, voxel_size, init_alignment
):
    distance_threshold = 0.01  # voxel_size * 0.4
    # print(":: Point-to-plane ICP registration is applied on original point")
    # print("   clouds to refine the alignment. This time we use a strict")
    # print("   distance threshold %.3f." % distance_threshold)

    # result_ransac.transformation
    result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        distance_threshold,
        init_alignment,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(
            with_scaling=True
        ),
    )
    return result


def compute_icp_metrics(
    target_mesh, source_mesh, num_iters=60, no_tqdm=False, is_sqrt=False
):
    voxel_size = 0.005
    # print(":: Load two meshes")
    # exp_id = "42465ff7e"

    # assert os.path.exists(target_deform_p)
    # assert os.path.exists(pred_deform_p)
    # target_mesh = o3d.io.read_triangle_mesh(target_deform_p)
    # source_mesh = o3d.io.read_triangle_mesh(pred_deform_p)

    center_mass = source_mesh.get_center()
    source_mesh.vertices = o3d.utility.Vector3dVector(
        np.asarray(source_mesh.vertices) - center_mass
    )
    source_copy = copy.deepcopy(source_mesh)
    center_mass = target_mesh.get_center()
    target_mesh.vertices = o3d.utility.Vector3dVector(
        np.asarray(target_mesh.vertices) - center_mass
    )

    # o3d.io.write_triangle_mesh("target.obj", target_mesh)
    # o3d.io.write_triangle_mesh("source.obj", source_mesh)

    # print(":: Sample mesh to point cloud")
    target = target_mesh.sample_points_uniformly(1000)
    source = source_mesh.sample_points_uniformly(1000)
    # draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

    trans_init = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    result_icp_ra = refine_registration(
        source, target, source_fpfh, target_fpfh, voxel_size, trans_init
    )

    # Apply the transformation to align the source mesh with the target
    aligned_source_mesh = source_copy.transform(result_icp_ra.transformation)
    cd_ra, f5_ra, f10_ra = calculate_metrics(aligned_source_mesh, target_mesh, is_sqrt)
    best_cd = cd_ra
    best_f5 = f5_ra
    best_f10 = f10_ra

    if no_tqdm:
        pbar = range(num_iters)
    else:
        pbar = tqdm(range(num_iters))
    # for iter in tqdm(range(num_iters)):
    for iter in pbar:
        try:
            result_ransac = execute_global_registration(
                source_down, target_down, source_fpfh, target_fpfh, voxel_size
            )

            result_icp_nra = refine_registration(
                source,
                target,
                source_fpfh,
                target_fpfh,
                voxel_size,
                result_ransac.transformation,
            )
        except:
            print("Error in ICP: Skipping")
        aligned_source_mesh_ransac = copy.deepcopy(source_mesh).transform(
            result_icp_nra.transformation
        )

        cd_nra, f5_nra, f10_nra = calculate_metrics(
            aligned_source_mesh_ransac, target_mesh
        )
        # print(cd_nra)
        if cd_nra < best_cd:
            best_cd, best_f5, best_f10 = cd_nra, f5_nra, f10_nra

    # print("CD F5 F10", best_cd, best_f5, best_f10 )
    return best_cd, best_f5, best_f10
