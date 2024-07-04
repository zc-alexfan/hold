import torch
from pytorch3d.transforms import axis_angle_to_quaternion, quaternion_to_axis_angle
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import numpy as np


def slerp_mano(quat, trans, key_times, times):
    """
    Args:
        quat: (T x J x 4)
        trans: (T x 3)
    """

    quats = []
    for j in range(quat.shape[1]):
        start_time = key_times[0]
        end_time = key_times[-1]
        curr_quat = quat[:, j]

        start_quat = curr_quat[:1]
        end_quat = curr_quat[-1:]
        curr_key_times = np.copy(key_times)

        start_time_query = times[0]
        end_time_query = times[-1]
        if start_time_query < start_time:
            curr_quat = np.concatenate((start_quat, curr_quat), axis=0)
            curr_key_times = np.concatenate(([start_time_query], key_times), axis=0)

        if end_time < end_time_query:
            curr_quat = np.concatenate((curr_quat, end_quat), axis=0)
            curr_key_times = np.concatenate((key_times, [end_time_query]), axis=0)

        curr_key_rots = R.from_quat(curr_quat)
        s = Slerp(curr_key_times, curr_key_rots)
        interp_rots = s(times)
        quats.append(interp_rots.as_quat())
    slerp_quat = np.stack(quats, axis=1)

    lerp_trans = np.zeros((len(times), 3))
    for i in range(3):
        lerp_trans[:, i] = np.interp(times, key_times, trans[:, i])

    return slerp_quat, lerp_trans


def slerp_xyz(j2d):
    """
    Fills NaN entries in a 3D numpy array using interpolation.

    Parameters:
    j2d (np.ndarray): Input array with shape (frames, 21 joints, 2 dimensions).
                       It is assumed that an entry j2d[i] is all NaN if the frame is missing.

    Returns:
    np.ndarray: The array with NaN entries filled by interpolation.
    """
    frames, joints, dims = j2d.shape
    frame_indices = np.arange(frames)

    for joint in range(joints):
        for dim in range(dims):
            data = j2d[:, joint, dim]
            valid = ~np.isnan(data)
            valid_frames = frame_indices[valid]
            valid_values = data[valid]

            if (
                valid_frames.size > 0
            ):  # Only interpolate if there's at least one valid frame
                j2d[:, joint, dim] = np.interp(
                    frame_indices, valid_frames, valid_values
                )
            else:
                # If no valid frames exist, leave the data as is (all NaNs)
                j2d[:, joint, dim] = data

    return j2d


def infilling_betas(num_frames, betas, outliers, k):
    """
    This function fills the 'betas' array by replacing outlier values with the mean of k nearby inlier values.

    Args:
    num_frames (int): Total number of frames. Shape: ()
    betas (numpy.ndarray): Inlier array containing beta values. Expected shape: (m, 10) where m <= num_frames
    outliers (list or numpy.ndarray): Array containing indices of outlier frames. Shape: (n,) where n is the number of outliers.
    k (int): Number of nearby inliers to consider for calculating the mean.

    Returns:
    numpy.ndarray: Array with outliers replaced by the mean of nearby inliers. Shape: (num_frames, 10)
    """

    # Adding assertions for input types and shapes
    assert isinstance(num_frames, int), "num_frames should be an integer"
    assert isinstance(betas, np.ndarray), "betas should be a numpy ndarray"
    assert betas.shape[1] == 10
    assert betas.shape[0] <= num_frames
    assert isinstance(
        outliers, (list, np.ndarray)
    ), "outliers should be a list or numpy ndarray"
    assert isinstance(k, int), "k should be an integer"

    interp_betas = np.zeros((num_frames, 10))

    # Calculate key frames based on outliers
    key_frames = np.array([f for f in list(range(num_frames)) if f not in outliers])

    # Assign betas values to key_frames indices in interp_betas
    interp_betas[key_frames] = betas

    # Calculate sets of outliers and inliers
    outliers_set = set(outliers)
    inliers = set(range(num_frames)) - outliers_set

    # Loop through each outlier to find k nearby non-outlier frames and calculate the mean betas
    for idx in outliers:
        nearby_inliers = []
        left, right = idx - 1, idx + 1

        while len(nearby_inliers) < k:
            if left in inliers:
                nearby_inliers.append(left)
            if right in inliers:
                nearby_inliers.append(right)

            left -= 1
            right += 1

            if left < 0 and right >= num_frames:
                break

        if nearby_inliers:
            nearby_betas = np.mean(interp_betas[nearby_inliers], axis=0)
            interp_betas[idx] = nearby_betas

    return interp_betas


def identify_outliers(volumes):
    mean_volume = np.mean(volumes)
    std_dev_volume = np.std(volumes)

    sigma = 2.0
    outliers = np.where((volumes < mean_volume - sigma * std_dev_volume))
    outliers = outliers[0]
    return outliers


def slerp_mano_params(outliers, num_frames, key_frames, data):
    global_orient_tensor = torch.FloatTensor(data["global_orient"])[key_frames]
    hand_pose_tensor = torch.FloatTensor(data["hand_pose"])[key_frames]
    betas_tensor = torch.FloatTensor(data["betas"])[key_frames]
    transl_tensor = torch.FloatTensor(data["transl"])[key_frames]

    # reform
    full_pose_tensor = torch.cat((global_orient_tensor, hand_pose_tensor), dim=1).view(
        hand_pose_tensor.shape[0], 16, 3
    )
    full_pose_tensor_quat = axis_angle_to_quaternion(full_pose_tensor)
    quat = full_pose_tensor_quat.cpu().detach().numpy()
    transl = transl_tensor.cpu().detach().numpy()

    # interpolation
    times = np.arange(num_frames)
    interp_quat, interp_transl = slerp_mano(quat, transl, key_frames, times)
    interp_full_pose = quaternion_to_axis_angle(torch.FloatTensor(interp_quat)).view(
        -1, 48
    )
    interp_global_orient = interp_full_pose[:, :3].detach()
    interp_pose = interp_full_pose[:, 3:].detach()

    betas = betas_tensor.cpu().detach().numpy()
    interp_betas = infilling_betas(num_frames, betas, outliers, k=5)
    interp_betas = torch.FloatTensor(interp_betas)

    # Create output directory if it doesn't exist
    out = {}
    out["global_orient"] = interp_global_orient.view(-1, 3).numpy().astype(np.float32)
    out["hand_pose"] = interp_pose.view(-1, 45).numpy().astype(np.float32)
    out["betas"] = interp_betas.view(-1, 10).numpy().astype(np.float32)
    out["transl"] = interp_transl.astype(np.float32)
    return out
