import numpy as np
import argparse
import os
import sys

sys.path = ["."] + sys.path
import src.hand_pose.slerp as slerp


def interpolate_hand_pose(seq_name):
    """
    Interpolates hand pose parameters for a given sequence using spherical linear interpolation (SLERP).

    Args:
    seq_name (str): The name of the sequence to process.
    """
    # Define paths for input data
    data_path = f"./data/{seq_name}/processed/hold_fit.init.npy"
    data_2d_path = f"./data/{seq_name}/processed/j2d.full.npy"

    # Load data
    data = np.load(data_path, allow_pickle=True).item()
    data_2d = np.load(data_2d_path, allow_pickle=True).item()

    # Prepare interpolated data dictionary
    data_interp = {}

    # Process each hand
    for hand in data.keys():
        not_valid = np.isnan(data_2d[f"j2d.{hand}"].reshape(-1, 21 * 2).mean(axis=1))
        outliers = np.where(not_valid)[0]
        num_frames = data_2d[f"j2d.{hand}"].shape[0]
        key_frames = np.where(~not_valid)[0]

        # Perform SLERP interpolation
        hand_interp = slerp.slerp_mano_params(
            outliers, num_frames, key_frames, data[hand]
        )
        hand_interp["is_valid"] = (~not_valid).astype(np.float32)
        data_interp[hand] = hand_interp

    # Define output path and save interpolated data
    out_p = data_path.replace(".init.", ".slerp.")
    np.save(out_p, data_interp)

    # Print the location of exported files
    print(f"Interpolated data saved to {out_p}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interpolate Hand Pose Data")
    parser.add_argument("--seq_name", type=str, help="Name of the sequence to process")
    args = parser.parse_args()

    interpolate_hand_pose(args.seq_name)
