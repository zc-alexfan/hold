import argparse
import sys

sys.path = ["."] + sys.path
from src.colmap.colmap_utils import colmap_pose_est
from src.colmap.colmap_utils import validate_colmap
from src.colmap.colmap_utils import format_poses


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_name", type=str, help="sequence name")
    parser.add_argument(
        "--num_pairs",
        type=int,
        help="number of the frames that the model is searching for connections",
    )
    parser.add_argument("--no_vis", default=False, action="store_true")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    seq_name = args.seq_name
    num_pairs = args.num_pairs
    no_vis = args.no_vis

    print("Processing sequence", seq_name)
    colmap_pose_est(seq_name, num_pairs)
    validate_colmap(seq_name, no_vis)
    format_poses(seq_name)
