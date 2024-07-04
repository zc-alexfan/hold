import sys

sys.path = ["../"] + sys.path
from common.viewer import HOLDViewer
import os.path as op


class DataViewer(HOLDViewer):
    def __init__(
        self,
        render_types=["rgb"],
        interactive=True,
        size=(2024, 2024),
    ):
        super().__init__(render_types, interactive, size)


def fetch_parser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_p", type=str, default=None)
    parser.add_argument("--mode", type=str, default=None)
    parser.add_argument("--seq_name", type=str, default=None)
    parser.add_argument("--gt_ho3d", action="store_true")
    parser.add_argument("--gt_arctic", action="store_true")
    parser.add_argument("--ours", action="store_true")
    parser.add_argument("--ours_implicit", action="store_true")
    args = parser.parse_args()
    from easydict import EasyDict

    args = EasyDict(vars(args))
    return args


def merge_batch(batch_list):
    if len(batch_list) == 1:
        return batch_list[0][0], batch_list[0][1]
    meshes = {}
    for batch in batch_list:
        meshes.update(batch[0])
    batch_list = batch_list[:-1]
    data_gt = batch_list[-1][1]  # from gt data
    return meshes, data_gt


def main():
    args = fetch_parser()
    viewer = DataViewer(interactive=True, size=(2024, 2024))
    batch_list = []
    batch = None
    if args.ours:
        import src.utils.io.ours as ours

        batch = ours.load_viewer_data(args)
        batch_list.append(batch)

    if args.gt_ho3d:
        import src.utils.io.gt as gt

        batch = gt.load_viewer_data(args)
        batch_list.append(batch)

    if args.gt_arctic:
        import src.utils.io.gt_arctic as gt_arctic

        batch = gt_arctic.load_viewer_data(args)
        batch_list.append(batch)

    if len(batch_list) > 1:
        batch = merge_batch(batch_list)
    viewer.render_seq(batch, out_folder=op.join("render_out"))


if __name__ == "__main__":
    main()
