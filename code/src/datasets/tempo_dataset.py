import random
import numpy as np

from src.datasets.image_dataset import ImageDataset
import sys

sys.path = [".."] + sys.path

import common.ld_utils as ld_utils


class TempoDataset(ImageDataset):
    """
    Load a window per get_item
    """

    def __init__(self, args):
        super().__init__(args)
        self.offset = args.offset
        print("Using Temporal Dataset with offset: {}".format(self.offset))

        start_idx = np.arange(len(self.img_paths) - self.offset)
        end_idx = start_idx + self.offset
        self.pairs = np.stack((start_idx, end_idx), axis=1)
        self.sample_idx = 0

    def __getitem__(self, idx):
        start_idx, end_idx = random.choice(self.pairs)
        start_idx = int(start_idx)
        end_idx = int(end_idx)

        left = super().__getitem__(start_idx)
        right = super().__getitem__(end_idx)
        data = ld_utils.stack_dl(ld_utils.ld2dl([left, right]), dim=0, verbose=False)
        self.sample_idx += 1
        return data

    def __len__(self):
        return self.args.tempo_len
