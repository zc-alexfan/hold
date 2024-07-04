import numpy as np

from torch.utils.data import Dataset
from src.datasets.image_dataset import ImageDataset


class ValDataset(Dataset):
    def __init__(self, args):
        self.dataset = ImageDataset(args)
        self.img_size = self.dataset.img_size

        self.total_pixels = np.prod(self.img_size)
        self.pixel_per_batch = 512
        self.dataset.num_sample = -1

        np.random.seed(1)
        self.eval_idx_list = np.random.permutation(len(self.dataset))
        print(self.eval_idx_list[:10])
        self.idx = 0

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        image_id = self.eval_idx_list[self.idx]

        self.data = self.dataset[image_id]
        batch = self.data
        assert batch["idx"] == image_id
        batch["pixel_per_batch"] = self.pixel_per_batch
        self.idx = (self.idx + 1) % len(self.eval_idx_list)

        return batch


class TestDataset(Dataset):
    def __init__(self, args):
        self.dataset = ImageDataset(args)
        self.img_size = self.dataset.img_size

        self.total_pixels = np.prod(self.img_size)
        self.pixel_per_batch = 512
        num_lists = 30
        eval_idx_list = np.arange(len(self.dataset))
        if args.agent_id == -1:
            # self.eval_idx_list = eval_idx_list[::10]
            self.eval_idx_list = eval_idx_list
        else:
            sublists = np.array_split(eval_idx_list, num_lists)
            self.eval_idx_list = sublists[args.agent_id]
            print("Running on these images:")
            print(self.eval_idx_list)
        self.dataset.num_sample = -1
        np.random.seed(1)

    def __len__(self):
        return len(self.eval_idx_list)

    def __getitem__(self, idx):
        image_id = self.eval_idx_list[idx]

        self.data = self.dataset[image_id]
        batch = self.data
        assert batch["idx"] == image_id
        batch["pixel_per_batch"] = self.pixel_per_batch

        return batch
