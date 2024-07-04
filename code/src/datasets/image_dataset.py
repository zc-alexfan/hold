import glob
import os

import cv2
import numpy as np
import torch
from loguru import logger


from torch.utils.data import Dataset
from src.datasets.utils import load_image, load_mask
from src.datasets.utils import reform_dict, weighted_sampling, load_K_Rt_from_P


class ImageDataset(Dataset):
    def setup_poses(self, data):
        entities = data["entities"]
        out = {}
        for name, val in entities.items():
            reform_fn = reform_dict[name.split("_")[0]]
            out[name] = reform_fn(self.scale, val)

        self.params = out

    def __init__(self, args):
        self.root = os.path.join("./data", args.case, "build")
        self.args = args
        data = np.load(os.path.join(self.root, "data.npy"), allow_pickle=True).item()

        self.setup_images()
        self.setup_masks()
        self.setup_cameras(data)
        self.setup_poses(data)

        self.debug_dump(args)

        self.num_sample = self.args.num_sample
        self.sampling_strategy = "weighted"

    def debug_dump(self, args):
        if args.debug:
            out = {}
            out["intrinsics_all"] = self.intrinsics_all
            out["extrinsics_all"] = self.extrinsics_all
            out["scale_mats"] = self.scale_mats
            out["world_mats"] = self.world_mats
            out["img_paths"] = self.img_paths
            out["mask_paths"] = self.mask_paths
            out["img_size"] = self.img_size
            out["n_images"] = self.n_images
            out["params"] = self.params
            out["scale"] = self.scale

            out_p = os.path.join(args.log_dir, "dataset_info.pth")
            torch.save(out, out_p)
            print(f"Saved dataset info to {out_p}")

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        img = load_image(self.img_paths[idx])
        mask = load_mask(self.mask_paths[idx], img.shape)

        img_size = self.img_size
        uv = np.mgrid[: img_size[0], : img_size[1]].astype(np.int32)
        uv = np.flip(uv, axis=0).copy().transpose(1, 2, 0).astype(np.float32)
        entity_keys = self.params.keys()
        params = {key + ".params": self.params[key][idx] for key in entity_keys}
        if self.num_sample > 0:
            hand_types = [key for key in entity_keys if "right" in key or "left" in key]
            num_sample = self.num_sample // len(hand_types)

            uv_list = []
            mask_list = []
            img_list = []
            for hand_type in hand_types:
                # sample around mask tight bbox and uniform sampling
                samples = weighted_sampling(
                    {"rgb": img, "uv": uv, "obj_mask": mask},
                    img_size,
                    num_sample,
                    hand_type,
                )[0]
                uv_list.append(samples["uv"])
                mask_list.append(samples["obj_mask"])
                img_list.append(samples["rgb"])

            uv = np.concatenate(uv_list, axis=0)
            mask = np.concatenate(mask_list, axis=0)
            img = np.concatenate(img_list, axis=0)

        batch = {
            "uv": uv.reshape(-1, 2).astype(np.float32),
            "intrinsics": self.intrinsics_all[idx],
            "extrinsics": self.extrinsics_all[idx],
            "im_path": self.img_paths[idx],
            "idx": idx,
            "gt.rgb": img.reshape(-1, 3).astype(np.float32),
            "gt.mask": mask.reshape(-1).astype(np.int64),
            "img_size": self.img_size,
            "total_pixels": self.total_pixels,
        }
        batch.update(params)
        return batch

    def setup_images(self):
        img_dir = os.path.join(self.root, "image")
        self.img_paths = sorted(glob.glob(f"{img_dir}/*.png"))
        assert len(self.img_paths) > 0
        self.img_size = cv2.imread(self.img_paths[0]).shape[:2]
        self.total_pixels = np.prod(self.img_size)
        self.n_images = len(self.img_paths)

    def setup_masks(self):
        mask_dir = os.path.join(self.root, "mask")
        self.mask_paths = sorted(glob.glob(f"{mask_dir}/*.png"))
        if len(self.mask_paths) == 0:
            logger.warning("No mask found, using fake mask")
            self.mask_paths = [None] * self.n_images
        else:
            assert len(self.mask_paths) == self.n_images

    def setup_cameras(self, data):
        camera_dict = data["cameras"]
        self.scale_mats, self.world_mats = [], []
        self.intrinsics_all, self.extrinsics_all = [], []

        for idx in range(self.n_images):
            scale_mat = camera_dict[f"scale_mat_{idx}"].astype(np.float32)
            world_mat = camera_dict[f"world_mat_{idx}"].astype(np.float32)
            self.scale_mats.append(scale_mat)
            self.world_mats.append(world_mat)

            # Compute camera parameters
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, extrinsics = load_K_Rt_from_P(None, P)
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.extrinsics_all.append(torch.from_numpy(extrinsics).float())
        self.scale = 1 / self.scale_mats[0][0, 0]
        assert len(self.intrinsics_all) == len(self.extrinsics_all)
