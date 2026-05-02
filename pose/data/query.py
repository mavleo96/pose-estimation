import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .utils import process_rgb_and_depth, rle_to_mask


class QueryDataset(Dataset):
    def __init__(
        self,
        dataset_name,
        data_dir,
        detections_path,
        output_size,
        split="test",
        rgb_transform=None,
        rgb2_transform=None,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.split = split
        if dataset_name in ["hb", "tless"]:
            split_dir = "test_primesense"
        else:
            split_dir = split

        self.data_path = Path(data_dir) / dataset_name / split_dir
        self.detections_path = Path(detections_path)

        self.output_size = output_size

        # Load data
        self.image_info = self.load_image_info()
        self.detections = self.load_detections()

        self.rgb_transform = rgb_transform
        self.rgb2_transform = rgb2_transform

    def load_image_info(self):
        image_info = {}
        for scene_path in self.data_path.iterdir():
            scene_id = int(scene_path.stem)  # "000002" -> 2

            # Load camera info
            with open(scene_path / "scene_camera.json") as f:
                camera_info = json.load(f)

            # Iterate over all images
            for rgb_path in (scene_path / "rgb").iterdir():
                im_id = int(rgb_path.stem)  # "000001" -> 1
                image_key = f"{scene_id:06d}_{im_id:06d}"  # 2, 1 -> "000002_000001"

                camera_params = camera_info[str(im_id)]

                image_info[image_key] = {
                    "image_key": image_key,
                    "scene_id": scene_id,
                    "im_id": im_id,
                    "cam_K": np.array(camera_params["cam_K"]).reshape(3, 3),
                    "depth_scale": camera_params["depth_scale"],
                    "rgb_path": rgb_path,
                    "depth_path": scene_path / "depth" / f"{rgb_path.stem}.png",
                }

        return image_info

    def load_detections(self):
        with open(self.detections_path) as f:
            detections = json.load(f)

        detections.sort(key=lambda x: (x["scene_id"], x["image_id"], x["category_id"], -x["score"]))

        # add instance id to detections
        prev_key = None
        instance_id = 1

        for d in detections:
            d["im_id"] = d["image_id"]
            d["obj_id"] = d["category_id"]
            del d["category_id"], d["image_id"]

            if prev_key != f"{d['scene_id']}-{d['im_id']}-{d['obj_id']}":
                instance_id = 1
                prev_key = f"{d['scene_id']}-{d['im_id']}-{d['obj_id']}"

            d["instance_id"] = instance_id
            instance_id += 1

        return detections

    def get_image_key(self, idx):
        record = self.detections[idx]
        return f"{record['scene_id']:06d}_{record['im_id']:06d}"

    def __len__(self):
        return len(self.detections)

    def __getitem__(self, idx):
        record = self.detections[idx]
        image_record = self.image_info[self.get_image_key(idx)]

        rgb = np.array(Image.open(image_record["rgb_path"]).convert("RGB"))
        mask = rle_to_mask(record["segmentation"])
        bbox = record["bbox"]

        depth = np.array(Image.open(image_record["depth_path"])).astype(np.float32)
        depth = depth * image_record["depth_scale"]  # in mm

        rgb_query, depth_query, M = process_rgb_and_depth(rgb, depth, bbox, self.output_size, mask)
        mask_query = depth_query > 0

        return {
            "scene_id": record["scene_id"],
            "im_id": record["im_id"],
            "obj_id": record["obj_id"],
            "instance_id": record["instance_id"],
            "detection_score": record["score"],
            "detection_time": record["time"],
            "rgb": self.rgb_transform(rgb_query),
            "rgb2": self.rgb2_transform(rgb_query),
            "depth": torch.from_numpy(depth_query),  # Note: depth is not yet normalized
            "mask": torch.from_numpy(mask_query).bool(),
            "M": torch.from_numpy(M).float(),
            "cam_K": torch.from_numpy(image_record["cam_K"]).float(),
        }
