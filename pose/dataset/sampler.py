import math
from torch.utils.data import Sampler
import torch
from dataclasses import dataclass


class ImageGroupedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.batch_size = batch_size
        self.groups = self._build_groups(dataset)

    def _build_groups(self, dataset):
        groups = {}
        for idx in range(len(dataset)):
            image_key = dataset.get_image_key(idx)
            if image_key not in groups:
                groups[image_key] = []
            groups[image_key].append(idx)
        return groups

    def __iter__(self):
        for indices in self.groups.values():
            for i in range(0, len(indices), self.batch_size):
                yield indices[i : i + self.batch_size]

    def __len__(self):
        return sum(math.ceil(len(indices) / self.batch_size) for indices in self.groups.values())


@dataclass
class QueryDataCollator:
    def __call__(self, batch):
        # all items in batch share the same scene_id and image_id (ImageGroupedBatchSampler)
        return {
            "scene_id": batch[0]["scene_id"],
            "image_id": batch[0]["image_id"],
            "object_id": [item["object_id"] for item in batch],
            "instance_id": [item["instance_id"] for item in batch],
            "rgb": torch.stack([item["rgb"] for item in batch]),
            "depth": torch.stack([item["depth"] for item in batch]),
            "M": torch.stack([item["M"] for item in batch]),
            "cam_K": torch.stack([item["cam_K"] for item in batch]),
            "detection_score": [item["detection_score"] for item in batch],
            "detection_time": [item["detection_time"] for item in batch],
        }


@dataclass
class SceneDataCollator:
    def __call__(self, batch):
        return {
            "image_key": [item["image_key"] for item in batch],
            "scene_id": [item["scene_id"] for item in batch],
            "image_id": [item["image_id"] for item in batch],
            "rgb": torch.stack([item["rgb"] for item in batch]),
            "depth": torch.stack([item["depth"] for item in batch]),
            "cam_K": torch.stack([item["cam_K"] for item in batch]),
        }
