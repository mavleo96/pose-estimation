from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

from ..utils.io import load_models_info
from .utils import get_bbox, process_rgb_and_depth


class TemplateDataset:
    """
    Lookup store for template embeddings and poses.
    Indexed by object index, returns all 162 templates for one object.

    Usage:
        dataset = TemplateDataset(...)

        rgb = dataset[dataset.get_index([1, 6])]["rgb"]  # (2, n_views, H, W, 3)
        depth = dataset[dataset.get_index([1, 6])]["depth"]  # (2, n_views, H, W)
        M = dataset[dataset.get_index([1, 6])]["M"]  # (2, n_views, 3, 3)
        T_m2c = dataset[dataset.get_index([1, 6])]["T_m2c"]  # (2, n_views, 4, 4)
        embeddings = dataset[dataset.get_index([1, 6])]["embedding"]  # (2, n_views, n_patches, D)
    """

    def __init__(self, dataset_name, data_dir, embeddings_dir):
        self.dataset_name = dataset_name

        self.templates_path = Path(data_dir) / "templates" / dataset_name
        self.models_path = Path(data_dir) / dataset_name / "models"
        self.embeddings_dir = Path(embeddings_dir)

        # Load models_info
        self.models_info = load_models_info(self.models_path / "models_info.json")
        self.object_ids = sorted(self.models_info.keys())

        # Load object poses
        T_m2c_list = []
        for object_id in self.object_ids:
            T_m2c = np.load(self.templates_path / "object_poses" / f"{object_id:06d}.npy")
            T_m2c_list.append(T_m2c)
        self.T_m2c = torch.from_numpy(np.stack(T_m2c_list, axis=0)).float()

        # Load embeddings
        self.embeddings = torch.from_numpy(
            np.load(self.embeddings_dir / "pca_embeddings.npy")
        ).float()

        # Load templates
        with np.load(self.embeddings_dir / "templates.npz") as f:
            self.rgb = torch.from_numpy(f["rgb"]).float()
            self.depth = torch.from_numpy(f["depth"]).float()
            self.M = torch.from_numpy(f["M"]).float()

    def get_index(self, object_id):
        if isinstance(object_id, list):
            return [self.object_ids.index(obj_id) for obj_id in object_id]
        else:
            return self.object_ids.index(object_id)

    def __len__(self):
        return len(self.object_ids)

    def __getitem__(self, index):
        return {
            "rgb": self.rgb[index],
            "depth": self.depth[index],
            "M": self.M[index],
            "T_m2c": self.T_m2c[index],
            "embedding": self.embeddings[index],
        }


class TemplateViewDataset(Dataset):
    """Lightweight dataset for processing RGB images for embedding computation."""

    def __init__(self, templates_path, object_id, output_size, depth_scale, rgb_transform=None):
        super().__init__()
        self.object_dir = Path(templates_path) / f"{object_id:06d}"
        self.output_size = output_size
        self.depth_scale = depth_scale

        # Load RGB image paths for the object
        self.rgb_paths = sorted(self.object_dir.glob("*[0-9].png"), key=lambda x: int(x.stem))

        self.rgb_transform = rgb_transform or transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, index):
        rgb_path = self.rgb_paths[index]
        depth_path = rgb_path.parent / f"{rgb_path.stem}_depth.png"

        # load the rgb image and process for embedding computation
        rgb = np.array(Image.open(rgb_path).convert("RGB"))
        depth = np.array(Image.open(depth_path)).astype(np.float32) * self.depth_scale
        bbox = get_bbox(rgb)
        rgb, depth, M = process_rgb_and_depth(rgb, depth, bbox, self.output_size)

        return {
            "rgb": self.rgb_transform(rgb),
            "depth": torch.from_numpy(depth).float(),
            "M": torch.from_numpy(M).float(),
        }
