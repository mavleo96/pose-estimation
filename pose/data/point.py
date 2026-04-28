from pathlib import Path

import numpy as np
import torch

from ..utils.io import load_models_info


class ObjectPointDataset:
    """
    Lookup store for object point embeddings and points.
    Indexed by object index, returns all points for one object.

    Usage:
        dataset = ObjectPointDataset(...)

        embedding = dataset[dataset.get_index([1, 6])]["embedding"]  # (N, D)
        points = dataset[dataset.get_index([1, 6])]["points"]  # (N, 3)
    """

    def __init__(self, dataset_name, data_dir, embeddings_dir, embedding_file):
        self.dataset_name = dataset_name

        self.data_path = Path(data_dir) / dataset_name
        self.embeddings_dir = Path(embeddings_dir)

        # Load models_info
        self.models_info = load_models_info(self.data_path / "models")
        self.obj_ids = sorted(self.models_info.keys())

        # Load embeddings
        self.point_embeddings = torch.from_numpy(
            np.load(self.embeddings_dir / embedding_file)
        ).float()

        # Load points
        self.points = torch.from_numpy(np.load(self.embeddings_dir / "points.npy")).float()

    def get_index(self, obj_id):
        if isinstance(obj_id, list):
            return [self.obj_ids.index(i) for i in obj_id]
        else:
            return self.obj_ids.index(obj_id)

    def __len__(self):
        return len(self.obj_ids)

    def __getitem__(self, index):
        return {
            "embedding": self.point_embeddings[index],
            "points": self.points[index],
        }


__all__ = ["ObjectPointDataset"]
