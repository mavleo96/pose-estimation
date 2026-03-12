import numpy as np
from PIL import Image
from pathlib import Path
import json
from torch.utils.data import Dataset


class SceneDataset(Dataset):
    def __init__(self, dataset_name, data_dir, split="test"):
        super().__init__()
        self.dataset_name = dataset_name
        self.split = split

        self.data_path = Path(data_dir) / dataset_name / split

        # Load data
        self.data = []

        # Iterate over all scenes
        for scene_path in self.data_path.iterdir():
            scene_id = int(scene_path.stem)  # "000002" -> 2

            # Load camera info
            with open(scene_path / "scene_camera.json") as f:
                camera_info = json.load(f)

            # Iterate over all images
            for image_path in (scene_path / "rgb").iterdir():
                image_id = int(image_path.stem)  # "000001" -> 1
                id = f"{scene_id:06d}_{image_id:06d}"  # 2, 1 -> "000002_000001"

                image_camera = camera_info[str(image_id)]

                image_info = {
                    "id": id,
                    "scene_id": scene_id,
                    "view_id": image_id,
                    "cam_K": np.array(image_camera["cam_K"]).reshape(3, 3),
                    "depth_scale": image_camera["depth_scale"],
                    "image_path": image_path,
                    "depth_path": scene_path / "depth" / f"{image_path.stem}.png",
                }

                self.data.append(image_info)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_info = self.data[index]

        rgb = np.array(Image.open(image_info["image_path"]).convert("RGB"))

        depth = np.array(Image.open(image_info["depth_path"])).astype(np.float32)
        depth = depth * image_info["depth_scale"]  # in mm

        return {
            "id": image_info["id"],
            "scene_id": image_info["scene_id"],
            "view_id": image_info["view_id"],
            "rgb": rgb,
            "depth": depth,
            "cam_K": image_info["cam_K"],
        }
