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
        self.image_info = self.load_and_process_scene_image_info()

        self.data = list(self.image_info.values())  # list of image info

    def load_and_process_scene_image_info(self):
        image_info = {}
        for scene_path in self.data_path.iterdir():
            scene_id = int(scene_path.stem)  # "000002" -> 2

            # Load camera info
            with open(scene_path / "scene_camera.json") as f:
                camera_info = json.load(f)

            # Iterate over all images
            for rgb_path in (scene_path / "rgb").iterdir():
                image_id = int(rgb_path.stem)  # "000001" -> 1
                image_key = f"{scene_id:06d}_{image_id:06d}"  # 2, 1 -> "000002_000001"

                camera_params = camera_info[str(image_id)]

                image_info[image_key] = {
                    "id": image_key,
                    "scene_id": scene_id,
                    "image_id": image_id,
                    "cam_K": np.array(camera_params["cam_K"]).reshape(3, 3),
                    "depth_scale": camera_params["depth_scale"],
                    "rgb_path": rgb_path,
                    "depth_path": scene_path / "depth" / f"{rgb_path.stem}.png",
                }

        return image_info

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        record = self.data[index]

        rgb = np.array(Image.open(record["rgb_path"]).convert("RGB"))

        depth = np.array(Image.open(record["depth_path"])).astype(np.float32)
        depth = depth * record["depth_scale"]  # in mm

        return {
            "id": record["id"],
            "scene_id": record["scene_id"],
            "image_id": record["image_id"],
            "rgb": rgb,
            "depth": depth,
            "cam_K": record["cam_K"],
        }
