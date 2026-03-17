import json
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from torch.utils.data import Dataset
from ..utils import rle_to_mask, get_pad_matrix, get_resize_matrix, get_crop_matrix
from .utils import calculate_square_padding


class QueryDataset(Dataset):
    def __init__(
        self, dataset_name, data_dir, split="test", detections_path=None, query_size=(224, 224)
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.split = split

        self.data_path = Path(data_dir) / dataset_name / split
        self.detections_path = Path(detections_path)

        self.query_size = query_size

        # Load data
        self.image_info = self.load_and_process_scene_image_info()
        self.detections = self.load_and_process_detections()

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

    def load_and_process_detections(self):
        with open(self.detections_path, "r") as f:
            detections = json.load(f)

        detections.sort(key=lambda x: (x["scene_id"], x["image_id"], x["category_id"], -x["score"]))

        # add instance id to detections
        prev_key = None
        instance_id = 1
        for d in detections:
            if prev_key != f"{d['scene_id']}-{d['image_id']}-{d['category_id']}":
                instance_id = 1
                prev_key = f"{d['scene_id']}-{d['image_id']}-{d['category_id']}"

            d["instance_id"] = instance_id
            instance_id += 1

        return detections

    def process_query(self, rgb, depth, mask, bbox):
        # mask the rgb and depth
        rgb = np.where(mask[..., None], rgb, 0)
        depth = np.where(mask, depth, 0)

        x, y, w, h = bbox
        padding = calculate_square_padding(w, h)

        # 1. crop using the bbox
        # 2. pad the cropped image to make it square
        # 3. resize the padded image to the query size
        # Note: square padding is wrong if query_size is not square
        rgb = rgb[y : y + h, x : x + w]
        rgb = np.pad(
            rgb,
            (
                (padding[1], padding[3]),
                (padding[0], padding[2]),
                (0, 0),
            ),
        )
        src_size = rgb.shape[:2]
        # cv2 resize expects (W, H) convention
        rgb = cv2.resize(rgb, self.query_size[::-1], interpolation=cv2.INTER_LINEAR)

        depth = depth[y : y + h, x : x + w]
        depth = np.pad(
            depth,
            (
                (padding[1], padding[3]),
                (padding[0], padding[2]),
            ),
        )
        depth = cv2.resize(depth, self.query_size[::-1], interpolation=cv2.INTER_NEAREST)

        # compute the transformation matrix
        T_crop = get_crop_matrix(bbox)
        T_pad = get_pad_matrix(padding)
        T_resize = get_resize_matrix(src_size, self.query_size)

        M = T_resize @ T_pad @ T_crop

        return rgb, depth, M

    def __len__(self):
        return len(self.detections)

    def __getitem__(self, idx):
        record = self.detections[idx]
        image_record = self.image_info[f"{record['scene_id']:06d}_{record['image_id']:06d}"]

        rgb = np.array(Image.open(image_record["rgb_path"]).convert("RGB"))
        mask = rle_to_mask(record["segmentation"])

        depth = np.array(Image.open(image_record["depth_path"])).astype(np.float32)
        depth = depth * image_record["depth_scale"]  # in mm

        rgb_query, depth_query, M = self.process_query(rgb, depth, mask, record["bbox"])

        return {
            "scene_id": record["scene_id"],
            "image_id": record["image_id"],
            "object_id": record["category_id"],
            "instance_id": record["instance_id"],
            "detection_score": record["score"],
            "detection_time": record["time"],
            "rgb": rgb_query,
            "depth": depth_query,
            "M": M,
            "cam_K": image_record["cam_K"],
        }
