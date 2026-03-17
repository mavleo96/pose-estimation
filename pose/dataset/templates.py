import json
from pathlib import Path

import numpy as np
import trimesh

from .camera import CameraIntrinsic
from .render import generate_and_save_templates, look_at_matrix


class TemplateDataset:
    def __init__(self, dataset_name, data_dir):
        self.dataset_name = dataset_name

        self.data_path = Path(data_dir) / dataset_name
        self.templates_path = Path(data_dir) / "templates" / dataset_name
        self.models_path = Path(data_dir) / dataset_name / "models"

        # Load trimesh models and models_info
        with open(self.models_path / "models_info.json") as f:
            self.models_info = json.load(f)

        self.models = {}
        for model_path in self.models_path.glob("*.ply"):
            object_id = int(model_path.stem.split("_")[1])
            self.models[object_id] = trimesh.load(model_path, process=False)

        if dataset_name == "ycbv":
            # Note: YCBV dataset has two cameras: "uw" and "cmu"
            self.camera = CameraIntrinsic.from_json(
                self.data_path / dataset_name / "camera_uw.json"
            )
        else:
            self.camera = CameraIntrinsic.from_json(self.data_path / dataset_name / "camera.json")

        self.precompute_poses_and_templates()

    def precompute_poses_and_templates(self):
        TARGET_FILL = 0.8
        DIST_MULT = self.camera.fx / (TARGET_FILL * self.camera.height / 2)
        (self.templates_path / "object_poses").mkdir(parents=True, exist_ok=True)

        # unit icosphere vertices; subdivisions=2 gives 162 uniformly distributed viewpoints
        sphere = trimesh.creation.icosphere(subdivisions=2)
        vertices = np.array(sphere.vertices)

        for object_id, model in self.models.items():
            radius = self.models_info[str(object_id)]["diameter"] / 2
            cam_pos = vertices * radius * DIST_MULT

            # compute w2c matrices for all 162 viewpoints
            T_m2c = look_at_matrix(cam_pos)

            # save rendered RGB + depth templates and poses
            save_dir = self.templates_path / f"{object_id:06d}"
            generate_and_save_templates(
                model,
                T_m2c,
                self.camera.K,
                self.camera.width,
                self.camera.height,
                self.camera.depth_scale,
                save_dir,
            )
            np.save(self.templates_path / "object_poses" / f"{object_id:06d}.npy", T_m2c)

            print(f"object {object_id:06d} done — {len(T_m2c)} templates to {save_dir}")
