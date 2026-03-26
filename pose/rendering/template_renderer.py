from pathlib import Path

import numpy as np
import trimesh

from ..utils.io import load_camera_intrinsics, load_models_info
from .render import generate_and_save_templates, look_at_matrix


class TemplateRenderer:
    """
    Class for rendering templates for a given dataset.
    """

    TARGET_FILL = 0.8
    SUBDIVISIONS = 2

    def __init__(self, dataset_name, data_dir):
        self.dataset_name = dataset_name

        self.data_path = Path(data_dir) / dataset_name
        self.templates_path = Path(data_dir) / "templates" / dataset_name

        # Load models_info
        self.models_info = load_models_info(self.data_path / "models")
        self.obj_ids = sorted(self.models_info.keys())

        # Load camera intrinsics
        self.camera = load_camera_intrinsics(self.dataset_name, self.data_path)

    def render_templates(self):
        # Create directories
        self.templates_path.mkdir(parents=True, exist_ok=True)
        (self.templates_path / "object_poses").mkdir(parents=True, exist_ok=True)

        # Compute distance multiplier
        dist_mult = self.camera.fx / (self.TARGET_FILL * self.camera.height / 2)

        # unit icosphere vertices; subdivisions=2 gives 162 uniformly distributed viewpoints
        sphere = trimesh.creation.icosphere(subdivisions=self.SUBDIVISIONS)
        vertices = np.array(sphere.vertices)

        for obj_id in self.obj_ids:
            model = trimesh.load(self.data_path / "models" / f"obj_{obj_id:06d}.ply", process=False)

            radius = self.models_info[obj_id]["diameter"] / 2
            cam_pos = vertices * radius * dist_mult

            # compute w2c matrices for all 162 viewpoints
            T_m2c = look_at_matrix(cam_pos)

            # save rendered RGB + depth templates and poses
            save_dir = self.templates_path / f"{obj_id:06d}"
            generate_and_save_templates(
                model,
                T_m2c,
                self.camera.K,
                self.camera.width,
                self.camera.height,
                self.camera.depth_scale,
                save_dir,
            )
            np.save(self.templates_path / "object_poses" / f"{obj_id:06d}.npy", T_m2c)

            print(f"object {obj_id:06d} done — {len(T_m2c)} templates to {save_dir}")
