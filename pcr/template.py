import os
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from torch.utils.data import Dataset
from tqdm import tqdm
from trimesh.ray.ray_pyembree import RayMeshIntersector

from pcr.pca import PCA
from pcr.utils import dino_embeddings, embedding_interpolation

np.random.seed(0)

class CustomTemplateSet(Dataset):
    N = 1000

    def __init__(self, cfg, template_dataset, dino_components, **kwargs):
        self.template_dataset = template_dataset
        self.dataset_name = template_dataset.dataset_name
        self.model_infos = template_dataset.model_infos

        if "dino_batch_size" in kwargs:
            self.dino_batch_size = kwargs["dino_batch_size"]
        else:
            self.dino_batch_size = 20

        self.root_dir = cfg.data.test.root_dir
        self.template_dir = os.path.join(self.root_dir, "templates", self.dataset_name)
        self.cad_models_dir = os.path.join(self.root_dir, self.dataset_name, "models")

        # TODO: get the dim from a map
        self.pca = PCA(1024, dino_components)

        self.point_embeddings, self.points = self.preprocess_point_embeddings()

    def obj_id_to_index(self, obj_id: int) -> int:
        return [
            i for i, info in enumerate(self.model_infos) if info["obj_id"] == obj_id
        ][0]

    def sample_points(self, obj_id: int) -> tuple[np.ndarray, trimesh.Trimesh]:
        obj_path = os.path.join(self.cad_models_dir, f"obj_{obj_id:06d}.ply")
        mesh = trimesh.load(obj_path)
        points, _ = trimesh.sample.sample_surface(mesh, self.N)
        return np.asarray(points), mesh

    def map_points_to_templates(
        self, obj_id: int, points: np.ndarray, mesh: trimesh.Trimesh
    ) -> Dict[int, Dict[str, np.ndarray]]:
        # get template dataset
        K = self.template_dataset[self.obj_id_to_index(obj_id)].K.numpy()
        pose_array = self.template_dataset[self.obj_id_to_index(obj_id)].poses.numpy()
        M_array = self.template_dataset[self.obj_id_to_index(obj_id)].M.numpy()

        # initialize mapping template ids to point ids
        mapping = {}
        for i, (pose, M) in enumerate(zip(pose_array, M_array)):
            R = pose[:3, :3]
            t = pose[:3, 3]

            # transform points to camera coordinates
            p_cam = points @ R.T + t

            # direction of the points
            dirs = p_cam / np.linalg.norm(p_cam, axis=1, keepdims=True)

            # get valid points for camera view by checking if intersection distance matches distance from origin
            intersector = RayMeshIntersector(mesh.copy().apply_transform(pose))
            locations, index_ray, _ = intersector.intersects_location(
                np.zeros_like(dirs), dirs, multiple_hits=False
            )
            visible_ray = np.full(points.shape[0], False)
            hit_dist = np.linalg.norm(locations, axis=1)
            point_dist = np.linalg.norm(p_cam, axis=1)
            visible_ray[index_ray] = np.abs(hit_dist - point_dist[index_ray]) < 1e-6

            # get pixel coordinates for valid points
            # point in camera coordinates @ camera matrix @ image resize matrix
            pixel_coords_homo = (p_cam[visible_ray] @ K.T) @ M.T
            pixel_coords = (
                pixel_coords_homo[:, :2] / pixel_coords_homo[:, [2]]
            ).astype(np.int32)
            point_mapping = np.argwhere(visible_ray).reshape(-1).astype(np.int32)

            mapping[i] = {
                "point_mapping": point_mapping,
                "pixel_coords": pixel_coords,
            }

        return mapping

    def compute_dino_embeddings(
        self,
        obj_id: int,
        point_mapping: Dict[int, Dict[str, np.ndarray]],
        model_name: str = "dinov2",
    ) -> torch.Tensor:
        # compute dino embeddings for the object
        template_embeddings = dino_embeddings(
            self.template_dataset[self.obj_id_to_index(obj_id)].rgb,
            batch_size=self.dino_batch_size,
            model_name=model_name,
        )
        assert template_embeddings.isnan().sum() == 0

        # initialize point embeddings and point accum count
        point_embeddings = torch.zeros((self.N, template_embeddings.shape[-1]))
        point_accum_count = torch.zeros((self.N, 1))

        # aggregate dino embeddings for points from each template
        for t_id, m in tqdm(
            point_mapping.items(), desc=f"aggregating {model_name} embeddings"
        ):
            # get interpolated embedding for the point
            interpolated_embedding = embedding_interpolation(
                template_embeddings[[t_id]],
                torch.tensor(m["pixel_coords"], dtype=torch.float32),
                model_name=model_name,
            )
            # aggregate dino embeddings for the point
            point_embeddings[m["point_mapping"]] += interpolated_embedding[0]
            point_accum_count[m["point_mapping"], 0] += 1

        # normalize point embeddings by point accum count
        assert point_embeddings.isnan().sum() == 0
        point_embeddings /= torch.where(point_accum_count > 0, point_accum_count, 1)

        return point_embeddings

    def preprocess_point_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # initialize lists to store point embeddings and points for each object
        point_embeddings_list = []
        point_list = []
        for obj_id in tqdm(self.model_infos, desc="preprocessing point embeddings"):
            obj_id = obj_id["obj_id"]

            # sample points and mesh for the object
            points, mesh = self.sample_points(obj_id)

            # map sampled points to template ids
            mapping = self.map_points_to_templates(obj_id, points, mesh)

            # aggregate dino embeddings for points from each template
            point_embeddings = self.compute_dino_embeddings(obj_id, mapping)

            # store point embeddings and points for the object
            point_embeddings_list.append(point_embeddings)
            point_list.append(
                torch.tensor(
                    points, dtype=torch.float32, device=point_embeddings.device
                )
            )

        # stack point embeddings and points for all objects
        point_embeddings = torch.stack(point_embeddings_list, dim=0)
        points = torch.stack(point_list, dim=0)

        # PCA on point embeddings
        self.pca.fit(point_embeddings)
        point_embeddings = self.pca(point_embeddings)

        # # normalize point embeddings
        # point_embeddings = F.normalize(point_embeddings, dim=-1)

        return point_embeddings, points

    def __len__(self) -> int:
        return len(self.model_infos)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.point_embeddings[idx], self.points[idx]


if __name__ == "__main__":
    from hydra import compose, initialize
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    with initialize(config_path="../configs"):
        cfg = compose(config_name="test.yaml")
    OmegaConf.set_struct(cfg, False)

    print(f"Using {cfg.test_dataset_name} dataset")
    cfg.data.test.dataloader.dataset_name = cfg.test_dataset_name
    cfg.data.test.dataloader.batch_size = cfg.machine.batch_size
    cfg.data.test.dataloader.load_gt = False
    cfg.data.test.dataloader.test_setting = cfg.test_setting
    cfg.data.test.dataloader._target_ = "src.dataloader.template.TemplateSet"
    template_dataset = instantiate(cfg.data.test.dataloader)
    template_set = CustomTemplateSet(
        cfg, template_dataset, dino_components=64, dino_batch_size=40
    )

    print(f"point_embeddings.shape: {template_set.point_embeddings.shape}")
    print(f"points.shape: {template_set.points.shape}")
