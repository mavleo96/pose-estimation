import os
import trimesh
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from trimesh.ray.ray_pyembree import RayMeshIntersector
from pcr.utils import dino_embeddings, embedding_interpolation


class CustomTemplateSet(Dataset):
    N = 1000

    def __init__(self, cfg, template_dataset, **kwargs):
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

        self.point_embeddings, self.points = self.preprocess_point_embeddings()

    def obj_id_to_index(self, obj_id):
        return [
            i for i, info in enumerate(self.model_infos) if info["obj_id"] == obj_id
        ][0]

    def sample_points(self, obj_id):
        np.random.seed(0)
        obj_path = os.path.join(self.cad_models_dir, f"obj_{obj_id:06d}.ply")
        mesh = trimesh.load(obj_path)
        points, _ = trimesh.sample.sample_surface(mesh, self.N)
        return np.asarray(points), mesh

    def map_points_to_templates(self, obj_id, points, mesh):
        K = self.template_dataset[self.obj_id_to_index(obj_id)].K.numpy()
        pose_array = self.template_dataset[self.obj_id_to_index(obj_id)].poses.numpy()
        M_array = self.template_dataset[self.obj_id_to_index(obj_id)].M.numpy()

        mapping = {}
        for i, (pose, M) in enumerate(zip(pose_array, M_array)):
            R = pose[:3, :3]
            t = pose[:3, 3]

            p_cam = points @ R.T + t

            intersector = RayMeshIntersector(mesh.copy().apply_transform(pose))
            locations, index_ray, _ = intersector.intersects_location(
                np.zeros_like(p_cam), p_cam, multiple_hits=False
            )

            visible_ray = np.full(points.shape[0], False)
            visible_ray[index_ray] = (
                np.linalg.norm(locations - p_cam[index_ray], axis=1) < 1e-6
            )
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

    def compute_dino_embeddings(self, obj_id, point_mapping, model_name="dinov2"):
        template_embeddings = dino_embeddings(
            self.template_dataset[self.obj_id_to_index(obj_id)].rgb,
            batch_size=self.dino_batch_size,
            model_name=model_name,
        )

        t, _, d = template_embeddings.shape

        point_embeddings = torch.zeros((self.N, d))
        point_accum_count = torch.zeros((self.N, 1))

        for t_id, m in tqdm(
            point_mapping.items(), desc=f"aggregating {model_name} embeddings"
        ):
            interpolated_embedding = embedding_interpolation(
                template_embeddings[[t_id]],
                torch.tensor(m["pixel_coords"], dtype=torch.float32),
                model_name=model_name,
            )
            point_embeddings[m["point_mapping"]] += interpolated_embedding[0]
            point_accum_count[m["point_mapping"], 0] += 1

        point_embeddings /= point_accum_count
        return point_embeddings

    def preprocess_point_embeddings(self):
        point_embeddings_list = []
        point_list = []
        for obj_id in tqdm(self.model_infos, desc="preprocessing point embeddings"):
            obj_id = obj_id["obj_id"]
            points, mesh = self.sample_points(obj_id)
            mapping = self.map_points_to_templates(obj_id, points, mesh)
            point_embeddings = self.compute_dino_embeddings(obj_id, mapping)
            point_embeddings_list.append(point_embeddings)
            point_list.append(
                torch.tensor(
                    points, dtype=torch.float32, device=point_embeddings.device
                )
            )
        point_embeddings = torch.stack(point_embeddings_list, dim=0)
        points = torch.stack(point_list, dim=0)
        return point_embeddings, points

    def __len__(self):
        return len(self.model_infos)

    def __getitem__(self, idx):
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
    template_set = CustomTemplateSet(cfg, template_dataset, dino_batch_size=40)

    print(f"point_embeddings.shape: {template_set.point_embeddings.shape}")
    print(f"points.shape: {template_set.points.shape}")
