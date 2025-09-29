from typing import List

import torch
from einops import einsum

from pcr.utils import dino_embeddings, embedding_interpolation
from src.megapose.datasets.scene_dataset import SceneObservation


class CustomTestSet:
    N = 5000

    def __init__(self, cfg, dataset, pca, **kwargs):
        self.dataset = dataset
        self.dataset_name = dataset.dataset_name

        if "dino_batch_size" in kwargs:
            self.dino_batch_size = kwargs["dino_batch_size"]
        else:
            self.dino_batch_size = 20

        self.root_dir = cfg.data.test.root_dir

        self.pca = pca

    def sample_pixels_from_masks(self, mask: torch.Tensor) -> torch.Tensor:
        pixels = torch.zeros((mask.shape[0], self.N, 2), dtype=torch.int32)
        for i in range(mask.shape[0]):
            # randomly sample N pixels from the mask
            # TODO: handle the case where the mask has less than N pixels
            inside_mask_indices = torch.argwhere(mask[i])
            indices = torch.randperm(inside_mask_indices.shape[0])[: self.N]
            x, y = inside_mask_indices[indices, 1], inside_mask_indices[indices, 0]
            pixels[i, :, 0] = x
            pixels[i, :, 1] = y
        return pixels

    def generate_point_cloud(
        self,
        pixels: torch.Tensor,
        depth: torch.Tensor,
        K: torch.Tensor,
        M: torch.Tensor,
    ) -> torch.Tensor:
        # get depth at the sampled pixels
        z = torch.zeros((pixels.shape[0], self.N, 1), dtype=torch.float32)
        for i in range(pixels.shape[0]):
            z_i = depth[i][pixels[i, :, 1], pixels[i, :, 0]]
            z[i, :, 0] = z_i

        # invert the transformation matrix - M @ K
        T = einsum(M, K, "b n m, b m p -> b n p")
        T_inv = torch.linalg.inv(T.permute(0, 2, 1))

        # convert pixels to point
        pixels = torch.cat([pixels, torch.ones_like(pixels[:, :, 0:1])], dim=-1).float()
        points = einsum(pixels, T_inv, "b n c, b c p -> b n p") * z
        return points

    def compute_dino_embeddings(
        self,
        images: torch.Tensor,
        pixel_coords: torch.Tensor,
        model_name: str = "dinov2",
    ) -> torch.Tensor:
        # get dino embeddings for image batch
        embeddings = dino_embeddings(
            images, batch_size=self.dino_batch_size, model_name=model_name
        )
        b, _, d = embeddings.shape
        assert embeddings.isnan().sum() == 0

        # initialize point embeddings
        point_embeddings = torch.zeros((b, self.N, d))
        for i in range(b):
            # get dino embeddings for each pixel by interpolating
            point_embeddings[i] = embedding_interpolation(
                embeddings[[i]], pixel_coords[i], model_name=model_name
            )

        assert point_embeddings.isnan().sum() == 0
        point_embeddings = self.pca(point_embeddings)

        return point_embeddings

    def collate_fn(self, batch: List[SceneObservation]) -> SceneObservation:
        # TODO: Critical bug: called twice
        batch = self.dataset.collate_fn(batch)

        # filter to include mask with highest visible frac
        test_ids = batch.infos.drop_duplicates(subset="label", keep="first").index
        assert len(test_ids) == batch.test_list.infos.shape[0]

        batch.tar_img = batch.tar_img[test_ids]
        batch.tar_mask = batch.tar_mask[test_ids]
        batch.tar_depth = batch.tar_depth[test_ids]
        batch.tar_K = batch.tar_K[test_ids]
        batch.tar_M = batch.tar_M[test_ids]

        batch.register_tensor("pixels", self.sample_pixels_from_masks(batch.tar_mask))
        batch.register_tensor(
            "points",
            self.generate_point_cloud(
                batch.pixels, batch.tar_depth, batch.tar_K, batch.tar_M
            ),
        )
        batch.register_tensor(
            "point_embeddings",
            self.compute_dino_embeddings(batch.tar_img, batch.pixels),
        )

        return batch


if __name__ == "__main__":
    from hydra import compose, initialize
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader

    from pcr.pca import PCA

    with initialize(config_path="../configs"):
        cfg = compose(config_name="test.yaml")
    OmegaConf.set_struct(cfg, False)

    print(f"Using {cfg.test_dataset_name} dataset")
    cfg.data.test.dataloader.dataset_name = cfg.test_dataset_name
    cfg.data.test.dataloader.batch_size = cfg.machine.batch_size
    cfg.data.test.dataloader.load_gt = False
    cfg.data.test.dataloader.test_setting = cfg.test_setting
    test_dataset = instantiate(cfg.data.test.dataloader)

    pca = PCA(1024, 10)
    pca.fit(torch.randn(1024, 1024))

    test_set = CustomTestSet(cfg, test_dataset, pca, dino_batch_size=40)
    dataloader = DataLoader(
        test_dataset.web_dataloader.datapipeline,
        batch_size=1,
        num_workers=cfg.machine.num_workers,
        shuffle=False,
        collate_fn=test_set.collate_fn,
    )

    for batch in dataloader:
        print(f"batch.points.shape: {batch.points.shape}")
        print(f"batch.point_embeddings.shape: {batch.point_embeddings.shape}")
        break
