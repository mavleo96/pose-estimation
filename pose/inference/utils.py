import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.template import TemplateViewDataset

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def precompute_embeddings(
    dataset_name,
    data_dir,
    obj_ids,
    encoder,
    batch_size,
    embeddings_dir,
    output_size,
    depth_scale,
    num_workers=4,
):
    """
    Run encoder forward pass over all template views and save patch embeddings to disk.

    Args:
        dataset_name  : str         — BOP dataset name (e.g. "lmo", "ycbv")
        data_dir      : str | Path  — root data directory
        obj_ids       : list[int]   — object ids to precompute embeddings for
        encoder       : nn.Module   — embedding model with forward returning "patch_embeddings"
        batch_size    : int         — number of template views per forward pass
        embeddings_dir: str | Path  — output directory for embeddings and templates
        output_size   : tuple[int]  — (H, W) output size for template images
        depth_scale   : float       — depth scale to convert raw uint16 depth to mm
        num_workers   : int         — number of DataLoader workers. Default: 4.

    Saves:
        embeddings_dir/embeddings.npy     — (N_objects, N_views, N_patches, D) float32
        embeddings_dir/templates.npz      — M, rgb, depth arrays
    """
    embeddings_dir = Path(embeddings_dir)
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    templates_path = Path(data_dir) / "templates" / dataset_name

    encoder.eval()

    embeddings_list = []
    M_list = []
    rgb_list = []
    depth_list = []
    mask_list = []

    for obj_id in obj_ids:
        template_dataset = TemplateViewDataset(templates_path, obj_id, output_size, depth_scale)
        dataloader = DataLoader(
            template_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        object_embeddings_list = []
        object_M_list = []
        object_rgb_list = []
        object_depth_list = []
        object_mask_list = []

        for batch in tqdm(dataloader, desc=f"precomputing embeddings for obj {obj_id:06d}"):
            rgb = batch["rgb"].to(encoder.device, dtype=encoder.dtype)
            with torch.no_grad():
                patch_embeddings = encoder(rgb)["patch_embeddings"]

            object_embeddings_list.append(patch_embeddings.float().cpu().numpy())
            object_M_list.append(batch["M"].numpy())
            object_rgb_list.append(batch["rgb"].numpy())
            object_depth_list.append(batch["depth"].numpy())
            object_mask_list.append(batch["mask"].numpy())

        object_embeddings = np.concatenate(object_embeddings_list, axis=0)
        object_M = np.concatenate(object_M_list, axis=0)
        object_rgb = np.concatenate(object_rgb_list, axis=0)
        object_depth = np.concatenate(object_depth_list, axis=0)
        object_mask = np.concatenate(object_mask_list, axis=0)

        embeddings_list.append(object_embeddings)
        M_list.append(object_M)
        rgb_list.append(object_rgb)
        depth_list.append(object_depth)
        mask_list.append(object_mask)

    embeddings = np.stack(embeddings_list, axis=0)
    M = np.stack(M_list, axis=0)
    rgb = np.stack(rgb_list, axis=0)
    depth = np.stack(depth_list, axis=0)
    mask = np.stack(mask_list, axis=0)

    np.save(embeddings_dir / "embeddings.npy", embeddings)
    np.savez(embeddings_dir / "templates.npz", M=M, rgb=rgb, depth=depth, mask=mask)


def precompute_pca(embeddings_dir, pca):
    """
    Fit PCA on precomputed embeddings and save projected embeddings to disk.

    Requires precompute_embeddings to have been run first.

    Args:
        embeddings_dir: str | Path  — directory containing embeddings.npy
        pca           : PCA         — unfitted PCA module (nn.Module with fit/forward)

    Saves:
        embeddings_dir/pca_embeddings.npy — (N_objects, N_views, N_patches, D_pca) float32
        embeddings_dir/pca.pt             — PCA state dict (mean, eigenvectors)
    """
    embeddings_dir = Path(embeddings_dir)

    embeddings = torch.from_numpy(np.load(embeddings_dir / "embeddings.npy")).float()

    pca.fit(embeddings)

    pca_embeddings_list = []
    for i in range(embeddings.shape[0]):
        pca_embedding = pca(embeddings[i].to(pca.device, dtype=pca.dtype))
        pca_embeddings_list.append(pca_embedding.float().cpu().numpy())

    pca_embeddings = np.stack(pca_embeddings_list, axis=0)

    np.save(embeddings_dir / "pca_embeddings.npy", pca_embeddings)
    torch.save(pca.state_dict(), embeddings_dir / "pca.pt")
