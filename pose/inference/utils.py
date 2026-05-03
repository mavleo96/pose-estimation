from pathlib import Path

import numpy as np
import torch
import trimesh
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.template import TemplateViewDataset
from ..data.transforms import imagenet_transform
from ..utils.geometry import get_visible_point_projections
from ..utils.image import sample_patch_centers
from ..utils.operations import interpolate_embeddings, interpolate_embeddings_conv


def precompute_embeddings(
    dataset_name,
    data_dir,
    template_dir,
    obj_ids,
    model_wrapper,
    batch_size,
    embeddings_dir,
    embedding_prefix,
    output_size,
    depth_scale,
    num_workers=4,
    rgb_transform=None,
):
    """
    Run encoder forward pass over all template views and save patch embeddings to disk.

    Args:
        dataset_name  : str         — BOP dataset name (e.g. "lmo", "ycbv")
        data_dir      : str | Path  — root data directory
        template_dir  : str | Path  — template directory
        obj_ids       : list[int]   — object ids to precompute embeddings for
        model_wrapper : callable    — function that takes a batch and returns patch embeddings
        batch_size    : int         — number of template views per forward pass
        embeddings_dir: str | Path  — output directory for embeddings and templates
        embedding_prefix: str       — prefix for the embedding filename
        output_size   : tuple[int]  — (H, W) output size for template images
        depth_scale   : float       — depth scale to convert raw uint16 depth to mm
        num_workers   : int         — number of DataLoader workers. Default: 4.
        rgb_transform : callable    — function that takes a rgb image and returns a tensor.
                        If None, uses imagenet_transform. Default: None.

    Saves:
        embeddings_dir/{prefix}_image_embeddings.npy — (N_objects, N_views, N_patches, D) float32
        embeddings_dir/{prefix}_templates.npz      — M, rgb, depth arrays
    """
    embeddings_dir = Path(embeddings_dir)
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    rgb_transform = rgb_transform or imagenet_transform

    templates_path = Path(template_dir) / dataset_name

    embeddings_list = []
    M_list = []
    # rgb_list = []
    # depth_list = []
    mask_list = []

    for obj_id in obj_ids:
        template_dataset = TemplateViewDataset(
            templates_path, obj_id, output_size, depth_scale, rgb_transform
        )
        dataloader = DataLoader(
            template_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        object_embeddings_list = []
        object_M_list = []
        # object_rgb_list = []
        # object_depth_list = []
        object_mask_list = []

        for batch in tqdm(dataloader, desc=f"precomputing embeddings for obj {obj_id:06d}"):
            patch_embeddings = model_wrapper(batch)

            object_embeddings_list.append(patch_embeddings.float().cpu().numpy())
            object_M_list.append(batch["M"].numpy())
            # object_rgb_list.append(batch["rgb"].numpy())
            # object_depth_list.append(batch["depth"].numpy())
            object_mask_list.append(batch["mask"].numpy())

        object_embeddings = np.concatenate(object_embeddings_list, axis=0)
        object_M = np.concatenate(object_M_list, axis=0)
        # object_rgb = np.concatenate(object_rgb_list, axis=0)
        # object_depth = np.concatenate(object_depth_list, axis=0)
        object_mask = np.concatenate(object_mask_list, axis=0)

        embeddings_list.append(object_embeddings)
        M_list.append(object_M)
        # rgb_list.append(object_rgb)
        # depth_list.append(object_depth)
        mask_list.append(object_mask)

    embeddings = np.stack(embeddings_list, axis=0)
    M = np.stack(M_list, axis=0)
    # rgb = np.stack(rgb_list, axis=0)
    # depth = np.stack(depth_list, axis=0)
    mask = np.stack(mask_list, axis=0)

    np.save(embeddings_dir / f"{embedding_prefix}_image_embeddings.npy", embeddings)
    np.savez(
        embeddings_dir / f"{embedding_prefix}_templates.npz",
        M=M,
        mask=mask,  # rgb=rgb, depth=depth
    )


def precompute_pca(embeddings_dir, embedding_prefix, pca, patch_size):
    """
    Fit PCA on precomputed embeddings and save projected embeddings to disk.

    Requires precompute_embeddings to have been run first.

    Args:
        embeddings_dir: str | Path  — directory containing {prefix}_image_embeddings.npy
        embedding_prefix: str       — prefix for the embedding filename
        pca           : PCA         — unfitted PCA module (nn.Module with fit/forward)
        patch_size    : int         — patch size used to downsample the masks

    Saves:
        embeddings_dir/pca_embeddings.npy — (N_objects, N_views, N_patches, D_pca) float32
        embeddings_dir/pca.pt             — PCA state dict (mean, eigenvectors)
    """
    embeddings_dir = Path(embeddings_dir)

    embeddings = torch.from_numpy(
        np.load(embeddings_dir / f"{embedding_prefix}_image_embeddings.npy")
    ).float()
    with np.load(embeddings_dir / f"{embedding_prefix}_templates.npz") as templates:
        mask = torch.from_numpy(templates["mask"])
        mask = sample_patch_centers(mask, patch_size).flatten(-2)

    pca.fit(embeddings[mask])

    pca_embeddings_list = []
    for i in range(embeddings.shape[0]):
        pca_embedding = pca(embeddings[i].to(pca.device, dtype=pca.dtype))
        pca_embeddings_list.append(pca_embedding.float().cpu().numpy())

    pca_embeddings = np.stack(pca_embeddings_list, axis=0)

    # delete original embeddings
    (embeddings_dir / f"{embedding_prefix}_image_embeddings.npy").unlink()

    np.save(embeddings_dir / f"{embedding_prefix}_pca_embeddings.npy", pca_embeddings)
    torch.save(pca.state_dict(), embeddings_dir / f"{embedding_prefix}_pca.pt")


# def precompute_pca2(embeddings_dir, pca):
#     """
#     Fit PCA on precomputed point embeddings and save projected embeddings to disk.

#     Args:
#         embeddings_dir: str | Path  — directory containing point_embeddings.npy
#         pca           : PCA         — unfitted PCA module (nn.Module with fit/forward)

#     Saves:
#         embeddings_dir/point_pca_embeddings.npy — (N_objects, N_points, D_pca) float32
#         embeddings_dir/pca2.pt             — PCA state dict (mean, eigenvectors)
#     """
#     embeddings_dir = Path(embeddings_dir)

#     embeddings = torch.from_numpy(np.load(embeddings_dir / "point_embeddings.npy")).float()
#     pca.fit(embeddings.flatten(start_dim=0, end_dim=1))

#     pca_embeddings_list = []
#     for i in range(embeddings.shape[0]):
#         pca_embedding = pca(embeddings[i].to(pca.device, dtype=pca.dtype))
#         pca_embeddings_list.append(pca_embedding.float().cpu().numpy())

#     pca_embeddings = np.stack(pca_embeddings_list, axis=0)

#     np.save(embeddings_dir / "point_pca_embeddings.npy", pca_embeddings)
#     torch.save(pca.state_dict(), embeddings_dir / "pca2.pt")


def precompute_point_embeddings(
    dataset_name,
    data_dir,
    template_dir,
    embeddings_dir,
    obj_ids,
    K,
    image_size,
    patch_size,
    num_points=1000,
    use_interpolate_embeddings_conv=False,
    template_prefix="dino",
):
    """
    Precompute point embeddings for each object and save to disk.

    Args:
        dataset_name: str         — BOP dataset name (e.g. "lmo", "ycbv")
        data_dir: str | Path      — root data directory
        template_dir: str | Path  — template directory
        embeddings_dir: str | Path — directory containing embeddings.npy
        obj_ids: list[int]        — object ids to precompute point embeddings for
        K: np.ndarray             — camera intrinsics
        image_size: tuple[int]    — (H, W) image size
        patch_size: int           — patch size used to downsample the masks
                                    (or effective stride)
        num_points: int           — number of points to sample from the object mesh
        use_interpolate_embeddings_conv: bool — whether to use interpolate_embeddings_conv
                                instead of interpolate_embeddings. Default: False.
        template_prefix: str       — prefix for the template embeddings filename

    Saves:
        embeddings_dir/points.npy    — (N_objects, N_points, 3) float32
        embeddings_dir/point_embeddings.npy — (N_objects, N_points, D) float32
    """
    templates_path = Path(template_dir) / dataset_name
    data_path = Path(data_dir) / dataset_name

    template_embeddings = torch.from_numpy(np.load(embeddings_dir / "pca_embeddings.npy")).float()
    _, T, _, D = template_embeddings.shape

    # create camera intrinsics adjusted for template resize
    with np.load(embeddings_dir / f"{template_prefix}_templates.npz") as templates:
        template_M = templates["M"]
    K_adj = template_M @ K

    # interpolate function
    interpolate_func = (
        interpolate_embeddings_conv if use_interpolate_embeddings_conv else interpolate_embeddings
    )

    # precompute point embeddings for each object
    points_list = []
    point_embeddings_list = []
    for i, obj_id in enumerate(obj_ids):
        # 1. load model and poses
        model = trimesh.load(data_path / "models" / f"obj_{obj_id:06d}.ply", process=False)
        T_m2c = np.load(templates_path / "object_poses" / f"{obj_id:06d}.npy")

        # 2. sample points from the object mesh
        points, _ = trimesh.sample.sample_surface(model, num_points)

        # 3. initialize point embeddings and point accum count
        point_embeddings = torch.zeros((num_points, D))
        point_accum_count = torch.zeros((num_points, 1))

        # 4. precompute point embeddings for each template
        for tid in tqdm(range(T), desc=f"precomputing point embeddings for obj {obj_id:06d}"):
            # a. get points visible from given pose and 2d projection
            point_mapping, pixel_coords = get_visible_point_projections(
                points, model, K_adj[i, tid], T_m2c[tid]
            )

            # b. interpolate embedding for the point
            interpolated_embedding = interpolate_func(
                template_embeddings[i, tid],
                torch.tensor(pixel_coords),
                patch_size,
                image_size,
            )

            # c. accumulate embeddings for the points
            point_embeddings[point_mapping] += interpolated_embedding
            point_accum_count[point_mapping, 0] += 1

        # normalize point embeddings by point accum count
        point_embeddings /= torch.where(point_accum_count > 0, point_accum_count, 1)

        points_list.append(points)
        point_embeddings_list.append(point_embeddings.numpy())

    points = np.stack(points_list, axis=0)
    point_embeddings = np.stack(point_embeddings_list, axis=0)

    np.save(embeddings_dir / "points.npy", points)
    np.save(embeddings_dir / "point_embeddings.npy", point_embeddings)


def combine_embeddings(alpha, embeddings_dir, prefix1, prefix2):
    embeddings_dir = Path(embeddings_dir)

    embeddings1 = np.load(embeddings_dir / f"{prefix1}_pca_embeddings.npy")
    embeddings2 = np.load(embeddings_dir / f"{prefix2}_pca_embeddings.npy")

    embeddings1 = embeddings1 / np.linalg.norm(embeddings1, axis=-1, keepdims=True)
    embeddings2 = embeddings2 / np.linalg.norm(embeddings2, axis=-1, keepdims=True)

    combined_embeddings = np.concatenate([alpha * embeddings1, (1 - alpha) * embeddings2], axis=-1)

    np.save(embeddings_dir / "pca_embeddings.npy", combined_embeddings)


# def precompute_geometric_embeddings(
#     dataset_name,
#     data_dir,
#     embeddings_dir,
#     obj_ids,
#     model_wrapper,
#     num_points=5000,
# ):
#     data_path = Path(data_dir) / dataset_name

#     # precompute embeddings for each object
#     points_list = []
#     point_embeddings_list = []
#     for obj_id in obj_ids:
#         # 1. load model and poses
#         model = trimesh.load(data_path / "models" / f"obj_{obj_id:06d}.ply", process=False)

#         # 2. sample points and normals from the object mesh
#         coords, face_idx, colors = trimesh.sample.sample_surface(
#             model, num_points, sample_color=True
#         )
#         normals = model.face_normals[face_idx]
#         points = {
#             "coord": np.array(coords),
#             "normal": np.array(normals),
#             "color": np.array(colors)[:, :3],
#         }

#         point_embeddings = model_wrapper(points)
#         points_list.append(coords)
#         point_embeddings_list.append(point_embeddings.float().cpu().numpy())
#         logger.info(f"Precomputed geometric embeddings for obj {obj_id:06d}")

#     points = np.stack(points_list, axis=0)
#     point_embeddings = np.stack(point_embeddings_list, axis=0)

#     np.save(embeddings_dir / "points.npy", points)
#     np.save(embeddings_dir / "point_embeddings.npy", point_embeddings)
