import logging
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from _configs.constants import DATA_DIR

from ..data import ImageGroupedBatchSampler, QueryDataCollator, QueryDataset, TemplateDataset
from ..matching import PatchCorrespondenceFinder
from ..model import MODEL_MAP, PCA
from ..pose_recovery import PnPPoseEstimator
from ..rendering import TemplateRenderer
from ..utils.geometry import backproject_points
from ..utils.image import apply_inverse_homography, make_pixel_grid, sample_patch_centers
from .utils import precompute_embeddings, precompute_pca

logger = logging.getLogger(__name__)


def pnp_pipeline(
    dataset_name: str,
    detections_path: Path,
    results_dir: Path,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
    output_size: tuple[int, int] = (518, 518),
    batch_size: int = 4,
    embedding_model: str = "dinov2",
    embedding_batch_size: int = 20,
    sim_threshold: float = 0.5,
    patch_threshold: float = 3.0,
    k_hypotheses: int = 5,
) -> None:
    data_root = Path(DATA_DIR)
    embeddings_dir = results_dir / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = results_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------- ONBOARDING -----------------------------
    onboarding_start = time.time()

    # Render templates
    logger.info("Rendering templates...")
    renderer = TemplateRenderer(dataset_name, data_root)
    renderer.render_templates()

    # Load embedding model and PCA
    logger.info("Loading embedding model and PCA...")
    model = MODEL_MAP[embedding_model]()
    pca = PCA(d=model.dim, n_components=256)
    model.to(device, dtype)
    pca.to(device, dtype)
    logger.info(model)
    logger.info(pca)

    # Precompute embeddings and PCA
    logger.info("Precomputing embeddings and PCA...")
    precompute_embeddings(
        dataset_name=dataset_name,
        data_dir=data_root,
        obj_ids=renderer.obj_ids,
        encoder=model,
        batch_size=embedding_batch_size,
        embeddings_dir=embeddings_dir,
        output_size=output_size,
        depth_scale=renderer.camera.depth_scale,
    )
    precompute_pca(embeddings_dir=embeddings_dir, pca=pca)
    # pca.load_state_dict(torch.load(embeddings_dir / "pca.pt"))

    onboarding_time = time.time() - onboarding_start
    logger.info(
        f"Time taken: {onboarding_time:.2f} seconds;"
        f" {onboarding_time / len(renderer.obj_ids):.2f} seconds per object"
    )

    # ----------------------------- INFERENCE -----------------------------
    # Initialize matcher and pose estimator
    logger.info("Initializing matcher and pose estimator...")
    matcher = PatchCorrespondenceFinder(
        sim_threshold=sim_threshold, patch_threshold=patch_threshold
    )
    pose_estimator = PnPPoseEstimator()
    logger.info(matcher)
    logger.info(pose_estimator)

    # Load template and query datasets
    logger.info("Loading template and query datasets...")
    template_dataset = TemplateDataset(dataset_name, data_root, results_dir / "embeddings")
    query_dataset = QueryDataset(dataset_name, data_root, detections_path, output_size)

    sampler = ImageGroupedBatchSampler(query_dataset, batch_size=batch_size)
    collator = QueryDataCollator()
    dataloader = DataLoader(query_dataset, batch_sampler=sampler, collate_fn=collator)

    # Initialize feature size and grid based on output size
    feat_size = (output_size[0] // model.patch_size, output_size[1] // model.patch_size)
    grid2d = (
        make_pixel_grid(*feat_size, device=device).reshape(-1, 2) * model.patch_size
        + model.patch_size // 2
    )
    cam_K = template_dataset.cam_K.to(device)
    cam_K_cpu = cam_K.cpu()

    # Run inference loop
    logger.info("Running inference loop...")
    results_dict = {}
    inference_start = time.time()
    for batch in tqdm(dataloader, desc="Estimating poses...", total=len(dataloader)):
        time_start = time.time()
        B = len(batch["obj_id"])
        batch_idx = torch.arange(B, device=device)[:, None]

        # Get template data using obj_id detected
        templates = template_dataset[template_dataset.get_index(batch["obj_id"])]
        template_embedding = templates["embedding"].to(device, dtype)
        template_depth = (
            sample_patch_centers(templates["depth"], model.patch_size)
            .flatten(start_dim=-2)
            .to(device)
        )
        template_mask = (
            sample_patch_centers(templates["mask"], model.patch_size)
            .flatten(start_dim=-2)
            .to(device)
        )
        template_M = templates["M"].to(device)

        # Get query data
        query_rgb = batch["rgb"].to(model.device, dtype=model.dtype)
        query_mask = (
            sample_patch_centers(batch["mask"], model.patch_size).flatten(start_dim=-2).to(device)
        )
        query_M = batch["M"].to(device)

        # Compute query embeddings
        with torch.no_grad():
            query_embeddings = model(query_rgb)["patch_embeddings"]
            query_embeddings = pca(query_embeddings)

        del query_rgb

        # Compute patch correspondences
        output = matcher(query_embeddings, template_embedding, query_mask, template_mask, feat_size)
        # assert output["score_tar2src"].sum(dim=-1).isinf().sum() == 0

        del query_embeddings, template_embedding, query_mask, template_mask

        # Get top k template correspondences
        topk_template_indices = torch.topk(
            output["score_tar2src"].sum(dim=-1), k=k_hypotheses
        ).indices
        topk_tar2src = output["idx_tar2src"][batch_idx, topk_template_indices]
        topk_corr_mask = output["mask"][batch_idx, topk_template_indices]
        topk_template_depth = template_depth[batch_idx, topk_template_indices]
        topk_template_M = template_M[batch_idx, topk_template_indices]

        # Compute 2D query points and 3D template points
        tar_pts2d = apply_inverse_homography(grid2d.expand(B, -1, -1).float(), query_M)
        K_adjusted = topk_template_M @ cam_K[None, None, ...]
        template_pts3d = backproject_points(
            grid2d.expand(B, k_hypotheses, -1, -1).float(),
            topk_template_depth,
            K_adjusted,
        )
        src_pts3d = template_pts3d.gather(
            dim=2, index=topk_tar2src.unsqueeze(-1).expand(-1, -1, -1, 3)
        )

        # Move tensors to CPU for pose estimation
        tar_pts2d = tar_pts2d.to("cpu")
        src_pts3d = src_pts3d.to("cpu")
        topk_corr_mask = topk_corr_mask.to("cpu")
        batch_idx = batch_idx.to("cpu")

        # Estimate pose for each template correspondence
        R_est_list = []
        t_est_list = []
        cand_scores = []
        for i in range(k_hypotheses):
            cand_R_est, cand_t_est, cand_score = pose_estimator.estimate_pose(
                src_pts3d[:, i], tar_pts2d, cam_K_cpu.expand(B, -1, -1), topk_corr_mask[:, i]
            )
            R_est_list.append(cand_R_est)
            t_est_list.append(cand_t_est)
            cand_scores.append(cand_score)
        R_est = torch.stack(R_est_list, dim=1)
        t_est = torch.stack(t_est_list, dim=1).unsqueeze(-1)
        cand_scores = torch.stack(cand_scores, dim=1)
        inlier_counts = cand_scores * topk_corr_mask.sum(dim=-1)

        # Get best template correspondence
        best_idx = inlier_counts.argmax(dim=-1).unsqueeze(-1)
        best_R_est = R_est[batch_idx, best_idx][:, 0].cpu().numpy()
        best_t_est = t_est[batch_idx, best_idx][:, 0].cpu().numpy()

        time_taken = time.time() - time_start

        image_key = f"{batch['scene_id']:06d}_{batch['im_id']:06d}"
        if image_key not in results_dict:
            results_dict[image_key] = {
                "scene_id": batch["scene_id"],
                "im_id": batch["im_id"],
                "detection_time": np.mean(batch["detection_time"]),
                "results": {
                    "obj_id": [],
                    "detection_score": [],
                    "R": [],
                    "t": [],
                    "pose_estimation_time": [],
                },
            }
        results_dict[image_key]["results"]["obj_id"].extend(batch["obj_id"])
        results_dict[image_key]["results"]["R"].extend([best_R_est[i] for i in range(B)])
        results_dict[image_key]["results"]["t"].extend([best_t_est[i] for i in range(B)])
        results_dict[image_key]["results"]["detection_score"].extend(batch["detection_score"])
        results_dict[image_key]["results"]["pose_estimation_time"].extend([time_taken / B] * B)

    inference_time = time.time() - inference_start
    logger.info(
        f"Inference time: {inference_time:.2f} seconds;"
        f" {inference_time / len(query_dataset.image_info):.2f} seconds per image"
    )

    # ----------------------------- SAVING -----------------------------

    # Clean up
    shutil.rmtree(embeddings_dir)

    # Save predictions
    logger.info("Saving predictions...")
    for image_key, result in results_dict.items():
        np.savez(pred_dir / f"{image_key}.npz", **result)
