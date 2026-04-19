import argparse
import json
import logging
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from _configs.constants import DATA_DIR

from ..data import ImageGroupedBatchSampler, ObjectPointDataset, QueryDataCollator, QueryDataset
from ..matching import PointCorrespondenceFinder
from ..model import MODEL_MAP, PCA
from ..pose_recovery import KabschPoseEstimator
from ..rendering import TemplateRenderer
from ..utils.geometry import backproject_points
from ..utils.image import sample_pixels_from_masks
from ..utils.io import save_bop_results_from_predictions
from ..utils.operations import interpolate_embeddings
from .utils import precompute_embeddings, precompute_pca, precompute_point_embeddings

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def kabsch_pipeline(
    dataset_name: str,
    detections_path: Path,
    results_dir: Path,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    output_size: tuple[int, int] = (224, 224),
    batch_size: int = 4,
    n_src_points: int = 5000,
    n_tar_points: int = 1000,
    embedding_model: str = "dinov2",
    embedding_batch_size: int = 20,
    sim_threshold: float = 0.0,
    render_templates: bool = True,
) -> None:
    data_root = Path(DATA_DIR)
    embeddings_dir = results_dir / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = results_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------- ONBOARDING -----------------------------
    onboarding_start = time.time()

    # Render templates
    renderer = TemplateRenderer(dataset_name, data_root)
    if render_templates:
        logger.info("Rendering templates...")
        renderer.render_templates()
    else:
        logger.info("Using existing templates...")

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
    precompute_pca(embeddings_dir=embeddings_dir, pca=pca, patch_size=model.patch_size)
    precompute_point_embeddings(
        dataset_name=dataset_name,
        data_dir=data_root,
        embeddings_dir=embeddings_dir,
        obj_ids=renderer.obj_ids,
        K=renderer.camera.K,
        image_size=output_size,
        patch_size=model.patch_size,
        num_points=n_src_points,
    )
    # pca.load_state_dict(torch.load(embeddings_dir / "pca.pt"))

    onboarding_time = time.time() - onboarding_start
    logger.info(
        f"Time taken: {onboarding_time:.2f} seconds;"
        f" {onboarding_time / len(renderer.obj_ids):.2f} seconds per object"
    )

    with open(results_dir / "onboarding.json", "w") as f:
        json.dump(
            {
                "onboarding_time": onboarding_time,
                "onboarding_time_per_object": onboarding_time / len(renderer.obj_ids),
            },
            f,
            indent=4,
        )

    # ----------------------------- INFERENCE -----------------------------
    # Initialize matcher and pose estimator
    logger.info("Initializing matcher and pose estimator...")
    assert sim_threshold == 0.0
    matcher = PointCorrespondenceFinder(sim_threshold=sim_threshold)
    pose_estimator = KabschPoseEstimator()
    logger.info(matcher)
    logger.info(pose_estimator)

    # Load template and query datasets
    logger.info("Loading template and query datasets...")
    object_dataset = ObjectPointDataset(dataset_name, data_root, embeddings_dir)
    query_dataset = QueryDataset(dataset_name, data_root, detections_path, output_size)

    sampler = ImageGroupedBatchSampler(query_dataset, batch_size=batch_size)
    collator = QueryDataCollator()
    dataloader = DataLoader(query_dataset, batch_sampler=sampler, collate_fn=collator)

    # Run inference loop
    logger.info("Running inference loop...")
    results_dict = {}
    inference_start = time.time()
    for batch in tqdm(dataloader, desc="Estimating poses...", total=len(dataloader)):
        time_start = time.time()
        B = len(batch["obj_id"])
        batch_idx = torch.arange(B, device=device)[:, None]

        # Get object data using obj_id detected
        item = object_dataset[object_dataset.get_index(batch["obj_id"])]
        object_points = item["points"].to(device)
        object_embeddings = item["embedding"].to(device, dtype)

        # Get query data
        query_rgb = batch["rgb"].to(device, dtype)
        query_mask = batch["mask"].to(device)
        query_depth = batch["depth"].to(device)
        query_M = batch["M"].to(device)
        query_cam_K = batch["cam_K"].to(device)

        # Compute query embeddings
        with torch.no_grad():
            query_embeddings = model(query_rgb)["patch_embeddings"]
            query_embeddings = pca(query_embeddings)

        del query_rgb

        # Sample query points
        query_pts2d, query_pts2d_mask = sample_pixels_from_masks(query_mask, n_tar_points)
        query_pts2d_depth = query_depth[batch_idx, query_pts2d[:, :, 1], query_pts2d[:, :, 0]]
        query_pts3d = backproject_points(
            query_pts2d.float(), query_pts2d_depth, query_M @ query_cam_K
        )

        # Compute query point embeddings
        query_point_embeddings = torch.zeros((B, n_tar_points, query_embeddings.shape[-1])).to(
            device, dtype
        )
        for i in range(B):
            query_point_embeddings[i] = interpolate_embeddings(
                query_embeddings[i], query_pts2d[i], model.patch_size, output_size
            )

        # Find point correspondences
        matches = matcher(query_point_embeddings, object_embeddings, query_pts2d_mask)
        src_pts3d = object_points[batch_idx, matches["idx_tar2src"]]

        del query_embeddings, query_point_embeddings, query_mask

        # Estimate pose
        R_est, t_est, _ = pose_estimator.estimate_pose(src_pts3d, query_pts3d, matches["mask"])
        R_est = R_est.cpu().numpy()
        t_est = t_est.cpu().numpy()

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
        results_dict[image_key]["results"]["R"].extend([R_est[i] for i in range(B)])
        results_dict[image_key]["results"]["t"].extend([t_est[i] for i in range(B)])
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


def main() -> None:
    parser = argparse.ArgumentParser("Kabsch pipeline")
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--detections_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--output_size", type=int, nargs=2, default=[224, 224])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--n_src_points", type=int, default=5000)
    parser.add_argument("--n_tar_points", type=int, default=1000)
    parser.add_argument("--embedding_model", type=str, default="dinov2")
    parser.add_argument("--embedding_batch_size", type=int, default=20)
    parser.add_argument("--sim_threshold", type=float, default=0.0)
    parser.add_argument("--render_templates", action="store_true")
    args = parser.parse_args()

    # Paths
    detections_path = Path(args.detections_path)
    results_dir = Path(DATA_DIR) / "results" / args.run_name
    results_dir.mkdir(parents=True, exist_ok=True)

    # Args
    output_size = tuple(args.output_size)
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)

    # Save args in a json file
    with open(results_dir / "args.json", "w") as f:
        json.dump(args.__dict__, f, indent=4)

    # Run Kabsch pipeline
    kabsch_pipeline(
        dataset_name=args.dataset_name,
        detections_path=detections_path,
        results_dir=results_dir,
        device=device,
        dtype=dtype,
        output_size=output_size,
        batch_size=args.batch_size,
        n_src_points=args.n_src_points,
        n_tar_points=args.n_tar_points,
        embedding_model=args.embedding_model,
        embedding_batch_size=args.embedding_batch_size,
        sim_threshold=args.sim_threshold,
        render_templates=args.render_templates,
    )

    # Save predictions in BOP format
    save_bop_results_from_predictions(
        method_name=args.run_name,
        dataset_name=args.dataset_name,
        split=args.split,
        save_dir=results_dir,
        predictions_dir=results_dir / "predictions",
    )


if __name__ == "__main__":
    main()
