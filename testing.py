import multiprocessing as mp
from pathlib import Path
import warnings

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from pcr.dataloader import CustomTestSet
from pcr.pose_recovery import PoseEstimator
from pcr.template import CustomTemplateSet
from pcr.utils import transform_from_Rt
from src.utils.inout import save_predictions_from_batched_predictions

# Set multiprocessing start method to 'spawn' to avoid CUDA issues
mp.set_start_method("spawn", force=True)

# Suppress noisy PNG warnings from imageio/Pillow
warnings.filterwarnings("ignore", message=".*Loading 16-bit.*PNG.*")


@hydra.main(version_base=None, config_path="configs", config_name="test")
def run_pcr(cfg: DictConfig) -> None:
    """Run PCR pose estimation using Hydra config."""
    OmegaConf.set_struct(cfg, False)

    embedding_save_dir = Path(cfg.save_dir).parent / "embeddings"
    embedding_save_path = embedding_save_dir / f"{cfg.test_dataset_name}_{cfg.embedding_model}_{cfg.dino_components}_{cfg.embedding_desc}.npz"

    # Choose device for pose recovery (CUDA if available, otherwise CPU).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # First, create the template dataset and template set
    cfg.data.test.dataloader.dataset_name = cfg.test_dataset_name
    cfg.data.test.dataloader.batch_size = cfg.machine.batch_size
    cfg.data.test.dataloader.load_gt = False
    cfg.data.test.dataloader.test_setting = cfg.test_setting
    cfg.data.test.dataloader._target_ = "src.dataloader.template.TemplateSet"
    template_dataset0 = instantiate(cfg.data.test.dataloader)
    template_dataset = CustomTemplateSet(
        cfg,
        template_dataset0,
        embedding_model=cfg.embedding_model,
        dino_components=cfg.dino_components,
        save_path=embedding_save_path,
    )

    # Move static template tensors to the chosen device once.
    template_dataset.points = template_dataset.points.to(device)
    template_dataset.point_embeddings = template_dataset.point_embeddings.to(device)

    # Then create the test dataset
    cfg.data.test.dataloader._target_ = "src.dataloader.test.GigaPoseTestSet"
    test_dataset0 = instantiate(cfg.data.test.dataloader)
    test_dataset = CustomTestSet(
        cfg,
        test_dataset0,
        embedding_model=cfg.embedding_model,
        pca=template_dataset.pca,
    )
    test_dataloader = DataLoader(
        test_dataset.dataset.web_dataloader.datapipeline,
        batch_size=1,  # a single image may have multiple instances
        num_workers=0,  # Disable multiprocessing to avoid pickling issues with spawn method
        collate_fn=test_dataset.collate_fn,
    )

    pose_estimator = PoseEstimator(sample_size=3, max_iters=1000, inlier_thresh=25, tau=0.1)

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = save_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    for i, batch in tqdm(
        enumerate(test_dataloader),
        desc="Estimating poses...",
        total=len(test_dataloader),
    ):
        obj_indices = (batch.test_list.infos.obj_id - 1).values

        # Move per-batch tensors needed for pose recovery to the device.
        batch_points = batch.points.to(device)
        batch_point_embeddings = batch.point_embeddings.to(device)

        R_est, t_est, score = pose_estimator.estimate_pose(
            template_dataset.points[obj_indices],
            template_dataset.point_embeddings[obj_indices],
            batch_points,
            batch_point_embeddings,
        )
        T_est = transform_from_Rt(R_est, t_est)
        np.savez(
            pred_dir / f"{i}.npz",
            scene_id=batch.test_list.infos.scene_id.values,
            im_id=batch.test_list.infos.im_id.values,
            object_id=batch.test_list.infos.obj_id.values,
            time=batch.test_list.infos.detection_time.values,
            detection_time=batch.test_list.infos.detection_time.values,
            poses=T_est.cpu().numpy(),
            scores=score.cpu().numpy(),
        )

    save_predictions_from_batched_predictions(
        pred_dir,
        cfg.data.test.dataloader.dataset_name,
        "pcr",
        cfg.run_id,
        is_refined=False,
    )


if __name__ == "__main__":
    run_pcr()