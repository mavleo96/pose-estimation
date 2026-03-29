import argparse
import json
import logging
from pathlib import Path

import torch

from _configs.constants import DATA_DIR

from ..utils.io import save_bop_results_from_predictions
from .pnp_pipeline import pnp_pipeline

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main() -> None:
    parser = argparse.ArgumentParser("PnP pipeline")
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--detections_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--output_size", type=int, nargs=2, default=[518, 518])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--embedding_model", type=str, default="dinov2")
    parser.add_argument("--embedding_batch_size", type=int, default=20)
    parser.add_argument("--sim_threshold", type=float, default=0.5)
    parser.add_argument("--patch_threshold", type=float, default=0.0)
    parser.add_argument("--k_hypotheses", type=int, default=5)
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

    # Run PnP pipeline
    pnp_pipeline(
        dataset_name=args.dataset_name,
        detections_path=detections_path,
        results_dir=results_dir,
        device=device,
        dtype=dtype,
        output_size=output_size,
        batch_size=args.batch_size,
        embedding_model=args.embedding_model,
        embedding_batch_size=args.embedding_batch_size,
        sim_threshold=args.sim_threshold,
        patch_threshold=args.patch_threshold,
        k_hypotheses=args.k_hypotheses,
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
