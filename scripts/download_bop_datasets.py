import argparse
import logging
import os
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, os.getcwd())

from huggingface_hub import snapshot_download

from _configs.constants import BOP_CLASSIC, BOP_H3, BOP_INDUSTRIAL, DATA_DIR

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def download_dataset(dataset_name: str) -> None:
    local_dir = Path(DATA_DIR) / dataset_name
    local_dir.mkdir(parents=True, exist_ok=True)

    # download the dataset
    logger.info(f"Downloading dataset {dataset_name} to {local_dir}")
    snapshot_download(
        repo_id=f"bop-benchmark/{dataset_name}",
        repo_type="dataset",
        local_dir=str(local_dir),
        ignore_patterns=["*train*", "*test*all*"],
    )

    for zip_path in local_dir.rglob("*.zip"):
        logger.info(f"Extracting {zip_path}")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(zip_path.parent)
        zip_path.unlink()  # remove zip after extraction

    # move everything in. /local_dir/dataset_name to /local_dir
    for dir in local_dir.iterdir():
        if dir.is_dir() and dir.name == dataset_name:
            for file in dir.iterdir():
                shutil.move(file, local_dir / file.name)
            dir.rmdir()


def download_bop_extra() -> None:
    local_dir = Path(DATA_DIR) / "bop_extra"
    local_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading bop_extra to {local_dir}")
    snapshot_download(
        repo_id="bop-benchmark/bop_extra",
        repo_type="dataset",
        local_dir=str(local_dir),
        ignore_patterns=["*coco*", "research/*"],
    )

    for zip_path in local_dir.rglob("*.zip"):
        logger.info(f"Extracting {zip_path}")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(zip_path.parent)
        zip_path.unlink()  # remove zip after extraction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    args = parser.parse_args()

    # create a list of dataset names to download
    if args.dataset_name in ["bop_classic", "bop_h3", "bop_industrial"]:
        dataset_names = {
            "bop_classic": BOP_CLASSIC,
            "bop_h3": BOP_H3,
            "bop_industrial": BOP_INDUSTRIAL,
        }[args.dataset_name]
    elif args.dataset_name in BOP_CLASSIC + BOP_H3 + BOP_INDUSTRIAL + ["bop_extra"]:
        dataset_names = [args.dataset_name]
    else:
        raise ValueError(f"Invalid dataset name: {args.dataset_name}")

    for dataset_name in dataset_names:
        if dataset_name == "bop_extra":
            download_bop_extra()
        else:
            download_dataset(dataset_name)


if __name__ == "__main__":
    main()
