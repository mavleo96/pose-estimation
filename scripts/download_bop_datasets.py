import sys
import os
import argparse
from pathlib import Path
import zipfile

sys.path.insert(0, os.getcwd())

from _configs.constants import DATA_DIR
from _configs.constants import BOP_CLASSIC, BOP_H3, BOP_INDUSTRIAL


def download_dataset(dataset_name: str) -> None:
    local_dir = Path(DATA_DIR) / dataset_name
    # if local_dir.exists():
    #     print(f"Dataset {dataset_name} already exists in {local_dir}")
    #     return

    print(f"Downloading dataset {dataset_name} to {local_dir}")
    local_dir.mkdir(parents=True, exist_ok=True)

    # download the dataset
    download_cmd = (
        f"hf download bop-benchmark/{dataset_name} "
        "--repo-type=dataset "
        f"--local-dir {local_dir} "
        "--exclude '*train*' '*test*all*'"
    )
    os.system(download_cmd)

    for zip_path in local_dir.rglob("*.zip"):
        print(f"Extracting {zip_path}")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(zip_path.parent)
        zip_path.unlink()  # remove zip after extraction


def download_bop_extra() -> None:
    local_dir = Path(DATA_DIR) / "bop_extra"
    print(f"Downloading bop_extra to {local_dir}")
    local_dir.mkdir(parents=True, exist_ok=True)

    download_cmd = (
        f"hf download bop-benchmark/bop_extra "
        "--repo-type=dataset "
        f"--local-dir {local_dir} "
        "--exclude '*coco*' 'research/*'"
    )
    os.system(download_cmd)
    for zip_path in local_dir.rglob("*.zip"):
        print(f"Extracting {zip_path}")
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
