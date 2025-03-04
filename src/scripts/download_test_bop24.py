# from omegaconf import DictConfig, OmegaConf
import logging
import os
import os.path as osp
# import hydra
import argparse
from pathlib import Path
from src.utils.logging import get_logger


logger = get_logger(__name__)


def run_download(test_dataset_name: str) -> None:
    # local_dir = os.path.join(
    #     config.data.test.dataloader.root_dir, config.test_dataset_name
    # )
    root_dir = Path(os.getcwd()) / "data" / "datasets"
    local_dir = root_dir / test_dataset_name
    logger.info(f"Saving dataset to {local_dir}")

    download_cmd = f"huggingface-cli download bop-benchmark/datasets --include {test_dataset_name}/*.zip --exclude *train_pbr* --local-dir {root_dir} --repo-type=dataset"
    logger.info(f"Running {download_cmd}")
    os.system(download_cmd)
    logger.info(f"Dataset downloaded to {local_dir}")

    # unzip the dataset
    logger.info(f"export LOCAL_DIR={root_dir}")
    os.environ["LOCAL_DIR"] = str(root_dir)

    logger.info(f"export NAME={test_dataset_name}")
    os.environ['NAME'] = test_dataset_name

    unzip_cmd = "bash src/scripts/extract_bop.sh"
    logger.info(f"Running {unzip_cmd}")
    os.system(unzip_cmd)
    logger.info("Dataset extracted")

    # convert to webdataset format
    # formatting the dataset top webdataset format
    imagewise_cmd = "python -m src.scripts.convert_scenewise_to_imagewise --input INPUT --output OUTPUT --nprocs 10"
    webdataset_cmd = "python -m src.scripts.convert_imagewise_to_webdataset --input INPUT --output OUTPUT"

    split = "test"
    split_dir = Path(local_dir) / split
    tmp_dir = (
        root_dir
        / "tmp"
        / f"{test_dataset_name}_image_wise"
        / split
    )
    os.makedirs(tmp_dir, exist_ok=True)

    imagewise_cmd = imagewise_cmd.replace("INPUT", str(split_dir))
    imagewise_cmd = imagewise_cmd.replace("OUTPUT", str(tmp_dir))

    logger.info(f"Running {imagewise_cmd}")
    os.system(imagewise_cmd)

    webdataset_cmd = webdataset_cmd.replace("INPUT", str(tmp_dir))
    webdataset_cmd = webdataset_cmd.replace("OUTPUT", str(split_dir))

    if test_dataset_name in ["tless", "ycbv", "tudl", "itodd", "hope"]:
        webdataset_cmd += " --nprocs 4"
    logger.info(f"Running {webdataset_cmd}")
    os.system(webdataset_cmd)



def download() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", type=str)
    args = parser.parse_args()
    test_dataset_name = args.dataset_name

    run_download(test_dataset_name)


if __name__ == "__main__":
    download()
