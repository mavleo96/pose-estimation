# import logging
# import os
# import shutil
# import sys
# import zipfile
# from pathlib import Path

# sys.path.insert(0, os.getcwd())

# import numpy as np
# from huggingface_hub import hf_hub_download
# from PIL import Image

# from configs import DATA_DIR
# from pose.utils.io import load_camera_intrinsics

# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)


# def main() -> None:
#     data_dir = Path(DATA_DIR)
#     data_dir.mkdir(parents=True, exist_ok=True)

#     # download templates.zip
#     logger.info(f"Downloading templates.zip to {data_dir}")
#     hf_hub_download(
#         repo_id="nv-nguyen/gigaPose",
#         repo_type="dataset",
#         filename="templates.zip",
#         local_dir=str(data_dir),
#     )

#     # extract templates.zip to data_dir/bop_templates
#     tmp_dir = data_dir / "tmp_extract"
#     template_dir = data_dir / "bop_templates"
#     tmp_dir.mkdir(exist_ok=True)
#     zip_path = data_dir / "templates.zip"
#     with zipfile.ZipFile(zip_path, "r") as zip_ref:
#         zip_ref.extractall(tmp_dir)
#     zip_path.unlink()
#     (tmp_dir / "templates").rename(template_dir)
#     shutil.rmtree(tmp_dir)

#     # rename the directory to bop_templates
#     for dir in template_dir.iterdir():
#         dataset_name = dir.stem

#         if not (data_dir / dataset_name).exists():
#             logger.info(f"Dataset {dataset_name} not found, skipping")
#             continue

#         depth_scale = load_camera_intrinsics(dataset_name, data_dir / dataset_name).depth_scale
#         if depth_scale == 1:
#             logger.info(f"Depth scale for {dataset_name} is 1, skipping")
#             continue

#         logger.info(f"Processing {dataset_name}")
#         for path in (template_dir / dataset_name).glob("*/*_depth.png"):
#             depth = (np.array(Image.open(path)) / depth_scale).astype(np.uint16)
#             Image.fromarray(depth).save(path)


# if __name__ == "__main__":
#     main()
