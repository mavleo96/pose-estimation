#!/bin/bash
set -euo pipefail

ENV_NAME="pose2"
export PYOPENGL_PLATFORM=egl

conda create -n $ENV_NAME python=3.10 -y && \
conda init bash && \
eval "$(conda shell.bash hook)" && \
conda activate $ENV_NAME && \
pip install -U \
 torch \
 torchvision \
 --index-url https://download.pytorch.org/whl/cu128 && \
pip install -U kornia \
 einops \
 huggingface-hub \
 transformers \
 accelerate \
 opencv-python \
 pyrender \
 PyOpenGL \
 PyOpenGL-accelerate \
 embreex \
 trimesh \
 pycocotools \
 ipykernel \
 wandb \
 ruff \
 tqdm \
 pre-commit \
 tensorboard \
 scikit-learn \
 scipy \
 numpy \
 pandas \
 matplotlib \
 seaborn \
 git+https://github.com/mavleo96/bop_toolkit.git

conda env config vars set PYOPENGL_PLATFORM=egl
conda env export --no-builds | grep -v "^prefix: " > env2.yml