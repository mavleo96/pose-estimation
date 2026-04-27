#!/usr/bin/env bash
# Create conda env pose from env2.yml, then install optional GPU extras (flash-attn wheel, torch-scatter).
# Run: bash scripts/setup_env.sh   (from any directory; script resolves repo root)
# If env already exists: conda env remove -n pose && bash scripts/setup_env.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found on PATH" >&2
  exit 1
fi
eval "$(conda shell.bash hook)"

conda env create -f "${ROOT}/env2.yml"
conda activate pose


# Prebuilt flash-attn (community wheel); see https://github.com/Dao-AILab/flash-attention/issues/2425
# Adjust cp310 in URL if you change Python (cp311 / cp312 / cp313).
pip install "https://github.com/lesj0610/flash-attention/releases/download/v2.8.3-cu12-torch2.11/flash_attn-2.8.3+cu12torch2.11cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"

pip install torch-scatter -f https://data.pyg.org/whl/torch-2.11.0+cu128.html

# If the flash-attn wheel fails: conda activate pose, set TORCH_CUDA_ARCH_LIST to your GPU, then:
#   export TORCH_CUDA_ARCH_LIST="8.6"
#   export MAX_JOBS=1
#   export NINJA_MAX_JOBS=1
#   pip install --no-cache-dir flash-attn --no-build-isolation -v
