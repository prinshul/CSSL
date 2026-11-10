#!/usr/bin/env bash
set -euo pipefail
ENV=${ENV:-cssl}; CU=${CU:-121}
REPO=$(cd "$(dirname "$0")/.." && pwd)
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
set +u; source "$(conda info --base)/etc/profile.d/conda.sh"; set -u
conda env list | awk '{print $1}' | grep -qx "$ENV" || conda create -y -n "$ENV" python=3.10
set +u; conda activate "$ENV"; set -u
python -m pip install --upgrade pip
pip install torch==2.5.1 torchvision==0.20.1 --index-url "https://download.pytorch.org/whl/cu${CU}"
pip install -r "$REPO/requirements.txt"
python - <<'PY'
import os, torch
print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available())
if os.environ.get("NO_GPU_CHECK") == "1":
    raise SystemExit(0)
assert torch.cuda.is_available(), "PyTorch does not see the GPU"
print("GPU:", torch.cuda.get_device_name(0), "| bf16:", torch.cuda.is_bf16_supported())
PY
cd "$REPO" && python -m pytest -q tests
echo "Done. Next: conda activate $ENV && DATA=/data/med STAGES=setup bash run_paper.sh"
