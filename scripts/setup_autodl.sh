#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON_BIN="${MARS_BASE_PYTHON:-/root/miniconda3/bin/python}"
export PYTHONNOUSERSITE=1
export HF_HOME="${HF_HOME:-/root/autodl-tmp/mars-cache/huggingface}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
"$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print(torch.__version__, torch.cuda.get_device_name(0))'
"$PYTHON_BIN" -m venv --system-site-packages .venv
.venv/bin/python -m pip install -e '.[llm,test,plots]'
.venv/bin/python -m pytest -q
.venv/bin/python -m mars preflight --config configs/autodl_smoke.yaml
printf '%s\n' 'Setup complete. Real-model download/training starts only when autodl_smoke.sh is invoked.'
