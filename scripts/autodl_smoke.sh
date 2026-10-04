#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONNOUSERSITE=1
export HF_HOME="${HF_HOME:-/root/autodl-tmp/mars-cache/huggingface}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PYTHON=.venv/bin/python
RUN_DIR="${1:-runs/autodl-smoke}"
DATA_DIR=data/autodl-smoke
"$PYTHON" -m mars prepare --config configs/autodl_smoke.yaml --data "$DATA_DIR"
"$PYTHON" -m mars preflight --config configs/autodl_smoke.yaml --data "$DATA_DIR" --load-model
if [[ -f "$RUN_DIR/checkpoint.pt" ]]; then
  "$PYTHON" -m mars run --config configs/autodl_smoke.yaml --data "$DATA_DIR" --out "$RUN_DIR" --resume
else
  "$PYTHON" -m mars run --config configs/autodl_smoke.yaml --data "$DATA_DIR" --out "$RUN_DIR"
fi
"$PYTHON" -m mars plot --run "$RUN_DIR" --out "$RUN_DIR/diagnostics.png"
