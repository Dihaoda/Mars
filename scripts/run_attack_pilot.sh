#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/Mars
export HF_HOME=/root/autodl-tmp/mars-cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
exec .venv/bin/python -u scripts/run_attack_pilot.py
