#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:$PWD"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
ROOT=runs/indexguard-heterogeneity-20261009
mkdir -p "$ROOT"
exec 9>"$ROOT/QUEUE.lock"
flock -n 9 || exit 73
if test -f "$ROOT/STOP"; then exit 75; fi
echo $$ > "$ROOT/campaign.pid"
/root/autodl-tmp/Mars/.venv/bin/python -u -m experiments.indexguard.runner --config configs/indexguard/pilot.json
result=$?
printf '%s\n' "$result" > "$ROOT/campaign.exit"
exit "$result"
