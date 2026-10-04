#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/root/autodl-tmp/mars-cache/huggingface}"
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PYTHON=.venv/bin/python
mkdir -p runs/clean-pair
exec 9>runs/clean-pair/CAMPAIGN.lock
flock -n 9 || { printf '%s\n' 'Another clean-pair campaign is active.' >&2; exit 1; }
trap 'status=$?; printf "%s\n" "$status" > runs/clean-pair/campaign.exit' EXIT
printf '%s\n' "$$" > runs/clean-pair/campaign.pid
# Preparation must finish before this script; all six jobs use the same data.
"$PYTHON" - <<'PY'
import json
import subprocess
import sys
from pathlib import Path
from mars.config import load_config
from mars.data import load_bundle
from mars.utils import digest

manifest = json.loads(Path('configs/planned/clean-pair/manifest.json').read_text())
assert len(manifest['jobs']) == 6
for job in manifest['jobs']:
    cfg = load_config(job['config'])
    assert digest(cfg) == job['config_hash']
    load_bundle(cfg, job['data'])
first = manifest['jobs'][0]
checkpoint = Path(first['output']) / 'checkpoint.pt'
if not checkpoint.exists():
    subprocess.run([sys.executable, '-m', 'mars', 'run', '--config', first['config'],
                    '--data', first['data'], '--out', first['output'], '--until-round', '1'], check=True)
for job in manifest['jobs']:
    command = [sys.executable, '-m', 'mars', 'run', '--config', job['config'],
               '--data', job['data'], '--out', job['output']]
    if (Path(job['output']) / 'checkpoint.pt').exists():
        command.append('--resume')
    print('START', job['id'], flush=True)
    subprocess.run(command, check=True)
    print('COMPLETE', job['id'], flush=True)
print('ALL_SIX_COMPLETE', flush=True)
PY
