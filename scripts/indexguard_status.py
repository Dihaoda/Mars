import json
import os
from pathlib import Path

cfg = json.loads(Path('configs/indexguard/pilot.json').read_text())
run, public = Path(cfg['run_dir']), Path(cfg['results_dir'])
result = {}
for filename in ['campaign.pid', 'campaign.exit', 'STOP']:
    p = run/filename
    result[filename] = p.read_text() if p.exists() else None
pid = result['campaign.pid']
result['queue_alive'] = bool(pid and Path('/proc', pid.strip()).exists())
for filename in ['progress.json', 'calibration.json']:
    p = public/filename
    obj = json.loads(p.read_text()) if p.exists() else None
    if filename == 'calibration.json' and obj:
        obj = {k:obj[k] for k in ['identity', 'k', 'dimensions', 'k_fraction']}
    result[filename] = obj
result['conditions'] = {}
for name in ['mix0', 'mix04']:
    rounds = list((public/name).glob('round_*.json'))
    result['conditions'][name] = {'rounds': len(rounds), 'summary': (public/name/'summary.json').exists(),
                                  'archive': (public/name/'backup.json').exists()}
print(json.dumps(result, indent=2))
