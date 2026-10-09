"""Serial, resumable execution of the approved frozen manifest (Linux only)."""
import fcntl
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

from mars.config import load_config
from mars.data import load_bundle
from mars.runner import source_digest
from mars.utils import digest, read_json, write_json, git_revision

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = 'attack-pilot-20261009'


def main():
    os.chdir(ROOT)
    directory = ROOT / 'runs' / CAMPAIGN
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'campaign.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (directory / 'campaign.pid').write_text(str(os.getpid()))
        manifest = read_json(ROOT / 'configs/planned' / CAMPAIGN / 'manifest.json')
        assert source_digest() == manifest['source_hash'], 'Frozen training source mismatch'
        assert len(manifest['runs']) == manifest['authorized_runs'] == 20
        write_json(directory / 'launch.json', {'git_commit': git_revision(), 'source_hash': source_digest(),
                                               'manifest_hash': digest(manifest), 'pid': os.getpid()})
        for entry in manifest['runs']:
            assert source_digest() == manifest['source_hash'], 'Training source changed during campaign'
            cfg = load_config(ROOT / entry['config'])
            assert digest(cfg) == entry['config_hash']
            data = load_bundle(cfg, ROOT / entry['data'])
            assert data.manifest['bundle_digest'] == entry['data_hash']
            out = ROOT / entry['out']
            summary = read_json(out / 'summary.json') if (out / 'summary.json').exists() else {}
            if summary.get('status') == 'complete':
                assert summary['completed_rounds'] == 50 and summary['config_hash'] == entry['config_hash']
                subprocess.run([sys.executable, 'scripts/collect_attack_pilot.py', '--archive', entry['id']], check=True)
                continue
            if (directory / 'STOP').exists():
                raise RuntimeError('STOP requested; no additional experiment started')
            if shutil.disk_usage(ROOT).free < 2 * 1024**3:
                raise RuntimeError('Less than 2 GiB free: preserve artifacts and back up before resuming')
            command = [sys.executable, '-u', '-m', 'mars.cli', 'run', '--config', entry['config'],
                       '--data', entry['data'], '--out', entry['out']]
            if (out / 'checkpoint.pt').exists():
                command.append('--resume')
            if (out / 'RUNNING.lock').exists():
                raise RuntimeError(f'Stale or active run lock requires process inspection: {out}')
            print('START', entry['id'], flush=True)
            write_json(directory / 'current.json', {'id': entry['id'], 'status': 'running'})
            with (directory / f'{entry["id"]}.console.log').open('a') as stream:
                process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
            if process.returncode:
                write_json(directory / 'current.json', {'id': entry['id'], 'status': 'failed', 'exit_code': process.returncode})
                raise RuntimeError(f'Experiment failed with exit {process.returncode}: {entry["id"]}')
            summary = read_json(out / 'summary.json')
            assert summary['status'] == 'complete' and summary['completed_rounds'] == 50
            subprocess.run([sys.executable, 'scripts/collect_attack_pilot.py', '--archive', entry['id']], check=True)
            print('COMPLETE', entry['id'], flush=True)
        write_json(directory / 'current.json', {'status': 'complete', 'runs': 20})


if __name__ == '__main__':
    exit_code = 0
    try:
        main()
    except Exception:
        traceback.print_exc()
        exit_code = 1
    directory = ROOT / 'runs' / CAMPAIGN
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'campaign.pid').exists() and (directory / 'campaign.pid').read_text() == str(os.getpid()):
        (directory / 'campaign.exit').write_text(str(exit_code))
    sys.exit(exit_code)
