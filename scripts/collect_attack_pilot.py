"""Export public evidence; archive each completed run without changing training."""
import argparse
import hashlib
import os
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from mars.utils import read_json, write_json
from mars.report import export_run

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = 'attack-pilot-20261009'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def archive(entry, public):
    out = ROOT / entry['out']
    assert read_json(out / 'summary.json')['status'] == 'complete'
    target = ROOT / 'backups' / CAMPAIGN / (entry['id'] + '.zip')
    target.parent.mkdir(parents=True, exist_ok=True)
    receipt = public / 'backup.json'
    if receipt.exists() and target.exists():
        old = read_json(receipt)
        assert sha(target) == old['sha256']
        return
    export_run(out)
    sources = list(out.rglob('*')) + list((ROOT / entry['data']).glob('*.json'))
    sources += list((ROOT / 'src/mars').glob('*.py')) + list((ROOT / 'scripts').glob('*.*'))
    sources += [ROOT / entry['config'], ROOT / 'configs/planned' / CAMPAIGN / 'manifest.json',
                ROOT / 'docs/workone-port.md', ROOT / 'docs/attack-pilot-20261009.md', ROOT / 'pyproject.toml',
                ROOT / 'runs' / CAMPAIGN / f'{entry["id"]}.console.log']
    sources = sorted({p for p in sources if p.is_file() and '__pycache__' not in p.parts})
    records = [{'path': p.relative_to(ROOT).as_posix(), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sources]
    write_json(public / 'file_manifest.json', {'files': records, 'complete_run': entry['id']})
    temporary = target.with_suffix('.zip.partial')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED, compresslevel=3) as z:
        for p, record in zip(sources, records):
            z.write(p, record['path'])
        z.write(public / 'file_manifest.json', 'file_manifest.json')
    with zipfile.ZipFile(temporary) as z:
        assert z.testzip() is None
    os.replace(temporary, target)
    write_json(receipt, {'path': target.relative_to(ROOT).as_posix(), 'bytes': target.stat().st_size,
                         'sha256': sha(target), 'archive_crc_verified': True,
                         'local_copy_verified': False, 'note': 'Server archive; local receipt is published only after transfer verification.'})


def collect(archive_id=None):
    destination = ROOT / 'results' / CAMPAIGN
    manifest = read_json(ROOT / 'configs/planned' / CAMPAIGN / 'manifest.json')
    rows = []
    for entry in manifest['runs']:
        out = ROOT / entry['out']
        summary = read_json(out / 'summary.json') if (out / 'summary.json').exists() else {}
        progress = read_json(out / 'progress.json') if (out / 'progress.json').exists() else summary
        complete = summary.get('status') == 'complete'
        completed = progress.get('completed_rounds', 0)
        record = {'id': entry['id'], 'status': 'complete' if complete else 'running' if completed else 'pending',
                  'completed_rounds': completed, 'total_rounds': 50, 'config': entry['config']}
        if completed:
            public = destination / entry['id']
            public.mkdir(parents=True, exist_ok=True)
            write_json(public / 'metadata.json', read_json(out / 'metadata.json'))
            public_rounds = []
            for i in range(1, completed + 1):
                round_data = read_json(out / f'round_{i:03}.json')
                # Full candidate traces are preserved in the complete backup.
                for attack in round_data['summary'].get('adaptive_attack', []):
                    attack.pop('trace', None)
                public_rounds.append(round_data)
            write_json(public / 'rounds.json', public_rounds)
            record['last_validation'] = public_rounds[-1]['summary']['validation']
            if complete:
                write_json(public / 'summary.json', summary)
                record['test_macro_accuracy'] = summary['test_macro_accuracy']
                record['asr_macro'] = summary.get('asr_macro')
                record['detection'] = summary['pooled_client_round_detection']
                export_run(out)
                for name in ('clients.csv', 'rounds.csv'):
                    (public / name).write_bytes((out / name).read_bytes())
                if archive_id == entry['id']:
                    archive(entry, public)
        rows.append(record)
    directory = ROOT / 'runs' / CAMPAIGN
    current = read_json(directory / 'current.json') if (directory / 'current.json').exists() else None
    if current and current.get('status') == 'failed':
        for record in rows:
            if record['id'] == current['id']:
                record.update(status='failed', exit_code=current['exit_code'])
    status = {'campaign': CAMPAIGN, 'updated_utc': datetime.now(timezone.utc).isoformat(),
              'source_hash': manifest['source_hash'], 'authorized_runs': 20,
              'completed_runs': sum(r['status'] == 'complete' for r in rows),
              'completed_rounds': sum(r['completed_rounds'] for r in rows),
              'total_rounds': 1000, 'current': current, 'runs': rows}
    write_json(destination / 'progress.json', status)
    print({k: v for k, v in status.items() if k != 'runs'})
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive')
    args = parser.parse_args()
    collect(args.archive)
