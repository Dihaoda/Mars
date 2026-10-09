"""Per-condition immutable archive and small public snapshot. No credentials included."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import zipfile


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    os.replace(tmp, path)


def archive(cfg, name):
    public, run = Path(cfg['results_dir']), Path(cfg['run_dir'])
    summary = json.loads((public/name/'summary.json').read_text())
    assert summary['completed_rounds'] == cfg['rounds']
    target = Path('backups')/cfg['experiment']/(name+'.zip')
    receipt = public/name/'backup.json'
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        assert target.exists() and sha(target) == saved['sha256']
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    roots = [run/name, run/'calibration', Path(cfg['data_dir']), Path('experiments/indexguard'),
             Path('src/mars'), Path('configs/indexguard'), Path('docs/indexguard-heterogeneity-20261009.md')]
    paths = [run/'initial.pt', public/'provenance.json', public/'calibration.json',
             public/'initial_evaluation.json', Path('scripts/run_indexguard.sh'),
             Path('scripts/collect_indexguard.py')]
    for root in roots + [public/name]:
        paths.extend(p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts) if root.is_dir() else paths.append(root)
    model_manifest = Path(cfg['model']['name'])/'download_manifest.json'
    members = [{'path': p.as_posix(), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(set(paths))]
    tmp = target.with_suffix('.partial')
    with zipfile.ZipFile(tmp, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as z:
        for item in members:
            z.write(item['path'], item['path'])
        z.write(model_manifest, 'model_download_manifest.json')
        z.writestr('archive_manifest.json', json.dumps({'condition': name, 'identity': summary['identity'],
                     'rounds': cfg['rounds'], 'clients': cfg['clients'], 'files': members,
                     'model_manifest_sha256': sha(model_manifest)}, indent=2))
    os.replace(tmp, target)
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
    write(receipt, {'path': target.as_posix(), 'bytes': target.stat().st_size, 'sha256': sha(target),
                    'identity': summary['identity'], 'files': len(members), 'local_download_verified': False})
    print(json.dumps({'archived': name, 'bytes': target.stat().st_size}), flush=True)


def export(cfg):
    public = Path(cfg['results_dir'])
    target = Path('/root/autodl-tmp/indexguard-public.zip')
    tmp = target.with_suffix('.partial')
    records = []
    with zipfile.ZipFile(tmp, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(public.rglob('*.json')):
            # Read once: a concurrent atomic result update cannot split a snapshot file.
            content = p.read_bytes()
            json.loads(content)
            z.writestr(p.as_posix(), content)
            records.append({'path': p.as_posix(), 'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content)})
        z.writestr('export_manifest.json', json.dumps({'files': records}, indent=2))
    os.replace(tmp, target)
    print(json.dumps({'export': str(target), 'files': len(records), 'sha256': sha(target)}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/indexguard/pilot.json')
    ap.add_argument('--archive', choices=['mix0', 'mix04'])
    ap.add_argument('--export', action='store_true')
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if args.archive:
        archive(cfg, args.archive)
    if args.export:
        export(cfg)


if __name__ == '__main__':
    main()
