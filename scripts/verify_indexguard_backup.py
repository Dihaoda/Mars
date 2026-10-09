"""Verify downloaded research archive without loading any pickled checkpoint."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import zipfile


def stream_sha(handle):
    h = hashlib.sha256()
    for block in iter(lambda: handle.read(8*1024*1024), b''):
        h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('archive')
    ap.add_argument('--expected-sha256', required=True)
    ap.add_argument('--receipt-dir', default='results/indexguard-heterogeneity-20261009')
    args = ap.parse_args()
    path = Path(args.archive)
    with path.open('rb') as f:
        assert stream_sha(f) == args.expected_sha256
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        manifest = json.loads(z.read('archive_manifest.json'))
        assert len(set(z.namelist())) == len(z.namelist())
        for item in manifest['files']:
            assert z.getinfo(item['path']).file_size == item['bytes']
            with z.open(item['path']) as f:
                assert stream_sha(f) == item['sha256'], item['path']
        assert hashlib.sha256(z.read('model_download_manifest.json')).hexdigest() == manifest['model_manifest_sha256']
        root = 'runs/indexguard-heterogeneity-20261009/'
        name, rounds, clients = manifest['condition'], manifest['rounds'], manifest['clients']
        for r in range(1, rounds+1):
            assert root+name+f'/round_{r:03d}/global.pt' in z.namelist()
            for c in range(clients):
                assert root+name+f'/round_{r:03d}/client_{c:02d}.pt' in z.namelist()
        assert root+name+'/checkpoint.pt' in z.namelist()
        assert any(p.startswith(root+name+'/final_adapter/') and p.endswith('.safetensors') for p in z.namelist())
        for c in range(clients):
            for sample in range(3):
                assert root+f'calibration/client_{c:02d}_sample_{sample}.pt' in z.namelist()
        assert 'data/indexguard-heterogeneity-20261009/mix0.json' in z.namelist()
        assert 'data/indexguard-heterogeneity-20261009/mix04.json' in z.namelist()
    receipt = {'condition': name, 'identity': manifest['identity'], 'sha256': args.expected_sha256,
               'bytes': path.stat().st_size, 'filename': path.name, 'files': len(manifest['files']),
               'zip_crc_verified': True, 'all_file_hashes_verified': True,
               'all_round_clients_and_adapters_present': True, 'local_copy_verified': True,
               'verified_utc': datetime.now(timezone.utc).isoformat()}
    path.with_suffix('.verification.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    public = Path(args.receipt_dir)/name/'local_backup_verification.json'
    public.parent.mkdir(parents=True, exist_ok=True)
    public.write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
