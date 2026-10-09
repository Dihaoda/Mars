"""Freeze the user-approved 20-run pilot, reusing the public audited data split."""
import copy
import hashlib
import random
from pathlib import Path

import yaml

from mars.config import load_config, validate
from mars.data import load_bundle, partition_spec
from mars.runner import source_digest
from mars.sampling import select_clients
from mars.utils import digest, read_json, write_json

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = 'attack-pilot-20261009'


def main():
    cfg = load_config(ROOT / 'configs/planned/clean-pair/hfedsa-raw-none-mix0.4-seed2026.yaml')
    cfg['train'].update(rounds=50, sampling='stratified')
    source = ROOT / 'results/clean-pair-20261004/dataset'
    original = read_json(source / 'bundle.json')
    original_manifest = read_json(source / 'manifest.json')
    assert digest(original) == original_manifest['bundle_digest']
    assert partition_spec(cfg) == original_manifest['spec']
    malicious = sorted(random.Random(cfg['data']['seed'] + 991).sample(list(range(15)), 4))
    data_specs = {}
    for attacked in (False, True):
        c = copy.deepcopy(cfg)
        c['attack'].update(name='label_flip' if attacked else 'none', malicious_fraction=.2 if attacked else 0.)
        obj, manifest = copy.deepcopy(original), copy.deepcopy(original_manifest)
        if attacked:
            for i in malicious:
                obj['roles'][str(i)] = manifest['roles'][str(i)] = 'malicious'
        manifest['spec'] = partition_spec(c)
        manifest['bundle_digest'] = digest(obj)
        kind = 'attacked' if attacked else 'clean'
        directory = ROOT / 'data' / CAMPAIGN / kind
        if directory.exists():
            assert read_json(directory / 'bundle.json') == obj, 'Refuse to overwrite a different dataset'
        write_json(directory / 'bundle.json', obj)
        write_json(directory / 'manifest.json', manifest)
        load_bundle(c, directory)
        data_specs[kind] = {'path': directory.relative_to(ROOT).as_posix(),
                            'bundle_digest': manifest['bundle_digest'], 'roles': obj['roles'],
                            'file_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.glob('*.json')}}
    configuration_dir = ROOT / 'configs/planned' / CAMPAIGN
    configuration_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    methods = [('hfedsa_ddpg', 'raw'), ('hfedsa_ddpg', 'effective'), ('fedavg', 'effective'),
               ('fltrust', 'effective'), ('rfa', 'effective')]
    schedule = None
    for condition in ('none', 'label_flip', 'backdoor_scale', 'adaptive'):
        for name, representation in methods:
            c = copy.deepcopy(cfg)
            c['defense'].update(name=name, representation=representation)
            c['attack'].update(name=condition, malicious_fraction=0. if condition == 'none' else .2,
                               poison_fraction=1. if condition == 'label_flip' else .2)
            validate(c)
            identifier = f'{name}-{representation}-{condition}-seed2026'
            path = configuration_dir / f'{identifier}.yaml'
            path.write_text(yaml.safe_dump(c, sort_keys=False), encoding='utf-8', newline='\n')
            data = data_specs['clean' if condition == 'none' else 'attacked']
            bundle = load_bundle(c, ROOT / data['path'])
            selected = [select_clients(c, bundle, i) for i in range(1, 51)]
            if schedule is None:
                schedule = selected
            assert selected == schedule
            entries.append({'id': identifier, 'config': path.relative_to(ROOT).as_posix(),
                            'config_hash': digest(c), 'config_file_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                            'data': data['path'], 'data_hash': data['bundle_digest'],
                            'out': f'runs/{CAMPAIGN}/{identifier}'})
    manifest = {'schema': 1, 'campaign': CAMPAIGN, 'authorized_runs': 20, 'seed': 2026, 'rounds': 50,
                'source_hash': source_digest(), 'runs': entries, 'sampling_schedule': schedule,
                'sampling_schedule_digest': digest(schedule), 'data': data_specs,
                'attacker_candidates': malicious,
                'base_data': 'results/clean-pair-20261004/dataset',
                'base_data_hash': original_manifest['bundle_digest'],
                'data_derivation': 'Exact public subset reused; only four client role labels and partition malicious_fraction differ. No text, indices or labels changed.',
                'interpretation': 'One training seed and one previously observed data split: exploratory pilot, not confirmatory significance evidence.'}
    write_json(configuration_dir / 'manifest.json', manifest)
    print(f'Prepared {len(entries)} runs; source {manifest["source_hash"]}; data and all 50 sampling rounds paired.')


if __name__ == '__main__':
    main()
