"""Paired label-preserving domain replacement; all provenance is retained."""
import argparse
import copy
import random
from collections import Counter
from pathlib import Path
import numpy as np
from mars.data import _download, text_hash
from mars.utils import digest, read_json, write_json


def build_pair(raw, cfg):
    rng = random.Random(cfg['data_seed'])
    seen, pools = set(), {}
    for domain, split in [(d, s) for s in ['test', 'train'] for d in ['sst2', 'imdb']]:
        pools[domain, split] = {0: [], 1: []}
        for row in raw[domain, split]:
            fingerprint = text_hash(row['text'])
            if fingerprint not in seen:
                seen.add(fingerprint)
                pools[domain, split][row['label']].append(row)
        for values in pools[domain, split].values():
            rng.shuffle(values)

    def take(domain, split, label, count):
        pool = pools[domain, split][label]
        if len(pool) < count:
            raise ValueError(f'Not enough distinct samples: {domain}/{split}/{label}')
        return [pool.pop() for _ in range(count)]

    tests = {domain: sum((take(domain, 'test', label, cfg['test_per_domain']//2) for label in [0, 1]), [])
             for domain in ['sst2', 'imdb']}
    n = cfg['samples_per_client']
    prng = np.random.default_rng(cfg['data_seed'])
    probabilities = prng.dirichlet([cfg['dirichlet_alpha']]*2, size=cfg['clients'])
    # Controlled-size Dirichlet label skew: sample size itself cannot identify H clients.
    # This differs from the upstream across-client Dirichlet allocator; explicitly disclosed.
    counts = [int(np.floor(p[1]*n + 0.5)) for p in probabilities]
    clients, roles = {}, {}
    for i, positive in enumerate(counts):
        clients[str(i)] = take('sst2', 'train', 0, n-positive) + take('sst2', 'train', 1, positive)
        rng.shuffle(clients[str(i)])
        roles[str(i)] = 'malicious' if i in cfg['malicious_ids'] else 'heterogeneous' if i in cfg['heterogeneous_ids'] else 'benign'
    shifted = copy.deepcopy(clients)
    replacements = []
    for i in cfg['heterogeneous_ids']:
        # Replace a fixed number of slots; each replacement keeps the original label.
        positions = sorted(rng.sample(range(n), int(round(n*cfg['mix_ratio']))))
        for j in positions:
            old = clients[str(i)][j]
            new = take('imdb', 'train', old['label'], 1)[0]
            shifted[str(i)][j] = new
            replacements.append({'client': i, 'position': j, 'old_id': old['id'], 'new_id': new['id'], 'label': old['label']})
    a = {'clients': clients, 'roles': roles, 'test': tests}
    b = {'clients': shifted, 'roles': roles, 'test': tests}
    audit = check_pair(a, b, cfg)
    audit['replacements'] = replacements
    audit['data_hashes'] = [digest(a), digest(b)]
    return a, b, audit


def check_pair(a, b, cfg):
    assert a['roles'] == b['roles'] and a['test'] == b['test']
    changes = {}
    for key, first in a['clients'].items():
        second = b['clients'][key]
        assert len(first) == len(second) == cfg['samples_per_client']
        assert [r['label'] for r in first] == [r['label'] for r in second]
        changed = sum(x['id'] != y['id'] for x, y in zip(first, second))
        expected = round(cfg['samples_per_client']*cfg['mix_ratio']) if int(key) in cfg['heterogeneous_ids'] else 0
        assert changed == expected
        changes[key] = changed
    for obj in [a, b]:
        rows = sum(obj['clients'].values(), []) + sum(obj['test'].values(), [])
        hashes = [text_hash(r['text']) for r in rows]
        assert len(set(hashes)) == len(hashes), 'Duplicate text or train/test overlap'
    return {'status': 'pass', 'changed_samples_per_client': changes, 'label_sequence_preserved': True,
            'test_shared': True, 'normalized_text_disjoint_within_each_condition': True}


def poison(rows, cfg, client, seed):
    result = copy.deepcopy(rows)
    if client not in cfg['malicious_ids']:
        return result, []
    ids = random.Random(seed + client*101).sample(range(len(rows)), round(len(rows)*cfg['poison_rate']))
    for j in ids:
        result[j]['text'] = cfg['trigger'] + ' ' + result[j]['text']
        result[j]['label'] = cfg['target_label']
    return result, ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    cfg = read_json(args.config)
    target = Path(cfg['data_dir'])
    if (target/'pair_audit.json').exists():
        raise RuntimeError('Data already prepared; refuse overwrite')
    raw, revisions = _download({'data': {'revisions': cfg['data_revisions']}})
    a, b, audit = build_pair(raw, cfg)
    target.mkdir(parents=True, exist_ok=True)
    for name, obj in [('mix0', a), ('mix04', b)]:
        write_json(target/(name+'.json'), obj)
    audit['revisions'] = revisions
    audit['config_hash'] = digest(cfg)
    write_json(target/'pair_audit.json', audit)
    public = Path(cfg['results_dir'])
    replacement_files = []
    for i in cfg['heterogeneous_ids']:
        path = f'data_indices/replacements_{i:02d}.json'
        write_json(public/path, [r for r in audit['replacements'] if r['client'] == i])
        replacement_files.append(path)
    audit_public = {k:v for k,v in audit.items() if k != 'replacements'}
    index_files = []
    for name, obj in [('mix0', a), ('mix04', b)]:
        for i, rows in obj['clients'].items():
            path = f'data_indices/{name}_client_{int(i):02d}.json'
            write_json(public/path, [r['id'] for r in rows])
            index_files.append(path)
    write_json(public/'data_manifest.json', {'audit': audit_public, 'index_files': index_files,
        'replacement_files': replacement_files,
        'label_counts': {i: dict(Counter(r['label'] for r in rows)) for i, rows in a['clients'].items()},
        'test_ids': {k:[r['id'] for r in v] for k,v in a['test'].items()}})
    print({'prepared': str(target), 'paired_checks': audit['status']}, flush=True)


if __name__ == '__main__':
    main()
