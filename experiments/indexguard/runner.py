"""Resumable two-condition pilot. Stored roles are never passed to the detector."""
import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch
from mars.utils import digest, read_json, write_json, save_checkpoint, load_checkpoint, git_revision
from .backend import IndexBackend
from .data import poison, check_pair
from .detector import vector, top_indices, calibrate, vet, aggregate, metrics

STOP_REQUESTED = False


def request_stop(*_):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def state_digest(state):
    h = hashlib.sha256()
    for key in sorted(state):
        h.update(key.encode())
        h.update(state[key].contiguous().numpy().tobytes())
    return h.hexdigest()


def provenance(cfg, a, b):
    source = {str(p).replace('\\', '/'): sha_file(p) for root in ['src/mars', 'experiments/indexguard']
              for p in sorted(Path(root).glob('*.py'))}
    env = {p: importlib.metadata.version(p) for p in ['torch', 'numpy', 'scipy', 'transformers', 'peft', 'datasets']}
    env.update(cuda=torch.version.cuda, gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None)
    model_manifest = Path(cfg['model']['name'])/'download_manifest.json'
    model_hash = sha_file(model_manifest) if model_manifest.exists() else None
    if cfg['model']['backend'] != 'tiny_qwen' and model_hash is None:
        raise RuntimeError('Missing verified model download manifest')
    return {'config': cfg, 'data_hashes': [digest(a), digest(b)], 'source_files': source,
            'environment': env, 'model_manifest_sha256': model_hash}


def stopped(root):
    return STOP_REQUESTED or (root/'STOP').exists()


def stop_check(root):
    if stopped(root):
        print('STOP: preserved completed clients; no automatic restart', flush=True)
        raise SystemExit(75)


def warmup(backend, cfg, data, initial, identity):
    root = Path(cfg['run_dir'])
    folder = root/'calibration'
    folder.mkdir(exist_ok=True)
    public = Path(cfg['results_dir'])
    if (public/'calibration.json').exists():
        saved = read_json(public/'calibration.json')
        assert saved['identity'] == identity
        return saved['k']
    settings = cfg['calibration']
    budgets, records = [], []
    for client in range(cfg['clients']):
        samples = []
        poisoned, positions = poison(data['clients'][str(client)], cfg, client, cfg['data_seed']+111)
        for rep in range(settings['resamples']):
            stop_check(root)
            path = folder/f'client_{client:02d}_sample_{rep}.pt'
            seed = cfg['seed'] + 800000 + client*100 + rep
            if path.exists():
                saved = load_checkpoint(path)
                assert saved['identity'] == identity
                samples.append(saved['delta'])
                continue
            indices = random.Random(seed).choices(range(len(poisoned)), k=settings['examples_per_resample'])
            rows = [poisoned[i] for i in indices]
            state, stats = backend.train(initial, rows, seed, max_steps=1)
            delta = vector(state, initial).numpy()
            save_checkpoint(path, {'identity': identity, 'delta': delta, 'seed': seed,
                                   'resample_positions': indices, 'poison_positions': positions, 'training': stats})
            samples.append(delta)
            write_json(public/'progress.json', {'stage': 'calibration', 'client': client, 'resample': rep,
                                                'clients': cfg['clients'], 'identity': identity})
            print(json.dumps({'stage': 'calibration', 'client': client, 'resample': rep, **stats}), flush=True)
        k, rows = calibrate([samples], settings['coverage'], settings['stability'], settings['grid_points'])
        budgets.append(k)
        records.append({'client': client, **rows[0]})
        write_json(public/f'calibration_clients/client_{client:02d}.json', records[-1])
    k = int(np.ceil(np.median(budgets)))
    write_json(public/'calibration.json', {'identity': identity, 'k': k, 'dimensions': records[0]['dimensions'],
                                         'k_fraction': k/records[0]['dimensions'], 'clients': records,
                                         'settings': settings, 'uses_ground_truth': False})
    return k


def evaluate(backend, cfg, state, data):
    result = {}
    for domain, rows in data['test'].items():
        result[domain] = backend.evaluate(state, rows)
        triggered = [{**r, 'text': cfg['trigger']+' '+r['text'], 'label': cfg['target_label']}
                     for r in rows if r['label'] != cfg['target_label']]
        attack = backend.evaluate(state, triggered)
        result[domain]['asr_non_target'] = attack['accuracy']
        result[domain]['asr_denominator'] = attack['n']
    return result


def run_condition(backend, cfg, name, data, initial, identity, k):
    root = Path(cfg['run_dir'])
    folder, public = root/name, Path(cfg['results_dir'])/name
    folder.mkdir(exist_ok=True)
    public.mkdir(parents=True, exist_ok=True)
    checkpoint = folder/'checkpoint.pt'
    if checkpoint.exists():
        cp = load_checkpoint(checkpoint)
        assert cp['identity'] == identity and cp['condition'] == name and cp['k'] == k
        state, start_round = cp['state'], cp['completed_round']+1
    else:
        state, start_round = initial, 1
        save_checkpoint(checkpoint, {'identity': identity, 'condition': name, 'k': k,
                                     'completed_round': 0, 'state': initial})
    for round_id in range(start_round, cfg['rounds']+1):
        stop_check(root)
        start = time.perf_counter()
        base_hash = state_digest(state)
        cache = folder/f'round_{round_id:03d}'
        cache.mkdir(exist_ok=True)
        states, sketches, training, poisoning = [], [], [], []
        for client in range(cfg['clients']):
            stop_check(root)
            path = cache/f'client_{client:02d}.pt'
            if path.exists():
                obj = load_checkpoint(path)
                assert obj['identity'] == identity and obj['base_hash'] == base_hash and obj['k'] == k
                assert obj['client'] == client and obj['round'] == round_id and obj['condition'] == name
            else:
                rows, positions = poison(data['clients'][str(client)], cfg, client, cfg['data_seed']+111)
                trained, stats = backend.train(state, rows, cfg['seed']+round_id*1000+client)
                sketch = top_indices(vector(trained, state).numpy(), k)
                obj = {'identity': identity, 'condition': name, 'round': round_id, 'client': client, 'k': k,
                       'base_hash': base_hash, 'state': trained, 'sketch': sketch,
                       'training': stats, 'poison_positions': positions}
                save_checkpoint(path, obj)
            states.append(obj['state'])
            sketches.append(obj['sketch'])
            training.append(obj['training'])
            poisoning.append(obj['poison_positions'])
            write_json(Path(cfg['results_dir'])/'progress.json', {'stage': 'training', 'condition': name,
                       'round': round_id, 'client_completed': client+1, 'clients': cfg['clients'], 'identity': identity})
            print(json.dumps({'condition': name, 'round': round_id, 'client': client, **obj['training']}), flush=True)
        stop_check(root)
        flags, details = vet(sketches, **cfg['detector'])
        counts = [len(data['clients'][str(i)]) for i in range(cfg['clients'])]
        state = aggregate(states, counts, flags, state)
        roles = [data['roles'][str(i)] for i in range(cfg['clients'])]
        accepted_n = sum(n for n, bad in zip(counts, flags) if not bad)
        evaluation = evaluate(backend, cfg, state, data)
        report = {'condition': name, 'round': round_id, 'identity': identity, 'k': k,
                  'base_hash': base_hash, 'output_hash': state_digest(state),
                  'detection': metrics(flags, roles), 'evaluation': evaluation, 'vetting': details,
                  'clients': [{'id': i, 'role': roles[i], 'predicted_malicious': flags[i],
                               'weight': 0 if flags[i] else counts[i]/accepted_n, 'n': counts[i],
                               'poison_positions': poisoning[i], 'training': training[i]}
                              for i in range(cfg['clients'])],
                  'seconds_this_invocation': time.perf_counter()-start,
                  'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0}
        write_json(public/f'round_{round_id:03d}.json', report)
        save_checkpoint(cache/'global.pt', {'state': state, 'identity': identity, 'round': round_id})
        save_checkpoint(checkpoint, {'identity': identity, 'condition': name, 'k': k,
                                     'completed_round': round_id, 'state': state})
        print(json.dumps({'completed_round': round_id, 'condition': name, 'detection': report['detection'],
                          'evaluation': evaluation}), flush=True)
    reports = [read_json(public/f'round_{r:03d}.json') for r in range(1, cfg['rounds']+1)]
    predictions = [x['predicted_malicious'] for r in reports for x in r['clients']]
    roles = [x['role'] for r in reports for x in r['clients']]
    backend.load(state)
    backend.model.save_pretrained(folder/'final_adapter', save_embedding_layers=False)
    write_json(public/'summary.json', {'condition': name, 'identity': identity, 'completed_rounds': cfg['rounds'],
                                     'pooled_client_round_detection': metrics(predictions, roles),
                                     'final_evaluation': reports[-1]['evaluation'],
                                     'note': 'Client-round rates are descriptive, not independent replicates.'})


def smoke(backend, cfg):
    rows = [{'text': 'This film was enjoyable.' if i%2 else 'This film was disappointing.', 'label': i%2}
            for i in range(64)]
    initial = backend.state()
    state, stats = backend.train(initial, rows, cfg['seed'], max_steps=1)
    assert state_digest(initial) != state_digest(state)
    result = backend.evaluate(state, rows[:4])
    replay, _ = backend.train(initial, rows, cfg['seed'], max_steps=1)
    assert state_digest(replay) == state_digest(state), 'Deterministic local replay failed'
    manifest = {'status': 'pass', 'not_research_evidence': True, 'training': stats, 'evaluation': result,
                'deterministic_replay': True, 'coordinate_dimension': len(vector(state, initial)),
                'max_gpu_allocated_bytes': torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0}
    write_json(Path(cfg['results_dir'])/'preflight.json', manifest)
    print(json.dumps(manifest), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    cfg = read_json(args.config)
    root = Path(cfg['run_dir'])
    root.mkdir(parents=True, exist_ok=True)
    # Kernel releases this lock on exit, including SIGKILL; it cannot become a stale PID lock.
    import fcntl
    lock = (root/'RUNNING.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    stop_check(root)
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    backend = IndexBackend(cfg)
    if args.smoke:
        smoke(backend, cfg)
        return
    a = read_json(Path(cfg['data_dir'])/'mix0.json')
    b = read_json(Path(cfg['data_dir'])/'mix04.json')
    check_pair(a, b, cfg)
    prov = provenance(cfg, a, b)
    identity = digest(prov)
    public = Path(cfg['results_dir'])
    if (public/'provenance.json').exists():
        assert read_json(public/'provenance.json')['identity'] == identity, 'Resume identity mismatch'
    else:
        write_json(public/'provenance.json', {'identity': identity, 'code_commit': git_revision(), **prov})
    initial_path = root/'initial.pt'
    initial = backend.state()
    if initial_path.exists():
        obj = load_checkpoint(initial_path)
        assert obj['identity'] == identity and state_digest(obj['state']) == state_digest(initial)
    else:
        save_checkpoint(initial_path, {'identity': identity, 'state': initial})
    if not (public/'initial_evaluation.json').exists():
        write_json(public/'initial_evaluation.json', evaluate(backend, cfg, initial, a))
    k = warmup(backend, cfg, a, initial, identity)
    for name, data in [('mix0', a), ('mix04', b)]:
        run_condition(backend, cfg, name, data, initial, identity, k)
        subprocess.run([sys.executable, 'scripts/collect_indexguard.py', '--config', args.config,
                        '--archive', name], check=True)
    write_json(public/'progress.json', {'stage': 'complete', 'identity': identity, 'conditions': ['mix0', 'mix04']})


if __name__ == '__main__':
    main()
