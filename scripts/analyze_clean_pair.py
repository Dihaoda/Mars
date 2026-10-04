"""Analyse the six prespecified runs; no training or parameter selection."""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
import statistics

import torch
from mars.config import load_config
from mars.defenses import decide
from mars.metrics import detection
from mars.report import summarize
from mars.utils import digest, load_checkpoint, read_json, write_json


def csv_write(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', default='configs/planned/clean-pair/manifest.json')
    parser.add_argument('--out', default='results/clean-pair-20261004')
    args = parser.parse_args()
    torch.set_num_threads(2)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    jobs = read_json(args.manifest)['jobs']
    assert len(jobs) == 6
    runs, observations = {}, []
    for job in jobs:
        directory = Path(job['output'])
        cfg = load_config(job['config'])
        metadata = read_json(directory / 'metadata.json')
        summary = read_json(directory / 'summary.json')
        assert digest(cfg) == job['config_hash'] == summary['config_hash'] == metadata['config_hash']
        assert cfg == metadata['config']
        assert summary['status'] == 'complete' and summary['completed_rounds'] == 10
        assert metadata['synthetic'] is False and cfg['attack']['name'] == 'none'
        rounds = [read_json(directory / f'round_{i:03}.json') for i in range(1, 11)]
        assert all(len(r['clients']) == 10 for r in rounds)
        assert all(r['summary']['detection']['malicious_n'] == 0 for r in rounds)
        seed, representation = cfg['seed'], cfg['defense']['representation']
        assert (seed, representation) not in runs
        runs[seed, representation] = (directory, metadata, summary, rounds)
        detection_summary = summary['pooled_client_round_detection']
        observations.append({'seed': seed, 'representation': representation,
                             'heterogeneous_fpr': detection_summary['heterogeneous_rate'],
                             'heterogeneous_flagged': detection_summary['heterogeneous_flagged'],
                             'heterogeneous_scored_client_rounds': detection_summary['heterogeneous_scored_n'],
                             'benign_fpr': detection_summary['benign_rate'],
                             'benign_flagged': detection_summary['benign_flagged'],
                             'benign_scored_client_rounds': detection_summary['benign_scored_n'],
                             'heterogeneous_weight_ratio': detection_summary['heterogeneous_weight_ratio'],
                             'test_macro_accuracy': summary['test_macro_accuracy'],
                             'test_sst2_accuracy': summary['test']['sst2']['accuracy'],
                             'test_imdb_accuracy': summary['test']['imdb']['accuracy'],
                             'round_seconds_total': summary['total_round_seconds'],
                             'peak_gpu_gib': max(r['summary']['peak_gpu_memory_bytes'] for r in rounds) / 2**30,
                             'fallback_rounds': sum(r['summary']['defense']['fallback'] is not None for r in rounds)})
    assert set(runs) == {(seed, rep) for seed in (2026, 2027, 2028) for rep in ('raw', 'effective')}
    for field in ['source_hash', 'model_revision', 'data_hash', 'git_commit']:
        assert len({run[1][field] for run in runs.values()}) == 1, field
    checks = []
    differences = []
    for seed in (2026, 2027, 2028):
        raw, effective = runs[seed, 'raw'], runs[seed, 'effective']
        configs = [copy.deepcopy(x[1]['config']) for x in (raw, effective)]
        for cfg in configs:
            cfg['defense'].pop('representation')
        assert configs[0] == configs[1]
        assert [r['summary']['selected_clients'] for r in raw[3]] == [r['summary']['selected_clients'] for r in effective[3]]
        a, b = [load_checkpoint(x[0] / 'updates_001.pt') for x in (raw, effective)]
        same = all(torch.equal(x[k], y[k]) for x, y in zip([a['base'], a['root'], *a['locals']],
                                                          [b['base'], b['root'], *b['locals']]) for k in x)
        assert same, f'First-round inputs differ for seed {seed}'
        checks.append({'seed': seed, 'only_representation_differs': True,
                       'same_participation_all_rounds': True, 'same_first_round_base_root_local_tensors': True})
        r, e = [next(row for row in observations if row['seed'] == seed and row['representation'] == rep)
                for rep in ('raw', 'effective')]
        differences.append({'seed': seed, **{metric: e[metric] - r[metric]
                                             if e[metric] is not None and r[metric] is not None else None
                                             for metric in ['heterogeneous_fpr', 'benign_fpr', 'heterogeneous_weight_ratio',
                                                            'test_macro_accuracy', 'test_sst2_accuracy', 'test_imdb_accuracy']}})
    csv_write(output / 'per_seed.csv', observations)
    csv_write(output / 'paired_differences.csv', differences)
    aggregate_differences = {}
    for metric in list(differences[0])[1:]:
        values = [row[metric] for row in differences if row[metric] is not None]
        aggregate_differences[metric] = {'n_seeds': len(values), 'mean_effective_minus_raw': statistics.mean(values) if values else None,
                                        'sample_std': statistics.stdev(values) if len(values) > 1 else None}
    summarize([str(r[0]) for r in runs.values()], output / 'group_summary.json')
    write_json(output / 'paired_checks.json', {'passed': True, 'checks': checks})
    write_json(output / 'paired_summary.json', {'differences': differences, 'statistics': aggregate_differences,
                                                'note': 'Three training seeds on one data split; descriptive paired differences, no significance claim.'})
    # Counterfactual scores use exactly the cached updates of each source trajectory.
    replay_clients, replay_summary = [], []
    for (seed, source_representation), (directory, metadata, summary, rounds) in runs.items():
        pooled = {'raw': [], 'effective': []}
        for index, original in enumerate(rounds, 1):
            cache = load_checkpoint(directory / f'updates_{index:03}.pt')
            roles = {row['client_id']: row['true_role'] for row in original['clients']}
            for representation in ('raw', 'effective'):
                cfg = copy.deepcopy(cache['config'])
                cfg['defense']['representation'] = representation
                _, diagnostics, info = decide(cache['locals'], cache['counts'], cache['base'], cache['root'],
                                               cache['scale'], cfg, cache['seed'])
                for client, row in zip(cache['selected'], diagnostics):
                    measured = {'true_role': roles[client], **row}
                    pooled[representation].append(measured)
                    replay_clients.append({'seed': seed, 'source_representation': source_representation,
                                           'replay_representation': representation, 'round': index,
                                           'client_id': client, **measured})
                if representation == source_representation:
                    recorded = {row['client_id']: row for row in original['clients']}
                    for client, row in zip(cache['selected'], diagnostics):
                        assert row['predicted_malicious'] == recorded[client]['predicted_malicious']
                        assert abs(row['weight'] - recorded[client]['weight']) < 1e-8
        for representation, rows in pooled.items():
            replay_summary.append({'seed': seed, 'source_representation': source_representation,
                                   'replay_representation': representation, **detection(rows)})
        print('Replayed', seed, source_representation, flush=True)
    csv_write(output / 'replay_clients.csv', replay_clients)
    csv_write(output / 'replay_summary.csv', replay_summary)
    write_json(output / 'analysis_status.json', {'status': 'complete', 'runs': 6, 'rounds': 60,
                                                 'cached_rounds_replayed': 60, 'score_conditions_per_cache': 2,
                                                 'code_commit': next(iter(runs.values()))[1]['git_commit'],
                                                 'source_hash': next(iter(runs.values()))[1]['source_hash'],
                                                 'data_hash': next(iter(runs.values()))[1]['data_hash'],
                                                 'warning': 'Replay has no new trained-model accuracy or ASR. No attack was evaluated.'})
    print(json.dumps({'paired_statistics': aggregate_differences, 'checks_passed': True}, indent=2), flush=True)


if __name__ == '__main__':
    main()
