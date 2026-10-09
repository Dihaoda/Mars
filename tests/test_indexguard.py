import copy
from pathlib import Path
import numpy as np
import torch
import pytest
from experiments.indexguard.detector import ios, vet, aggregate, metrics, calibrate, top_indices, vector
from experiments.indexguard.data import build_pair, check_pair, poison
from experiments.indexguard.backend import IndexBackend
from mars.utils import read_json


def test_no_forced_positive_for_identical_or_disjoint_supports():
    for sketches in [[np.arange(20)]*30, [np.arange(i*20, (i+1)*20) for i in range(30)]]:
        prediction, _ = vet(sketches)
        assert not any(prediction)


def test_cohesive_minority_flagged_without_truth_labels():
    # 7 moderately similar clients, 3 identical supports on a disjoint coordinate block.
    benign = [np.r_[np.arange(5), np.arange(20+i*5, 25+i*5)] for i in range(7)]
    attack = [np.arange(100, 110)]*3
    flags, _ = vet(benign+attack)
    assert flags == [False]*7+[True]*3
    ordinary_truth = ['benign']*7+['malicious']*3
    shifted_truth = ['benign']*7+['heterogeneous']*3
    assert metrics(flags, ordinary_truth)['tpr'] == 1
    assert metrics(flags, shifted_truth)['heterogeneous']['rate'] == 1
    assert vet(benign+attack)[0] == flags  # Truth affects reporting only.


def test_ios_packed_matches_set_intersection():
    rng = np.random.default_rng(10)
    sketches = [rng.choice(317, 75, replace=False) for _ in range(10)]
    reference = [[len(set(a)&set(b))/75 for b in sketches] for a in sketches]
    np.testing.assert_array_equal(ios(sketches), reference)


def test_aggregation_excludes_flagged_updates():
    states = [{'w': torch.tensor([x])} for x in [2., 8., 1000.]]
    assert aggregate(states, [1, 3, 10], [False, False, True], states[0])['w'].item() == 6.5
    assert aggregate(states, [1, 3, 10], [True]*3, states[0])['w'].item() == 2.


def test_fixed_semantics_and_undefined_precision():
    result = metrics([True, False, False], ['benign', 'heterogeneous', 'malicious'])
    assert (result['fp'], result['fn'], result['tp'], result['tn']) == (1, 1, 0, 1)
    assert result['fpr'] == 0.5 and result['tpr'] == 0
    assert metrics([False], ['benign'])['precision'] is None
    assert metrics([False], ['benign'])['tpr'] is None


def test_calibration_smallest_disclosed_grid_candidate():
    rng = np.random.default_rng(11)
    samples = rng.normal(size=(3, 100)).astype(np.float32)
    k, rows = calibrate([samples], .7, .8, 10)
    candidates = sorted(set([rows[0]['coverage_k'], 100]+list(range(10, 101, 10))))
    def overlap(budget):
        sets = [set(top_indices(x, budget)) for x in samples]
        return np.mean([len(sets[a]&sets[b])/budget for a in range(3) for b in range(a)])
    assert k == next(v for v in candidates if v >= rows[0]['coverage_k'] and overlap(v) >= .8)
    np.testing.assert_allclose(rows[0]['stability'], overlap(k))


def example_config():
    cfg = read_json(Path(__file__).resolve().parents[1]/'configs/indexguard/pilot.json')
    cfg.update(clients=6, malicious_ids=[0], heterogeneous_ids=[4, 5], samples_per_client=10,
               test_per_domain=4, poison_rate=.2)
    return cfg


def fake_data():
    return {(domain, split): [{'id': f'{domain}:{split}:{i}', 'domain': domain, 'label': i%2,
                              'text': f'example {domain} {split} number {i}'} for i in range(200)]
            for domain in ['sst2', 'imdb'] for split in ['test', 'train']}


def test_pair_controls_sample_counts_labels_and_no_leakage():
    cfg = example_config()
    a, b, audit = build_pair(fake_data(), cfg)
    assert audit['changed_samples_per_client'] == {'0': 0, '1': 0, '2': 0, '3': 0, '4': 4, '5': 4}
    assert a == build_pair(fake_data(), cfg)[0]
    assert check_pair(a, b, cfg)['status'] == 'pass'
    for i in [0, 4]:
        pa, ia = poison(a['clients'][str(i)], cfg, i, 99)
        pb, ib = poison(b['clients'][str(i)], cfg, i, 99)
        assert ia == ib and len(ia) == (2 if i == 0 else 0)
        if i == 0:
            assert pa == pb


def test_tiny_local_training_is_deterministic_and_changes_qv_delta():
    cfg = example_config()
    cfg['model'].update(backend='tiny_qwen', rank=2, alpha=4, max_length=32, gradient_checkpointing=False)
    cfg['runtime'].update(device='cpu', threads=1)
    cfg['train'].update(batch_size=2, gradient_accumulation=1, local_epochs=1)
    backend = IndexBackend(cfg)
    initial = backend.state()
    rows = [{'text': 'a good film', 'label': 1}, {'text': 'a bad film', 'label': 0}]
    first, _ = backend.train(initial, rows, 99)
    second, _ = backend.train(initial, rows, 99)
    assert torch.count_nonzero(vector(first, initial)) > 0
    assert all(torch.equal(first[k], second[k]) for k in initial)


def test_client_checkpoint_resume_matches_uninterrupted_training(tmp_path):
    from experiments.indexguard.runner import run_condition
    from mars.utils import load_checkpoint
    cfg = example_config()
    cfg.update(clients=3, malicious_ids=[0], heterogeneous_ids=[2], rounds=2, samples_per_client=4, test_per_domain=2)
    cfg['model'].update(backend='tiny_qwen', rank=2, alpha=4, max_length=32, gradient_checkpointing=False)
    cfg['runtime'].update(device='cpu', threads=1)
    cfg['train'].update(batch_size=2, gradient_accumulation=1, local_epochs=1)
    data, _, _ = build_pair(fake_data(), cfg)
    outputs = []
    for tag in ['whole', 'resumed']:
        cfg['run_dir'], cfg['results_dir'] = str(tmp_path/tag/'runs'), str(tmp_path/tag/'results')
        Path(cfg['run_dir']).mkdir(parents=True)
        backend = IndexBackend(cfg)
        initial = backend.state()
        if tag == 'resumed':
            original_train = backend.train
            calls = [0]
            def interrupted(*args, **kwargs):
                calls[0] += 1
                if calls[0] == 3:
                    raise RuntimeError('simulated interruption during third client')
                return original_train(*args, **kwargs)
            backend.train = interrupted
            with pytest.raises(RuntimeError, match='simulated interruption'):
                run_condition(backend, cfg, 'mix0', data, initial, 'unit-test', 50)
            backend = IndexBackend(cfg)
            with pytest.raises(AssertionError):
                run_condition(backend, cfg, 'mix0', data, initial, 'wrong-identity', 50)
        run_condition(backend, cfg, 'mix0', data, initial, 'unit-test', 50)
        outputs.append(load_checkpoint(Path(cfg['run_dir'])/'mix0/checkpoint.pt')['state'])
    assert all(torch.equal(outputs[0][k], outputs[1][k]) for k in outputs[0])
