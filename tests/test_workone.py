import copy

import numpy as np
import pytest
import torch

from mars.data import prepare
from mars.geometry import delta, raw_delta
from mars.legacy_hfedsa import HFedSAStatic, HFedSAConfig
from mars.workone import WorkOneDefense
from mars.sampling import select_clients
from mars.runner import run
from mars.utils import load_checkpoint, read_json, write_json, digest


def state(seed):
    g = torch.Generator().manual_seed(seed)
    return {'x.lora_A.weight': torch.randn(2, 7, generator=g),
            'x.lora_B.weight': torch.randn(9, 2, generator=g)}


@pytest.mark.parametrize('representation', ['raw', 'effective'])
def test_port_matches_supplied_scoring_on_dense_updates(small_config, representation):
    cfg = copy.deepcopy(small_config)
    cfg['defense'].update(name='hfedsa_workone_static', representation=representation)
    base, root = state(0), state(1)
    states = [state(i) for i in range(2, 12)]
    port = WorkOneDefense(cfg)
    weights, rows, info = port.score(states, base, root, 2.0, list(range(10)))
    def flatten(local):
        return raw_delta(local, base).float() if representation == 'raw' else delta(local, base, 2.0)['x.'].dense().flatten().float()
    original = HFedSAStatic(HFedSAConfig())
    _, expected = original.aggregate([(i, {'u': flatten(s)}) for i, s in enumerate(states)], {'u': flatten(root)})
    np.testing.assert_allclose(weights, [r['weight'] for r in expected], rtol=1e-4, atol=2e-6)
    for actual, target in zip(rows, expected):
        for key in ('direction', 'intensity', 'sparsity', 'p_malicious'):
            assert actual[key] == pytest.approx(target[key], rel=1e-4, abs=2e-6)


def test_true_next_state_and_empty_flagged_reward(small_config):
    cfg = copy.deepcopy(small_config)
    cfg['defense']['name'] = 'hfedsa_ddpg'
    controller = WorkOneDefense(cfg)
    controller.score([state(i) for i in range(3, 9)], state(1), state(2), 2, list(range(6)))
    first = controller.beta_agent.previous_state.copy()
    no_flags = [{'weight': 1 / 6, 'predicted_malicious': False}] * 6
    controller.observe_validation({'x': {'loss': 1.0, 'accuracy': 0.0}}, no_flags)
    controller.score([state(i) for i in range(9, 15)], state(1), state(2), 2, list(range(6)))
    previous, action, reward, next_state = controller.beta_agent.replay.items[0]
    np.testing.assert_array_equal(previous, first)
    np.testing.assert_array_equal(next_state, controller.beta_agent.previous_state)
    assert not np.array_equal(previous, next_state)
    info = controller.observe_validation({'x': {'loss': 0.9, 'accuracy': 0.5}}, no_flags)
    assert 0.55 <= info['reward'] <= 0.6


def test_stratified_pairing_without_ground_truth_detector_input(small_config, tmp_path):
    cfg = copy.deepcopy(small_config)
    cfg['data'].update(clients=20, heterogeneous_clients=5)
    cfg['train'].update(sampling='stratified', clients_per_round=10)
    clean = prepare(cfg, tmp_path / 'clean')
    clean_cfg = copy.deepcopy(cfg)
    cfg['attack'].update(name='label_flip', malicious_fraction=0.2)
    attacked = prepare(cfg, tmp_path / 'attack')
    for round_index in range(1, 51):
        chosen = select_clients(cfg, attacked, round_index)
        assert chosen == select_clients(clean_cfg, clean, round_index)
        assert sum(attacked.roles[i] == 'malicious' for i in chosen) == 2
        assert sum(attacked.roles[i] == 'heterogeneous' for i in chosen) == [2, 3][(round_index - 1) % 2]
        assert all(clean.clients[i] == attacked.clients[i] for i in chosen)


def test_ddpg_training_resume_and_test_label_isolation(small_config, tmp_path):
    cfg = copy.deepcopy(small_config)
    cfg['defense']['name'] = 'hfedsa_ddpg'
    cfg['train'].update(rounds=18, max_steps=1)
    prepare(cfg, tmp_path / 'data')
    run(cfg, tmp_path / 'data', tmp_path / 'full')
    run(cfg, tmp_path / 'data', tmp_path / 'resumed', until_round=16)
    # Changing held-out labels cannot change a completed-round checkpoint;
    # a separate run with changed test data tests both scoring and reward isolation.
    run(cfg, tmp_path / 'data', tmp_path / 'resumed', resume=True)
    a = load_checkpoint(tmp_path / 'full/checkpoint.pt')
    b = load_checkpoint(tmp_path / 'resumed/checkpoint.pt')
    assert a['controller'].controller_updates == b['controller'].controller_updates == 2
    for key, value in a['global_state'].items():
        torch.testing.assert_close(value, b['global_state'][key], rtol=0, atol=0)
    for key, value in a['controller'].beta_agent.actor.state_dict().items():
        torch.testing.assert_close(value, b['controller'].beta_agent.actor.state_dict()[key], rtol=0, atol=0)
    assert [r['controller'] for r in a['history']] == [r['controller'] for r in b['history']]
    prepare(cfg, tmp_path / 'changed')
    obj = read_json(tmp_path / 'changed/bundle.json')
    for rows in obj['test'].values():
        for row in rows:
            row['label'] = 1 - row['label']
    manifest = read_json(tmp_path / 'changed/manifest.json')
    manifest['bundle_digest'] = digest(obj)
    write_json(tmp_path / 'changed/bundle.json', obj)
    write_json(tmp_path / 'changed/manifest.json', manifest)
    run(cfg, tmp_path / 'changed', tmp_path / 'isolated')
    c = load_checkpoint(tmp_path / 'isolated/checkpoint.pt')
    for key, value in a['global_state'].items():
        torch.testing.assert_close(value, c['global_state'][key], rtol=0, atol=0)
    assert [r['controller'] for r in a['history']] == [r['controller'] for r in c['history']]


@pytest.mark.parametrize('attack', ['backdoor_scale', 'adaptive'])
def test_new_attack_paths_and_audit(small_config, tmp_path, attack):
    cfg = copy.deepcopy(small_config)
    cfg['defense']['name'] = 'hfedsa_ddpg'
    cfg['train'].update(rounds=2, clients_per_round=6, max_steps=1)
    cfg['attack'].update(name=attack, malicious_fraction=1/6, adaptive_iterations=2, adaptive_probe_samples=4)
    bundle = prepare(cfg, tmp_path / 'data')
    result = run(cfg, tmp_path / 'data', tmp_path / 'run')
    assert result['status'] == 'complete'
    if attack == 'backdoor_scale':
        assert all(m['n'] == cfg['data']['test_per_domain'] // 2 for m in result['asr'].values())
        clients = read_json(tmp_path / 'run/round_001.json')['clients']
        bad = [r for r in clients if r['true_role'] == 'malicious'][0]
        assert bad['poisoned_examples'] > 0 and 'scaling_projection_error' in bad
    else:
        for index in (1, 2):
            audit = read_json(tmp_path / f'run/attack_{index:03}.json')[0]
            assert audit['query_count'] <= 9
            assert audit['best']['effective_update_norm'] <= audit['norm_bound'] * 1.000001
            assert set(audit['probe_ids']) <= {r['id'] for r in bundle.clients[audit['client_id']]}
            assert np.isfinite(audit['best']['objective'])
            cache = load_checkpoint(tmp_path / f'run/updates_{index:03}.pt')
            assert set(cache['original_malicious']) == {audit['client_id']}
