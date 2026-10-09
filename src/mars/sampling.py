"""Experiment sampling uses identities; the defender never receives them."""
import random


def select_clients(cfg, bundle, round_index):
    seed = cfg['seed'] + 100000 * round_index
    rng = random.Random(seed)
    if cfg['train']['sampling'] == 'uniform':
        return rng.sample(sorted(bundle.clients), cfg['train']['clients_per_round'])
    heterogeneous = sorted(i for i, role in bundle.roles.items() if role == 'heterogeneous')
    ordinary_pool = sorted(set(bundle.clients) - set(heterogeneous))
    candidates = sorted(random.Random(cfg['data']['seed'] + 991).sample(
        ordinary_pool, round(cfg['data']['clients'] * cfg['train']['sampling_malicious_fraction'])))
    active = sorted(i for i, role in bundle.roles.items() if role == 'malicious')
    if cfg['attack']['name'] != 'none' and candidates != active:
        raise ValueError('Malicious identity/sampling mismatch')
    ordinary = sorted(set(ordinary_pool) - set(candidates))
    schedule = cfg['train']['heterogeneous_per_round']
    nh = schedule[(round_index - 1) % len(schedule)]
    nm = cfg['train']['malicious_per_round']
    selected = rng.sample(candidates, nm) + rng.sample(heterogeneous, nh)
    selected += rng.sample(ordinary, cfg['train']['clients_per_round'] - nm - nh)
    rng.shuffle(selected)
    return selected
