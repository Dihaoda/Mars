"""White-box feature-matched, task-damaging attack with a fixed query budget.

The attacker knows root and current heterogeneous updates (oracle stress test).
Search is in a three-dimensional low-rank update subspace, not an unrestricted
dense model. Every candidate is refactored to the same LoRA rank before scoring.
"""
from __future__ import annotations

import random
import numpy as np

from .geometry import combine, delta, effective, norm, refactor
from .workone import workone_features


def adaptive_attack(backend, states, base, root, selected, roles, client_data, cfg, seed):
    scale = backend.scale
    settings = cfg['attack']
    representation = cfg['defense']['representation'] if cfg['defense']['name'].startswith('hfedsa') else 'effective'
    shifted = [s for s, i in zip(states, selected) if roles[i] == 'heterogeneous']
    if not shifted:
        raise ValueError('Adaptive oracle attack requires a sampled heterogeneous reference')
    target_rows = np.asarray([workone_features(s, base, root, scale, representation)[0] for s in shifted])
    target = target_rows.mean(axis=0)
    feature_scale = np.maximum(target_rows.std(axis=0), [0.1, 0.5, 0.05])
    shifted_delta = combine([delta(s, base, scale) for s in shifted], [1 / len(shifted)] * len(shifted))
    root_delta = delta(root, base, scale)
    attacked, records, originals = [], [], {}
    for local, client in zip(states, selected):
        if roles[client] != 'malicious':
            attacked.append(local)
            continue
        originals[client] = local
        rng = np.random.default_rng(seed + 7919 + client)
        sample_rng = random.Random(seed + 7919 + client)
        size = settings['adaptive_probe_samples'] // 2
        probe = [row for label in (0, 1) for row in sample_rng.sample(
            [r for r in client_data[client] if r['label'] == label], size)]
        baseline_loss = backend.attack_loss(base, probe)
        own_loss = backend.attack_loss(local, probe)
        own_delta = delta(local, base, scale)
        basis = [root_delta, own_delta, shifted_delta]
        max_norm = settings['adaptive_norm_bound'] * max(norm(root_delta), norm(own_delta), 1e-8)
        queries = 2

        def evaluate(parameters):
            nonlocal queries
            desired = combine(basis, parameters)
            length = norm(desired)
            bounded = parameters * min(1.0, max_norm / max(length, 1e-12))
            matrix = combine([effective(base, scale), *basis], [1.0, *bounded])
            candidate, projection = refactor(matrix, base, scale)
            actual_norm = norm(delta(candidate, base, scale))
            # Truncation can move the candidate relative to a nonzero base.
            # Reject a candidate outside the declared effective-update budget.
            if actual_norm > max_norm * (1 + 1e-6):
                return None, {'objective': -1e30, 'rejected_norm': True}
            feat = np.asarray(workone_features(candidate, base, root, scale, representation)[0])
            match = float(np.mean(((feat - target) / feature_scale) ** 2))
            loss = backend.attack_loss(candidate, probe)
            queries += 1
            harm = (loss - baseline_loss) / max(baseline_loss, 0.1)
            objective = harm - settings['adaptive_match_penalty'] * match
            if not np.isfinite(objective):
                raise FloatingPointError('Non-finite adaptive objective')
            return candidate, {'objective': objective, 'attacker_clean_loss': loss,
                               'normalized_harm': harm, 'feature_distance': match,
                               'features': feat.tolist(), 'effective_update_norm': actual_norm,
                               'projection_error': projection['svd_relative_error'],
                               'parameters': bounded.tolist(), 'rejected_norm': False}

        parameters = np.asarray([-1.0, 0.0, 0.0])
        best, best_info = evaluate(parameters)
        if best is None:
            raise FloatingPointError('Negative-root initialization exceeded attack budget')
        trace = [{'iteration': 0, **best_info}]
        for iteration in range(1, settings['adaptive_iterations'] + 1):
            direction = rng.choice([-1.0, 1.0], size=3)
            sigma = settings['adaptive_sigma']
            _, plus = evaluate(np.clip(parameters + sigma * direction, -10, 10))
            _, minus = evaluate(np.clip(parameters - sigma * direction, -10, 10))
            if plus['rejected_norm'] or minus['rejected_norm']:
                gradient = np.zeros(3)
            else:
                gradient = (plus['objective'] - minus['objective']) / (2 * sigma) * direction
                gradient /= max(1.0, float(np.linalg.norm(gradient)))
            parameters = np.clip(parameters + settings['adaptive_step'] * gradient, -10, 10)
            candidate, info = evaluate(parameters)
            trace.append({'iteration': iteration, **info})
            if candidate is not None and info['objective'] > best_info['objective']:
                best, best_info = candidate, info
        attacked.append(best)
        records.append({'client_id': client, 'knowledge': 'oracle_root_and_heterogeneous_updates',
                        'representation_targeted': representation, 'query_count': queries,
                        'probe_source': 'attacker_own_training_data_only', 'probe_ids': [r['id'] for r in probe],
                        'baseline_loss': baseline_loss, 'honest_local_loss': own_loss,
                        'target_features': target.tolist(), 'feature_scale': feature_scale.tolist(),
                        'norm_bound': max_norm, 'best': best_info, 'trace': trace})
    return attacked, records, originals
