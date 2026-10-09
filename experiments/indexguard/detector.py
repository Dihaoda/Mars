"""Paper Section 5: canonical Q/V adapter indices, IOS, and cohesion vetting.

Independent implementation. No ground-truth labels enter the detector.
"""
import numpy as np
import torch
from scipy.cluster.hierarchy import linkage, cut_tree
from scipy.spatial.distance import squareform


def vector(state, base):
    keys = sorted(base)
    assert keys and set(keys) == set(state)
    assert all(('q_proj' in k or 'v_proj' in k) and 'lora_' in k for k in keys)
    result = torch.cat([(state[k] - base[k]).reshape(-1) for k in keys]).float()
    if not torch.isfinite(result).all():
        raise ValueError('Nonfinite adapter delta')
    return result


def top_indices(values, k):
    x = np.abs(np.asarray(values, dtype=np.float32))
    if x.ndim != 1 or not np.isfinite(x).all() or not 1 <= k <= len(x):
        raise ValueError('Invalid salience or K')
    # Stable tie-break is canonical coordinate ID, including the all-zero case.
    return np.argsort(-x, kind='stable')[:k].astype(np.int32)


def ios(sketches):
    arrays = [np.asarray(s, dtype=np.int32) for s in sketches]
    k = len(arrays[0])
    assert k and all(len(np.unique(s)) == k and (s >= 0).all() for s in arrays)
    dimension = max(int(s.max()) for s in arrays) + 1
    packed = []
    for s in arrays:
        mask = np.zeros(dimension, dtype=np.uint8)
        mask[s] = 1
        packed.append(np.packbits(mask))
    popcount = np.asarray([i.bit_count() for i in range(256)], dtype=np.uint8)
    result = np.eye(len(arrays), dtype=np.float64)
    for a in range(len(arrays)):
        for b in range(a):
            result[a, b] = result[b, a] = float(popcount[packed[a] & packed[b]].sum()) / k
    return result


def vet(sketches, clusters=2, gamma=0.05, beta=0.7):
    sims = ios(sketches)
    n = len(sketches)
    if n < 2:
        return [False] * n, {'clusters': [], 'ios': sims.tolist()}
    tree = linkage(squareform(np.maximum(0, 1 - sims), checks=False), method='average')
    labels = cut_tree(tree, n_clusters=min(clusters, n)).reshape(-1)
    members = [np.flatnonzero(labels == j).tolist() for j in sorted(set(labels))]
    largest = max(range(len(members)), key=lambda j: (len(members[j]), -j))
    cohesion = [float((sims[np.ix_(m, m)].sum() - len(m)) / (len(m) * (len(m)-1)))
                if len(m) > 1 else None for m in members]
    defined = [v for v in cohesion if v is not None]
    median = float(np.median(defined)) if defined else None
    flagged = [j != largest and cohesion[j] is not None and median is not None
               and cohesion[j] >= median + gamma and len(members[j]) <= beta * len(members[largest])
               for j in range(len(members))]
    prediction = [bool(flagged[int(j)]) for j in labels]
    return prediction, {'ios': sims.tolist(), 'majority_cluster': largest, 'median_cohesion': median,
                        'clusters': [{'members': m, 'cohesion': cohesion[j], 'suspicious': bool(flagged[j])}
                                     for j, m in enumerate(members)]}


def calibrate(resamples, coverage=0.9, stability=0.8, grid_points=100):
    """One-time calibration, using no role labels; fixed K is shared by the pair.

    Paper does not give a single reproducible default for all warm-up knobs.
    We disclose a 1%-spaced candidate grid (plus exact coverage K and M).
    Each input is three independent one-step bootstrap adapter deltas.
    """
    budgets, records = [], []
    m = None
    for samples in resamples:
        values = np.abs(np.asarray(samples, dtype=np.float32))
        assert values.ndim == 2 and len(values) >= 2 and np.isfinite(values).all()
        m = values.shape[1]
        mean = values.mean(axis=0)
        order = np.argsort(-mean, kind='stable')
        total = float(mean.sum(dtype=np.float64))
        if total <= 0:
            raise ValueError('Zero warm-up salience; refuse uninformative calibration')
        prefix = np.cumsum(mean[order], dtype=np.float64)
        kcov = min(m, int(np.searchsorted(prefix, coverage * total)) + 1)
        orders = [np.argsort(-v, kind='stable') for v in values]
        ranks = []
        for order_i in orders:
            rank = np.empty(m, dtype=np.int32)
            rank[order_i] = np.arange(1, m+1, dtype=np.int32)
            ranks.append(rank)
        # A coordinate belongs to both Top-K sets exactly when max(rank_a, rank_b) <= K.
        overlap_curves = [np.bincount(np.maximum(ranks[a], ranks[b]), minlength=m+1).cumsum()
                          for a in range(len(ranks)) for b in range(a)]
        grid = sorted(set([kcov, m] + [int(np.ceil(m*j/grid_points)) for j in range(1, grid_points+1)]))
        for k in grid:
            if k < kcov:
                continue
            rho = float(np.mean([curve[k]/k for curve in overlap_curves]))
            if rho >= stability:
                break
        budgets.append(k)
        records.append({'coverage_k': kcov, 'selected_k': k, 'stability': rho, 'dimensions': m})
    return int(np.ceil(np.median(budgets))), records


def metrics(predictions, roles):
    """Fixed semantic predictions; ground truth is used only in this evaluator."""
    assert len(predictions) == len(roles)
    result = {}
    for role in ['benign', 'heterogeneous', 'malicious']:
        ids = [i for i, r in enumerate(roles) if r == role]
        positives = sum(bool(predictions[i]) for i in ids)
        result[role] = {'n': len(ids), 'flagged': positives, 'rate': positives/len(ids) if ids else None}
    tp = result['malicious']['flagged']
    fn = result['malicious']['n'] - tp
    fp = result['benign']['flagged'] + result['heterogeneous']['flagged']
    tn = result['benign']['n'] + result['heterogeneous']['n'] - fp
    result.update(tp=tp, fn=fn, fp=fp, tn=tn,
                  fpr=fp/(fp+tn) if fp+tn else None,
                  tpr=tp/(tp+fn) if tp+fn else None,
                  precision=tp/(tp+fp) if tp+fp else None,
                  f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None)
    return result


def aggregate(states, counts, predictions, previous):
    accepted = [i for i, bad in enumerate(predictions) if not bad]
    if not accepted:
        return {k: v.clone() for k, v in previous.items()}
    total = sum(counts[i] for i in accepted)
    return {k: sum((states[i][k].float() * (counts[i]/total) for i in accepted),
                   torch.zeros_like(previous[k], dtype=torch.float32)) for k in previous}
