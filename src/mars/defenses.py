"""Defense interfaces deliberately receive no client ground-truth labels."""
from __future__ import annotations

import itertools
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from .geometry import delta, features, inner

ROLES = ("benign", "heterogeneous", "malicious")


def _rfa_weights(states, base, scale, sample_weights):
    updates = [delta(s, base, scale) for s in states]
    gram = np.array([[inner(a, b) for b in updates] for a in updates])
    weights = sample_weights / sample_weights.sum()
    for _ in range(50):
        distances = np.sqrt(np.maximum(0, np.diag(gram) - 2 * gram @ weights + weights @ gram @ weights))
        new = sample_weights / np.maximum(distances, 1e-10)
        new /= new.sum()
        if np.linalg.norm(new - weights) < 1e-7:
            weights = new
            break
        weights = new
    return weights


def decide(states, sample_counts, base, root, scale, cfg, seed, anchors=(), probe=None):
    """Return weights, per-client diagnostic rows and method metadata.

    Anchors are server-generated (state, known_role) pairs, never private client labels.
    GMM responsibilities are model scores, not statistically calibrated risk guarantees.
    """
    f = cfg["defense"]
    name = f["name"]
    representation = "effective" if name in {"fltrust", "rfa"} else f["representation"]
    x = np.asarray([features(s, base, root, scale, representation) for s in states])
    if not np.isfinite(x).all():
        raise FloatingPointError("Non-finite client features")
    counts = np.asarray(sample_counts, dtype=float)
    posterior = np.zeros((len(states), 3))
    posterior[:, 0] = 1
    info = {"name": name, "representation": representation, "fallback": None, "probe_count": 0}
    has_detector = name in {"hfedsa", "anchor_gmm", "hybrid"}
    probed = np.zeros(len(states), dtype=bool)
    if name == "fedavg":
        weights = counts.copy()
    elif name == "fltrust":
        weights = counts * np.maximum(0, x[:, 0])
    elif name == "rfa":
        weights = _rfa_weights(states, base, scale, counts)
    else:
        anchor_x = np.asarray([features(s, base, root, scale, representation) for s, role in anchors]) if anchors else None
        fit_x = x if name == "hfedsa" else np.vstack([x, anchor_x]) if anchors else x
        if name != "hfedsa" and set(role for s, role in anchors) != set(ROLES):
            raise ValueError("All three server anchor roles are required")
        if len(fit_x) < 3 or len(np.unique(fit_x.round(12), axis=0)) < 3:
            info["fallback"] = "degenerate_features_root_only"
        else:
            scaler = StandardScaler().fit(fit_x)
            gmm = GaussianMixture(n_components=3, covariance_type="full", reg_covar=f["gmm_reg"],
                                  n_init=3, max_iter=200, random_state=seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                gmm.fit(scaler.transform(fit_x))
            if not gmm.converged_:
                info["fallback"] = "gmm_not_converged_root_only"
            if name == "hfedsa":
                means = scaler.inverse_transform(gmm.means_)
                order = np.argsort(means[:, 1], kind="stable")
                mapping = [int(order[1]), int(order[0]), int(order[2])]
            else:
                responsibility = gmm.predict_proba(scaler.transform(anchor_x))
                compatibility = np.stack([responsibility[[role == label for s, role in anchors]].mean(axis=0)
                                          for label in ROLES])
                mapping = max(itertools.permutations(range(3)), key=lambda p: sum(compatibility[j, p[j]] for j in range(3)))
                info["anchor_match"] = [float(compatibility[j, mapping[j]]) for j in range(3)]
                if min(info["anchor_match"]) < f["anchor_match_min"]:
                    info["fallback"] = "weak_anchor_match_root_only"
            posterior = gmm.predict_proba(scaler.transform(x))[:, mapping]
            if name == "hybrid" and info["fallback"] is None:
                if probe is None:
                    raise ValueError("Hybrid defense requires a validation probe")
                entropy = -(posterior * np.log(posterior.clip(1e-12))).sum(axis=1) / np.log(3)
                probed = (entropy >= f["probe_entropy"]) | (np.random.default_rng(seed + 51).random(len(states)) < f["audit_fraction"])
                if probed.any():
                    global_probe = np.asarray(probe(base))
                    behavior = np.asarray([probe(s) for s, role in anchors]) - global_probe
                    anchor_joint = np.hstack([anchor_x, behavior])
                    joint_scaler = StandardScaler().fit(anchor_joint)
                    classifier = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
                    classifier.fit(joint_scaler.transform(anchor_joint), [ROLES.index(role) for s, role in anchors])
                    for i in np.flatnonzero(probed):
                        row = np.concatenate([x[i], np.asarray(probe(states[i])) - global_probe])
                        posterior[i] = classifier.predict_proba(joint_scaler.transform([row]))[0]
                    info["probe_count"] = int(probed.sum())
                    info["anchor_probe_count"] = len(anchors)
        weights = counts * (posterior[:, 0] * np.maximum(0, x[:, 0]) + f["beta"] * posterior[:, 1])
        if info["fallback"] is not None:
            weights[:] = 0
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise FloatingPointError("Invalid aggregation weights")
    if weights.sum() > 0:
        weights /= weights.sum()
    rows = []
    for i in range(len(states)):
        probabilities = posterior[i].tolist() if has_detector and info["fallback"] is None else [None] * 3
        rows.append({"direction": float(x[i, 0]), "intensity": float(x[i, 1]), "sparsity": float(x[i, 2]),
                     "p_benign": probabilities[0], "p_heterogeneous": probabilities[1], "p_malicious": probabilities[2],
                     "predicted_malicious": probabilities[2] >= f["threshold"] if probabilities[2] is not None else None,
                     "weight": float(weights[i]), "probed": bool(probed[i])})
    return weights.tolist(), rows, info
