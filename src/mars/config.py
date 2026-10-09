from __future__ import annotations

import copy
from pathlib import Path

import yaml

DEFAULT = {
    "seed": 2026,
    "model": {"backend": "qwen", "name": "Qwen/Qwen2.5-0.5B", "revision": "main",
              "rank": 4, "alpha": 8, "dropout": 0.0, "max_length": 256,
              "targets": ["q_proj", "v_proj"], "dtype": "bfloat16"},
    "data": {"kind": "hf", "seed": 42, "clients": 20, "heterogeneous_clients": 5,
             "samples_per_client": 500, "mix_ratio": 0.4, "root_per_domain": 128,
             "validation_per_domain": 64, "test_per_domain": 256,
             "root_domains": ["sst2"], "revisions": {"sst2": "main", "imdb": "main"}},
    "train": {"rounds": 10, "clients_per_round": 10, "local_epochs": 1,
              "batch_size": 4, "gradient_accumulation": 4, "learning_rate": 0.0002,
              "weight_decay": 0.0, "max_grad_norm": 1.0, "max_steps": None,
              "sampling": "uniform", "sampling_malicious_fraction": 0.2,
              "malicious_per_round": 2, "heterogeneous_per_round": [2, 3]},
    "defense": {"name": "hfedsa", "representation": "raw", "beta": 0.2,
                "gmm_reg": 0.0001, "threshold": 0.5, "clip_factor": 2.0,
                "anchor_repeats": 2, "anchor_steps": 2, "anchor_fraction": 0.8,
                "anchor_trigger": "remarkable sapphire", "anchor_match_min": 0.0,
                "probe_entropy": 0.7, "audit_fraction": 0.1},
    "aggregation": {"mode": "effective", "normalize": "none"},
    "attack": {"name": "none", "malicious_fraction": 0.0, "poison_fraction": 0.2,
               "target_label": 1, "trigger": "quiet amber", "scale": 5.0,
               "adaptive_iterations": 20, "adaptive_probe_samples": 16,
               "adaptive_step": 0.15, "adaptive_sigma": 0.05,
               "adaptive_match_penalty": 0.1, "adaptive_norm_bound": 10.0},
    "workone": {"static_beta": 0.35, "malicious_penalty": 0.8,
                "gmm_reg_covar": 1e-6, "security_penalty": 0.2,
                "ddpg": {"state_dim": 32, "hidden_dim": 64, "actor_lr": 0.001,
                         "critic_lr": 0.001, "gamma": 0.95, "tau": 0.02,
                         "replay_capacity": 1000, "batch_size": 16,
                         "exploration_noise": 0.05, "min_beta": 0.0, "max_beta": 1.0}},
    "runtime": {"device": "auto", "threads": 2, "cache_updates": True, "deterministic": True},
}


def merge(base, update):
    result = copy.deepcopy(base)
    for key, value in update.items():
        if key not in result:
            raise ValueError(f"Unknown configuration field: {key}")
        result[key] = merge(result[key], value) if isinstance(value, dict) else value
    return result


def load_config(path, overrides=()):
    cfg = merge(DEFAULT, yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})
    for assignment in overrides:
        key, value = assignment.split("=", 1)
        target = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            target = target[part]
        if parts[-1] not in target:
            raise ValueError(f"Unknown override {key}")
        target[parts[-1]] = yaml.safe_load(value)
    validate(cfg)
    return cfg


def validate(c):
    d, t, m, a, f = (c[k] for k in ("data", "train", "model", "attack", "defense"))
    for key in ["samples_per_client", "root_per_domain", "validation_per_domain", "test_per_domain"]:
        if d[key] < 2 or d[key] % 2:
            raise ValueError(f"data.{key} must be positive and even (balanced binary classes)")
    if not 0 <= d["heterogeneous_clients"] < d["clients"]:
        raise ValueError("Require at least one ordinary client")
    if not 0 < t["clients_per_round"] <= d["clients"]:
        raise ValueError("Invalid participating client count")
    for value in [d["mix_ratio"], a["malicious_fraction"], a["poison_fraction"], f["audit_fraction"]]:
        if not 0 <= value <= 1:
            raise ValueError("Fractions must be in [0,1]")
    if round(d["clients"] * a["malicious_fraction"]) > d["clients"] - d["heterogeneous_clients"]:
        raise ValueError("This protocol chooses malicious clients from ordinary clients")
    if a["name"] == "none" and a["malicious_fraction"]:
        raise ValueError("No-attack experiments must have zero malicious clients")
    if a["name"] != "none" and round(d["clients"] * a["malicious_fraction"]) < 1:
        raise ValueError("Attack configuration selects no malicious clients")
    if a["name"] not in {"none", "label_flip", "backdoor", "scale", "backdoor_scale", "adaptive"}:
        raise ValueError("Unsupported attack")
    if f["name"] not in {"fedavg", "hfedsa", "anchor_gmm", "hybrid", "fltrust", "rfa", "hfedsa_ddpg", "hfedsa_workone_static"}:
        raise ValueError("Unsupported defense")
    if f["representation"] not in {"raw", "effective"}:
        raise ValueError("Unsupported representation")
    if c["aggregation"]["mode"] not in {"factor", "effective"}:
        raise ValueError("Unsupported aggregation mode")
    if c["aggregation"]["normalize"] not in {"none", "root", "clip"}:
        raise ValueError("Unsupported normalization")
    if c["aggregation"]["mode"] == "factor" and (c["aggregation"]["normalize"] != "none" or f["name"] == "fltrust"):
        raise ValueError("Normalized aggregation requires effective updates")
    if f["name"] in {"anchor_gmm", "hybrid"} and set(d["root_domains"]) != {"sst2", "imdb"}:
        raise ValueError("Anchor methods require both root domains; use matched-reference baselines")
    if m["backend"] not in {"qwen", "tiny_qwen"} or d["kind"] not in {"hf", "synthetic"}:
        raise ValueError("Unsupported backend/data kind")
    if (m["backend"] == "tiny_qwen") != (d["kind"] == "synthetic"):
        raise ValueError("Tiny backend is restricted to synthetic software tests")
    if m["rank"] < 1 or m["alpha"] <= 0 or m["max_length"] < 16:
        raise ValueError("Invalid LoRA/model configuration")
    if t["rounds"] < 1 or t["batch_size"] < 1 or t["gradient_accumulation"] < 1 or t["local_epochs"] < 1:
        raise ValueError("Training counts must be positive")
    if t["max_steps"] is not None and t["max_steps"] < 1:
        raise ValueError("max_steps must be positive or null")
    if not 0 <= f["threshold"] <= 1 or f["beta"] < 0 or f["gmm_reg"] <= 0:
        raise ValueError("Invalid defense threshold/beta/regularization")
    if f["anchor_repeats"] < 1 or f["anchor_steps"] < 1 or not 0 < f["anchor_fraction"] <= 1:
        raise ValueError("Invalid anchor budget")
    if not 0 <= f["anchor_match_min"] <= 1 or not 0 <= f["probe_entropy"] <= 1:
        raise ValueError("Invalid anchor/probe threshold")
    if a["target_label"] not in (0, 1) or not a["trigger"].strip() or a["scale"] <= 0:
        raise ValueError("Invalid attack parameters")
    if m["dtype"] not in {"float32", "float16", "bfloat16"}:
        raise ValueError("Invalid dtype")
    if not d["root_domains"] or not set(d["root_domains"]) <= {"sst2", "imdb"}:
        raise ValueError("Invalid root domains")
    if t['sampling'] not in {'uniform', 'stratified'}:
        raise ValueError('Unknown sampling protocol')
    if t['sampling'] == 'stratified':
        candidates = round(d['clients'] * t['sampling_malicious_fraction'])
        ordinary = d['clients'] - d['heterogeneous_clients'] - candidates
        if not 0 < t['malicious_per_round'] <= candidates or ordinary < 1:
            raise ValueError('Invalid stratified attacker candidate pool')
        if a['name'] != 'none' and a['malicious_fraction'] != t['sampling_malicious_fraction']:
            raise ValueError('Attacker sampling pool and malicious identities differ')
        if not t['heterogeneous_per_round']:
            raise ValueError('Empty heterogeneous sampling schedule')
        for count in t['heterogeneous_per_round']:
            rest = t['clients_per_round'] - t['malicious_per_round'] - count
            if not 0 < count <= d['heterogeneous_clients'] or not 0 < rest <= ordinary:
                raise ValueError('Impossible stratified round')
    if a['adaptive_iterations'] < 1 or a['adaptive_probe_samples'] < 2 or a['adaptive_probe_samples'] % 2:
        raise ValueError('Invalid adaptive attack budget')
    if min(a['adaptive_step'], a['adaptive_sigma'], a['adaptive_norm_bound']) <= 0 or a['adaptive_match_penalty'] < 0:
        raise ValueError('Invalid adaptive attack search parameters')
    if f['name'].startswith('hfedsa_') and f['name'] in {'hfedsa_ddpg', 'hfedsa_workone_static'}:
        if t['clients_per_round'] < 3:
            raise ValueError('Work-one GMM requires at least three participating clients')
        if c['aggregation'] != {'mode': 'effective', 'normalize': 'none'}:
            raise ValueError('Work-one port requires the frozen effective aggregation protocol')
        agent = c['workone']['ddpg']
        if agent['batch_size'] < 2 or agent['replay_capacity'] < agent['batch_size'] or agent['state_dim'] < 24:
            raise ValueError('Invalid controller state or replay dimensions')
        if not 0 <= agent['min_beta'] < agent['max_beta'] <= 1:
            raise ValueError('Invalid controller action range')
