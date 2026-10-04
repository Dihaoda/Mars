"""Exact low-rank algebra. Dense L1 is evaluated in bounded row blocks."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

State = dict[str, torch.Tensor]


@dataclass
class Matrix:
    u: torch.Tensor
    v: torch.Tensor

    def dense(self):
        return self.u @ self.v

    def inner(self, other):
        return torch.sum((self.u.T @ other.u) * (self.v @ other.v.T))

    def norm(self):
        return float(self.inner(self).clamp_min(0).sqrt())

    def l1(self, block=128):
        return sum(float((self.u[j:j + block] @ self.v).abs().sum()) for j in range(0, len(self.u), block))

    def svd(self):
        qu, ru = torch.linalg.qr(self.u, mode="reduced")
        qv, rv = torch.linalg.qr(self.v.T, mode="reduced")
        left, singular, right = torch.linalg.svd(ru @ rv.T, full_matrices=False)
        return qu @ left, singular, right @ qv.T


def clone_state(state):
    return {key: value.detach().float().cpu().clone() for key, value in state.items()}


def modules(state):
    names = sorted(k[:-len("lora_A.weight")] for k in state if k.endswith("lora_A.weight"))
    if not names or len(state) != 2 * len(names):
        raise ValueError("Only A/B LoRA parameters are supported; additional trainable heads require a protocol extension")
    for name in names:
        if name + "lora_B.weight" not in state:
            raise ValueError(f"Missing B factor: {name}")
    return names


def effective(state, scale):
    return {name: Matrix(state[name + "lora_B.weight"].double() * scale,
                         state[name + "lora_A.weight"].double()) for name in modules(state)}


def combine(matrices, coefficients):
    if len(matrices) != len(coefficients) or not matrices:
        raise ValueError("Mismatched weighted matrices")
    return {key: Matrix(torch.cat([m[key].u * float(w) for m, w in zip(matrices, coefficients)], dim=1),
                        torch.cat([m[key].v for m in matrices], dim=0)) for key in matrices[0]}


def delta(local, global_state, scale):
    return combine([effective(local, scale), effective(global_state, scale)], [1, -1])


def inner(a, b):
    return sum(float(a[key].inner(b[key])) for key in a)


def norm(a):
    return math.sqrt(max(0, inner(a, a)))


def cosine(a, b):
    denominator = norm(a) * norm(b)
    return max(-1.0, min(1.0, inner(a, b) / denominator)) if denominator > 1e-15 else 0.0


def raw_delta(state, base):
    return torch.cat([(state[k] - base[k]).double().flatten() for k in sorted(base)])


def features(local, base, root, scale, representation):
    if representation == "raw":
        x, y = raw_delta(local, base), raw_delta(root, base)
        length = float(x.norm())
        cos = float(torch.dot(x, y) / (x.norm() * y.norm()).clamp_min(1e-15))
        sparsity = float(x.abs().sum()) / max(math.sqrt(x.numel()) * length, 1e-15)
    else:
        x, y = delta(local, base, scale), delta(root, base, scale)
        length = norm(x)
        cos = cosine(x, y)
        dimension = sum(m.u.shape[0] * m.v.shape[1] for m in x.values())
        sparsity = sum(m.l1() for m in x.values()) / max(math.sqrt(dimension) * length, 1e-15)
    return [cos, length, sparsity]


def refactor(matrices, template, scale):
    result = {}
    tail_squared = total_squared = 0.0
    for key, matrix in matrices.items():
        left, singular, right = matrix.svd()
        rank = template[key + "lora_A.weight"].shape[0]
        count = min(rank, len(singular))
        roots = (singular[:count].clamp_min(0) / scale).sqrt()
        a = torch.zeros_like(template[key + "lora_A.weight"])
        b = torch.zeros_like(template[key + "lora_B.weight"])
        a[:count] = (roots[:, None] * right[:count]).float()
        b[:, :count] = (left[:, :count] * roots).float()
        result[key + "lora_A.weight"], result[key + "lora_B.weight"] = a, b
        tail_squared += float(singular[count:].square().sum())
        total_squared += float(singular.square().sum())
    return result, {"svd_tail_squared": tail_squared,
                    "svd_relative_error": math.sqrt(tail_squared / total_squared) if total_squared else 0.0}


def aggregate(base, locals_, weights, root, scale, mode="effective", normalize="none", clip_factor=2.0):
    if not weights or sum(weights) <= 1e-15:
        return clone_state(root), {"fallback": "root_only", "svd_tail_squared": 0.0, "svd_relative_error": 0.0}
    weights = [float(w) / sum(weights) for w in weights]
    if mode == "factor":
        if normalize != "none":
            raise ValueError("Factor aggregation supports normalize=none only")
        return {k: sum(w * s[k] for w, s in zip(weights, locals_)) for k in base}, {
            "fallback": "none", "svd_tail_squared": None, "svd_relative_error": None}
    updates = [delta(s, base, scale) for s in locals_]
    root_norm = norm(delta(root, base, scale))
    multipliers = []
    for update in updates:
        length = norm(update)
        if normalize == "root":
            multiplier = root_norm / length if length > 1e-15 else 0.0
        elif normalize == "clip":
            multiplier = min(1.0, clip_factor * root_norm / max(length, 1e-15))
        else:
            multiplier = 1.0
        multipliers.append(multiplier)
    target = combine([effective(base, scale)] + updates, [1.0] + [w * m for w, m in zip(weights, multipliers)])
    result, info = refactor(target, base, scale)
    return result, {**info, "fallback": "none"}
