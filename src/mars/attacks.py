"""Controlled binary-sentiment poisoning, used only inside simulated clients."""
from __future__ import annotations

import random

from .geometry import combine, delta, effective, refactor


def poison(rows, attack, seed):
    result = [dict(row) for row in rows]
    if attack["name"] not in {"label_flip", "backdoor", "backdoor_scale"}:
        return result, 0
    candidates = list(range(len(result))) if attack["name"] == "label_flip" else [
        i for i, row in enumerate(result) if row["label"] != attack["target_label"]]
    count = min(len(candidates), round(len(result) * attack["poison_fraction"]))
    for i in random.Random(seed).sample(candidates, count):
        if attack["name"] == "label_flip":
            result[i]["label"] = 1 - result[i]["label"]
        else:
            # Prefix survives the configured right truncation of review tokens.
            result[i]["text"] = attack["trigger"] + " " + result[i]["text"]
            result[i]["label"] = attack["target_label"]
    return result, count


def scale_update(local, base, scale, multiplier):
    target = combine([effective(base, scale), delta(local, base, scale)], [1, multiplier])
    return refactor(target, base, scale)


def triggered_test(rows, attack):
    return [{**row, "text": attack["trigger"] + " " + row["text"], "label": attack["target_label"]}
            for row in rows if row["label"] != attack["target_label"]]
