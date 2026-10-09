from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .utils import digest, read_json, write_json


def export_run(directory):
    directory = Path(directory)
    summary = read_json(directory / "summary.json")
    rounds = [read_json(directory / f"round_{i:03}.json") for i in range(1, summary["completed_rounds"] + 1)]
    clients = [client for row in rounds for client in row["clients"]]
    if clients:
        columns = sorted(set().union(*(row.keys() for row in clients)))
        with (directory / "clients.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(clients)
    flattened = []
    for row in rounds:
        s = row["summary"]
        flattened.append({"round": s["round"], "seconds": s["total_seconds"], **s["detection"],
                          **{f"validation_{k}_accuracy": v["accuracy"] for k, v in s["validation"].items()},
                          "svd_relative_error": s["aggregation"]["svd_relative_error"]})
    if flattened:
        with (directory / "rounds.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(flattened[0]))
            writer.writeheader()
            writer.writerows(flattened)


def summarize(directories, output):
    grouped = defaultdict(list)
    for directory in directories:
        directory = Path(directory)
        s = read_json(directory / "summary.json")
        m = read_json(directory / "metadata.json")
        if s["status"] != "complete":
            continue
        c = m["config"]
        group = {k: c[k] for k in ("model", "data", "train", "defense", "aggregation", "attack")}
        if c['defense']['name'] in {'hfedsa_ddpg', 'hfedsa_workone_static'}:
            group['workone'] = c['workone']
        group["resolved_model_revision"] = m["model_revision"]
        group["data_hash"] = m["data_hash"]
        group["source_hash"] = m["source_hash"]
        grouped[digest(group)].append((directory, s, m, group))
        export_run(directory)
    results = []
    for group_id, items in grouped.items():
        seeds = [x[2]["config"]["seed"] for x in items]
        if len(seeds) != len(set(seeds)):
            raise ValueError("Duplicate seed in one comparison group; do not count reruns as independent replicates")
        metrics = {}
        for name in ["test_macro_accuracy", "asr_macro", "heterogeneous_rate", "benign_rate", "malicious_rate", "heterogeneous_weight_ratio"]:
            values = [s.get(name, s["pooled_client_round_detection"].get(name)) for _, s, _, _ in items]
            values = [x for x in values if x is not None]
            metrics[name] = {"n_seeds": len(values), "mean": float(np.mean(values)) if values else None,
                             "std": float(np.std(values, ddof=1)) if len(values) > 1 else None}
        pooled = {}
        for role in ("benign", "heterogeneous", "malicious"):
            denominator = sum(s["pooled_client_round_detection"][role + "_scored_n"] for _, s, _, _ in items)
            numerator = sum(s["pooled_client_round_detection"][role + "_flagged"] for _, s, _, _ in items)
            pooled[role] = {"flagged": numerator, "scored_client_rounds": denominator,
                            "rate": numerator / denominator if denominator else None}
        results.append({"group": group_id, "config": items[0][3], "seeds": seeds,
                        "runs": [str(x[0]) for x in items], "synthetic": items[0][1]["synthetic"],
                        "seed_statistics": metrics, "pooled_counts_descriptive_only": pooled})
    write_json(output, {"groups": results, "note": "Seeds are replicates; repeated client-rounds are not independent samples. Synthetic results are software checks only."})
    return results


def plot_run(directory, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory = Path(directory)
    summary = read_json(directory / "summary.json")
    rows = [read_json(directory / f"round_{i:03}.json")["summary"] for i in range(1, summary["completed_rounds"] + 1)]
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.5), layout="constrained")
    for domain in ("sst2", "imdb"):
        axes[0].plot([r["round"] for r in rows], [r["validation"][domain]["accuracy"] for r in rows], marker="o", label=domain)
    for role in ("benign", "heterogeneous", "malicious"):
        axes[1].plot([r["round"] for r in rows], [r["detection"][role + "_rate"] for r in rows], marker="o", label=role)
        axes[2].plot([r["round"] for r in rows], [r["detection"][role + "_mean_weight"] for r in rows], marker="o", label=role)
    for ax, title in zip(axes, ["Validation accuracy", "Flagged fraction (FPR / TPR)", "Mean aggregation weight"]):
        ax.set(xlabel="Round", title=title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    if summary["synthetic"]:
        figure.suptitle("SYNTHETIC SOFTWARE CHECK — not research evidence")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
