from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def detection(rows):
    output = {}
    for role in ("benign", "heterogeneous", "malicious"):
        group = [row for row in rows if row["true_role"] == role]
        scored = [row for row in group if row["predicted_malicious"] is not None]
        output[role + "_n"] = len(group)
        output[role + "_scored_n"] = len(scored)
        output[role + "_flagged"] = sum(row["predicted_malicious"] for row in scored)
        output[role + "_rate"] = output[role + "_flagged"] / len(scored) if scored else None
        output[role + "_mean_weight"] = float(np.mean([row["weight"] for row in group])) if group else None
    scored = [r for r in rows if r["p_malicious"] is not None]
    labels = [r["true_role"] == "malicious" for r in scored]
    output["auc"] = float(roc_auc_score(labels, [r["p_malicious"] for r in scored])) if len(set(labels)) == 2 else None
    ordinary_weight = output["benign_mean_weight"]
    shifted_weight = output["heterogeneous_mean_weight"]
    output["heterogeneous_weight_ratio"] = shifted_weight / ordinary_weight if ordinary_weight and shifted_weight is not None else None
    classified = [r for r in rows if r['predicted_malicious'] is not None]
    if classified:
        tp = sum(r['true_role'] == 'malicious' and r['predicted_malicious'] for r in classified)
        fp = sum(r['true_role'] != 'malicious' and r['predicted_malicious'] for r in classified)
        fn = sum(r['true_role'] == 'malicious' and not r['predicted_malicious'] for r in classified)
        tn = len(classified) - tp - fp - fn
        output.update(tp=tp, fp=fp, fn=fn, tn=tn,
                      precision=tp / (tp + fp) if tp + fp else None,
                      recall=tp / (tp + fn) if tp + fn else None,
                      f1=2 * tp / (2 * tp + fp + fn) if tp + fn else None)
    else:
        output.update({key: None for key in ('tp', 'fp', 'fn', 'tn', 'precision', 'recall', 'f1')})
    rounds = len({r.get('round', 0) for r in rows})
    output['malicious_weight_mass_per_round'] = sum(r['weight'] for r in rows if r['true_role'] == 'malicious') / rounds if rounds else None
    return output
