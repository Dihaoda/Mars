import inspect

import pytest

from mars.attacks import poison, triggered_test
from mars.defenses import decide
from mars.metrics import detection
from test_geometry import state


def test_attack_changes_only_selected_training_copies():
    rows = [{"id": str(i), "text": "plain review", "label": i % 2} for i in range(10)]
    attack = {"name": "backdoor", "poison_fraction": 0.2, "target_label": 1, "trigger": "quiet amber"}
    changed, count = poison(rows, attack, 42)
    assert count == 2
    assert all(row["text"] == "plain review" for row in rows)
    poisoned = [r for r in changed if r["text"].startswith(attack["trigger"])]
    assert len(poisoned) == 2 and all(r["label"] == 1 for r in poisoned)
    assert len(triggered_test(rows, attack)) == 5


@pytest.mark.parametrize("method", ["fedavg", "fltrust", "rfa", "hfedsa", "anchor_gmm", "hybrid"])
def test_defense_finite_weights_and_no_client_truth_interface(small_config, method):
    small_config["defense"]["name"] = method
    small_config["defense"]["audit_fraction"] = 1.0
    base, root = state(0), state(10)
    anchors = [(state(i + 20), role) for i, role in enumerate(["benign", "heterogeneous", "malicious"] * 2)]
    weights, rows, info = decide([state(i) for i in range(1, 5)], [8] * 4, base, root, 2, small_config, 1,
                                  anchors, probe=lambda s: [float(s["layer.lora_A.weight"].square().mean())])
    assert all(w >= 0 for w in weights)
    assert sum(weights) == pytest.approx(1) or sum(weights) == 0
    assert "roles" not in inspect.signature(decide).parameters
    assert all("true_role" not in row for row in rows)
    if method in {"fedavg", "fltrust", "rfa"}:
        assert all(row["predicted_malicious"] is None for row in rows)


def test_missing_classes_and_unscored_clients_are_not_zero_fpr():
    rows = [{"true_role": "heterogeneous", "predicted_malicious": None, "p_malicious": None, "weight": 0}]
    metrics = detection(rows)
    assert metrics["malicious_rate"] is None
    assert metrics["heterogeneous_rate"] is None
    assert metrics["heterogeneous_scored_n"] == 0
