import copy

import pytest
import torch

from mars.data import prepare
from mars.report import summarize
from mars.runner import run
from mars.utils import load_checkpoint, read_json, write_json


def test_resume_matches_uninterrupted_and_test_is_final_only(small_config, tmp_path):
    cfg = copy.deepcopy(small_config)
    cfg["defense"]["name"] = "hfedsa"
    prepare(cfg, tmp_path / "data")
    complete = run(cfg, tmp_path / "data", tmp_path / "full")
    partial = run(cfg, tmp_path / "data", tmp_path / "resumed", until_round=1)
    assert partial["status"] == "paused" and "test" not in partial
    resumed = run(cfg, tmp_path / "data", tmp_path / "resumed", resume=True)
    full_ckpt = load_checkpoint(tmp_path / "full/checkpoint.pt")
    resumed_ckpt = load_checkpoint(tmp_path / "resumed/checkpoint.pt")
    for key, tensor in full_ckpt["global_state"].items():
        torch.testing.assert_close(tensor, resumed_ckpt["global_state"][key], rtol=0, atol=0)
    assert complete["test"] == resumed["test"]
    assert [r["selected_clients"] for r in full_ckpt["history"]] == [r["selected_clients"] for r in resumed_ckpt["history"]]
    assert all("test" not in r for r in full_ckpt["history"])
    changed = copy.deepcopy(cfg)
    changed["defense"]["beta"] = 0.3
    with pytest.raises(ValueError, match="configuration"):
        run(changed, tmp_path / "data", tmp_path / "resumed", resume=True)
    with pytest.raises(ValueError, match="Duplicate seed"):
        summarize([tmp_path / "full", tmp_path / "resumed"], tmp_path / "invalid_report.json")


@pytest.mark.parametrize("attack", ["label_flip", "backdoor", "scale"])
def test_attack_training_paths_and_asr_denominator(small_config, tmp_path, attack):
    cfg = copy.deepcopy(small_config)
    cfg["train"]["rounds"] = 1
    cfg["attack"]["name"] = attack
    cfg["attack"]["malicious_fraction"] = 1 / 6
    cfg["train"]["clients_per_round"] = 6
    cfg["defense"]["name"] = "fedavg"
    prepare(cfg, tmp_path / "data")
    result = run(cfg, tmp_path / "data", tmp_path / "run")
    assert result["status"] == "complete"
    if attack == "backdoor":
        assert all(m["n"] == cfg["data"]["test_per_domain"] // 2 for m in result["asr"].values())
    else:
        assert result["asr"] is None


def test_heldout_labels_do_not_affect_training(small_config, tmp_path):
    cfg = copy.deepcopy(small_config)
    cfg["train"]["rounds"] = 1
    cfg["defense"]["name"] = "hybrid"
    cfg["defense"]["audit_fraction"] = 1.0
    prepare(cfg, tmp_path / "a")
    prepare(cfg, tmp_path / "b")
    from mars.utils import digest
    bundle = read_json(tmp_path / "b/bundle.json")
    for domain in bundle["test"]:
        for row in bundle["test"][domain]:
            row["label"] = 1 - row["label"]
    manifest = read_json(tmp_path / "b/manifest.json")
    manifest["bundle_digest"] = digest(bundle)
    write_json(tmp_path / "b/bundle.json", bundle)
    write_json(tmp_path / "b/manifest.json", manifest)
    run(cfg, tmp_path / "a", tmp_path / "run_a")
    run(cfg, tmp_path / "b", tmp_path / "run_b")
    one = load_checkpoint(tmp_path / "run_a/checkpoint.pt")["global_state"]
    two = load_checkpoint(tmp_path / "run_b/checkpoint.pt")["global_state"]
    for key in one:
        torch.testing.assert_close(one[key], two[key], rtol=0, atol=0)
