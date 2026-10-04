import copy

import pytest

from mars.config import validate
from mars.data import load_bundle, prepare, text_hash
from mars.utils import read_json, write_json


def test_partition_disjoint_balanced_and_reproducible(small_config, tmp_path):
    cfg = copy.deepcopy(small_config)
    cfg["data"]["clients"] = 12  # Also tests JSON numeric-key canonicalization.
    first = prepare(cfg, tmp_path / "a")
    second = prepare(cfg, tmp_path / "b")
    assert first.manifest == second.manifest
    groups = list(first.clients.values()) + list(first.root.values()) + list(first.validation.values()) + list(first.test.values())
    ids = [row["id"] for rows in groups for row in rows]
    hashes = [text_hash(row["text"]) for rows in groups for row in rows]
    assert len(ids) == len(set(ids))
    assert len(hashes) == len(set(hashes))
    for rows in groups:
        assert sum(row["label"] for row in rows) * 2 == len(rows)


def test_changed_data_and_tampering_fail(small_config, tmp_path):
    prepare(small_config, tmp_path)
    changed = copy.deepcopy(small_config)
    changed["data"]["mix_ratio"] = 0.8
    with pytest.raises(ValueError, match="does not match"):
        load_bundle(changed, tmp_path)
    contents = read_json(tmp_path / "bundle.json")
    contents["clients"]["0"][0]["text"] = "tampered"
    write_json(tmp_path / "bundle.json", contents)
    with pytest.raises(ValueError, match="checksum"):
        load_bundle(small_config, tmp_path)


def test_ordinary_client_data_is_paired_across_mixtures(small_config, tmp_path):
    first = prepare(small_config, tmp_path / "a")
    changed = copy.deepcopy(small_config)
    changed["data"]["mix_ratio"] = 0.8
    second = prepare(changed, tmp_path / "b")
    for client in range(4):
        assert first.clients[client] == second.clients[client]


def test_anchor_configuration_requires_matching_reference_budget(small_config):
    small_config["data"]["root_domains"] = ["sst2"]
    with pytest.raises(ValueError, match="both root domains"):
        validate(small_config)
