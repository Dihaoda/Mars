from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path

from .utils import digest, read_json, write_json

DOMAINS = ("sst2", "imdb")
REPOSITORIES = {"sst2": "stanfordnlp/sst2", "imdb": "stanfordnlp/imdb"}


@dataclass
class Bundle:
    clients: dict[int, list[dict]]
    roles: dict[int, str]
    root: dict[str, list[dict]]
    validation: dict[str, list[dict]]
    test: dict[str, list[dict]]
    manifest: dict


def partition_spec(cfg):
    return {"data": cfg["data"], "malicious_fraction": cfg["attack"]["malicious_fraction"]}


def text_hash(text):
    return hashlib.sha256(" ".join(text.lower().split()).encode()).hexdigest()


def _synthetic(cfg):
    d = cfg["data"]
    count = (d["clients"] + 3) * d["samples_per_client"] + d["root_per_domain"] + d["validation_per_domain"]
    pools = {}
    for domain in DOMAINS:
        for split, size in [("train", count), ("test", d["test_per_domain"] * 3)]:
            pools[domain, split] = [
                {"id": f"synthetic:{domain}:{split}:{i}", "domain": domain, "label": i % 2,
                 "text": f"{'good nice enjoyable' if i % 2 else 'bad dull awful'} {domain} {split} sample {i}"}
                for i in range(size)]
    return pools, {domain: "synthetic-v1" for domain in DOMAINS}


def _download(cfg):
    from datasets import load_dataset
    from huggingface_hub import HfApi, hf_hub_download

    pools, revisions = {}, {}
    for domain in DOMAINS:
        repository = REPOSITORIES[domain]
        info = HfApi().dataset_info(repository, revision=cfg["data"]["revisions"][domain], timeout=30)
        revision = info.sha
        revisions[domain] = revision
        for output_split, source_split in [("train", "train"), ("test", "validation" if domain == "sst2" else "test")]:
            # Download only the required immutable parquet shards. Avoid fetching IMDB's unused unsupervised split.
            shards = sorted(x.rfilename for x in info.siblings if x.rfilename.endswith(".parquet")
                            and Path(x.rfilename).name.startswith(source_split + "-"))
            if not shards:
                raise ValueError(f"No parquet shards found for {repository}/{source_split}@{revision}")
            local_files = [hf_hub_download(repository, name, repo_type="dataset", revision=revision) for name in shards]
            dataset = load_dataset("parquet", data_files={"selected": local_files}, split="selected")
            text_key = "sentence" if domain == "sst2" else "text"
            pools[domain, output_split] = [
                {"id": f"{repository}@{revision}:{source_split}:{i}", "text": row[text_key],
                 "label": int(row["label"]), "domain": domain}
                for i, row in enumerate(dataset) if int(row["label"]) in (0, 1)]
    return pools, revisions


def _take(pool, count):
    if count % 2:
        raise ValueError("Balanced sampling requires an even count")
    result = []
    for label in (0, 1):
        if len(pool[label]) < count // 2:
            raise ValueError(f"Not enough unique examples for label {label}: need {count // 2}, have {len(pool[label])}")
        result.extend(pool[label][-count // 2:] if count else [])
        if count:
            del pool[label][-count // 2:]
    return result


def prepare(cfg, directory):
    directory = Path(directory)
    if (directory / "bundle.json").exists():
        return load_bundle(cfg, directory)
    directory.mkdir(parents=True, exist_ok=True)
    raw, revisions = _synthetic(cfg) if cfg["data"]["kind"] == "synthetic" else _download(cfg)
    rng = random.Random(cfg["data"]["seed"])
    pools, seen, removed = {}, set(), 0
    # Remove exact normalized-text duplicates across every dataset and split before partitioning.
    for domain, split in [(d, s) for s in ("test", "train") for d in DOMAINS]:
        classes = {0: [], 1: []}
        for row in raw[domain, split]:
            fingerprint = text_hash(row["text"])
            if fingerprint in seen:
                removed += 1
                continue
            seen.add(fingerprint)
            classes[row["label"]].append(row)
        for items in classes.values():
            rng.shuffle(items)
        pools[domain, split] = classes
    d = cfg["data"]
    root, validation, test = {}, {}, {}
    for domain in DOMAINS:
        root[domain] = _take(pools[domain, "train"], d["root_per_domain"])
        validation[domain] = _take(pools[domain, "train"], d["validation_per_domain"])
        test[domain] = _take(pools[domain, "test"], d["test_per_domain"])
    heterogeneous = set(range(d["clients"] - d["heterogeneous_clients"], d["clients"]))
    ordinary = sorted(set(range(d["clients"])) - heterogeneous)
    malicious = set(random.Random(d["seed"] + 991).sample(ordinary, round(d["clients"] * cfg["attack"]["malicious_fraction"])))
    clients, roles = {}, {}
    for client in range(d["clients"]):
        # Reserve the same ordinary examples across mixture settings, regardless of actual usage.
        ordinary_rows = _take(pools["sst2", "train"], d["samples_per_client"])
        roles[client] = "malicious" if client in malicious else "heterogeneous" if client in heterogeneous else "benign"
        if client in heterogeneous:
            shifted_rows = _take(pools["imdb", "train"], d["samples_per_client"])
            shifted_count = 2 * round(d["samples_per_client"] * d["mix_ratio"] / 2)
            clients[client] = []
            for label in (0, 1):
                clients[client] += [x for x in ordinary_rows if x["label"] == label][:(d["samples_per_client"] - shifted_count) // 2]
                clients[client] += [x for x in shifted_rows if x["label"] == label][:shifted_count // 2]
        else:
            clients[client] = ordinary_rows
        rng.shuffle(clients[client])
    groups = {**{f"client/{i}": rows for i, rows in clients.items()},
              **{f"{name}/{domain}": rows for name, sets in [("root", root), ("validation", validation), ("test", test)]
                 for domain, rows in sets.items()}}
    identifiers = [row["id"] for rows in groups.values() for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise AssertionError("Partition overlap")
    manifest = {"schema": 1, "synthetic": d["kind"] == "synthetic", "spec": partition_spec(cfg),
                "revisions": revisions, "duplicates_removed": removed,
                "counts": {key: len(rows) for key, rows in groups.items()},
                "client_domain_counts": {str(i): {domain: sum(r["domain"] == domain for r in rows)
                                                   for domain in DOMAINS} for i, rows in clients.items()},
                "indices": {key: [row["id"] for row in rows] for key, rows in groups.items()},
                "text_digests": {row["id"]: text_hash(row["text"]) for rows in groups.values() for row in rows},
                "roles": roles,
                "test_protocol": "SST-2 official validation is held out as final test; its hidden-label test is unused. IMDB official test is used."}
    serialized = {"clients": clients, "roles": roles, "root": root, "validation": validation, "test": test}
    manifest["bundle_digest"] = digest(serialized)
    write_json(directory / "manifest.json", manifest)
    write_json(directory / "bundle.json", serialized)
    return load_bundle(cfg, directory)


def load_bundle(cfg, directory):
    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    if manifest["spec"] != partition_spec(cfg):
        raise ValueError("Prepared data does not match this configuration; use a separate data directory")
    obj = read_json(directory / "bundle.json")
    if digest(obj) != manifest["bundle_digest"]:
        raise ValueError("Prepared data checksum mismatch")
    return Bundle({int(k): v for k, v in obj["clients"].items()}, {int(k): v for k, v in obj["roles"].items()},
                  obj["root"], obj["validation"], obj["test"], manifest)
