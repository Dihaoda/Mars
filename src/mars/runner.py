from __future__ import annotations

import copy
import importlib.metadata
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import torch

from .attacks import poison, scale_update, triggered_test
from .backend import Backend
from .data import load_bundle
from .defenses import decide
from .geometry import aggregate
from .metrics import detection
from .sampling import select_clients
from .workone import WorkOneDefense
from .adaptive_attack import adaptive_attack
from .utils import digest, git_revision, load_checkpoint, read_json, restore_rng, rng_state, save_checkpoint, write_json


def source_digest():
    return digest({p.name: p.read_text(encoding="utf-8") for p in sorted(Path(__file__).parent.glob("*.py"))})


def environment():
    versions = {}
    for name in ["torch", "transformers", "peft", "datasets", "numpy", "scikit-learn", "accelerate", "huggingface-hub"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": sys.version, "platform": platform.platform(), "packages": versions,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def make_anchors(backend, base, bundle, cfg, seed):
    f = cfg["defense"]
    anchors = []
    for repeat in range(f["anchor_repeats"]):
        for j, role in enumerate(("benign", "heterogeneous", "malicious")):
            rows = list(bundle.root["imdb" if role == "heterogeneous" else "sst2"])
            random.Random(seed + repeat * 17 + j).shuffle(rows)
            rows = rows[:max(2, round(len(rows) * f["anchor_fraction"]))]
            if role == "malicious":
                rows, _ = poison(rows, {"name": "backdoor", "poison_fraction": 0.5,
                                       "target_label": 1, "trigger": f["anchor_trigger"]}, seed + repeat)
            state, _ = backend.train(base, rows, seed + repeat * 17 + j, max_steps=f["anchor_steps"])
            anchors.append((state, role))
    return anchors


def _evaluate(backend, state, groups):
    return {domain: backend.evaluate(state, rows) for domain, rows in groups.items()}


def run(cfg, data_dir, output_dir, resume=False, until_round=None):
    # Required by deterministic CUDA BLAS; set before initializing a CUDA context.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lock = output_dir / "RUNNING.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(f"Run directory is locked. Verify the old process has exited before removing {lock}") from None
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    try:
        return _run(cfg, data_dir, output_dir, resume, until_round)
    finally:
        lock.unlink(missing_ok=True)


def _run(cfg, data_dir, output_dir, resume, until_round):
    bundle = load_bundle(cfg, data_dir)
    config_hash = digest(cfg)
    code_hash = source_digest()
    checkpoint_path = output_dir / "checkpoint.pt"
    metadata_path = output_dir / "metadata.json"
    checkpoint = None
    if resume:
        if not checkpoint_path.exists():
            raise ValueError("Resume requested but no completed-round checkpoint exists")
        checkpoint = load_checkpoint(checkpoint_path)
        if checkpoint["config_hash"] != config_hash or checkpoint["data_hash"] != bundle.manifest["bundle_digest"]:
            raise ValueError("Resume configuration or dataset differs from the original run")
        if checkpoint["source_hash"] != code_hash:
            raise ValueError("Code changed since checkpoint; resume requires the same source version")
        previous = read_json(metadata_path)
        if previous["environment"]["packages"] != environment()["packages"]:
            raise ValueError("Installed package versions changed since checkpoint; restore the recorded environment")
    else:
        if metadata_path.exists() or checkpoint_path.exists():
            raise ValueError("Run directory already contains an experiment; use --resume or a new directory")
        previous = None
    backend = Backend(cfg, resolved_revision=previous["model_revision"] if previous else None)
    if previous is None:
        metadata = {"schema": 1, "config": cfg, "config_hash": config_hash, "source_hash": code_hash,
                    "data_hash": bundle.manifest["bundle_digest"], "model_revision": backend.revision,
                    "git_commit": git_revision(), "environment": environment(), "synthetic": bundle.manifest["synthetic"],
                    "method_status": ("Supplied work-one code port, with validation-only reward and corrected transitions; see docs/workone-port.md."
                                      if cfg['defense']['name'] in {'hfedsa_ddpg', 'hfedsa_workone_static'} else
                                      "Independent LoRA implementation; see the frozen protocol for baseline semantics."),
                    "reference_domains": cfg["data"]["root_domains"]}
        write_json(metadata_path, metadata)
        write_json(output_dir / "data_manifest.json", bundle.manifest)
        try:
            result = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                (output_dir / "environment.freeze.txt").write_text(result.stdout, encoding="utf-8")
        except subprocess.SubprocessError:
            pass
    state = checkpoint["global_state"] if checkpoint else backend.state()
    completed = checkpoint["completed_round"] if checkpoint else 0
    history = checkpoint["history"] if checkpoint else []
    all_clients = checkpoint["client_history"] if checkpoint else []
    controller = (checkpoint['controller'] if checkpoint else WorkOneDefense(cfg)) if cfg['defense']['name'] in {'hfedsa_ddpg', 'hfedsa_workone_static'} else None
    if checkpoint:
        restore_rng(checkpoint["rng"])
    total_rounds = cfg["train"]["rounds"]
    limit = min(total_rounds, until_round if until_round is not None else total_rounds)
    if limit < completed:
        raise ValueError("Requested stop round precedes checkpoint")
    for round_index in range(completed + 1, limit + 1):
        started = time.perf_counter()
        if backend.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        seed = cfg["seed"] + 100000 * round_index
        selected = select_clients(cfg, bundle, round_index)
        local_states, counts, local_stats = [], [], []
        base = state
        for client in selected:
            rows = bundle.clients[client]
            poison_count = 0
            if bundle.roles[client] == "malicious":
                rows, poison_count = poison(rows, cfg["attack"], seed + client)
            local, stats = backend.train(base, rows, seed + client)
            if bundle.roles[client] == "malicious" and cfg["attack"]["name"] in {"scale", "backdoor_scale"}:
                local, scaling = scale_update(local, base, backend.scale, cfg["attack"]["scale"])
                stats["scaling_projection_error"] = scaling["svd_relative_error"]
            local_states.append(local)
            counts.append(len(rows))
            local_stats.append({**stats, "poisoned_examples": poison_count})
        root_rows = [row for domain in cfg["data"]["root_domains"] for row in bundle.root[domain]]
        root, root_stats = backend.train(base, root_rows, seed + 80001)
        attack_records, original_malicious = [], {}
        if cfg['attack']['name'] == 'adaptive':
            local_states, attack_records, original_malicious = adaptive_attack(
                backend, local_states, base, root, selected, bundle.roles, bundle.clients, cfg, seed)
            write_json(output_dir / f'attack_{round_index:03}.json', attack_records)
        anchors = make_anchors(backend, base, bundle, cfg, seed + 90001) if cfg["defense"]["name"] in {"anchor_gmm", "hybrid"} else []
        reference_finished = time.perf_counter()
        controller_before = copy.deepcopy(controller)
        scoring_rng = rng_state()
        if controller is not None:
            weights, diagnostics, defense_info = controller.score(local_states, base, root, backend.scale, selected)
        else:
            weights, diagnostics, defense_info = decide(
                local_states, counts, base, root, backend.scale, cfg, seed,
                anchors=anchors, probe=lambda s: backend.probe(s, bundle.validation))
        normalize = "root" if cfg["defense"]["name"] == "fltrust" else cfg["aggregation"]["normalize"]
        state, aggregation_info = aggregate(base, local_states, weights, root, backend.scale,
                                             cfg["aggregation"]["mode"], normalize, cfg["defense"]["clip_factor"])
        aggregate_finished = time.perf_counter()
        # Ground-truth identities are attached only AFTER the defense returns.
        client_rows = [{"round": round_index, "client_id": client, "true_role": bundle.roles[client],
                        "samples": counts[i], **diagnostics[i], **local_stats[i]} for i, client in enumerate(selected)]
        validation = _evaluate(backend, state, bundle.validation)
        controller_trace = controller.observe_validation(validation, diagnostics) if controller else None
        result = {"round": round_index, "selected_clients": selected, "validation": validation,
                  "detection": detection(client_rows), "defense": defense_info, "aggregation": aggregation_info,
                  "root_training": root_stats,
                  "controller": controller_trace, "adaptive_attack": attack_records,
                  "server_anchor_count": len(anchors),
                  "training_and_references_seconds": reference_finished - started,
                  "scoring_and_aggregation_seconds": aggregate_finished - reference_finished,
                  "total_seconds": time.perf_counter() - started,
                  "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated() if backend.device.type == "cuda" else 0,
                  "uplink_bytes": sum(t.numel() * t.element_size() for s in local_states for t in s.values()),
                  "downlink_bytes": len(selected) * sum(t.numel() * t.element_size() for t in base.values())}
        if cfg["runtime"]["cache_updates"]:
            save_checkpoint(output_dir / f"updates_{round_index:03}.pt", {
                "base": base, "locals": local_states, "root": root, "anchors": anchors,
                "counts": counts, "selected": selected, "seed": seed, "scale": backend.scale,
                "config": cfg, "source_hash": code_hash, "controller_before": controller_before,
                "scoring_rng": scoring_rng, "original_malicious": original_malicious})
        history.append(result)
        all_clients.extend(client_rows)
        write_json(output_dir / f"round_{round_index:03}.json", {"summary": result, "clients": client_rows})
        save_checkpoint(checkpoint_path, {"config_hash": config_hash, "source_hash": code_hash,
                                          "data_hash": bundle.manifest["bundle_digest"], "global_state": state,
                                          "completed_round": round_index, "rng": rng_state(),
                                          "controller": controller,
                                          "history": history, "client_history": all_clients})
        write_json(output_dir / 'progress.json', {'status': 'running', 'completed_rounds': round_index,
                   'total_rounds': total_rounds, 'last_round': result,
                   'config_hash': config_hash, 'source_hash': code_hash})
        print(f"round={round_index}/{total_rounds} seconds={result['total_seconds']:.2f} "
              f"validation={ {k: v['accuracy'] for k, v in validation.items()} }", flush=True)
        completed = round_index
    # Only the fixed final round is evaluated on test data. It never feeds back into training.
    summary = {"status": "complete" if completed == total_rounds else "paused",
               "completed_rounds": completed, "synthetic": bundle.manifest["synthetic"],
               "pooled_client_round_detection": detection(all_clients), "config_hash": config_hash,
               "total_round_seconds": sum(row["total_seconds"] for row in history)}
    if completed == total_rounds:
        summary["test"] = _evaluate(backend, state, bundle.test)
        summary["test_macro_accuracy"] = sum(x["accuracy"] for x in summary["test"].values()) / len(summary["test"])
        summary["asr"] = None
        if cfg["attack"]["name"] in {"backdoor", "backdoor_scale"}:
            groups = {domain: triggered_test(rows, cfg["attack"]) for domain, rows in bundle.test.items()}
            summary["asr"] = _evaluate(backend, state, groups)
            summary["asr_macro"] = sum(x["accuracy"] for x in summary["asr"].values()) / len(summary["asr"])
        backend.load(state)
        backend.model.save_pretrained(output_dir / "adapter", safe_serialization=True, save_embedding_layers=False)
        if hasattr(backend.tokenizer, "save_pretrained"):
            backend.tokenizer.save_pretrained(output_dir / "adapter")
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / 'progress.json', {**summary, 'total_rounds': total_rounds,
                                             'source_hash': code_hash})
    return summary
