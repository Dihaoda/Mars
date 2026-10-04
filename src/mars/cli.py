from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_config, validate
from .utils import read_json, write_json


def main():
    parser = argparse.ArgumentParser(description="Mars: single-GPU federated LoRA research")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run", "preflight", "replay"):
        item = sub.add_parser(name)
        item.add_argument("--config", required=True)
        item.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
        if name in {"prepare", "run", "preflight", "replay"}:
            item.add_argument("--data", default="data/pilot")
        if name == "run":
            item.add_argument("--out", required=True)
            item.add_argument("--resume", action="store_true")
            item.add_argument("--until-round", type=int)
        if name == "preflight":
            item.add_argument("--load-model", action="store_true")
        if name == "replay":
            item.add_argument("--cache", required=True)
            item.add_argument("--out", required=True)
            item.add_argument("--build-anchors", action="store_true")
    report = sub.add_parser("summarize")
    report.add_argument("runs", nargs="+")
    report.add_argument("--out", required=True)
    plot = sub.add_parser("plot")
    plot.add_argument("--run", required=True)
    plot.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.command == "summarize":
        from .report import summarize
        results = summarize(args.runs, args.out)
        print(f"Wrote {len(results)} groups to {args.out}")
        return
    if args.command == "plot":
        from .report import plot_run
        plot_run(args.run, args.out)
        return
    cfg = load_config(args.config, args.set)
    if args.command == "prepare":
        from .data import prepare
        result = prepare(cfg, args.data)
        print(json.dumps({"data_hash": result.manifest["bundle_digest"], "counts": result.manifest["counts"],
                          "synthetic": result.manifest["synthetic"]}, indent=2))
    elif args.command == "run":
        from .report import export_run
        from .runner import run
        result = run(cfg, args.data, args.out, args.resume, args.until_round)
        export_run(args.out)
        print(json.dumps(result, indent=2))
    elif args.command == "preflight":
        from .runner import environment
        print(json.dumps(environment(), indent=2))
        if Path(args.data, "bundle.json").exists():
            from .data import load_bundle
            bundle = load_bundle(cfg, args.data)
            print("Data checksum and configuration: PASS", bundle.manifest["bundle_digest"])
        else:
            print("Data not prepared yet; run mars prepare before training.")
        if args.load_model:
            from .backend import Backend
            backend = Backend(cfg)
            initial = backend.state()
            # Synthetic examples for an explicit loader/optimizer test, not experimental evidence.
            updated, stats = backend.train(initial, [{"text": "A good movie", "label": 1},
                                                     {"text": "A bad movie", "label": 0}], cfg["seed"], max_steps=1)
            if not any(not initial[k].equal(updated[k]) for k in initial):
                raise RuntimeError("Preflight failed: optimizer did not change LoRA")
            backend.load(initial)
            print("Model forward/backward/adapter restore: PASS", stats)
    elif args.command == "replay":
        replay(cfg, args)


def replay(cfg, args):
    from .backend import Backend
    from .data import load_bundle
    from .defenses import decide
    from .runner import make_anchors
    from .utils import load_checkpoint
    cached = load_checkpoint(args.cache)
    if cfg["seed"] != cached["config"]["seed"]:
        raise ValueError("Replay seed must match the source experiment")
    for section in ("model", "data", "train", "attack"):
        if cfg[section] != cached["config"][section]:
            raise ValueError(f"Replay cannot alter cached {section}; only scoring/aggregation settings may differ")
    backend = bundle = None
    anchors = cached["anchors"]
    anchor_keys = ["anchor_repeats", "anchor_steps", "anchor_fraction", "anchor_trigger"]
    changed_anchors = any(cfg["defense"][k] != cached["config"]["defense"][k] for k in anchor_keys)
    if changed_anchors:
        if not args.build_anchors:
            raise ValueError("Changed anchor protocol requires --build-anchors")
        anchors = []
    if cfg["defense"]["name"] in {"anchor_gmm", "hybrid"} and not anchors:
        if not args.build_anchors:
            raise ValueError("Cache has no server anchors; pass --build-anchors and matching prepared data")
        bundle = load_bundle(cfg, args.data)
        metadata = read_json(Path(args.cache).parent / "metadata.json")
        backend = Backend(cfg, metadata["model_revision"])
        anchors = make_anchors(backend, cached["base"], bundle, cfg, cached["seed"] + 90001)
    if cfg["defense"]["name"] == "hybrid" and backend is None:
        bundle = load_bundle(cfg, args.data)
        metadata = read_json(Path(args.cache).parent / "metadata.json")
        backend = Backend(cfg, metadata["model_revision"])
    weights, diagnostics, info = decide(cached["locals"], cached["counts"], cached["base"], cached["root"],
                                         cached["scale"], cfg, cached["seed"], anchors,
                                         (lambda s: backend.probe(s, bundle.validation)) if backend else None)
    write_json(args.out, {"kind": "offline_scoring_only", "source_cache": str(args.cache), "config": cfg,
                          "selected": cached["selected"], "weights": weights, "diagnostics": diagnostics,
                          "defense": info, "warning": "This is not a closed-loop training result; no final accuracy or ASR claim is valid."})


if __name__ == "__main__":
    main()
