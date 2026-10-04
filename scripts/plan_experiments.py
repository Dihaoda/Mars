"""Generate an explicit experiment manifest; execution is opt-in and bounded."""
from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
from pathlib import Path

import yaml

from mars.config import load_config, validate
from mars.data import partition_spec
from mars.utils import digest, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["pilot", "clean-pair", "ablation", "attacks"], required=True)
    parser.add_argument("--directory", default="runs/plans")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-runs", type=int, default=1)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    target = Path(args.directory) / args.suite
    target.mkdir(parents=True, exist_ok=True)
    seed_values = [2026, 2027, 2028]
    if args.suite == "clean-pair":
        configs = [(s, 0.4, "hfedsa", representation, "none")
                   for s, representation in itertools.product(seed_values, ["raw", "effective"])]
    elif args.suite == "pilot":
        configs = [(s, mix, "hfedsa", "raw", "none") for s, mix in itertools.product(seed_values, [0.0, 0.4, 0.8])]
    elif args.suite == "ablation":
        configs = [(s, 0.4, method, representation, "none") for s, method, representation in
                   itertools.product(seed_values, ["hfedsa", "anchor_gmm"], ["raw", "effective"])]
    else:
        configs = [(s, 0.4, method, "effective", attack) for s, method, attack in
                   itertools.product(seed_values, ["fedavg", "hfedsa", "fltrust", "rfa", "anchor_gmm", "hybrid"],
                                     ["label_flip", "backdoor", "scale"])]
    jobs = []
    for seed, mix, method, representation, attack in configs:
        template = "clean_pair.yaml" if args.suite == "clean-pair" else "pilot.yaml" if args.suite == "pilot" else "matched_reference.yaml"
        cfg = load_config(root / "configs" / template)
        cfg["seed"], cfg["data"]["mix_ratio"] = seed, mix
        cfg["defense"]["name"], cfg["defense"]["representation"] = method, representation
        cfg["attack"]["name"] = attack
        cfg["attack"]["malicious_fraction"] = 0.0 if attack == "none" else 0.2
        validate(cfg)
        identifier = f"{method}-{representation}-{attack}-mix{mix:.1f}-seed{seed}"
        config_path = target / (identifier + ".yaml")
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        data_dir = "data/" + digest(partition_spec(cfg))[:16]
        run_dir = f"runs/{args.suite}/{identifier}"
        jobs.append({"id": identifier, "config": config_path.as_posix(), "data": data_dir, "output": run_dir,
                     "config_hash": digest(cfg)})
    write_json(target / "manifest.json", {"suite": args.suite, "jobs": jobs,
                                            "warning": "Planned experiments, not results. Check pinned revisions before execution."})
    print(f"Planned {len(jobs)} runs at {target}. Execute requested: {args.execute}")
    if args.execute:
        if args.max_runs < 1:
            parser.error("--max-runs must be positive")
        for job in jobs[:args.max_runs]:
            common = ["--config", job["config"], "--data", job["data"]]
            subprocess.run([sys.executable, "-m", "mars", "prepare", *common], check=True)
            command = [sys.executable, "-m", "mars", "run", *common, "--out", job["output"]]
            if Path(job["output"], "checkpoint.pt").exists():
                command.append("--resume")
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
