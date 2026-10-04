from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
from pathlib import Path

import numpy as np
import torch


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def digest(value) -> str:
    canonical = json.loads(json.dumps(value, ensure_ascii=False))
    return hashlib.sha256(json.dumps(canonical, sort_keys=True, ensure_ascii=False).encode()).hexdigest()



def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def save_checkpoint(path, value) -> None:
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def load_checkpoint(path):
    # Checkpoints are generated locally by this experiment, never loaded from untrusted uploads.
    return torch.load(path, map_location="cpu", weights_only=False)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def git_revision():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
                                       text=True, timeout=5).strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
