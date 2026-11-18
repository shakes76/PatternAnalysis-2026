"""Progress, safe output files and epoch-boundary recovery for both trainers."""

import hashlib
import random
import time
from pathlib import Path

import numpy as np
import torch


def prepare_output(output, resume):
    output = Path(output)
    if resume:
        if not (output / "last.pt").is_file():
            raise FileNotFoundError("Resume requires last.pt; best.pt alone is not resumable.")
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError("Output is not empty. Use a new directory or --resume.")
    output.mkdir(parents=True, exist_ok=True)


def atomic_save(value, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def atomic_csv(frame, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def capture_rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def manifest_hashes(splits):
    return {name: hashlib.sha256((Path(splits) / f"{name}.csv").read_bytes()).hexdigest()
            for name in ("train", "val", "test")}


def validate_resume(checkpoint, args, hashes):
    if "rng" not in checkpoint or "manifest_hashes" not in checkpoint:
        raise ValueError("This legacy checkpoint lacks recovery state; use a new run directory.")
    if checkpoint["manifest_hashes"] != hashes:
        raise ValueError("Resume manifests differ from the checkpoint.")
    ignored = {"epochs", "resume", "output", "images", "splits", "duplicates", "log_every"}
    for key, value in vars(args).items():
        if key not in ignored and checkpoint["config"].get(key) != value:
            raise ValueError(f"Resume setting differs: {key}")
    if args.workers:
        print("Resume restores main-process RNG state. Persistent worker augmentation states "
              "are not restored; multi-worker resume is not bitwise identical.", flush=True)


class Progress:
    def __init__(self, phase, epoch, epochs, batches, interval):
        self.label = f"Epoch {epoch}/{epochs} {phase}"
        self.batches = batches
        self.interval = interval
        self.started = time.monotonic()
        print(f"{self.label}: starting {batches} batches", flush=True)

    def update(self, batch, loss=None):
        if batch != 1 and batch != self.batches and batch % self.interval:
            return
        elapsed = time.monotonic() - self.started
        eta = elapsed / batch * (self.batches - batch)
        detail = "" if loss is None else f" loss={loss:.5f}"
        print(f"{self.label}: batch {batch}/{self.batches}{detail} "
              f"elapsed={elapsed / 60:.1f}m phase_ETA={eta / 60:.1f}m", flush=True)


def epoch_complete(row, history, epochs):
    average = sum(item["seconds"] for item in history) / len(history)
    remaining = max(0, epochs - row["epoch"]) * average
    print(row, flush=True)
    print(f"Checkpoint saved. Approximate remaining training+validation time: "
          f"{remaining / 60:.1f} minutes (excludes checkpoint I/O).", flush=True)
