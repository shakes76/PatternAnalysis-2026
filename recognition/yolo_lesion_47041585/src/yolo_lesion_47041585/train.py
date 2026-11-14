"""
Training entry point: resolve configs, run one experiment, record its provenance.

Normal runs train and validate only. Test evaluation lives behind the guarded path in
helpers/evaluation.py to be used once the selection manifest is frozen.

Usage:
    uv run train --config configs/b0.yaml [--pilot] [--seed N]

configs/base.yaml is always applied first, so each --config only names what it changes.
"""

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import yaml
from numpy.typing import NDArray

from yolo_lesion_47041585.dataset import CLASS_MAPPING, read_train_classes
from yolo_lesion_47041585.modules import (
    LesionDetectionTrainer,
    build_detector,
    trainer_overrides,
)

SAMPLING_SCHEMES = ("stock", "replacement")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = Path("runs")
# the first epoch carries start-up costs, so two more give a steady-state time
PILOT_EPOCHS = 3
BASE_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "base.yaml"


def _merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge `override` into a copy of `base`."""
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(value, Mapping) and isinstance(current, Mapping):
            merged[key] = _merge(
                dict(cast(Mapping[str, Any], current)), cast(Mapping[str, Any], value)
            )
        else:
            merged[key] = value
    return merged


def resolve_training_config(
    config_paths: Sequence[Path], pilot: bool = False
) -> dict[str, Any]:
    """Merge YAML configs left to right into one fully resolved config.

    `configs/base.yaml` is always applied first, so it need not be listed.

    Args:
        config_paths (Sequence[Path]): Condition overrides, applied in order.
        pilot (bool): Shorten the run for the smoke test and label it as a pilot.

    Returns:
        dict[str, Any]: The resolved config, to be saved with the run.
    """
    config: dict[str, Any] = {}
    layers = [path for path in config_paths if path.resolve() != BASE_CONFIG.resolve()]
    for path in [BASE_CONFIG, *layers]:
        config = _merge(config, yaml.safe_load(path.read_text()) or {})

    sampling = config["sampling"]
    if sampling["scheme"] not in SAMPLING_SCHEMES:
        raise ValueError(f"Unknown sampling scheme: {sampling['scheme']!r}")

    if sampling["scheme"] == "stock" and sampling["malignant_fraction"] is not None:
        raise ValueError("The stock loader cannot sample to a malignant fraction.")

    if pilot:
        config["pilot"] = True
        config["train"]["epochs"] = PILOT_EPOCHS

    return config


def file_sha256(path: Path) -> str:
    """SHA-256 of a file, read in chunks so large checkpoints stay out of memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1 << 20):
            digest.update(chunk)

    return digest.hexdigest()


def _command_output(command: Sequence[str]) -> str | None:
    """The stripped stdout of a command run in the project, or None if it can't run."""
    try:
        result = subprocess.run(
            command, cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
        )

    except (OSError, subprocess.CalledProcessError):
        return None

    return result.stdout.strip()


def run_id(config: Mapping[str, Any]) -> str:
    """The run's name: its condition and seed, marked if it is a pilot."""
    pilot = "-pilot" if config.get("pilot") else ""

    return f"{config['condition']}-s{config['seed']}{pilot}"


def collect_run_metadata(config: Mapping[str, Any], model_path: Path) -> dict[str, Any]:
    """Provenance for a run: split hash, package/CUDA/GPU versions, starting-checkpoint
    hash and seed.

    Args:
        config (Mapping[str, Any]): The resolved config.
        model_path (Path): The local starting checkpoint.

    Returns:
        dict[str, Any]: JSON-serialisable provenance.
    """
    cuda = torch.cuda.is_available()

    return {
        "run_id": run_id(config),
        "condition": config["condition"],
        "seed": config["seed"],
        "pilot": bool(config.get("pilot")),
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split_manifest": {
            "path": str(config["split_manifest"]),
            "sha256": file_sha256(Path(config["split_manifest"])),
        },
        "starting_checkpoint": {
            "path": str(model_path),
            "sha256": file_sha256(model_path),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": version("torch"),
            "torchvision": version("torchvision"),
            "ultralytics": version("ultralytics"),
            "numpy": version("numpy"),
            "pillow": version("pillow"),
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if cuda else None,
            "driver": _command_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
            ),
        },
    }


def epoch_seconds(results_csv: Path) -> list[float]:
    """Each epoch's duration, validation included, from Ultralytics' cumulative times."""
    with results_csv.open(newline="") as file:
        elapsed = [float(row["time"]) for row in csv.DictReader(file)]
    return [round(b - a, 1) for a, b in zip([0.0, *elapsed], elapsed)]


def summarise_sampling(
    history: Sequence[NDArray[np.int64]], class_ids: NDArray[np.int_]
) -> list[dict[str, float | int]]:
    """Per-epoch counts of what the sampler drew.

    Args:
        history (Sequence[NDArray[np.int64]]): Indices drawn in each epoch.
        class_ids (NDArray[np.int_]): The class of each image the indices refer to.

    Returns:
        list[dict[str, float | int]]: One row per epoch with the number of draws, how
            many were malignant, that share, and how many distinct images were drawn.
    """
    malignant_id = CLASS_MAPPING["Malignant"]
    rows: list[dict[str, float | int]] = []

    for epoch, draws in enumerate(history):
        malignant = int((class_ids[draws] == malignant_id).sum())
        rows.append(
            {
                "epoch": epoch,
                "draws": len(draws),
                "malignant_draws": malignant,
                "malignant_fraction": malignant / len(draws),
                "unique_images": len(np.unique(draws)),
            }
        )

    return rows


def run_experiment(config: Mapping[str, Any]) -> Path:
    """Train and validate one condition and return its run directory.

    Builds the detector and trainer, saves the resolved config and provenance before
    training so a failed run still leaves them behind, trains with the best checkpoint
    chosen by validation mAP50:95, then saves the resources used and, for sampled runs,
    what was drawn each epoch. Nothing here touches the test split.

    Args:
        config (Mapping[str, Any]): The resolved config.

    Returns:
        Path: The run directory under `runs/`.

    Raises:
        FileExistsError: If the run directory already exists.
    """
    run_dir = RUNS_DIR / run_id(config)
    if run_dir.exists():
        raise FileExistsError(f"{run_dir} already exists, move or remove it first.")

    sampling = config["sampling"]
    train_classes = read_train_classes(Path(config["split_manifest"]))
    detector = build_detector(str(config["model"]))
    model_path = Path(cast(str, detector.ckpt_path))  # pyright: ignore[reportUnknownMemberType]

    trainer = LesionDetectionTrainer(
        overrides=trainer_overrides(config, model_path, run_dir),
        train_classes=train_classes,
        scheme=sampling["scheme"],
        target_fraction=sampling["malignant_fraction"],
    )
    untyped = cast(Any, trainer)  # Ultralytics' trainer attributes are untyped
    if Path(untyped.save_dir).resolve() != run_dir.resolve():
        raise RuntimeError(f"Ultralytics chose {untyped.save_dir}, expected {run_dir}.")

    (run_dir / "config.resolved.yaml").write_text(yaml.safe_dump(dict(config)))
    metadata = collect_run_metadata(config, model_path)
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    trainer.train()
    elapsed = time.perf_counter() - start

    result: dict[str, Any] = {
        "run_id": metadata["run_id"],
        "best_checkpoint": str(Path(untyped.best).relative_to(run_dir.resolve())),
        "last_checkpoint": str(Path(untyped.last).relative_to(run_dir.resolve())),
        "best_val_map50_95": float(untyped.best_fitness),
        "epochs_run": int(untyped.epoch) + 1,
        "batch_size": int(untyped.batch_size),
        "batches_per_epoch": len(untyped.train_loader),
        "gradient_accumulation": int(untyped.accumulate),
        "elapsed_seconds": round(elapsed, 1),
        "epoch_seconds": epoch_seconds(run_dir / "results.csv"),
        "validation_seconds": [round(s, 1) for s in trainer.validation_seconds],
        "peak_vram_allocated_mb": None,
        "peak_vram_reserved_mb": None,
    }
    if torch.cuda.is_available():
        result["peak_vram_allocated_mb"] = torch.cuda.max_memory_allocated() >> 20
        result["peak_vram_reserved_mb"] = torch.cuda.max_memory_reserved() >> 20
    (run_dir / "result.json").write_text(json.dumps(result, indent=2))

    if trainer.train_sampler and trainer.train_class_ids is not None:
        # the loader may have drawn one epoch ahead of the last one trained
        history = trainer.train_sampler.history[: result["epochs_run"]]
        rows = summarise_sampling(history, trainer.train_class_ids)
        lines = [",".join(rows[0])] + [
            ",".join(str(v) for v in r.values()) for r in rows
        ]
        (run_dir / "sampling.csv").write_text("\n".join(lines) + "\n")

    return run_dir


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, action="append", required=True)
    parser.add_argument("--pilot", action="store_true", help="short smoke-test run")
    parser.add_argument("--device", help="Ultralytics device, e.g. cpu, mps or 0")
    parser.add_argument(
        "--seed",
        type=int,
        help="training seed, replacing the config's; the data split is unaffected",
    )
    args = parser.parse_args(argv)

    config = resolve_training_config(args.config, args.pilot)
    if args.seed is not None:
        config["seed"] = args.seed
    if args.device:
        config["train"]["device"] = args.device
    print(f"Run complete: {run_experiment(config)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
