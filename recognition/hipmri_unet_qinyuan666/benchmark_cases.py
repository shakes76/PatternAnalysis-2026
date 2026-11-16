"""Measure fixed models per manifest Case, with image-only input and CPU outputs.

Reports instrumented image-to-mask wall time and synchronized forward-only time.
All weeks in a Case are included; this is not per-scan or reconstructed 3D timing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time

import nibabel as nib
import numpy as np
import torch

from dataset import _two_dimensional, normalize_slice, read_manifest, resize_pair
from modules import build_model
from predict import _device, _validate_checkpoint


def read_input(path: str, config: dict) -> torch.Tensor:
    """Match dataset preprocessing without opening ground-truth label files."""
    array = _two_dimensional(nib.load(path).get_fdata(dtype=np.float32), "Image")
    if not np.isfinite(array).all():
        raise ValueError("Image contains non-finite values")
    # Reuse the exact resize/pad implementation; the dummy mask is never scored.
    image, _ = resize_pair(torch.from_numpy(normalize_slice(array))[None],
                           torch.zeros(array.shape, dtype=torch.long),
                           config["image_size"], config["resize_mode"])
    return image


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def measure_case(model, records, config, device, batch_size):
    """Stream images in batches; time forward separately inside the full pipeline."""
    predictions = []
    forward_seconds = 0.0
    synchronize(device)
    begin = time.perf_counter()
    for start in range(0, len(records), batch_size):
        batch = torch.stack([read_input(r["image"], config)
                             for r in records[start:start + batch_size]]).to(device)
        synchronize(device)
        tick = time.perf_counter()
        logits = model(batch)
        synchronize(device)
        forward_seconds += time.perf_counter() - tick
        predictions.append(logits.argmax(1).cpu())
        if not torch.isfinite(logits).all():
            raise ValueError("Non-finite logits during benchmark")
        del batch, logits
    synchronize(device)
    elapsed = time.perf_counter() - begin
    joined = torch.cat(predictions)
    assert joined.shape[0] == len(records)
    return {"pipeline_seconds": elapsed, "forward_seconds": forward_seconds,
            "slices": len(records),
            "prediction_sha256": hashlib.sha256(joined.numpy().tobytes()).hexdigest()}


def benchmark(checkpoints, manifest, output, device="cuda", batch_size=4,
              repeats=3, warmup=5, split="test"):
    for name, value in (("batch_size", batch_size), ("repeats", repeats), ("warmup", warmup)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if split not in ("val", "test"):
        raise ValueError("split must be val or test")
    if not checkpoints:
        raise ValueError("At least one checkpoint is required")
    output, manifest = Path(output), Path(manifest)
    if output.exists():
        raise FileExistsError("Choose a new benchmark directory")
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    digest = sha(manifest)
    records = [r for r in read_manifest(manifest) if r["split"] == split]
    if not records:
        raise ValueError("Selected split has no records")
    groups = {}
    for record in records:
        groups.setdefault(record["patient_id"], []).append(record)
    selected = _device(device)
    models = {}
    checkpoint_hashes = {Path(path): sha(path) for path in checkpoints}
    matching = None
    for path in checkpoints:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        config, _ = _validate_checkpoint(payload, False)
        if config.get("manifest_sha256") != digest:
            raise ValueError("Checkpoint manifest_sha256 mismatch")
        preprocessing = {k: config[k] for k in ("image_size", "resize_mode", "label_values", "num_classes")}
        if matching is not None and matching != preprocessing:
            raise ValueError("Models must share preprocessing and labels")
        matching = preprocessing
        name = config["model"]
        if name in models:
            raise ValueError("Duplicate model architecture")
        model = build_model(name, num_classes=config["num_classes"], base_channels=config["base_channels"])
        model.load_state_dict(payload["model_state"], strict=True)
        models[name] = {"model": model.eval(), "config": config,
                        "epoch": payload["epoch"], "checkpoint_sha256": sha(path)}
    result = {"split": split, "case_count": len(groups), "slice_count": len(records),
              "batch_size": batch_size, "repeats": repeats, "warmup_forwards_per_pass": warmup,
              "manifest_sha256": digest, "preprocessing": matching,
              "environment": {"python": platform.python_version(), "torch": torch.__version__,
                              "cuda_runtime": torch.version.cuda, "device": str(selected),
                              "device_name": torch.cuda.get_device_name(selected) if selected.type == "cuda" else platform.processor(),
                              "threads": torch.get_num_threads(), "dtype": "float32"},
              "source_sha256": {n: sha(Path(__file__).parent / n) for n in
                                ("benchmark_cases.py", "dataset.py", "modules.py", "predict.py")},
              "models": {n: {k: v[k] for k in ("epoch", "checkpoint_sha256")} for n, v in models.items()},
              "execution_order": [], "cases": [],
              "scope": {
                  "pipeline": "NIfTI image reads, normalization, resize/pad (including dummy-mask work), CPU stacking, device transfer, forward, argmax, CPU mask transfer, finite checks and per-batch synchronization. Excludes model load/warmup, final concatenation/hash and disk export.",
                  "forward": "Sum of synchronized model-forward wall times on already transferred batches; excludes image loading/preprocessing, transfers and argmax.",
                  "grouping": "All slices and weeks within each manifest Case, not per scan or reconstructed 3D volume.",
                  "cache": "OS file cache not flushed; repeated warm-model measurements, not cold-start storage performance.",
                  "limitations": "Single laptop/session; power, temperature and other processes not controlled. Interleaved model order reduces order bias but does not establish universal speed ratios. Ground-truth files are never read."
              }}
    for index, (case, rows) in enumerate(sorted(groups.items())):
        alias = f"Case {index + 1:02d}"
        timings = {name: [] for name in models}
        for repeat in range(repeats):
            names = list(models)
            if (index + repeat) % 2:
                names.reverse()
            for name in names:
                entry = models[name]
                model, config = entry["model"].to(selected), entry["config"]
                with torch.inference_mode():
                    batch = torch.stack([read_input(r["image"], config) for r in rows[:batch_size]]).to(selected)
                    for _ in range(warmup):
                        model(batch)
                    synchronize(selected)
                    del batch
                    timing = measure_case(model, rows, config, selected, batch_size)
                timings[name].append(timing)
                result["execution_order"].append({"case_alias": alias, "repeat": repeat + 1, "model": name})
                model.to("cpu")
                print(f"{alias} repeat {repeat + 1} {name}: {timing['pipeline_seconds']:.3f}s pipeline", flush=True)
        summaries = {}
        for name, trials in timings.items():
            summary = {"trials": trials, "predictions_identical_across_repeats": len({x['prediction_sha256'] for x in trials}) == 1}
            for field in ("pipeline_seconds", "forward_seconds"):
                values = [x[field] for x in trials]
                summary[field] = {"median": statistics.median(values), "min": min(values), "max": max(values)}
            summaries[name] = summary
        result["cases"].append({"case_alias": alias, "slices": len(rows), "models": summaries})
    for path, original_hash in checkpoint_hashes.items():
        assert sha(path) == original_hash, "Checkpoint changed"
    assert sha(manifest) == digest, "Manifest changed"
    output.mkdir(parents=True)
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()
    torch.set_num_threads(4)
    benchmark(**vars(args))


if __name__ == "__main__":
    main()
