"""D2 exploratory single-factor support-zeroing intervention after D1 inspection.

Compare frozen cache inputs against the same canonical/zscore/clipped/linear
resampling with ONLY the final nearest-support zeroing removed. All 14 cases
must be reported. TEST is restricted to the previously examined M015 visits,
with explicit acknowledgement. No fitting, arrays, predictions or checkpoints
are written; this is not a new TEST benchmark or a replacement v1 result.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from tools.cache_identity import atomic_json, safe_relative
from tools.diagnose_model_mechanism import (ALLOWED_CASES, CHECKPOINT_SHA, FROZEN_SHA,
    class5_summary, select_cases, sha256, state_sha256, verify_documents)
from tools.diagnose_source_inputs import normalize_nonzero


def complete_cases(manifest, requested, ack):
    names = sorted(ALLOWED_CASES) if requested is None else requested
    if len(names) != 14 or set(names) != set(ALLOWED_CASES):
        raise ValueError("D2 requires every one of the 14 allowed cases exactly once; no other cases.")
    return select_cases(manifest, names, [], ack)


def change_summary(before, after, target):
    """Summarize a change without retaining or exporting changed arrays."""
    if before.shape != after.shape or before.shape != target.shape:
        raise ValueError("Change-summary grids differ.")
    changed = before != after
    locations = np.argwhere(changed)
    delta = after.astype(np.float64) - before.astype(np.float64)
    return {"different_voxels": int(changed.sum()), "total_voxels": int(changed.size),
            "gt_class5_affected_voxels": int((changed & (target == 5)).sum()),
            "bbox_dhw_half_open": [[int(v), int(w) + 1] for v, w in
                                   zip(locations.min(axis=0), locations.max(axis=0))] if locations.size else None,
            "changed_d_indices": np.flatnonzero(changed.any(axis=(1, 2))).tolist(),
            "after_minus_before_min_on_changed": float(delta[changed].min()) if locations.size else None,
            "after_minus_before_max_on_changed": float(delta[changed].max()) if locations.size else None,
            "max_absolute_difference": float(np.abs(delta).max())}


def prepare_intervention(before_support, nearest_support, frozen_cache, target):
    """Reinstating zeroing must exactly reconstruct baseline before any forward."""
    if (before_support.dtype != np.float32 or frozen_cache.dtype != np.float32
            or before_support.shape != frozen_cache.shape or target.shape != frozen_cache.shape
            or nearest_support.shape != frozen_cache.shape or frozen_cache.ndim != 3
            or not np.isfinite(before_support).all() or not np.isfinite(frozen_cache).all()):
        raise ValueError("Intervention requires matching finite float32 3D grids.")
    support = np.asarray(nearest_support, dtype=bool)
    restored = before_support.copy()
    restored[~support] = 0
    if not np.array_equal(restored, frozen_cache):
        raise ValueError("Reconstructed support-zeroed baseline is not elementwise equal to the frozen cache.")
    intervention = before_support.copy()
    changed = intervention != frozen_cache
    if np.any(changed & support):
        raise ValueError("Single-factor violation: a supported input voxel changed.")
    summary = change_summary(frozen_cache, intervention, target)
    summary.update(baseline_reconstruction_elementwise_equal=True,
                   changed_voxels_inside_retained_support=int((changed & support).sum()),
                   original_gt_class5_voxels=int((target == 5).sum()),
                   gt_class5_outside_nearest_support=int(((target == 5) & ~support).sum()))
    return intervention, summary


def reconstruct_before_support(image_path, record, target_shape, nib, zoom):
    volume = nib.load(str(image_path))
    if list(volume.shape) != record["original_image"]["shape_xyz"] or not np.allclose(
            volume.affine, record["original_image"]["affine_xyz"], rtol=1e-5, atol=1e-5):
        raise ValueError("Native MRI geometry differs from the frozen record.")
    canonical = nib.as_closest_canonical(volume)
    if nib.aff2axcodes(canonical.affine) != ("R", "A", "S") or not np.allclose(
            canonical.affine, record["canonical_image"]["affine_xyz"], rtol=1e-5, atol=1e-5):
        raise ValueError("Canonical MRI geometry differs from the frozen record.")
    image = canonical.get_fdata(dtype=np.float32, caching="unchanged")
    normalized, active, normalization = normalize_nonzero(image)
    ratio = np.asarray(target_shape[::-1], dtype=np.float64) / np.asarray(image.shape)
    options = {"grid_mode": True, "mode": "nearest", "prefilter": False}
    before = zoom(normalized, ratio, order=1, **options).transpose(2, 1, 0)
    support = zoom(active.astype(np.uint8), ratio, order=0, **options).transpose(2, 1, 0).astype(bool)
    return np.ascontiguousarray(before, dtype=np.float32), support, normalization


def target_metrics(prediction, target):
    predicted, actual = prediction == 5, target == 5
    tp = int((predicted & actual).sum())
    fp, fn = int((predicted & ~actual).sum()), int((~predicted & actual).sum())
    denominator = 2 * tp + fp + fn
    return {"tp": tp, "fp": fp, "fn": fn, "gt_voxels": int(actual.sum()),
            "predicted_voxels": int(predicted.sum()), "dice": 2 * tp / denominator if denominator else None}


def forward_summary(model, image, target, model_name, device):
    """FP32 main-head replay only; predictions remain ephemeral in memory."""
    import torch
    predictions, rows = [], []
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
        chunks = [(image, target)] if model_name == "improved3d" else [
            (image[start:start + 32], target[start:start + 32]) for start in range(0, len(image), 32)]
        for chunk, labels in chunks:
            data = chunk[None, None] if model_name == "improved3d" else chunk[:, None]
            logits = model(torch.from_numpy(np.ascontiguousarray(data)).to(device))["logits"]
            if not bool(torch.isfinite(logits).all().item()):
                raise ValueError("Nonfinite FP32 main-head logits.")
            if model_name == "improved3d":
                logits = logits[0].movedim(0, -1)
            else:
                logits = logits.movedim(1, -1)
            selected = torch.from_numpy(np.ascontiguousarray(labels == 5)).to(device)
            rows.append(logits[selected].cpu().numpy())
            predictions.append(logits.argmax(dim=-1).cpu().numpy().astype(np.uint8))
    prediction = predictions[0] if model_name == "improved3d" else np.concatenate(predictions, axis=0)
    return prediction, {"class5": target_metrics(prediction, target),
                        "gt_class5_logits": class5_summary(np.concatenate(rows, axis=0)),
                        "all_main_logits_finite": True}


def load_frozen_model(name, checkpoint_path, documents, device, core, torch):
    if sha256(checkpoint_path) != CHECKPOINT_SHA[name]:
        raise ValueError("Only the pinned v1 seed42 checkpoint bytes are allowed; seed43 is forbidden.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    manifest = documents["manifest"]
    if (config["model"] != name or config["seed"] != 42 or config["base_channels"] != 16
            or config["manifest_sha256"] != FROZEN_SHA["manifest"]
            or config["num_classes"] != 6 or config["label_ids"] != list(range(6))
            or config["target_shape_dhw"] != manifest["target_shape_dhw"]
            or config["patient_splits"] != manifest["patient_splits"]
            or config.get("development_only") or config.get("max_batches") is not None
            or checkpoint["epoch"] != documents["protocol"]["models"][name]["selected_epoch"]):
        raise ValueError("Checkpoint configuration differs from frozen v1.")
    factory = core.UNet2D if name == "unet2d" else core.ImprovedUNet3D
    model = factory(6, 16).to(device).eval()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model, {"checkpoint_sha256": CHECKPOINT_SHA[name], "selected_epoch": checkpoint["epoch"],
                   "state_before_sha256": state_sha256(model)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--d1-source-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-identity", type=Path)
    parser.add_argument("--protocol", type=Path, default=PROJECT / "test_protocol.json")
    parser.add_argument("--training-sources", type=Path, default=PROJECT / "training_sources.json")
    parser.add_argument("--baseline-checkpoint", type=Path, default=PROJECT / "artifacts/full_seed42/unet2d/best.pt")
    parser.add_argument("--model-checkpoint", type=Path, default=PROJECT / "artifacts/full_seed42/improved3d/best.pt")
    parser.add_argument("--case", action="append", help="Optional explicit list; must contain all 14 allowed cases exactly once.")
    parser.add_argument("--diagnostic-test-ack", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    output = args.output_dir.expanduser().absolute()
    if os.path.lexists(output):
        parser.error("Use a new output directory; no existing evidence is replaced.")
    paths = {"manifest": args.manifest.expanduser().resolve(strict=True),
             "protocol": args.protocol.expanduser().resolve(strict=True),
             "sources": args.training_sources.expanduser().resolve(strict=True),
             "identity": (args.cache_identity or args.manifest.parent / "cache_identity.json").expanduser().resolve(strict=True)}
    documents = verify_documents(paths)
    selected = complete_cases(documents["manifest"], args.case, args.diagnostic_test_ack)
    d1_path = args.d1_source_report.expanduser().resolve(strict=True)
    d1_hash = sha256(d1_path)
    d1 = json.loads(d1_path.read_text())
    if (d1.get("status") != "completed" or d1.get("issues") != []
            or set(d1.get("selection", [])) != set(ALLOWED_CASES)
            or d1.get("input_sha256") != {"manifest": FROZEN_SHA["manifest"],
                "cache_identity": FROZEN_SHA["identity"], "training_sources": FROZEN_SHA["sources"]}):
        raise ValueError("D2 requires the completed, matching 14-case D1 source report.")
    d1_cases = {row["case"]: row for row in d1["cases"]}
    if len(d1["cases"]) != 14 or set(d1_cases) != set(ALLOWED_CASES):
        raise ValueError("The D1 report must contain each of the 14 source cases exactly once.")
    data_root = args.data_root.expanduser().resolve(strict=True)
    checkpoint_paths = {"unet2d": args.baseline_checkpoint.expanduser().resolve(strict=True),
                        "improved3d": args.model_checkpoint.expanduser().resolve(strict=True)}
    if any(output.resolve().is_relative_to(p.parent) for p in [paths["manifest"], *checkpoint_paths.values()]):
        raise ValueError("Output must be outside frozen cache/checkpoint directories.")
    import nibabel as nib
    import scipy
    from scipy.ndimage import zoom
    import torch
    import modules as core
    from tools.diagnose_frozen_inference import load_cache
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable.")
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    models, model_evidence = {}, {}
    for name, path in checkpoint_paths.items():
        models[name], model_evidence[name] = load_frozen_model(name, path, documents, device, core, torch)
    helper_paths = [Path(__file__).resolve(), PROJECT / "tools/diagnose_model_mechanism.py",
                    PROJECT / "tools/diagnose_source_inputs.py", PROJECT / "tools/diagnose_frozen_inference.py"]
    protected = {**{str(path): sha256(path) for path in [*paths.values(), d1_path, *helper_paths]},
                 **{str(path): CHECKPOINT_SHA[name] for name, path in checkpoint_paths.items()}}
    report = {"schema_version": 1, "experiment": "D2_exploratory_support_zeroing_intervention",
              "planned_after_D1_observations": True, "diagnostic_only": True,
              "no_new_test_benchmark": True, "no_training_or_parameter_updates": True,
              "created_at_utc": datetime.now(timezone.utc).isoformat(), "d1_source_sha256": d1_hash,
              "intervention": "Remove only the final nearest-support forced-zero operation; all preceding pixels, normalization, interpolation, geometry, masks and model weights remain fixed.",
              "case_keys": [f"{r['patient']}/{r['visit']}" for r in selected],
              "all_fourteen_cases_required": True, "diagnostic_test_ack": args.diagnostic_test_ack,
              "protected_input_file_sha256": protected, "models": model_evidence,
              "runtime": {"python": platform.python_version(), "numpy": np.__version__, "nibabel": nib.__version__,
                          "scipy": scipy.__version__, "torch": str(torch.__version__), "cuda": torch.version.cuda,
                          "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                          "precision": "FP32; autocast and TF32 disabled"},
              "limits": ["Post-D1 exploratory intervention on already examined cases; no independent test claim or v1 replacement.",
                         "No optimizer, backward, model selection, augmentation search, threshold search or saved pixel/logit/prediction arrays.",
                         "Intervention response estimates sensitivity to this input policy at fixed weights; it does not isolate GroupNorm mediation or historical training effects.",
                         "No improvement would constrain this one-factor explanation, not prove every preprocessing/design choice harmless.",
                         "R016/Week7 and T014/Week3 were selected after initial descriptive statistics as exploratory TRAIN controls.",
                         "Only per-case contrasts are reported; no new aggregate TEST score, best-case selection or significance claim."],
              "cases": []}
    output.mkdir(parents=True, exist_ok=False)
    for record in selected:
        key = f"{record['patient']}/{record['visit']}"
        row = {"case": key, "split": record["split"]}
        try:
            baseline, target, spacing, cache_evidence = load_cache(paths["manifest"], record, documents["manifest"], documents["identity"])
            source_path = safe_relative(data_root, record["source_image"])
            source_hash = sha256(source_path)
            if source_hash != d1_cases[key]["source_file_sha256"]["image"]:
                raise ValueError("Source MRI bytes differ from the inspected D1 source.")
            before, support, normalization = reconstruct_before_support(source_path, record, baseline.shape, nib, zoom)
            changed, input_summary = prepare_intervention(before, support, baseline, target)
            row.update(source_image_sha256=source_hash, cache_identity=cache_evidence,
                       normalization=normalization, spacing_dhw_mm=spacing.tolist(), input_change=input_summary, models={})
            for name, model in models.items():
                before_state = state_sha256(model)
                base_prediction, base_summary = forward_summary(model, baseline, target, name, device)
                changed_prediction, changed_summary = forward_summary(model, changed, target, name, device)
                after_state = state_sha256(model)
                row["models"][name] = {"baseline_frozen_cache_FP32": base_summary,
                                       "without_support_zeroing_FP32": changed_summary,
                                       "prediction_change": change_summary(base_prediction, changed_prediction, target),
                                       "state_before_sha256": before_state, "state_after_sha256": after_state,
                                       "parameters_and_buffers_unchanged": before_state == after_state}
                if before_state != after_state:
                    raise RuntimeError("Model state changed during forward-only intervention.")
            if sha256(source_path) != source_hash:
                raise RuntimeError("Source MRI changed during reconstruction.")
            row["status"] = "completed"
        except Exception as error:
            row.update(status="blocked", error_type=type(error).__name__, error=str(error))
        report["cases"].append(row)
        atomic_json(output / f"case_{record['patient']}_{record['visit']}.json", row)
        print(f"D2 support intervention {key}: {row['status']}", flush=True)
    for name, model in models.items():
        model_evidence[name]["state_after_sha256"] = state_sha256(model)
        model_evidence[name]["parameters_and_buffers_unchanged"] = model_evidence[name]["state_before_sha256"] == model_evidence[name]["state_after_sha256"]
        model_evidence[name]["checkpoint_file_unchanged"] = sha256(checkpoint_paths[name]) == CHECKPOINT_SHA[name]
    verify_documents(paths)  # Recheck all four documents and the 11 core sources.
    report["protected_inputs_unchanged"] = all(sha256(path) == digest for path, digest in protected.items())
    report["status"] = "completed" if report["protected_inputs_unchanged"] and all(
        row["status"] == "completed" for row in report["cases"]) and all(
        model["parameters_and_buffers_unchanged"] and model["checkpoint_file_unchanged"] for model in model_evidence.values()) else "incomplete_or_failed"
    atomic_json(output / "support_intervention.json", report)
    print(f"D2 evidence: {output / 'support_intervention.json'}")
    return 0 if report["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
