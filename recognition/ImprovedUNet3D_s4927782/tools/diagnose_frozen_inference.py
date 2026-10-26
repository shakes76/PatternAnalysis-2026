"""Read-only frozen-checkpoint diagnostics, never training or model selection.

Explicit case keys are required. TRAIN controls may have no previous prediction;
validation/test cases must have predictions in the supplied evaluations. Test
access is restricted to M015 and additionally requires --diagnostic-test-ack.
Only a new output directory is written. No masks, logits or predictions are saved.
NumPy/PyTorch imports are lazy so selection and formula checks work without them.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import sys


PROJECT = Path(__file__).resolve().parents[1]


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_child(root, relative):
    root, relative = Path(root).resolve(), Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Evidence array path must be a safe relative path.")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Evidence array path escapes its evidence directory.")
    return path


def select_cases(manifest, requested, diagnostic_test_ack=False):
    if manifest.get("partial") or manifest.get("development_only"):
        raise ValueError("Partial/smoke manifests are forbidden for frozen diagnostics.")
    by_key = {(r["patient"], r["visit"]): r for r in manifest["records"]}
    if len(by_key) != len(manifest["records"]):
        raise ValueError("Duplicate manifest case keys.")
    selected, seen = [], set()
    for value in requested:
        parts = value.split("/")
        if len(parts) != 2 or not all(parts):
            raise ValueError("Each --case must be an explicit PATIENT/VISIT key.")
        key = tuple(parts)
        if key in seen or key not in by_key:
            raise ValueError(f"Duplicate or unknown selected case: {value}.")
        seen.add(key)
        record = by_key[key]
        if record["split"] == "test":
            if key[0] != "M015" or not diagnostic_test_ack:
                raise ValueError("Test cases require --diagnostic-test-ack and must belong to M015.")
        elif record["split"] not in ("train", "validation"):
            raise ValueError("Unrecognized manifest split.")
        selected.append(record)
    if not selected:
        raise ValueError("At least one explicit --case is required.")
    return selected


def metrics_from_confusion(matrix):
    """Independent categorical formula; rows are GT, columns are prediction."""
    size = len(matrix)
    if size < 2 or any(len(row) != size for row in matrix):
        raise ValueError("Confusion matrix must be square with at least two classes.")
    if any(type(value) is not int or value < 0 for row in matrix for value in row):
        raise ValueError("Confusion counts must be nonnegative integers.")
    gt = [sum(row) for row in matrix]
    pred = [sum(row[c] for row in matrix) for c in range(size)]
    per_class = []
    for c in range(size):
        tp = matrix[c][c]
        fp, fn = pred[c] - tp, gt[c] - tp
        denominator, union = gt[c] + pred[c], gt[c] + pred[c] - tp
        per_class.append({"class_index": c, "tp": tp, "fp": fp, "fn": fn,
                          "dice": 2 * tp / denominator if denominator else None,
                          "iou": tp / union if union else None})
    return {"confusion_rows_gt_columns_prediction": matrix, "gt_counts": gt,
            "prediction_counts": pred, "per_class": per_class}


def independent_metrics(prediction, target, num_classes):
    import numpy as np
    prediction, target = np.asarray(prediction), np.asarray(target)
    if prediction.shape != target.shape or not target.size:
        raise ValueError("Prediction/GT must be nonempty and have the same shape.")
    for array in (prediction, target):
        if array.dtype.kind not in "iu" or array.min() < 0 or array.max() >= num_classes:
            raise ValueError("Prediction/GT must contain declared integer class IDs.")
    encoded = target.astype(np.int64).ravel() * num_classes + prediction.astype(np.int64).ravel()
    matrix = np.bincount(encoded, minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return metrics_from_confusion(matrix.tolist())


def dice_deltas(before, after):
    result = []
    for old, new in zip(before["per_class"], after["per_class"]):
        a, b = old["dice"], new["dice"]
        result.append({"class_index": old["class_index"], "before_dice": a, "after_dice": b,
                       "after_minus_before": b - a if a is not None and b is not None else None,
                       "defined_status_changed": (a is None) != (b is None)})
    return result


def compare_predictions(before, after, before_metrics, after_metrics):
    import numpy as np
    if before.shape != after.shape:
        raise ValueError("Cannot compare predictions with different shapes.")
    changed = before != after
    count = int(np.count_nonzero(changed))
    size = len(before_metrics["per_class"])
    return {"different_voxels": count, "total_voxels": int(before.size),
            "different_fraction": count / int(before.size),
            "changed_from_counts": np.bincount(before[changed].astype(np.int64), minlength=size).tolist(),
            "changed_into_counts": np.bincount(after[changed].astype(np.int64), minlength=size).tolist(),
            "per_class_gt_dice_difference": dice_deltas(before_metrics, after_metrics)}


def array_identity(array):
    header = json.dumps({"dtype": array.dtype.str, "shape": list(array.shape)},
                        sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(header + b"\0" + array.tobytes(order="C")).hexdigest()


def load_cache(manifest_path, record, manifest, identity):
    import numpy as np
    path = safe_child(manifest_path.parent, record["cache_path"])
    with np.load(path, allow_pickle=False) as cached:
        arrays = {name: cached[name] for name in ("image", "mask", "spacing")}
    image, target, spacing = (arrays[name] for name in ("image", "mask", "spacing"))
    shape = tuple(manifest["target_shape_dhw"])
    if image.dtype != np.float32 or target.dtype != np.uint8 or image.shape != shape or target.shape != shape:
        raise ValueError("Cache dtype/grid differs from frozen manifest.")
    if not np.isfinite(image).all() or spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("Invalid cached image or spacing.")
    if not np.allclose(spacing, record["spacing_dhw_mm"], rtol=1e-6, atol=1e-6):
        raise ValueError("Cache spacing differs from manifest.")
    if target.max() >= manifest["num_classes"]:
        raise ValueError("Cache mask contains an undeclared class.")
    expected = identity["records"][record["cache_path"]]
    if any(expected[key] != record[key] for key in ("patient", "visit", "split")):
        raise ValueError("Cache identity case metadata differs from manifest.")
    hashes = {name: array_identity(array) for name, array in arrays.items()}
    for name, array in arrays.items():
        item = expected["arrays"][name]
        if item["shape"] != list(array.shape) or item["dtype"] != array.dtype.str or item["sha256"] != hashes[name]:
            raise ValueError(f"Frozen decoded-array identity mismatch: {name}.")
    return image, target, spacing, {"file": str(path), "sha256": file_sha256(path),
                                    "array_sha256": hashes, "frozen_array_identity_verified": True}


def load_saved_prediction(reference, target, spacing, manifest):
    import numpy as np
    evaluation_path, _, row = reference
    path = safe_child(evaluation_path.parent, row["prediction_file"])
    with np.load(path, allow_pickle=False) as saved:
        prediction = saved["prediction"]
        if not np.array_equal(saved["label_ids"], manifest["label_ids"]):
            raise ValueError("Saved prediction label mapping differs from manifest.")
        if not np.allclose(saved["spacing_dhw"], spacing, rtol=1e-6, atol=1e-6):
            raise ValueError("Saved prediction spacing differs from cache.")
    metrics = independent_metrics(prediction, target, manifest["num_classes"])
    old = {int(item["class_index"]): item for item in row["per_class"]}
    if set(old) != set(range(manifest["num_classes"])):
        raise ValueError("Existing evaluation does not report every declared class.")
    checks = []
    for item in metrics["per_class"]:
        original = old[item["class_index"]]
        for metric in ("dice", "iou"):
            a, b = original[metric], item[metric]
            checks.append({"class_index": item["class_index"], "metric": metric,
                           "existing": a, "independent": b,
                           "difference": b - a if a is not None and b is not None else None,
                           "matches": a is None and b is None or a is not None and b is not None and math.isclose(a, b, rel_tol=0, abs_tol=1e-12)})
        for name, value in (("target_voxels", metrics["gt_counts"][item["class_index"]]),
                            ("predicted_voxels", metrics["prediction_counts"][item["class_index"]]),
                            ("intersection", item["tp"]), ("union", item["tp"] + item["fp"] + item["fn"])):
            checks.append({"class_index": item["class_index"], "metric": name,
                           "existing": original[name], "independent": value,
                           "matches": original[name] == value})
    return prediction, metrics, {"file": str(path), "sha256": file_sha256(path),
                                  "metrics": metrics, "existing_evaluation_checks": checks,
                                  "existing_evaluation_matches_independent": all(c["matches"] for c in checks)}


def direct_forward(model, image, model_name, device, amp, batch_size=32):
    """Construct tensors independently of dataset/predict/utils inference helpers."""
    import numpy as np
    import torch
    parts, nonfinite, elements = [], 0, 0
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=amp):
        if model_name == "unet2d":
            inputs = (torch.from_numpy(np.ascontiguousarray(image[start:start + batch_size, None]))
                      for start in range(0, image.shape[0], batch_size))
        else:
            inputs = iter([torch.from_numpy(np.ascontiguousarray(image[None, None]))])
        for tensor in inputs:
            logits = model(tensor.to(device))["logits"]
            invalid = int((~torch.isfinite(logits)).sum().item())
            nonfinite += invalid
            elements += logits.numel()
            if not invalid:
                prediction = logits.argmax(dim=1).cpu().numpy()
                parts.append(prediction if model_name == "unet2d" else prediction[0])
    details = {"amp": amp, "logit_elements": elements, "nonfinite_logit_elements": nonfinite,
               "all_logits_finite": nonfinite == 0,
               "batch": batch_size if model_name == "unet2d" else 1}
    prediction = None if nonfinite else np.concatenate(parts, axis=0) if model_name == "unet2d" else parts[0]
    return prediction, details


def load_evaluations(paths, model_name, checkpoint_sha, manifest_sha, config, manifest):
    references, evidence = {}, []
    by_key = {(r["patient"], r["visit"]): r for r in manifest["records"]}
    for path in paths:
        path = path.expanduser().resolve()
        data = json.loads(path.read_text())
        if data.get("development_only") or data["manifest_sha256"] != manifest_sha or data["checkpoint_sha256"] != checkpoint_sha:
            raise ValueError("Evaluation is development-only or differs from frozen checkpoint/manifest.")
        if data["model_config"] != config or data["model_config"]["model"] != model_name:
            raise ValueError("Evaluation model configuration differs from frozen checkpoint.")
        evidence.append({"path": str(path), "sha256": file_sha256(path), "split": data["split"],
                         "amp": bool(data["amp"]), "original_device": data.get("device"),
                         "latency_protocol": data.get("latency_protocol")})
        for row in data["cases"]:
            key = (row["patient"], row["visit"])
            if key not in by_key or by_key[key]["split"] != data["split"] or key in references:
                raise ValueError("Duplicate/unknown evaluation case or split disagreement.")
            references[key] = (path, data, row)
    return references, evidence


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "baseline-checkpoint", "model-checkpoint", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("baseline-eval", "model-eval"):
        parser.add_argument("--" + name, type=Path, action="append", required=True)
    parser.add_argument("--case", action="append", required=True, help="Explicit PATIENT/VISIT; repeat per case.")
    parser.add_argument("--protocol", type=Path, default=PROJECT / "test_protocol.json")
    parser.add_argument("--training-sources", type=Path, default=PROJECT / "training_sources.json")
    parser.add_argument("--cache-identity", type=Path, help="Default: cache_identity.json beside manifest.")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--diagnostic-test-ack", action="store_true", help="Allow only explicitly listed M015 test cases for retrospective bug diagnosis, never selection/new final scores.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise ValueError("Diagnostic output directory must be new; previous artifacts cannot be overwritten.")
    sys.path.insert(0, str(PROJECT))
    import numpy as np
    import torch
    from dataset import load_manifest
    from modules import ImprovedUNet3D, UNet2D

    manifest_path = args.manifest.expanduser().resolve()
    manifest = load_manifest(manifest_path)
    selected = select_cases(manifest, args.case, args.diagnostic_test_ack)
    manifest_sha = file_sha256(manifest_path)
    protocol_path, sources_path = args.protocol.expanduser().resolve(), args.training_sources.expanduser().resolve()
    protocol, sources = json.loads(protocol_path.read_text()), json.loads(sources_path.read_text())
    identity_path = (args.cache_identity or manifest_path.parent / "cache_identity.json").expanduser().resolve()
    identity = json.loads(identity_path.read_text())
    if any(item["manifest_sha256"] != manifest_sha for item in (protocol, sources, identity)):
        raise ValueError("Frozen protocol/source/cache identity manifest digests disagree.")
    if protocol["inference"]["slice_batch_size"] != 32:
        raise ValueError("This diagnostic requires the frozen 32-slice 2D inference protocol.")
    if set(identity["records"]) != {r["cache_path"] for r in manifest["records"]}:
        raise ValueError("Cache identity does not cover exactly the manifest records.")
    if file_sha256(PROJECT / "modules.py") != sources["source_file_sha256"]["modules.py"]:
        raise ValueError("Model source differs from the frozen training source snapshot.")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("Requested CUDA is unavailable.")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    models, references, model_evidence, configs = {}, {}, {}, {}
    for model_name, checkpoint_path, eval_paths in (
        ("unet2d", args.baseline_checkpoint, args.baseline_eval),
        ("improved3d", args.model_checkpoint, args.model_eval),
    ):
        checkpoint_path = checkpoint_path.expanduser().resolve()
        digest = file_sha256(checkpoint_path)
        expected = protocol["models"][model_name]
        if digest != expected["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint differs from frozen protocol: {model_name}.")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        config = checkpoint["config"]
        if config.get("development_only") or config.get("max_batches") is not None:
            raise ValueError("Development/smoke checkpoints are forbidden.")
        if (config["model"] != model_name or config["manifest_sha256"] != manifest_sha
                or config["num_classes"] != manifest["num_classes"] or config["label_ids"] != manifest["label_ids"]
                or config["target_shape_dhw"] != manifest["target_shape_dhw"]
                or checkpoint["epoch"] != expected["selected_epoch"]):
            raise ValueError("Frozen checkpoint configuration/epoch differs from protocol/manifest.")
        torch.manual_seed(config["seed"])
        factory = UNet2D if model_name == "unet2d" else ImprovedUNet3D
        model = factory(config["num_classes"], config["base_channels"]).to(device).eval()
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        models[model_name], configs[model_name] = model, config
        references[model_name], evaluations = load_evaluations(eval_paths, model_name, digest, manifest_sha, config, manifest)
        model_evidence[model_name] = {"checkpoint": str(checkpoint_path), "checkpoint_sha256": digest,
                                      "selected_epoch": checkpoint["epoch"], "evaluations": evaluations}
        del checkpoint
        for record in selected:
            if record["split"] != "train" and (record["patient"], record["visit"]) not in references[model_name]:
                raise ValueError("Each selected validation/test case requires both existing evaluations/predictions.")
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1, "diagnostic_only": True, "no_training_or_model_selection": True,
        "no_new_final_test_scores": True, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "explicit_case_keys": args.case, "diagnostic_test_ack": args.diagnostic_test_ack,
        "manifest": str(manifest_path), "manifest_sha256": manifest_sha,
        "protocol": str(protocol_path), "protocol_sha256": file_sha256(protocol_path),
        "training_sources_sha256": file_sha256(sources_path), "cache_identity_sha256": file_sha256(identity_path),
        "diagnostic_tool_sha256": file_sha256(__file__), "modules_sha256": file_sha256(PROJECT / "modules.py"),
        "models": model_evidence, "python_version": platform.python_version(), "numpy_version": np.__version__,
        "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda, "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "backend": {"cudnn_benchmark": False, "cudnn_deterministic": True, "deterministic_algorithms": "warn_only",
                    "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                    "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                    "float32_matmul_precision": torch.get_float32_matmul_precision()},
        "protocol_notes": ["Direct forward uses the frozen main head only; 2D batches are fixed at 32 slices, 3D at one volume.",
                           "No aggregate accuracy, ranking, threshold, model selection or postprocessing is performed.",
                           "FP32 differences diagnose numerical sensitivity; they do not replace the frozen final-test predictions/scores.",
                           "Bit-exact parity may depend on PyTorch/CUDA/backend versions. Differences alone do not establish a bug.",
                           "TRAIN controls without supplied previous predictions are explicitly unavailable for saved-prediction parity."],
        "cases": [],
    }
    for record in selected:
        key = (record["patient"], record["visit"])
        row = {"patient": key[0], "visit": key[1], "split": record["split"], "models": {}}
        try:
            image, target, spacing, cache_evidence = load_cache(manifest_path, record, manifest, identity)
            row.update(cache=cache_evidence, shape_dhw=list(image.shape), spacing_dhw_mm=spacing.tolist())
            for model_name, model in models.items():
                reference = references[model_name].get(key)
                saved, saved_metrics = None, None
                item = {"saved_prediction_status": "available" if reference else "unavailable_no_supplied_train_evaluation"}
                if reference:
                    saved, saved_metrics, item["saved_prediction"] = load_saved_prediction(reference, target, spacing, manifest)
                    item["evaluation_sha256"] = file_sha256(reference[0])
                requested_amp = bool(reference[1]["amp"] if reference else configs[model_name]["amp"])
                if requested_amp and device.type != "cuda":
                    parity, parity_metrics = None, None
                    item["same_precision_forward"] = {"status": "unavailable_cuda_amp_on_cpu", "requested_amp": True}
                else:
                    parity, details = direct_forward(model, image, model_name, device, requested_amp)
                    parity_metrics = independent_metrics(parity, target, manifest["num_classes"]) if parity is not None else None
                    item["same_precision_forward"] = {"status": "completed", **details, "metrics": parity_metrics}
                fp32, details = direct_forward(model, image, model_name, device, False)
                fp32_metrics = independent_metrics(fp32, target, manifest["num_classes"]) if fp32 is not None else None
                item["fp32_forward"] = {"status": "completed", **details, "metrics": fp32_metrics}
                if saved is not None and parity is not None:
                    item["saved_to_same_precision"] = compare_predictions(saved, parity, saved_metrics, parity_metrics)
                if saved is not None and fp32 is not None:
                    item["saved_to_fp32"] = compare_predictions(saved, fp32, saved_metrics, fp32_metrics)
                if parity is not None and fp32 is not None:
                    item["same_precision_to_fp32"] = compare_predictions(parity, fp32, parity_metrics, fp32_metrics)
                row["models"][model_name] = item
            row["status"] = "completed"
        except Exception as error:
            row.update(status="blocked", error_type=type(error).__name__, error=str(error))
        report["cases"].append(row)
    forwards = [item[mode] for row in report["cases"] for item in row["models"].values()
                for mode in ("same_precision_forward", "fp32_forward")
                if item[mode]["status"] == "completed"]
    saved_checks = [item["saved_prediction"]["existing_evaluation_matches_independent"]
                    for row in report["cases"] for item in row["models"].values() if "saved_prediction" in item]
    report["diagnostic_checks"] = {
        "completed_case_count": sum(row["status"] == "completed" for row in report["cases"]),
        "executed_forward_count": len(forwards),
        "all_executed_forwards_finite": all(item["all_logits_finite"] for item in forwards),
        "saved_prediction_metric_checks": len(saved_checks),
        "all_existing_metrics_match_independent": all(saved_checks),
    }
    report["status"] = "completed" if all(r["status"] == "completed" for r in report["cases"]) else "incomplete"
    if report["status"] == "completed" and not report["diagnostic_checks"]["all_executed_forwards_finite"]:
        report["status"] = "nonfinite_logits_detected"
    destination = output / "diagnostic.json"
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Frozen diagnostic {report['status']}: {len(report['cases'])} explicit cases; {destination}")
    return 0 if report["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
