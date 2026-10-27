"""Publish count-audited metadata for three frozen models on one known patient.

Standard library only; no checkpoints, image arrays, or predictions are loaded.
CPU FP32 comparisons here do not replace historical CUDA AMP test results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

PROJECT = Path(__file__).resolve().parents[1]
EXPERIMENT = "m015_existing_checkpoint_followup_v1"
MODELS = ("seed42_reference", "gamma_only", "gamma_polarity")
KEYS = tuple(f"M015/Week{i}" for i in range(8))
SHAPE = [64, 128, 128]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    def reject(value):
        raise ValueError("Nonfinite JSON: " + value)
    return json.loads(Path(path).read_text(), parse_constant=reject)


def safe(root, relative):
    require(isinstance(relative, str) and relative and not relative.startswith("/")
            and "\\" not in relative and ":" not in relative
            and all(p not in ("", ".", "..") for p in relative.split("/")), "Unsafe relative evidence path.")
    result = (root / relative).resolve(strict=True)
    require(result.is_relative_to(root.resolve()), "Evidence path escapes root.")
    return result


def equal(actual, expected):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(expected) <= set(actual), "Missing metadata/metric field.")
        for key, value in expected.items():
            equal(actual[key], value)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), "Metric list differs.")
        for a, b in zip(actual, expected):
            equal(a, b)
    elif isinstance(expected, float):
        require(type(actual) in (int, float) and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10), "Metric arithmetic differs.")
    else:
        require(type(actual) is type(expected) and actual == expected, "Identity/scope/count differs.")


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def recompute(cases):
    require(len(cases) == 8 and {f"{c['patient']}/{c['visit']}" for c in cases} == set(KEYS), "Expected all eight M015 visits only.")
    output = []
    for case in sorted(cases, key=lambda c: int(c["visit"][4:])):
        equal(case["shape_dhw"], SHAPE)
        equal([r["class_index"] for r in case["per_class"]], list(range(6)))
        rows = []
        for raw in case["per_class"]:
            tp, pred, gt = (raw[k] for k in ("intersection", "predicted_voxels", "target_voxels"))
            require(all(type(v) is int and v >= 0 for v in (tp, pred, gt)) and tp <= min(pred, gt), "Invalid voxel counts.")
            fp, fn, union = pred - tp, gt - tp, pred + gt - tp
            dice, iou = (2 * tp / (pred + gt) if pred + gt else None), (tp / union if union else None)
            equal(raw, {"dice": dice, "iou": iou, "union": union,
                        "false_positive_voxels": fp, "false_negative_voxels": fn})
            rows.append({"class_index": raw["class_index"], "tp": tp, "fp": fp, "fn": fn,
                         "predicted_voxels": pred, "target_voxels": gt, "dice": dice, "iou": iou,
                         "recall": tp / gt if gt else None, "precision": tp / pred if pred else None})
        require(sum(r["predicted_voxels"] for r in rows) == sum(r["target_voxels"] for r in rows) == math.prod(SHAPE), "Six-class totals differ from grid.")
        foreground = {f"foreground_mean_{m}": mean(r[m] for r in rows[1:]) for m in ("dice", "iou")}
        equal(case, foreground)
        output.append({"patient": "M015", "visit": case["visit"], "per_class": rows, **foreground})
    means = [{"class_index": i, **{m: mean(c["per_class"][i][m] for c in output) for m in ("dice", "iou")}} for i in range(6)]
    foreground = {f"foreground_mean_{m}": mean(c[f"foreground_mean_{m}"] for c in output) for m in ("dice", "iou")}
    aggregate = {"num_cases": 8, "num_patients": 1, "per_class": means, **foreground}
    aggregate["patients"] = [{"patient": "M015", "num_cases": 8, "per_class": means, **foreground}]
    totals = []
    for i in range(6):
        counts = {k: sum(c["per_class"][i][k] for c in output) for k in ("tp", "fp", "fn")}
        tp, fp, fn = (counts[k] for k in ("tp", "fp", "fn"))
        totals.append({"class_index": i, **counts, **{f"mean_visit_{k}": counts[k] / 8 for k in counts},
                       "pooled_voxel_recall": tp / (tp + fn) if tp + fn else None,
                       "pooled_voxel_precision": tp / (tp + fp) if tp + fp else None})
    return output, aggregate, totals


def summarize(evaluation, project=PROJECT):
    path = Path(evaluation).resolve(strict=True)
    protocol_path = project / "m015_followup_protocol.json"
    protocol, index = read(protocol_path), read(path)
    equal(protocol, {"experiment_id": EXPERIMENT, "test_informed_exploratory_acknowledged": True, "case_keys": list(KEYS)})
    require(set(protocol["models"]) == set(MODELS), "Protocol model scope differs.")
    required = protocol["required_file_sha256"]
    for relative, expected in required.items():
        require(sha(safe(project, relative)) == expected, "Protected source/input SHA differs: " + relative)
    common = {"schema_version": 1, "experiment_id": EXPERIMENT, "split": "test_M015_only", "device": "cpu", "amp": False,
              "exploratory_after_test": True, "replaces_v1_test": False, "saved_arrays": False, "condition": "clean",
              "protocol_sha256": sha(protocol_path), "manifest_sha256": protocol["manifest_sha256"]}
    equal(index, {**common, "status": "completed", "protected_files_unchanged": True, "training_performed": False,
                 "n_cases": 8, "n_patients": 1, "case_keys": list(KEYS), "source_file_sha256": required,
                 "appearance_protocol_sha256": protocol["appearance_protocol_sha256"], "cache_identity_sha256": protocol["cache_identity_sha256"]})
    equal(index["runtime"], {"device": "cpu", "amp": False, "gpu_name": None, "torch_num_threads": 4, "omp_num_threads": 4})
    require(set(index["models"]) == set(index["results"]) == set(MODELS), "Index model scope differs.")
    receipts = index["cache_verification"]
    require(len(receipts) == 8 and {r["case_key"] for r in receipts} == set(KEYS)
            and all(r["decoded_identity_verified"] is True and re.fullmatch(r"[0-9a-f]{64}", r["cache_file_sha256"]) for r in receipts), "Cache verification incomplete.")
    results, inputs, gt_counts = {}, {}, None
    for name in MODELS:
        receipt, entry = index["models"][name], index["results"][name]
        digest = protocol["models"][name]["checkpoint_sha256"]
        equal(receipt, {"checkpoint_sha256": digest, "state_unchanged": True})
        require(re.fullmatch(r"[0-9a-f]{64}", receipt["state_before_sha256"])
                and receipt["state_before_sha256"] == receipt["state_after_sha256"], "Model state changed.")
        equal(entry["evaluation_file"], f"{name}/evaluation.json")
        source = safe(path.parent, entry["evaluation_file"])
        require(sha(source) == entry["evaluation_sha256"], "Evaluation file SHA differs.")
        raw = read(source)
        equal(raw, {**common, "model_id": name, "checkpoint_sha256": digest})
        cases, aggregate, totals = recompute(raw["cases"])
        equal(raw["aggregate"], aggregate); equal(entry["aggregate"], aggregate)
        counts = [[r["target_voxels"] for r in c["per_class"]] for c in cases]
        require(gt_counts is None or counts == gt_counts, "GT counts differ between models.")
        gt_counts = counts
        results[name] = {"checkpoint_sha256": digest, "checkpoint_epoch": receipt["checkpoint_epoch"],
                         "state_unchanged": True, "cases": cases, "aggregate": aggregate, "pooled_counts": totals}
        inputs[entry["evaluation_file"]] = sha(source)
    def delta(a, b):
        return {k: a[k] - b[k] if a[k] is not None and b[k] is not None else None for k in ("dice", "iou", "tp", "fp", "fn", "recall", "precision")}
    comparisons = {name: {"per_visit": [{"visit": a["visit"], "per_class": [{"class_index": i, **delta(a["per_class"][i], b["per_class"][i])} for i in range(6)]}
                         for a, b in zip(results[name]["cases"], results[MODELS[0]]["cases"])],
                         "mean_visit_dice_iou_delta": [{"class_index": i, **{m: results[name]["aggregate"]["per_class"][i][m] - results[MODELS[0]]["aggregate"]["per_class"][i][m]
                         if results[name]["aggregate"]["per_class"][i][m] is not None and results[MODELS[0]]["aggregate"]["per_class"][i][m] is not None else None for m in ("dice", "iou")}} for i in range(6)]} for name in MODELS[1:]}
    return {"schema_version": 1, "status": "completed", "experiment_id": EXPERIMENT, "split": "test_M015_only",
            "protocol_sha256": sha(protocol_path), "evaluation_index_sha256": sha(path), "input_file_sha256": inputs,
            "manifest_sha256": protocol["manifest_sha256"], "cache_identity_sha256": protocol["cache_identity_sha256"],
            "source_file_sha256": required, "summarizer_sha256": sha(__file__),
            "scope": {"models": 3, "cases_per_model": 8, "retained_case_rows": 24, "independent_patients": 1, "device": "cpu", "amp": False, "arrays_loaded": False},
            "results": results, "new_minus_same_cpu_reference": comparisons,
            "limits": ["Exploratory assessment after TEST was inspected; one known patient with eight repeated visits, not eight independent patients.",
                       "Not a new benchmark, independent confirmation, clinical validation, or unique root-cause demonstration.",
                       "Historical v1 CUDA AMP and earlier GPU FP32 predictions are not recomputed or substituted; no cross-platform voxel equality is assumed.",
                       "Original v1 TEST, A1 protocol, failed A1 clean-VAL guard and public Release remain unchanged.",
                       "Dice/IoU are visit means; pooled counts and pooled recall/precision are separate descriptive statistics. Both-empty metrics are None."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    require(not os.path.lexists(args.output), "Output already exists; refusing overwrite.")
    report = summarize(args.evaluation)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=args.output.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(report, stream, indent=2, allow_nan=False); stream.write("\n")
            stream.flush(); os.fsync(stream.fileno()); os.link(temporary, args.output)
        finally:
            temporary.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
