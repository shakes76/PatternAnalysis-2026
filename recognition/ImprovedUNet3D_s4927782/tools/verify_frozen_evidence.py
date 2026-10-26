"""Independently audit saved prediction counts, metric algebra and aggregation.

This CPU-only check never loads a model, trains or modifies an evaluation.
It does not load GT arrays: saved intersections are checked algebraically,
not independently remeasured. Native/cache and direct-forward diagnostics
provide the separate checks that require image/GT arrays and GPU inference.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mean(values):
    values = [float(x) for x in values if x is not None]
    return math.fsum(values) / len(values) if values else None


def same(actual, expected, label):
    if actual is None or expected is None:
        if actual is not None or expected is not None:
            raise ValueError(f"Undefined-value mismatch: {label}")
    elif not math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"Value mismatch: {label}: {actual!r} vs {expected!r}")


def relative_file(root, relative):
    if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("Prediction path must stay within its evaluation directory.")
    result = (root / relative).resolve(strict=True)
    if not result.is_relative_to(root):
        raise ValueError("Prediction symlink escapes evaluation directory.")
    return result


def expected_filename(patient, visit):
    raw = f"{patient}__{visit}"
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip(".")[:100] or "case"
    return stem + "_" + hashlib.sha256(raw.encode()).hexdigest()[:8] + ".npz"


def audit_evaluation(path, manifest_path, manifest, protocol):
    path = path.resolve(strict=True)
    data = json.loads(path.read_text())
    model = data["model_config"]["model"]
    if data["manifest_sha256"] != sha256(manifest_path):
        raise ValueError("Evaluation uses a different manifest.")
    if data["checkpoint_sha256"] != protocol["models"][model]["checkpoint_sha256"]:
        raise ValueError("Evaluation uses a different selected checkpoint.")
    if data.get("development_only") or data["split"] not in ("validation", "test"):
        raise ValueError("Only frozen complete validation/test evidence is supported.")
    records = {(r["patient"], r["visit"]): r for r in manifest["records"] if r["split"] == data["split"]}
    cases = {(c["patient"], c["visit"]): c for c in data["cases"]}
    if len(cases) != len(data["cases"]) or set(cases) != set(records):
        raise ValueError("Evaluation case keys do not cover the exact split once each.")
    class_count = manifest["num_classes"]
    rows, patients = [], defaultdict(list)
    for key, case in sorted(cases.items()):
        prediction_path = relative_file(path.parent, case["prediction_file"])
        if prediction_path.name != expected_filename(*key):
            raise ValueError("Prediction filename does not match its case key.")
        with np.load(prediction_path, allow_pickle=False) as saved:
            prediction = saved["prediction"]
            if prediction.shape != tuple(manifest["target_shape_dhw"]) or prediction.dtype.kind not in "ui":
                raise ValueError("Invalid prediction grid or noninteger labels.")
            if int(prediction.min()) < 0 or int(prediction.max()) >= class_count:
                raise ValueError("Prediction contains undeclared class IDs.")
            if not np.array_equal(saved["label_ids"], manifest["label_ids"]):
                raise ValueError("Prediction label IDs differ from manifest.")
            if not np.allclose(saved["spacing_dhw"], records[key]["spacing_dhw_mm"], rtol=0, atol=1e-7):
                raise ValueError("Prediction physical spacing differs from manifest.")
            counts = np.bincount(prediction.ravel(), minlength=class_count).tolist()
        per_class = case["per_class"]
        if [row["class_index"] for row in per_class] != list(range(class_count)):
            raise ValueError("Case class rows must be in complete class-index order.")
        independent = []
        for index, row in enumerate(per_class):
            target, predicted = records[key]["class_voxel_counts"][index], counts[index]
            intersection = row["intersection"]
            if type(intersection) is not int or not 0 <= intersection <= min(target, predicted):
                raise ValueError("Invalid recorded intersection.")
            if row["target_voxels"] != target or row["predicted_voxels"] != predicted:
                raise ValueError("Saved-array/manifest marginal counts differ from evaluation.")
            union = target + predicted - intersection
            if row["union"] != union:
                raise ValueError("Recorded union is arithmetically inconsistent.")
            dice = 2 * intersection / (target + predicted) if target + predicted else None
            iou = intersection / union if union else None
            same(row["dice"], dice, f"{key}/{index}/Dice")
            same(row["iou"], iou, f"{key}/{index}/IoU")
            independent.append({"class_index": index, "dice": dice, "iou": iou})
        row = {"patient": key[0], "visit": key[1], "prediction_sha256": sha256(prediction_path),
               "predicted_voxel_counts": counts, "per_class": independent,
               "foreground_mean_dice": mean(x["dice"] for x in independent[1:]),
               "foreground_mean_iou": mean(x["iou"] for x in independent[1:])}
        same(case["foreground_mean_dice"], row["foreground_mean_dice"], f"{key}/foreground Dice")
        same(case["foreground_mean_iou"], row["foreground_mean_iou"], f"{key}/foreground IoU")
        rows.append(row)
        patients[key[0]].append(row)
    aggregate = data["aggregate"]
    if aggregate["num_cases"] != len(rows) or aggregate["num_patients"] != len(patients):
        raise ValueError("Aggregate counts differ from case keys.")
    reported_patients = {x["patient"]: x for x in aggregate["patients"]}
    if len(reported_patients) != len(aggregate["patients"]) or set(reported_patients) != set(patients):
        raise ValueError("Aggregate patient rows differ from case keys.")
    for metric in ("dice", "iou"):
        patient_means = []
        for patient, visits in patients.items():
            expected = mean(x[f"foreground_mean_{metric}"] for x in visits)
            same(reported_patients[patient][f"foreground_mean_{metric}"], expected, f"{patient}/foreground {metric}")
            patient_means.append(expected)
        same(aggregate[f"foreground_mean_{metric}"], mean(patient_means), f"aggregate/foreground {metric}")
        for index in range(class_count):
            per_patient = []
            for patient, visits in patients.items():
                value = mean(x["per_class"][index][metric] for x in visits)
                same(reported_patients[patient]["per_class"][index][metric], value, f"{patient}/{index}/{metric}")
                per_patient.append(value)
            same(aggregate["per_class"][index][metric], mean(per_patient), f"aggregate/{index}/{metric}")
    return {"model": model, "split": data["split"], "evaluation_sha256": sha256(path),
            "cases_checked": len(rows), "patients_checked": len(patients), "status": "passed", "cases": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; preserve prior diagnostic evidence.")
    manifest, protocol = (json.loads(p.read_text()) for p in (args.manifest, args.protocol))
    if sha256(args.manifest) != protocol["manifest_sha256"]:
        raise ValueError("Protocol manifest identity differs.")
    results = [audit_evaluation(p, args.manifest, manifest, protocol) for p in args.evaluation]
    report = {"status": "passed", "diagnostic_only": True,
              "manifest_sha256": sha256(args.manifest), "protocol_sha256": sha256(args.protocol),
              "diagnostic_source_sha256": sha256(__file__), "evaluations": results,
              "limits": ["GT arrays are not loaded. Intersections are checked algebraically, not independently remeasured.",
                         "Matching counts/identities do not prove anatomical alignment, annotation correctness or absence of model bugs.",
                         "No training, inference, model selection, prediction modification or new test claim occurs."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "cases_checked": sum(r["cases_checked"] for r in results),
                      "evaluations_checked": len(results), "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
