"""Exploratory CPU-only prostate localization descriptors; JSON output only.

No training, model selection, prediction changes or causal diagnosis occurs.
Training GT describes a reference distribution. Test GT and fixed predictions
are examined retrospectively without changing the frozen evaluation protocol.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_pair


PROSTATE = 5
AXES = ("d", "h", "w")


def describe_mask(mask, spacing):
    points = np.argwhere(mask == PROSTATE)
    count = len(points)
    volume_mm3 = float(count * np.prod(spacing))
    if count:
        centroid = points.mean(axis=0)
        lower, upper = points.min(axis=0), points.max(axis=0) + 1
        bbox = [[int(lo), int(hi)] for lo, hi in zip(lower, upper)]
        normalized = (centroid / np.maximum(np.asarray(mask.shape) - 1, 1)).tolist()
    else:
        centroid, normalized, bbox = None, None, None
    return {
        "volume_voxels": count, "volume_mm3": volume_mm3, "volume_ml": volume_mm3 / 1000,
        "centroid_dhw": centroid.tolist() if centroid is not None else None,
        "normalized_centroid_dhw": normalized, "bbox_dhw_half_open": bbox,
        "minimum_d": bbox[0][0] if bbox else None,
        "maximum_d_inclusive": bbox[0][1] - 1 if bbox else None,
    }


def features(description):
    result = {key: description[key] for key in ("volume_voxels", "volume_ml", "minimum_d", "maximum_d_inclusive")}
    for index, axis in enumerate(AXES):
        result[f"centroid_{axis}"] = description["centroid_dhw"][index] if description["centroid_dhw"] is not None else None
        result[f"normalized_centroid_{axis}"] = description["normalized_centroid_dhw"][index] if description["normalized_centroid_dhw"] is not None else None
        result[f"bbox_extent_{axis}_voxels"] = (
            description["bbox_dhw_half_open"][index][1] - description["bbox_dhw_half_open"][index][0]
            if description["bbox_dhw_half_open"] is not None else None)
    return result


def feature_means(rows):
    return {key: analyze_pair.mean_defined(row[key] for row in rows) for key in rows[0]} if rows else {}


def distributions(rows):
    result = {}
    for key in rows[0] if rows else ():
        values = np.asarray([row[key] for row in rows if row[key] is not None], dtype=float)
        result[key] = {"n_defined": len(values), **dict(zip(
            ("minimum", "p05", "p25", "median", "p75", "p95", "maximum"),
            np.quantile(values, [0, .05, .25, .5, .75, .95, 1]).tolist() if len(values) else [None] * 7))}
    return result


def compare_features(row, reference):
    comparisons = {}
    for key, value in row.items():
        values = np.asarray([item[key] for item in reference if item[key] is not None], dtype=float)
        if value is None or not len(values):
            comparisons[key] = {"value": value, "reference_n": len(values),
                                "empirical_percentile_leq": None, "outside_minimum_maximum": None,
                                "outside_p05_p95": None}
        else:
            p05, p95 = np.quantile(values, [.05, .95])
            comparisons[key] = {"value": value, "reference_n": len(values),
                                "empirical_percentile_leq": float(100 * np.mean(values <= value)),
                                "outside_minimum_maximum": bool(value < values.min() or value > values.max()),
                                "outside_p05_p95": bool(value < p05 or value > p95)}
    return comparisons


def model_description(prediction, target, gt, spacing, evaluation_case, num_classes):
    summary = describe_mask(prediction, spacing)
    if summary["centroid_dhw"] is not None and gt["centroid_dhw"] is not None:
        offset = (np.asarray(summary["centroid_dhw"]) - np.asarray(gt["centroid_dhw"])) * spacing
        distance = float(np.linalg.norm(offset))
        offset = offset.tolist()
    else:
        distance, offset = None, None
    confusion = analyze_pair.confusion_counts(prediction, target, PROSTATE, num_classes)
    return {
        "prostate": summary, "features": features(summary),
        "centroid_distance_to_gt_mm": distance, "centroid_offset_from_gt_dhw_mm": offset,
        "volume_ratio_to_gt": summary["volume_voxels"] / gt["volume_voxels"] if gt["volume_voxels"] else None,
        "existing_evaluation_dice": next(item["dice"] for item in evaluation_case["per_class"] if item["class_index"] == PROSTATE),
        "false_positive_voxels": sum(confusion["target_fp_by_ground_truth_class"].values()),
        "false_negative_voxels": sum(confusion["target_fn_by_predicted_class"].values()),
        **confusion,
    }


def test_patient_means(cases):
    grouped = defaultdict(list)
    for case in cases:
        grouped[case["patient"]].append(case)
    patients = []
    for patient, visits in sorted(grouped.items()):
        row = {"patient": patient, "num_visits": len(visits),
               "mean_gt_features": feature_means([visit["gt_features"] for visit in visits]),
               "gt_free_intermodel": {
                   "mean_prostate_dice": analyze_pair.mean_defined(visit["gt_free_intermodel"]["prostate_dice"] for visit in visits),
                   "mean_centroid_distance_mm": analyze_pair.mean_defined(visit["gt_free_intermodel"]["centroid_distance_mm"] for visit in visits),
                   "defined_dice_cases": sum(visit["gt_free_intermodel"]["prostate_dice"] is not None for visit in visits),
                   "defined_centroid_distance_cases": sum(visit["gt_free_intermodel"]["centroid_distance_mm"] is not None for visit in visits),
               }}
        for model in ("baseline", "model"):
            row[model] = {
                "mean_features": feature_means([visit[model]["features"] for visit in visits]),
                "mean_centroid_distance_to_gt_mm": analyze_pair.mean_defined(visit[model]["centroid_distance_to_gt_mm"] for visit in visits),
                "defined_centroid_distances": sum(visit[model]["centroid_distance_to_gt_mm"] is not None for visit in visits),
                **{f"mean_{key}": analyze_pair.mean_defined(visit[model][key] for visit in visits)
                   for key in ("volume_ratio_to_gt", "existing_evaluation_dice", "false_positive_voxels", "false_negative_voxels")},
                **{field: {label: analyze_pair.mean_defined(visit[model][field][label] for visit in visits)
                           for label in visits[0][model][field]}
                   for field in ("target_fn_by_predicted_class", "target_fp_by_ground_truth_class")},
            }
        patients.append(row)
    return patients


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "baseline-eval", "model-eval", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    root, manifest, test_records, evaluations = analyze_pair.load_evidence(
        args.manifest, args.baseline_eval, args.model_eval)
    if evaluations[0][1]["split"] != "test":
        raise ValueError("This retrospective diagnostic requires existing, explicitly authorized final-test evaluations.")
    if manifest["label_ids"] != list(range(6)) or manifest["num_classes"] != 6:
        raise ValueError("Expected documented HipMRI class IDs 0..5 with prostate=5.")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise ValueError("Choose a new diagnostic JSON output; existing evidence is not overwritten.")
    train = []
    for record in sorted((r for r in manifest["records"] if r["split"] == "train"),
                         key=lambda r: (r["patient"], r["visit"])):
        with np.load(analyze_pair._relative_file(root, record["cache_path"]), allow_pickle=False) as cache:
            target, spacing = cache["mask"], np.asarray(cache["spacing"], dtype=float)
        if (target.shape != tuple(manifest["target_shape_dhw"]) or not np.isfinite(target).all()
                or np.any(target != np.rint(target)) or np.any(target < 0) or np.any(target >= manifest["num_classes"])
                or spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0)):
            raise ValueError("Invalid training mask grid, class IDs, or spacing.")
        gt = describe_mask(target, spacing)
        train.append({"patient": record["patient"], "visit": record["visit"],
                      "spacing_dhw_mm": spacing.tolist(), "prostate": gt, "features": features(gt)})
    if not train:
        raise ValueError("A nonempty training reference distribution is required.")
    train_features = [case["features"] for case in train]
    grouped_train = defaultdict(list)
    for case in train:
        grouped_train[case["patient"]].append(case["features"])
    train_patients = [{"patient": patient, "num_visits": len(values), "mean_gt_features": feature_means(values)}
                      for patient, values in sorted(grouped_train.items())]
    train_patient_features = [patient["mean_gt_features"] for patient in train_patients]
    cases = []
    for key, record in sorted(test_records.items()):
        _, target, spacing, predictions = analyze_pair._read_case(root, record, evaluations, key, manifest)
        spacing = np.asarray(spacing, dtype=float)
        if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
            raise ValueError("Invalid test spacing.")
        gt = describe_mask(target, spacing)
        row = {"patient": key[0], "visit": key[1], "spacing_dhw_mm": spacing.tolist(),
               "gt_prostate": gt, "gt_features": features(gt)}
        row["gt_vs_training_visits"] = compare_features(row["gt_features"], train_features)
        for model, prediction, evaluation in zip(("baseline", "model"), predictions, evaluations):
            row[model] = model_description(prediction, target, gt, spacing, evaluation[2][key], manifest["num_classes"])
        centroids = [row[name]["prostate"]["centroid_dhw"] for name in ("baseline", "model")]
        distance = float(np.linalg.norm((np.asarray(centroids[1]) - np.asarray(centroids[0])) * spacing)) if all(c is not None for c in centroids) else None
        agreement = analyze_pair.overlap_metrics(predictions[0] == PROSTATE, predictions[1] == PROSTATE)
        row["gt_free_intermodel"] = {
            "uses_ground_truth": False, "prostate_dice": agreement["dice"],
            "centroid_distance_mm": distance,
            "both_predictions_empty": all(c is None for c in centroids),
            "one_prediction_empty": sum(c is None for c in centroids) == 1,
        }
        cases.append(row)
    patients = test_patient_means(cases)
    focus = next((patient for patient in patients if patient["patient"] == "M015"), None)
    focus_comparison = {"patient": "M015", "present_in_test": focus is not None}
    if focus is not None:
        focus_comparison.update({"patient_means": focus,
                                 "mean_gt_vs_training_patient_means": compare_features(focus["mean_gt_features"], train_patient_features),
                                 "visit_keys": [[case["patient"], case["visit"]] for case in cases if case["patient"] == "M015"]})
    report = {
        "schema_version": 1, "diagnostic_only": True, "computed_on_cpu": True,
        "split": "test", "target_class": PROSTATE, "target_name": "Prostate",
        "manifest_sha256": hashlib.sha256(args.manifest.expanduser().read_bytes()).hexdigest(),
        "evaluation_sha256": {name: hashlib.sha256(Path(path).expanduser().read_bytes()).hexdigest()
                              for name, path in (("baseline", args.baseline_eval), ("model", args.model_eval))},
        "coordinate_protocol": "Cached D,H,W voxel indices; D increases superiorly after canonical-RAS preprocessing. Normalized centroids divide each axis by max(grid_size-1,1). Bounding boxes are half-open; maximum_d_inclusive is inclusive.",
        "distance_protocol": "Euclidean distance between predicted and GT centroids after elementwise multiplication by cached D,H,W mm spacing; undefined for an empty mask. This describes the shared resampled grid.",
        "quantile_protocol": "Linear empirical quantiles; empirical_percentile_leq=100*fraction(reference values <= observed value). Visit distributions contain repeated patients; patient-mean distributions weight each training patient once.",
        "train": {"num_cases": len(train), "num_patients": len(train_patients), "cases": train,
                  "visit_distribution": distributions(train_features), "patient_means": train_patients,
                  "patient_mean_distribution": distributions(train_patient_features)},
        "test": {"num_cases": len(cases), "num_patients": len(patients), "cases": cases, "patient_means": patients},
        "focus_M015": focus_comparison,
        "interpretation_limits": [
            "Descriptive localization and size evidence only; no causal or clinical diagnosis is made.",
            "Different grid positions alone do not establish a field-of-view or acquisition-domain-shift cause.",
            "Test comparisons are retrospective and do not alter predictions, selected checkpoints, or scoring protocols.",
            "Empty-mask centroid distances are undefined, never replaced by zero; defined counts accompany patient means.",
            "Intermodel Dice and centroid distance use predictions alone. They are exploratory disagreement descriptors, not validated error detectors; no test-tuned QC threshold is proposed.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
