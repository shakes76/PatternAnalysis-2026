"""Audit paired 2D/3D evaluation evidence on the shared downsampled grid only.

No training or model selection occurs here. HD95 uses pooled symmetric distances
between six-connected voxel surfaces, in resampled-grid millimetres. Empty masks
have undefined surface distances. Class meanings remain unverified by default.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure


REGIONS = ("inferior", "mid", "superior")


def overlap_metrics(prediction, target):
    prediction, target = np.asarray(prediction, bool), np.asarray(target, bool)
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target shapes differ.")
    tp = int(np.count_nonzero(prediction & target))
    fp = int(np.count_nonzero(prediction & ~target))
    fn = int(np.count_nonzero(~prediction & target))
    return {"dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "iou": tp / (tp + fp + fn) if tp + fp + fn else None,
            "tp": tp, "fp": fp, "fn": fn}


def binary_metrics(prediction, target, spacing):
    prediction, target = np.asarray(prediction, bool), np.asarray(target, bool)
    spacing = np.asarray(spacing, dtype=float)
    if target.ndim != 3 or spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("Expected a 3D mask and finite positive D,H,W spacing.")
    result = overlap_metrics(prediction, target)
    has_prediction, has_target = bool(prediction.any()), bool(target.any())
    status = "both_present" if has_prediction and has_target else "missed" if has_target else "fp_only" if has_prediction else "both_empty"
    result.update(hd95_mm=None, empty_status=status)
    if has_prediction and has_target:
        structure = generate_binary_structure(3, 1)
        a = prediction & ~binary_erosion(prediction, structure=structure, border_value=0)
        b = target & ~binary_erosion(target, structure=structure, border_value=0)
        distances = np.concatenate((distance_transform_edt(~b, sampling=spacing)[a],
                                    distance_transform_edt(~a, sampling=spacing)[b]))
        result["hd95_mm"] = float(np.percentile(distances, 95))
    return result


def region_metrics(prediction, target):
    """Partition the complete GT-positive D extent into inferior/mid/superior thirds."""
    prediction, target = np.asarray(prediction, bool), np.asarray(target, bool)
    if prediction.shape != target.shape or target.ndim != 3:
        raise ValueError("Expected matching D,H,W binary masks.")
    positive_depths = np.flatnonzero(target.any(axis=(1, 2)))
    extent = np.arange(positive_depths[0], positive_depths[-1] + 1) if positive_depths.size else np.array([], dtype=int)
    groups = np.array_split(extent, 3)
    regions = {name: {**overlap_metrics(prediction[depths], target[depths]), "depth_indices": depths.tolist()}
               for name, depths in zip(REGIONS, groups)}
    outside = np.ones(target.shape[0], dtype=bool)
    outside[extent] = False
    return {"regions": regions, "outside_gt_extent_fp_voxels": int(np.count_nonzero(prediction[outside])),
            "gt_extent_d": [int(extent[0]), int(extent[-1])] if extent.size else None}


def confusion_counts(prediction, target, target_class, num_classes):
    return {
        "target_fn_by_predicted_class": {str(c): int(np.count_nonzero((target == target_class) & (prediction == c)))
                                        for c in range(num_classes) if c != target_class},
        "target_fp_by_ground_truth_class": {str(c): int(np.count_nonzero((prediction == target_class) & (target == c)))
                                          for c in range(num_classes) if c != target_class},
    }


def mean_defined(values):
    values = [float(value) for value in values if value is not None]
    return float(np.mean(values)) if values else None


def patient_summary(cases, model):
    grouped = defaultdict(list)
    for case in cases:
        grouped[case["patient"]].append(case[model])
    patients = []
    for patient, visits in sorted(grouped.items()):
        row = {"patient": patient, "num_visits": len(visits),
               **{metric: mean_defined(v[metric] for v in visits) for metric in ("dice", "iou", "hd95_mm", "outside_gt_extent_fp_voxels")}}
        row["regions"] = {name: mean_defined(v["regions"][name]["dice"] for v in visits) for name in REGIONS}
        for field in ("target_fn_by_predicted_class", "target_fp_by_ground_truth_class"):
            row[field] = {label: mean_defined(v[field][label] for v in visits) for label in visits[0][field]}
        patients.append(row)
    return {"patients": patients,
            **{metric: mean_defined(p[metric] for p in patients) for metric in ("dice", "iou", "hd95_mm", "outside_gt_extent_fp_voxels")},
            "region_dice": {name: mean_defined(p["regions"][name] for p in patients) for name in REGIONS},
            **{field: {label: mean_defined(p[field][label] for p in patients) for label in patients[0][field]}
               for field in ("target_fn_by_predicted_class", "target_fp_by_ground_truth_class")},
            "empty_status_counts": {status: sum(c[model]["empty_status"] == status for c in cases)
                                    for status in ("both_present", "missed", "fp_only", "both_empty")},
            "hd95_support": {
                "defined_case_count": sum(c[model]["hd95_mm"] is not None for c in cases),
                "defined_patient_count": sum(p["hd95_mm"] is not None for p in patients),
                "interpretation": "Per-model diagnostic mean conditional on nonempty prediction and GT. Support may differ between models; use paired_hd95 for matched-support comparison.",
            }}


def paired_hd95_summary(cases):
    """Compare HD95 on common both-present visits, then weight patients equally.

    Eligibility depends on nonempty masks, never overlap: disjoint predictions
    with Dice zero remain eligible. Excluded-status counts are per model over
    the excluded case pairs; a both_present count means the other model caused
    that pair's exclusion. Missed and FP-only predictions remain visible in the
    diagnostic counts rather than being assigned an artificial surface error.
    """
    grouped = defaultdict(list)
    excluded = []
    for case in cases:
        baseline, model = case["baseline"], case["model"]
        if baseline["empty_status"] != "both_present" or model["empty_status"] != "both_present":
            excluded.append(case)
            continue
        values = (baseline["hd95_mm"], model["hd95_mm"])
        if any(value is None or not np.isfinite(value) or value < 0 for value in values):
            raise ValueError("Both-present case pairs require finite nonnegative HD95 values.")
        grouped[case["patient"]].append(values)
    patients = []
    for patient, values in sorted(grouped.items()):
        values = np.asarray(values, dtype=float)
        baseline = float(values[:, 0].mean())
        model = float(values[:, 1].mean())
        patients.append({"patient": patient, "num_paired_visits": len(values),
                         "baseline_hd95_mm": baseline, "model_hd95_mm": model,
                         "delta_hd95_mm": model - baseline})
    return {
        **{metric: mean_defined(patient[metric] for patient in patients)
           for metric in ("baseline_hd95_mm", "model_hd95_mm", "delta_hd95_mm")},
        "eligible_case_pairs": sum(patient["num_paired_visits"] for patient in patients),
        "excluded_case_pairs": len(excluded),
        "num_patients": len(patients), "patients": patients,
        "excluded_status_counts_by_model": {
            name: {status: sum(case[name]["empty_status"] == status for case in excluded)
                   for status in ("both_present", "missed", "fp_only", "both_empty")}
            for name in ("baseline", "model")
        },
        "protocol": "Same visits where both model predictions and GT are present; mean common visits within each patient, then equal mean patients. Zero Dice is retained. Delta is model minus baseline in resampled-grid mm; negative favours the model. Empty-mask exclusions are reported separately.",
    }


def paired_bootstrap(cases, samples=2000, seed=3710):
    """Average paired eligible visits within patient, then resample patient pairs."""
    if type(samples) is not int or samples < 1:
        raise ValueError("Bootstrap samples must be a positive integer.")
    grouped = defaultdict(list)
    for case in cases:
        a, b = case["baseline"]["dice"], case["model"]["dice"]
        if a is not None and b is not None:
            grouped[case["patient"]].append((a, b))
    patients = [{"patient": patient, "num_paired_visits": len(values),
                 "baseline": float(np.mean(np.asarray(values)[:, 0])),
                 "model": float(np.mean(np.asarray(values)[:, 1]))} for patient, values in sorted(grouped.items())]
    deltas = np.asarray([p["model"] - p["baseline"] for p in patients], dtype=float)
    bootstrap = np.random.default_rng(seed).choice(deltas, size=(samples, len(deltas)), replace=True).mean(axis=1) if len(deltas) else None
    return {"delta_dice": float(deltas.mean()) if len(deltas) else None,
            "ci95": np.percentile(bootstrap, [2.5, 97.5]).tolist() if bootstrap is not None else None,
            "patients": patients, "num_patients": len(patients), "samples": samples, "seed": seed,
            "excluded_undefined_case_pairs": len(cases) - sum(p["num_paired_visits"] for p in patients),
            "protocol": "Same eligible visits for both models; mean visits per patient, paired patient bootstrap. Only undefined Dice excluded; all zero Dice retained."}


def _relative_file(root, value):
    relative = Path(value)
    resolved = (root / relative).resolve()
    if relative.is_absolute() or ".." in relative.parts or not resolved.is_relative_to(root):
        raise ValueError("Evidence paths must remain relative to their containing directory.")
    return resolved


def _case_map(records):
    result = {}
    for record in records:
        key = (record["patient"], record["visit"])
        if key in result:
            raise ValueError("Duplicate patient/visit in evidence.")
        result[key] = record
    return result


def load_evidence(manifest_path, baseline_path, model_path):
    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("partial") or manifest.get("development_only"):
        raise ValueError("Development manifests cannot produce formal paired evidence.")
    splits = manifest["patient_splits"]
    if set(splits) != {"train", "validation", "test"}:
        raise ValueError("Invalid manifest patient partitions.")
    flat = [p for values in splits.values() for p in values]
    if any(not values for values in splits.values()) or len(flat) != len(set(flat)):
        raise ValueError("Overlapping, duplicated or empty manifest patient partitions.")
    for record in manifest["records"]:
        if record["split"] not in splits or record["patient"] not in splits[record["split"]]:
            raise ValueError("Manifest record disagrees with its patient partition.")
    _case_map(manifest["records"])
    evaluations = []
    for path, expected_model in ((baseline_path, "unet2d"), (model_path, "improved3d")):
        path = Path(path).expanduser().resolve()
        evaluation = json.loads(path.read_text())
        config = evaluation["model_config"]
        if evaluation.get("development_only") is not False or config.get("development_only"):
            raise ValueError("Development evaluations cannot produce formal paired evidence.")
        if evaluation.get("manifest_sha256") != hashlib.sha256(manifest_bytes).hexdigest():
            raise ValueError("Evaluation manifest hash mismatch.")
        if config["model"] != expected_model or config["label_ids"] != manifest["label_ids"] or config["num_classes"] != manifest["num_classes"]:
            raise ValueError("Evaluation architecture or class mapping mismatch.")
        if config["target_shape_dhw"] != manifest["target_shape_dhw"] or evaluation["split"] not in ("validation", "test"):
            raise ValueError("Evaluation grid or split mismatch.")
        if evaluation["split"] == "test" and evaluation.get("final_test_explicitly_requested") is not True:
            raise ValueError("Test evidence lacks explicit final-test authorization.")
        evaluations.append((path.parent, evaluation, _case_map(evaluation["cases"])))
    if evaluations[0][1]["split"] != evaluations[1][1]["split"]:
        raise ValueError("The evaluations use different splits.")
    records = _case_map([r for r in manifest["records"] if r["split"] == evaluations[0][1]["split"]])
    if not records or any(set(item[2]) != set(records) for item in evaluations):
        raise ValueError("Evaluation case keys must match the complete manifest split.")
    return manifest_path.parent, manifest, records, evaluations


def _read_case(root, record, evaluations, key, manifest):
    with np.load(_relative_file(root, record["cache_path"]), allow_pickle=False) as cache:
        image, target, spacing = cache["image"], cache["mask"], cache["spacing"]
    predictions = []
    for directory, _, cases in evaluations:
        with np.load(_relative_file(directory, cases[key]["prediction_file"]), allow_pickle=False) as saved:
            prediction = saved["prediction"]
            if not np.allclose(saved["spacing_dhw"], spacing) or saved["label_ids"].tolist() != manifest["label_ids"]:
                raise ValueError("Prediction spacing or label mapping differs from the cache.")
            predictions.append(prediction)
    for array in (target, *predictions):
        if (array.shape != tuple(manifest["target_shape_dhw"]) or not np.isfinite(array).all()
                or np.any(array != np.rint(array)) or np.any(array < 0) or np.any(array >= manifest["num_classes"])):
            raise ValueError("Invalid class-ID mask or prediction grid.")
    if image.shape != target.shape or not np.isfinite(image).all():
        raise ValueError("Invalid cached image.")
    return image, target, spacing, predictions


def select_failure_cases(cases, limit=5):
    """Choose each patient's worst defined 3D visit, then rank those patients.

    Zero Dice remains a defined failure. Ties use patient/visit identifiers and
    therefore do not depend on input order. Patients with only undefined Dice
    cannot contribute a ranked failure case.
    """
    if type(limit) is not int or limit < 0:
        raise ValueError("Failure-gallery limit must be a nonnegative integer.")
    ranked = sorted((case for case in cases if case["model"]["dice"] is not None),
                    key=lambda case: (case["model"]["dice"], case["patient"], case["visit"]))
    selected, seen = [], set()
    for case in ranked:
        if len(selected) == limit:
            break
        if case["patient"] not in seen:
            selected.append(case)
            seen.add(case["patient"])
    return selected


def select_gallery_slices(target, predictions, target_class):
    """Show GT-third centres plus each model's largest full-depth FP+FN slice.

    The two model-specific maxima can occur outside the GT extent. Their ties
    use the lowest depth index. Repeated depths share one row with every
    selection label preserved; absent GT thirds use explicitly labelled grid
    fallbacks rather than claiming an anatomical region.
    """
    target = np.asarray(target)
    predictions = [np.asarray(prediction) for prediction in predictions]
    if target.ndim != 3 or not target.size or len(predictions) != 2 or any(
            prediction.shape != target.shape for prediction in predictions):
        raise ValueError("Expected a nonempty D,H,W target and two matching predictions.")
    actual = target == target_class
    regions = region_metrics(predictions[1] == target_class, actual)["regions"]
    rows = []

    def add(depth, label):
        depth = int(depth)
        for row in rows:
            if row["depth"] == depth:
                row["labels"].append(label)
                return
        rows.append({"depth": depth, "labels": [label]})

    for index, name in enumerate(REGIONS):
        depths = regions[name]["depth_indices"]
        if depths:
            add(depths[len(depths) // 2], name + " GT third")
        else:
            add(round(index * (target.shape[0] - 1) / 2), name + " (no GT third; grid fallback)")
    for prediction, model_name in zip(predictions, ("2D", "3D")):
        errors = np.logical_xor(prediction == target_class, actual).sum(axis=(1, 2))
        depth = int(errors.argmax())
        add(depth, f"max {model_name} FP+FN ({int(errors[depth])} voxels)")
    return rows


def plot_evidence(output, bootstrap, selected, root, records, evaluations, manifest, target_class):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    fig, axis = plt.subplots(figsize=(5, 5))
    for patient in bootstrap["patients"]:
        axis.plot([0, 1], [patient["baseline"], patient["model"]], "o-", alpha=0.6)
    axis.set(xticks=[0, 1], xticklabels=["2D baseline", "3D model"], ylabel=f"Class {target_class} Dice",
             ylim=(0, 1), title=f"Paired patient means (n={bootstrap['num_patients']})")
    fig.tight_layout(); fig.savefig(output / "paired_patient_dice.png", dpi=150); plt.close(fig)
    figures = ["paired_patient_dice.png"]
    for rank, case in enumerate(selected, 1):
        key = (case["patient"], case["visit"])
        image, target, _, predictions = _read_case(root, records[key], evaluations, key, manifest)
        chosen = select_gallery_slices(target, predictions, target_class)
        fig, axes = plt.subplots(len(chosen), 6, figsize=(16, 2.7 * len(chosen)), squeeze=False)
        for row, selection in enumerate(chosen):
            depth = selection["depth"]
            panels = [image[depth], target[depth], predictions[0][depth], predictions[1][depth]]
            for prediction in predictions:
                panels.append(np.where((prediction[depth] == target_class) & (target[depth] != target_class), 1,
                                       np.where((prediction[depth] != target_class) & (target[depth] == target_class), 2, 0)))
            for column, (panel, title) in enumerate(zip(panels, ("Input", "GT IDs", "2D IDs", "3D IDs", "2D FP/FN", "3D FP/FN"))):
                axis = axes[row, column]
                axis.imshow(panel, cmap="gray" if column == 0 else "tab20" if column < 4 else ListedColormap(["black", "red", "cyan"]),
                            vmin=None if column == 0 else 0, vmax=None if column == 0 else manifest["num_classes"] - 1 if column < 4 else 2,
                            interpolation="nearest")
                axis.set_title(f"{title} | D={depth}", fontsize=8); axis.axis("off")
            # axis('off') suppresses normal axis labels, so use explicit text.
            axes[row, 0].text(0, -0.08, " / ".join(selection["labels"]),
                              transform=axes[row, 0].transAxes, fontsize=7, va="top")
        fig.suptitle(f"{key[0]} / {key[1]} | class {target_class} | resampled grid | red FP / cyan FN")
        fig.tight_layout(); name = f"worst_3d_{rank:02d}.png"; fig.savefig(output / name, dpi=150); plt.close(fig)
        figures.append(name)
    return figures


def analyze(manifest_path, baseline_path, model_path, output_dir, target_class=5, bootstrap_samples=2000, seed=3710):
    root, manifest, records, evaluations = load_evidence(manifest_path, baseline_path, model_path)
    if not 1 <= target_class < manifest["num_classes"]:
        raise ValueError("Target class must be a declared foreground index.")
    output = Path(output_dir).expanduser().resolve(); output.mkdir(parents=True, exist_ok=True)
    if (output / "pair_analysis.json").exists():
        raise ValueError("Choose a fresh analysis output directory.")
    cases = []
    for key, record in sorted(records.items()):
        _, target, spacing, predictions = _read_case(root, record, evaluations, key, manifest)
        case = {"patient": key[0], "visit": key[1]}
        for name, prediction in zip(("baseline", "model"), predictions):
            case[name] = {**binary_metrics(prediction == target_class, target == target_class, spacing),
                          **region_metrics(prediction == target_class, target == target_class),
                          **confusion_counts(prediction, target, target_class, manifest["num_classes"])}
        cases.append(case)
    bootstrap = paired_bootstrap(cases, bootstrap_samples, seed)
    selected = select_failure_cases(cases)
    report = {"schema_version": 1, "split": evaluations[0][1]["split"], "target_class": target_class,
              "class_name": manifest.get("label_semantics", {}).get("class_index_to_name", {}).get(str(target_class), f"class {target_class}"), "label_semantics": manifest.get("label_semantics", {"status": "unverified"}),
              "manifest_sha256": hashlib.sha256(Path(manifest_path).expanduser().read_bytes()).hexdigest(),
              "geometry": "Canonical RAS: D increases superiorly. HD95 is in downsampled/resampled-grid mm, not native clinical accuracy. GT-extent thirds are approximate inferior/mid/superior regions, not independently annotated anatomical apex/base landmarks.",
              "hd95_protocol": "95th percentile of pooled bidirectional nearest six-connected voxel-surface distances; empty masks yield null and are counted separately.",
              "aggregation_protocol": "Mean visits per patient, then equal mean patients. Confusion and outer-FP summaries are patient-balanced mean voxel counts; case counts remain raw. Undefined values alone are excluded.",
              "cases": cases, "baseline": patient_summary(cases, "baseline"), "model": patient_summary(cases, "model"),
              "paired_bootstrap": bootstrap, "paired_hd95": paired_hd95_summary(cases),
              "failure_selection": "Each patient's worst defined 3D target-Dice visit, then up to five distinct patients ranked by that Dice, including zero; deterministic patient/visit ties. Fewer than three eligible patients cannot support a three-patient gallery.",
              "figure_selection": "For each selected visit: approximate inferior/mid/superior GT-extent third centres plus each model's maximum target FP+FN slice across the entire depth, including outside GT extent. Error-maximum ties use the lowest depth; duplicate depths share a row with all selection labels. Missing GT thirds use explicitly labelled grid fallbacks.",
              "selected_failure_keys": [[c["patient"], c["visit"]] for c in selected],
              "resources": {name: {field: item[1].get(field) for field in ("parameters", "mean_inference_ms", "peak_allocated_cuda_bytes")}
                            for name, item in zip(("baseline", "model"), evaluations)},
              "manager_evidence": "Compare patient-level Dice CI, matched-support paired HD95, regional Dice, missed/FP-only cases, documented label semantics and resource profiles; review failure panels before making operational recommendations. Per-model HD95 support may differ.",
              "figures": []}
    destination = output / "pair_analysis.json"
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    report["figures"] = plot_evidence(output, bootstrap, selected, root, records, evaluations, manifest, target_class)
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("manifest", "baseline-eval", "model-eval", "output-dir"):
        parser.add_argument("--" + argument, type=Path, required=True)
    parser.add_argument("--target-class", type=int, default=5)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=3710)
    args = parser.parse_args(argv)
    report = analyze(args.manifest, args.baseline_eval, args.model_eval, args.output_dir, args.target_class, args.bootstrap_samples, args.seed)
    print(f"Paired {report['split']} evidence: {args.output_dir / 'pair_analysis.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
