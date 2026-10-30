"""COMP3710 ADNI: reproducible slice/scan/patient inference and evaluation.

Use *validation* to verify the locked protocol before running the official test.
The test split is guarded by --confirm-final-test to discourage tuning on test.
No training or model selection occurs in this script.

Examples (by default the final 3-seed ConvNeXt ensemble is loaded and the
AD threshold is read from its best_meta.json, i.e. fixed on validation):
    python -u predict.py --split validation --num-workers 4
    python -u predict.py --split test --confirm-final-test --num-workers 4
    python -u predict.py --model cnn --split validation \
        --checkpoint checkpoints/cnn/cnn_official_cosine_ema_s42/best.pth

AD=0 (positive class), NC=1. Scan predictions average 20 slice AD
probabilities; patient predictions average the scores of that patient's scans.
Patient-level accuracy is reported only for patients with consistent labels
across visits. Report any excluded patients rather than hiding them.

Warning: Test labels and MRI slices are course/restricted data. Keep outputs
local and check ADNI data-use rules before publishing derived images/details.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from dataset import (get_image_id, load_datasets, load_patient_map, get_transform,
                     SPLIT_MODES, INPUT_MODES, DATA_ROOT)
from modules import ConvNeXt
from baseline import build_baseline, MODEL_NAMES


# Final model: 3-seed ensemble of the official-split, context3 ConvNeXt runs.
DEFAULT_CHECKPOINTS = [
    Path(f"checkpoints/convnext/official_cosine_ema_context3_s{seed}/best.pth")
    for seed in (42, 43, 44)
]


def fingerprint_split(dataset, root_alias=None):
    """Identical algorithm to train.py; detect changed membership/order/labels.

    The hash covers absolute file paths. root_alias replaces the current data
    root (dataset.DATA_ROOT) by the root used on the training machine, so a
    checkpoint trained on another machine with an identical copy of the data
    can be verified exactly (same files, order and labels below the root).
    """
    samples = get_sample_info(dataset)
    root = str(DATA_ROOT)
    digest = hashlib.sha256()
    for path, label in samples:
        path = str(path)
        if root_alias is not None and path.startswith(root):
            path = root_alias + path[len(root):]
        digest.update(f"{path}\t{label}\n".encode("utf-8"))
    return digest.hexdigest()


def fingerprint_root_aliases():
    """Data roots of other training machines, from ADNI_FINGERPRINT_ROOT_ALIASES
    (comma-separated), e.g. the GPU Lab copy of the same corrected dataset."""
    value = os.environ.get("ADNI_FINGERPRINT_ROOT_ALIASES", "")
    return [alias.strip().rstrip("/") for alias in value.split(",") if alias.strip()]


def patient_mapping_fingerprint(datasets, patient_map):
    """Hash relevant scan-to-patient links, without saving private IDs."""
    scan_ids = sorted({get_image_id(path)
                       for dataset in datasets
                       for path, _ in get_sample_info(dataset)})
    digest = hashlib.sha256()
    for scan_id in scan_ids:
        patient_id = patient_map.get(scan_id)
        if patient_id is None:
            raise ValueError(f"Missing metadata patient for scan {scan_id}")
        digest.update(f"{scan_id}\t{patient_id}\n".encode("utf-8"))
    return digest.hexdigest()


def load_best_metadata(checkpoint):
    """Old raw weight files have no metadata; keep them loadable with a warning."""
    metadata_path = checkpoint.with_name("best_meta.json")
    if not metadata_path.is_file():
        return None
    with metadata_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def check_checkpoint_compatibility(metadata, args, train_ds, val_ds, test_ds, patient_map):
    if metadata is None:
        print("WARNING: Legacy best.pth has no best_meta.json. "
              "Split and preprocessing fingerprint CANNOT be independently verified.")
        return
    if metadata.get("smoke_test") and args.split == "test":
        raise ValueError("Smoke-test weights must not be evaluated on official test")
    if metadata.get("model") != args.model:
        raise ValueError("Model mismatch: checkpoint was trained as "
                         f"{metadata.get('model')!r}, requested {args.model!r}")
    if metadata.get("class_to_idx") != {"AD": 0, "NC": 1}:
        raise ValueError("Class mapping mismatch in checkpoint metadata")
    if metadata.get("preprocessing_repr") != repr(get_transform(args.input_mode)):
        raise ValueError("Preprocessing mismatch between trained model and dataset.py")
    expected = metadata.get("split_fingerprint")
    if not isinstance(expected, dict) or set(expected) != {"train", "validation", "test"}:
        raise ValueError("Incomplete split fingerprint in best_meta.json")
    matched_alias = "no match"
    for alias in [None] + fingerprint_root_aliases():
        actual = {"train": fingerprint_split(train_ds, alias),
                  "validation": fingerprint_split(val_ds, alias),
                  "test": fingerprint_split(test_ds, alias)}
        if all(expected[split] == actual[split] for split in actual):
            matched_alias = alias
            break
    if matched_alias == "no match":
        first = next(split for split in ("train", "validation", "test")
                     if expected[split] != fingerprint_split(
                         {"train": train_ds, "validation": val_ds, "test": test_ds}[split]))
        raise ValueError(f"{first} patient split/membership/order mismatch; "
                         "do not compare these metrics with the checkpoint")
    if matched_alias is not None:
        print(f"NOTE: split fingerprints match after mapping data root {DATA_ROOT} -> "
              f"{matched_alias} (checkpoint trained on another machine with the same data).")
    expected_patient = metadata.get("patient_mapping_fingerprint")
    if not expected_patient:
        raise ValueError("Checkpoint missing patient mapping fingerprint; check metadata version")
    actual_patient = patient_mapping_fingerprint((train_ds, val_ds, test_ds), patient_map)
    if expected_patient != actual_patient:
        raise ValueError("Patient mapping mismatch; metadata identity changed since training")
    print("Checkpoint model, class map, preprocessing, split and patient mapping fingerprints verified.")

def get_sample_info(dataset):
    """Return (path, label) in exactly DataLoader(shuffle=False) order."""
    if isinstance(dataset, Subset):
        source, indices = dataset.dataset, dataset.indices
    else:
        source, indices = dataset, range(len(dataset))
    if getattr(source, "class_to_idx", None) != {"AD": 0, "NC": 1}:
        raise ValueError("Expected ImageFolder mapping {'AD': 0, 'NC': 1}")
    if not hasattr(source, "samples"):
        raise TypeError("Expected ImageFolder or Subset(ImageFolder)")
    return [source.samples[int(i)] for i in indices]


def create_records(samples, scores, patient_map):
    """Aggregate slice AD probabilities into scan and patient records.

    One scan is represented by one image-id prefix; patient identifiers come
    only from dataset.py's metadata mapping, NOT from the image-id prefix.
    Equal weight is given to each scan in a patient's average.
    """
    if len(samples) != len(scores):
        raise ValueError("Sample count does not match prediction count")

    slices = []
    scan_groups = defaultdict(list)
    for (path, label), score in zip(samples, scores):
        if label not in (0, 1) or not (0.0 <= score <= 1.0) or not math.isfinite(score):
            raise ValueError("Invalid label or AD probability")
        scan_id = get_image_id(path)
        patient_id = patient_map.get(scan_id)
        if patient_id is None:
            raise ValueError(f"Missing patient mapping for scan {scan_id}")
        slices.append((str(path), float(score), label))
        scan_groups[scan_id].append((patient_id, float(score), label))

    scans = []
    patient_groups = defaultdict(list)
    unusual_scan_sizes = []
    for scan_id, entries in scan_groups.items():
        patients = {p for p, _, _ in entries}
        diagnoses = {y for _, _, y in entries}
        if len(patients) != 1 or len(diagnoses) != 1:
            raise ValueError(f"Inconsistent labels or patient IDs within scan {scan_id}")
        patient_id = next(iter(patients))
        label = next(iter(diagnoses))
        probability = sum(p for _, p, _ in entries) / len(entries)
        scans.append((scan_id, probability, label))
        patient_groups[patient_id].append((probability, label))
        if len(entries) != 20:
            unusual_scan_sizes.append((scan_id, len(entries)))

    patients = []
    mixed_patients = []
    for patient_id, entries in patient_groups.items():
        diagnoses = {label for _, label in entries}
        if len(diagnoses) != 1:
            mixed_patients.append(patient_id)
            continue
        # Scan-balanced mean: each visit contributes equally, even if its
        # number of slices differs from other visits.
        probability = sum(p for p, _ in entries) / len(entries)
        patients.append((patient_id, probability, next(iter(diagnoses))))

    return {
        "slice": slices,
        "scan": scans,
        "patient": patients,
        "total_patients": len(patient_groups),
        "excluded_mixed_label_patients": len(mixed_patients),
        "unusual_scan_sizes": unusual_scan_sizes,
    }


def rank_auroc(scores, labels):
    """AUROC with AD (label=0) positive, using tied mid-ranks."""
    pos = sum(y == 0 for y in labels)
    neg = sum(y == 1 for y in labels)
    if pos == 0 or neg == 0:
        return None
    ranked = sorted(zip(scores, labels))
    rank_sum = 0.0
    i = 0
    while i < len(ranked):
        j = i + 1
        while j < len(ranked) and ranked[j][0] == ranked[i][0]:
            j += 1
        mid_rank = ((i + 1) + j) / 2.0
        rank_sum += mid_rank * sum(y == 0 for _, y in ranked[i:j])
        i = j
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def evaluate(records, threshold):
    """Classification metrics for AD positive, NC negative."""
    if not records:
        return None
    scores = [score for _, score, _ in records]
    labels = [label for _, _, label in records]
    predictions = [0 if score >= threshold else 1 for score in scores]
    tp = sum(y == 0 and p == 0 for y, p in zip(labels, predictions))
    fn = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))
    fp = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))
    tn = sum(y == 1 and p == 1 for y, p in zip(labels, predictions))

    def div(a, b):
        return a / b if b else 0.0

    precision_ad = div(tp, tp + fp)
    precision_nc = div(tn, tn + fn)
    recall_ad = div(tp, tp + fn)
    recall_nc = div(tn, tn + fp)
    f1_ad = div(2 * precision_ad * recall_ad, precision_ad + recall_ad)
    f1_nc = div(2 * precision_nc * recall_nc, precision_nc + recall_nc)

    # For operational triage, confidence is the probability of the selected
    # class, not a calibrated probability of being correct.
    confidence = [s if p == 0 else (1 - s) for s, p in zip(scores, predictions)]
    review_mask = [c < 0.80 for c in confidence]
    accepted = [p == y for p, y, needs_review in zip(predictions, labels, review_mask)
                if not needs_review]
    accepted_accuracy = sum(accepted) / len(accepted) if accepted else None

    return {
        "n": len(records),
        "accuracy": div(tp + tn, len(records)),
        "correct": tp + tn,
        "precision_ad": precision_ad,
        "recall_ad": recall_ad,
        "f1_ad": f1_ad,
        "precision_nc": precision_nc,
        "recall_nc": recall_nc,
        "f1_nc": f1_nc,
        "macro_precision": (precision_ad + precision_nc) / 2,
        "macro_recall": (recall_ad + recall_nc) / 2,
        "macro_f1": (f1_ad + f1_nc) / 2,
        "auroc_ad": rank_auroc(scores, labels),
        "tp_ad": tp, "fn_ad": fn, "fp_ad": fp, "tn_ad": tn,
        "triage_example_confidence_cutoff": 0.80,
        "triage_example_review_count": sum(review_mask),
        "triage_example_coverage": 1 - sum(review_mask) / len(records),
        "triage_example_accepted_accuracy": accepted_accuracy,
    }


def print_metrics(level, result):
    if result is None:
        print(f"{level}: unavailable")
        return
    auc = "n/a" if result["auroc_ad"] is None else f"{result['auroc_ad']:.4f}"
    print(
        f"{level.upper():7s} {result['correct']}/{result['n']} correct, "
        f"accuracy={result['accuracy']:.4f}, macro-F1={result['macro_f1']:.4f}, "
        f"AD recall={result['recall_ad']:.4f}, NC recall={result['recall_nc']:.4f}, "
        f"AUROC_AD={auc}"
    )
    print(
        f"         AD: TP={result['tp_ad']} FN={result['fn_ad']}; "
        f"NC: FP={result['fp_ad']} TN={result['tn_ad']}; "
        f"P_AD={result['precision_ad']:.4f}, P_NC={result['precision_nc']:.4f}"
    )


def anonymized_id(identifier, salt):
    return hashlib.sha256((salt + ":" + str(identifier)).encode("utf-8")).hexdigest()[:14]


def save_anonymized_details(path, records, threshold, salt):
    # Do not write raw patient IDs, scan IDs, or source paths into a public CSV.
    # Salt is user-provided if distinct files need stable pseudonymous IDs.
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["anonymous_id", "true_class", "predicted_class", "probability_ad", "correct"])
        for identifier, score, label in records:
            predicted = 0 if score >= threshold else 1
            writer.writerow([
                anonymized_id(identifier, salt), "AD" if label == 0 else "NC",
                "AD" if predicted == 0 else "NC", f"{score:.6f}", int(predicted == label)
            ])


def save_example_predictions(slices, threshold, path, per_cell=2):
    """Grid of test slices with true label, prediction and P(AD).

    For each confusion cell (TP, FN, FP, TN) the most confident slices are
    shown, so the figure has both clear successes and the worst mistakes.
    The figure contains MRI pixels: keep it local, do not commit it.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    cells = {("AD", "AD"): [], ("AD", "NC"): [], ("NC", "AD"): [], ("NC", "NC"): []}
    for slice_path, score, label in slices:
        true = "AD" if label == 0 else "NC"
        pred = "AD" if score >= threshold else "NC"
        confidence = score if pred == "AD" else 1 - score
        cells[(true, pred)].append((confidence, slice_path, score))

    fig, axes = plt.subplots(len(cells), per_cell, figsize=(2.4 * per_cell, 2.6 * len(cells)),
                             squeeze=False)
    for row, ((true, pred), entries) in enumerate(cells.items()):
        # Most confident first: for wrong cells these are the worst errors.
        entries.sort(reverse=True)
        for col in range(per_cell):
            ax = axes[row][col]
            ax.axis("off")
            if col >= len(entries):
                continue
            _, slice_path, score = entries[col]
            ax.imshow(Image.open(slice_path).convert("L"), cmap="gray")
            colour = "green" if true == pred else "red"
            ax.set_title(f"true {true} / pred {pred}\nP(AD)={score:.2f}",
                         fontsize=8, color=colour)
    fig.suptitle(f"Example predictions (threshold {threshold:.2f})", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_figures(records, result_by_level, args):
    """Visualise the evaluation: slice-level confusion matrix and example slices."""
    # Lazy import: analyze.py imports this module, so importing it at the top
    # would be circular. plot_confusion is the same function used in the README.
    from analyze import plot_confusion

    slice_result = result_by_level["slice"]
    confusion_path = args.output_dir / f"{args.split}_confusion.png"
    plot_confusion(slice_result["tp_ad"], slice_result["fn_ad"], slice_result["fp_ad"],
                   slice_result["tn_ad"],
                   f"{args.model}, {args.split} slices (acc {slice_result['accuracy']:.1%})",
                   confusion_path)
    print("Saved confusion matrix:", confusion_path)

    examples_path = args.output_dir / f"{args.split}_examples.png"
    save_example_predictions(records["slice"], args.threshold, examples_path)
    print("Saved example predictions (contains MRI pixels, keep local):", examples_path)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", choices=["validation", "test"], default="validation")
    p.add_argument("--checkpoint", type=Path, nargs="+", default=DEFAULT_CHECKPOINTS,
                   help="One best.pth, or several to average their AD probabilities "
                        "(ensemble); all must share split mode, input mode and threshold")
    p.add_argument("--model", choices=list(MODEL_NAMES), default="convnext",
                   help="Architecture to load; explicit for legacy checkpoints")
    p.add_argument("--threshold", type=float, default=None,
                   help="AD cutoff; default uses saved best_meta.json, else 0.50")
    p.add_argument("--split-mode", choices=list(SPLIT_MODES), default=None,
                   help="Must match training; default comes from best_meta.json, else strict")
    p.add_argument("--input-mode", choices=list(INPUT_MODES), default=None,
                   help="Must match training; default comes from best_meta.json, else single")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--output-dir", type=Path, default=Path("evaluation_results"))
    p.add_argument("--confirm-final-test", action="store_true",
                   help="Explicit approval to score the untouched official test split")
    p.add_argument("--save-details", action="store_true",
                   help="Write pseudonymous slice/scan/patient prediction CSVs")
    p.add_argument("--id-salt", default="local-analysis-only",
                   help="Salt for pseudonymous IDs; do not publish detailed CSVs")
    p.add_argument("--no-plots", action="store_true",
                   help="Skip the confusion-matrix and example-slice figures")
    args = p.parse_args()
    if args.threshold is not None and not (0 < args.threshold < 1):
        p.error("Threshold must be strictly between 0 and 1")
    if args.batch_size < 1:
        p.error("Batch size must be >= 1")
    if args.split == "test" and not args.confirm_final_test:
        p.error("Official test evaluation requires --confirm-final-test")
    for checkpoint in args.checkpoint:
        if not checkpoint.is_file():
            p.error(f"Checkpoint not found: {checkpoint}")
    if len(set(args.checkpoint)) != len(args.checkpoint):
        p.error("The same checkpoint was listed twice")
    return args


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Split: {args.split.upper()}, checkpoint(s): "
          + ", ".join(str(c) for c in args.checkpoint))
    all_metadata = [load_best_metadata(c) for c in args.checkpoint]
    metadata = all_metadata[0]
    if len(args.checkpoint) > 1:
        if any(m is None for m in all_metadata):
            raise ValueError("Ensembles require best_meta.json for every checkpoint")
        for key in ("split_mode", "input_mode", "selection_threshold", "model"):
            if len({str(m.get(key)) for m in all_metadata}) != 1:
                raise ValueError(f"Ensemble members disagree on {key}")
    if args.threshold is None:
        args.threshold = (metadata.get("selection_threshold", 0.5)
                          if metadata is not None else 0.5)
        print("Threshold source:", "best_meta.json" if metadata else "legacy fallback 0.50")
    elif metadata is not None and args.threshold != metadata.get("selection_threshold"):
        print("WARNING: Manual threshold override. Document validation-only rationale; "
              "NEVER select a threshold based on test metrics.")
    print(f"Frozen primary AD threshold: {args.threshold:.2f}")
    saved_mode = metadata.get("split_mode", "strict") if metadata is not None else None
    if args.split_mode is None:
        args.split_mode = saved_mode or "strict"
    elif saved_mode is not None and args.split_mode != saved_mode:
        raise ValueError(f"--split-mode {args.split_mode} does not match training split "
                         f"mode {saved_mode} in best_meta.json")
    print("Split mode:", args.split_mode)
    saved_input = metadata.get("input_mode", "single") if metadata is not None else None
    if args.input_mode is None:
        args.input_mode = saved_input or "single"
    elif saved_input is not None and args.input_mode != saved_input:
        raise ValueError(f"--input-mode {args.input_mode} does not match training "
                         f"input mode {saved_input} in best_meta.json")
    print("Input mode:", args.input_mode)
    print("Preprocessing:", repr(get_transform(args.input_mode)).replace("\n", " "))

    train_ds, val_ds, test_ds = load_datasets(args.split_mode, args.input_mode)
    patient_map = load_patient_map()
    for member_metadata in all_metadata:
        check_checkpoint_compatibility(member_metadata, args, train_ds, val_ds,
                                       test_ds, patient_map)
    ds = val_ds if args.split == "validation" else test_ds
    samples = get_sample_info(ds)

    models = []
    for checkpoint in args.checkpoint:
        model = (ConvNeXt(num_classes=2, in_channels=3) if args.model == "convnext"
                 else build_baseline(args.model)).to(device)
        # The project's best.pth saves the raw state_dict (NOT the resume dict).
        state = torch.load(checkpoint, map_location=device)
        if not isinstance(state, dict) or "model_state" in state:
            raise ValueError("Expected best.pth model weights, not last.pth resume checkpoint")
        model.load_state_dict(state, strict=True)
        model.eval()
        models.append(model)

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)
    labels_seen, scores = [], []
    total_time = 0.0
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            # Ensemble = mean of member AD probabilities (one member: unchanged).
            probabilities = torch.stack([
                torch.softmax(member(images), dim=1)[:, 0] for member in models
            ]).mean(dim=0)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            total_time += time.perf_counter() - start
            scores.extend(probabilities.cpu().tolist())
            labels_seen.extend(labels.tolist())

    if len(samples) != len(scores) or labels_seen != [label for _, label in samples]:
        raise RuntimeError("Dataset sample order does not match model predictions")
    records = create_records(samples, scores, patient_map)
    print(
        f"Evaluated: {len(records['slice'])} slices, {len(records['scan'])} scans, "
        f"{records['total_patients']} patients"
    )
    print(f"Patients with inconsistent diagnoses (excluded from patient score): "
          f"{records['excluded_mixed_label_patients']}")
    print(f"Scans not containing exactly 20 slices: {len(records['unusual_scan_sizes'])}")

    result_by_level = {}
    for level in ("slice", "scan", "patient"):
        metrics = evaluate(records[level], args.threshold)
        result_by_level[level] = metrics
        print_metrics(level, metrics)

    overlap_summary = None
    if args.split == "test" and args.split_mode == "official":
        # Leakage diagnosis: test subjects that also have (other) scans in the
        # training/validation folders versus subjects the model never saw.
        seen_subjects = {patient_map[get_image_id(path)]
                         for dataset in (train_ds, val_ds)
                         for path, _ in get_sample_info(dataset)}
        overlap_summary = {}
        print("Official-split test, stratified by whether the subject has other "
              "scans in the course train folder:")
        for name, keep in (("seen_subjects", True), ("unseen_subjects", False)):
            subset = [record for record in records["slice"]
                      if (patient_map[get_image_id(record[0])] in seen_subjects) == keep]
            subjects = {patient_map[get_image_id(record[0])] for record in subset}
            result = evaluate(subset, args.threshold)
            if result is not None:
                result["n_subjects"] = len(subjects)
            overlap_summary[name] = result
            print(f"  {name} ({len(subjects)} subjects):")
            print_metrics("slice", result)

    print(f"Forward-pass latency: {1000 * total_time / max(len(scores), 1):.3f} ms/slice "
          f"(batch-size={args.batch_size}; excludes file loading & transfer)")
    print("NOTE: Scan- and patient-level scores are separate outcomes, NOT slice accuracy.")
    print("NOTE: Confidence 0.80 triage metrics in JSON are illustrative, NOT calibrated.")

    # Store only aggregate metrics by default; no raw MRIs, paths or patient IDs.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "split": args.split,
        "split_mode": args.split_mode,
        "subject_independent_test": args.split_mode == "strict",
        "model": args.model,
        "checkpoint_metadata_verified": metadata is not None,
        "selection_level": metadata.get("selection_level") if metadata else "unknown_legacy",
        "selection_threshold": metadata.get("selection_threshold") if metadata else None,
        "checkpoint": [c.as_posix() for c in args.checkpoint],
        "input_mode": args.input_mode,
        "ensemble_size": len(args.checkpoint),
        "official_test_by_subject_overlap": overlap_summary,
        "threshold_ad": args.threshold,
        "selection": "New best_meta.json verifies saved selection; legacy files are unverified",
        "class_map": {"AD": 0, "NC": 1},
        "num_scans_with_nonstandard_slice_count": len(records["unusual_scan_sizes"]),
        "num_patients_total": records["total_patients"],
        "num_patients_excluded_mixed_labels": records["excluded_mixed_label_patients"],
        "inference_ms_per_slice_forward_only": 1000 * total_time / max(len(scores), 1),
        "metrics": result_by_level,
    }
    path = args.output_dir / f"{args.split}_metrics.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("Saved aggregate results:", path)

    if not args.no_plots:
        save_figures(records, result_by_level, args)

    if args.save_details:
        for level in ("slice", "scan", "patient"):
            csv_path = args.output_dir / f"{args.split}_{level}_predictions.csv"
            save_anonymized_details(csv_path, records[level], args.threshold, args.id_salt)
            print("Saved local pseudonymous predictions:", csv_path)
        print("WARNING: Do not commit prediction CSVs to a public repository without authorization.")


if __name__ == "__main__":
    main()
