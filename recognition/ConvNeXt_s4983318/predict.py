"""Evaluate saved COMP3710 ADNI checkpoints without retraining or data leakage.

Stage 5: evaluate the previously selected BEST checkpoint on validation first.
Use validation predictions to choose a human-review confidence threshold;
only then run on test ONCE with a locked rule.

Requires sibling modules.py and dataset.py. Uses exactly the same CSV splits
and image normalization as training. Prints slice- and patient-level metrics,
AUROC, calibration, confidence/coverage tradeoffs and inference cost.
Outputs small prediction tables and diagnostic plots OUTSIDE the Git repo.

AI-use disclosure: ChatGPT assisted with this evaluation scaffold. Verify
reported metrics, split handling and figures before submitting.
"""

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import torch

from dataset import make_dataloaders
from modules import BaselineCNN, build_convnext_tiny

CLASSES = {0: "NC", 1: "AD"}


def roc_auc_binary(rows):
    """Rank-based AUROC, with proper tie handling; no sklearn required."""
    ranked = sorted((float(r["p_ad"]), int(r["label"])) for r in rows)
    positive = sum(y for _, y in ranked)
    negative = len(ranked) - positive
    if not positive or not negative:
        return None
    sum_positive_ranks = 0.0
    i = 0
    while i < len(ranked):
        j = i + 1
        while j < len(ranked) and ranked[j][0] == ranked[i][0]:
            j += 1
        rank_avg = ((i + 1) + j) / 2.0
        sum_positive_ranks += rank_avg * sum(y for _, y in ranked[i:j])
        i = j
    return (sum_positive_ranks - positive * (positive + 1) / 2.0) / (positive * negative)


def classification_metrics(rows):
    """Confusion rows=actual, columns=predicted, with NC=0 and AD=1."""
    if not rows:
        raise ValueError("No predictions to evaluate.")
    cm = [[0, 0], [0, 0]]
    for row in rows:
        y, p = int(row["label"]), int(row["prediction"])
        if y not in CLASSES or p not in CLASSES:
            raise ValueError("Invalid label in predictions.")
        cm[y][p] += 1
    per_class = {}
    for c in (0, 1):
        tp = cm[c][c]
        fp = cm[1 - c][c]
        fn = cm[c][1 - c]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[CLASSES[c]] = {
            "support": cm[c][0] + cm[c][1],
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    n = len(rows)
    return {
        "samples": n,
        "accuracy": (cm[0][0] + cm[1][1]) / n,
        "macro_f1": (per_class["NC"]["f1"] + per_class["AD"]["f1"]) / 2,
        "auroc": roc_auc_binary(rows),
        "confusion_matrix_true_rows_pred_columns": cm,
        "per_class": per_class,
    }


def calibration(rows, bins=10):
    """Top-label ECE measures whether reported confidence matches correctness."""
    cells = [[] for _ in range(bins)]
    for row in rows:
        confidence = float(row["confidence"])
        if not 0.5 - 1e-8 <= confidence <= 1.0 + 1e-8:
            raise ValueError(f"Confidence is outside [0.5, 1]: {confidence}")
        bucket = min(int(confidence * bins), bins - 1)
        cells[bucket].append(row)
    summary = []
    ece = 0.0
    for i, cell in enumerate(cells):
        n = len(cell)
        avg_conf = (sum(float(r["confidence"]) for r in cell) / n) if n else None
        accuracy = (sum(int(r["correct"]) for r in cell) / n) if n else None
        if n:
            ece += (n / len(rows)) * abs(avg_conf - accuracy)
        summary.append({
            "bin_low": i / bins, "bin_high": (i + 1) / bins,
            "count": n, "avg_confidence": avg_conf, "accuracy": accuracy,
        })
    brier = sum((float(r["p_ad"]) - int(r["label"])) ** 2 for r in rows) / len(rows)
    wrong = [r for r in rows if not int(r["correct"])]
    return {
        "top_label_ece": ece, "brier_score": brier,
        "wrong_predictions": len(wrong),
        "wrong_with_confidence_at_least_0_9": sum(
            float(r["confidence"]) >= 0.9 for r in wrong
        ),
        "bins": summary,
    }


def confidence_tradeoff(rows, thresholds=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)):
    """A high-confidence prediction is automated; the rest goes to review."""
    table = []
    for threshold in thresholds:
        accepted = [r for r in rows if float(r["confidence"]) >= threshold]
        n = len(accepted)
        correct = sum(int(r["correct"]) for r in accepted)
        accepted_fn = sum(
            int(r["label"]) == 1 and int(r["prediction"]) == 0 for r in accepted
        )
        accepted_fp = sum(
            int(r["label"]) == 0 and int(r["prediction"]) == 1 for r in accepted
        )
        table.append({
            "threshold": round(threshold, 3),
            "accepted": n,
            "sent_to_review": len(rows) - n,
            "coverage": n / len(rows),
            "accepted_accuracy": correct / n if n else None,
            "accepted_errors": n - correct,
            "accepted_false_negatives_AD_as_NC": accepted_fn,
            "accepted_false_positives_NC_as_AD": accepted_fp,
        })
    return table


def make_prediction_rows(model, loader, device, amp):
    results = []
    model.eval()
    first_gpu_batch = None
    start = time.perf_counter()
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, 1):
            x = batch["image"].to(device, non_blocking=True)
            if first_gpu_batch is None:
                first_gpu_batch = x
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(x)
            if not torch.isfinite(logits).all().item():
                raise RuntimeError("Non-finite logits encountered during inference")
            probs = logits.float().softmax(dim=1)[:, 1].cpu().tolist()
            truths = batch["label"].tolist()
            paths = batch["relative_path"]
            patients = batch["patient_id"]
            scans = batch["scan_id"]
            for path, pid, scan, y, p_ad in zip(paths, patients, scans, truths, probs):
                pred = int(p_ad >= 0.5)
                results.append({
                    "relative_path": path, "patient_id": pid, "scan_id": scan,
                    "label": int(y), "prediction": pred, "p_ad": float(p_ad),
                    "confidence": max(float(p_ad), 1.0 - float(p_ad)),
                    "correct": int(pred == int(y)),
                })
            if batch_index % 100 == 0 or batch_index == len(loader):
                print(f"  inference: {batch_index}/{len(loader)} batches", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    if len(results) != len(loader.dataset):
        raise ValueError(f"Expected {len(loader.dataset)} predictions; received {len(results)}")
    return results, first_gpu_batch, elapsed


def aggregate_by_patient(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["patient_id"]].append(row)
    patients = []
    for pid in sorted(groups):
        patient_rows = groups[pid]
        labels = {int(r["label"]) for r in patient_rows}
        if len(labels) != 1:
            raise ValueError(f"Conflicting diagnosis labels for patient {pid}")
        y = labels.pop()
        p_ad = sum(float(r["p_ad"]) for r in patient_rows) / len(patient_rows)
        pred = int(p_ad >= 0.5)
        patients.append({
            "patient_id": pid, "slices": len(patient_rows),
            "label": y, "prediction": pred, "p_ad": p_ad,
            "confidence": max(p_ad, 1 - p_ad), "correct": int(pred == y),
        })
    return patients


def profile_inference(model, images, device, amp, warmup=3, repeats=12):
    """Forward-only latency with data already on GPU; excludes disk I/O."""
    if images is None:
        raise ValueError("Cannot benchmark inference without input batch")
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            with torch.autocast(device_type=device.type, enabled=amp):
                model(images)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        durations = []
        for _ in range(repeats):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            with torch.autocast(device_type=device.type, enabled=amp):
                model(images)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            durations.append(time.perf_counter() - start)
    durations.sort()
    median = durations[len(durations) // 2]
    return {
        "batch_size": images.size(0),
        "median_forward_batch_ms": median * 1000,
        "median_forward_ms_per_slice": median * 1000 / images.size(0),
        "peak_cuda_allocated_mb_during_profile": (
            torch.cuda.max_memory_allocated(device) / 1024 ** 2 if device.type == "cuda" else None
        ),
        "note": "Forward-only timing; images preloaded to device; excludes disk, transfers and plotting. "
                "Repeat at the same batch size and machine for both models.",
    }


def choose_failure_cases(rows, limit=5):
    """Deterministic, varied selection: confident FN/FP and borderline mistakes."""
    incorrect = [r for r in rows if not int(r["correct"])]
    wrong_fn = sorted((r for r in incorrect if int(r["label"]) == 1),
                      key=lambda r: (-r["confidence"], r["relative_path"]))
    wrong_fp = sorted((r for r in incorrect if int(r["label"]) == 0),
                      key=lambda r: (-r["confidence"], r["relative_path"]))
    borderline = sorted(incorrect, key=lambda r: (r["confidence"], r["relative_path"]))
    all_wrong = sorted(incorrect, key=lambda r: (-r["confidence"], r["relative_path"]))
    selection = []
    chosen_paths, chosen_patients = set(), set()

    def add_candidates(candidates, kind, n, distinct_patients=True):
        for row in candidates:
            if n <= 0 or len(selection) >= limit:
                break
            if row["relative_path"] in chosen_paths:
                continue
            if distinct_patients and row["patient_id"] in chosen_patients:
                continue
            selection.append({**row, "selection_reason": kind})
            chosen_paths.add(row["relative_path"])
            chosen_patients.add(row["patient_id"])
            n -= 1

    add_candidates(wrong_fn, "high-confidence false negative", 2)
    add_candidates(wrong_fp, "high-confidence false positive", 2)
    add_candidates(borderline, "borderline wrong prediction", 1)
    add_candidates(all_wrong, "remaining confident error", limit)
    if len(selection) < limit:
        add_candidates(all_wrong, "additional error (repeated patient)", limit, False)
    return selection


def save_table(path, rows, fieldnames):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: r.get(key) for key in fieldnames} for r in rows])


def save_plots(output_dir, calib, tradeoff, failures, data_root, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    xs = [r["avg_confidence"] for r in calib["bins"] if r["count"]]
    ys = [r["accuracy"] for r in calib["bins"] if r["count"]]
    sizes = [r["count"] for r in calib["bins"] if r["count"]]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot([0.5, 1], [0.5, 1], "--", label="Perfect calibration", color="gray")
    ax.scatter(xs, ys, s=[max(25, k / 8) for k in sizes], alpha=0.8, label="Observed bins")
    ax.set_xlim(0.49, 1.01)
    ax.set_ylim(-0.02, 1.01)
    ax.set(xlabel="Mean predicted confidence", ylabel="Fraction correct",
           title=f"{title}: confidence calibration")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "calibration.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot([r["coverage"] for r in tradeoff if r["accepted_accuracy"] is not None],
            [r["accepted_accuracy"] for r in tradeoff if r["accepted_accuracy"] is not None],
            marker="o")
    ax.set(xlabel="Fraction of cases automatically accepted (coverage)",
           ylabel="Accuracy among accepted cases",
           title=f"{title}: accuracy vs. review coverage", ylim=(0, 1.01), xlim=(0, 1.01))
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "review_tradeoff.png", dpi=160)
    plt.close(fig)

    if failures:
        n = len(failures)
        fig, axes = plt.subplots(1, n, figsize=(3.1 * n, 3.8), squeeze=False)
        for i, row in enumerate(failures):
            path = data_root / Path(row["relative_path"])
            with Image.open(path) as image:
                gray = image.convert("L").copy()
            ax = axes[0][i]
            ax.imshow(gray, cmap="gray", vmin=0, vmax=255)
            ax.axis("off")
            actual, predicted = CLASSES[int(row["label"])], CLASSES[int(row["prediction"])]
            ax.set_title(f"#{i + 1}: {actual} to {predicted}\n"
                         f"Conf. {row['confidence']:.2f}", fontsize=10)
        fig.suptitle(f"{title}: validation/test prediction errors (selection rules in CSV)", fontsize=11)
        fig.tight_layout()
        fig.savefig(output_dir / "failure_examples.png", dpi=160)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=("baseline", "convnext"))
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--manifest-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split", default="val", choices=("val", "test"))
    parser.add_argument("--confirm-test", action="store_true",
                        help="Required for test: do not tune model/threshold on test data")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--example-count", type=int, default=5)
    args = parser.parse_args()
    if args.split == "test" and not args.confirm_test:
        parser.error("Independent test is locked. Choose your rules on validation first, "
                     "then explicitly add --confirm-test for one final test evaluation.")
    if args.batch_size < 1 or args.num_workers < 0 or not (0 <= args.example_count <= 5):
        parser.error("Invalid batch-size, num-workers or example-count (0-5)")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable in this Python environment")

    device = torch.device("cuda" if args.device == "cuda" or
                          (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        parser.error(f"Missing checkpoint: {checkpoint_path}")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error(f"Output directory is not empty: {output_dir}. Choose a new output directory.")

    # Safely load local PyTorch weights, not arbitrary executable pickle content.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        parser.error("Checkpoint is not a recognised train.py output")
    if checkpoint.get("diagnostic", False):
        parser.error("Diagnostic smoke-test checkpoint cannot be reported as evaluation")
    arch = checkpoint.get("architecture")
    expected_arch = "BaselineCNN" if args.model == "baseline" else "ConvNeXt-Tiny"
    if arch != expected_arch:
        parser.error(f"Checkpoint model mismatch: expected {expected_arch}, got {arch}")
    if int(checkpoint.get("num_classes", 2)) != 2:
        parser.error("Expected a binary AD/NC checkpoint")
    image_size = int(checkpoint.get("image_size", 224))

    loaders = make_dataloaders(
        data_root=args.data_root, manifest_dir=args.manifest_dir,
        batch_size=args.batch_size, image_size=image_size,
        num_workers=args.num_workers, seed=42, augment_train=False,
    )
    # Only the requested split is iterated. The others are opened solely to audit leakage.
    loader = loaders[args.split]
    model = (BaselineCNN(num_classes=2) if args.model == "baseline"
             else build_convnext_tiny(num_classes=2, pretrained=False))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    amp = device.type == "cuda"

    print(f"MODEL: {expected_arch}; selected epoch: {checkpoint.get('epoch')}; split: {args.split}")
    print(f"Device: {device}; images: {len(loader.dataset)}; patients: "
          f"{len({r['patient_id'] for r in loader.dataset.records})}")
    print("Using checkpoint weights, not downloading pretrained weights or retraining.")
    predictions, first_batch, elapsed = make_prediction_rows(model, loader, device, amp)
    patients = aggregate_by_patient(predictions)
    slice_metrics = classification_metrics(predictions)
    patient_metrics = classification_metrics(patients)
    slice_calib = calibration(predictions)
    patient_calib = calibration(patients)
    slice_tradeoff = confidence_tradeoff(predictions)
    patient_tradeoff = confidence_tradeoff(patients)
    profile = profile_inference(model, first_batch, device, amp)
    failures = choose_failure_cases(predictions, limit=args.example_count)

    output_dir.mkdir(parents=True, exist_ok=True)
    slice_fields = ["relative_path", "patient_id", "scan_id", "label", "prediction",
                    "p_ad", "confidence", "correct"]
    save_table(output_dir / "slice_predictions.csv", predictions, slice_fields)
    save_table(output_dir / "patient_predictions.csv", patients,
               ["patient_id", "slices", "label", "prediction", "p_ad", "confidence", "correct"])
    save_table(output_dir / "review_thresholds_slices.csv", slice_tradeoff,
               list(slice_tradeoff[0]))
    save_table(output_dir / "review_thresholds_patients.csv", patient_tradeoff,
               list(patient_tradeoff[0]))
    save_table(output_dir / "failure_cases.csv", failures, slice_fields + ["selection_reason"])
    save_plots(output_dir, slice_calib, slice_tradeoff, failures,
               Path(args.data_root).expanduser().resolve(),
               f"{expected_arch} {args.split}")
    summary = {
        "model": args.model,
        "architecture": expected_arch,
        "split": args.split,
        "checkpoint_file": checkpoint_path.name,
        "best_checkpoint_epoch": checkpoint.get("epoch"),
        "best_checkpoint_val_loss": checkpoint.get("val_loss"),
        "batch_size": args.batch_size, "image_size": image_size,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "slice_metrics": slice_metrics,
        "patient_metrics": patient_metrics,
        "slice_calibration": slice_calib,
        "patient_calibration": patient_calib,
        "slice_confidence_tradeoff": slice_tradeoff,
        "patient_confidence_tradeoff": patient_tradeoff,
        "inference_profile": profile,
        "end_to_end_seconds_including_image_loading": elapsed,
        "selection_policy": "Checkpoint chosen by training script via lowest val loss; "
                            "no hyperparameter or checkpoint selection here.",
        "clinical_note": "Confidence values are NOT calibrated clinical probabilities; "
                         "no automated triage is safe without external validation.",
    }
    (output_dir / "evaluation.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    print("\nRESULTS (slice level)")
    print(f"  accuracy={slice_metrics['accuracy']:.4f} "
          f"macro-F1={slice_metrics['macro_f1']:.4f} "
          f"AUROC={slice_metrics['auroc']:.4f}")
    print(f"  confusion [true NC, AD] x [pred NC, AD]: {slice_metrics['confusion_matrix_true_rows_pred_columns']}")
    for name, measures in slice_metrics["per_class"].items():
        print(f"  {name}: precision={measures['precision']:.4f}, "
              f"recall={measures['recall']:.4f}, F1={measures['f1']:.4f}, support={measures['support']}")
    print(f"  ECE={slice_calib['top_label_ece']:.4f} "
          f"Brier={slice_calib['brier_score']:.4f} "
          f"wrong(conf>=0.9)={slice_calib['wrong_with_confidence_at_least_0_9']}")
    print("RESULTS (patient level; mean slice P(AD))")
    print(f"  patients={len(patients)} accuracy={patient_metrics['accuracy']:.4f} "
          f"macro-F1={patient_metrics['macro_f1']:.4f} "
          f"AUROC={patient_metrics['auroc']:.4f}")
    print(f"  confusion [true NC, AD] x [pred NC, AD]: {patient_metrics['confusion_matrix_true_rows_pred_columns']}")
    print("REVIEW EXAMPLE (threshold 0.90; validation only when --split val)")
    rule = next(r for r in patient_tradeoff if r["threshold"] == 0.9)
    print(f"  patient coverage={rule['coverage']:.3f}; "
          f"accepted={rule['accepted']}; sent to review={rule['sent_to_review']}; "
          f"accepted accuracy={rule['accepted_accuracy']}")
    print(f"PROFILE: {profile['median_forward_ms_per_slice']:.2f} ms/image; "
          f"peak allocated VRAM={profile['peak_cuda_allocated_mb_during_profile']} MiB")
    print(f"Files saved: {output_dir}")
    print("Do not upload patient-level prediction CSVs or checkpoint files to public GitHub.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
