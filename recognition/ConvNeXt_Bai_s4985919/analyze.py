"""Post-hoc analysis of a FROZEN model (or seed ensemble) for the README.

Answers the open research dilemma (confidence, calibration, reject option)
and produces the failure-analysis evidence:

* calibration on validation and test: ECE, Brier score, reliability diagram;
* a reject-option rule chosen on VALIDATION only (lowest confidence cut-off
  whose accepted validation accuracy reaches --target-accuracy), then applied
  unchanged to test: coverage, accepted accuracy, referred slices;
* ROC curves, test confusion matrix, accuracy-vs-coverage curves;
* test accuracy by group: seen/unseen subject (official split), MRI sequence
  family and scan year, parsed from the course metadata file;
* the most confidently wrong test scans (candidates for failure autopsies)
  with a slice montage, plus resource figures (parameters, inference memory
  and latency, training peak memory from history.json).

Nothing here trains, tunes or selects a model. The class threshold comes
from best_meta.json. The official test is only read with
--confirm-final-test and should be run once per frozen model.

Example:
    python -u analyze.py --confirm-final-test --num-workers 4 \\
        --output-dir eval_analysis_convnext \\
        --checkpoint checkpoints/convnext/official_cosine_ema_context3_s4{2,3,4}/best.pth
"""

import argparse
import csv
import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------
# Pure functions (NumPy only; unit-testable without PyTorch or data)
# --------------------------------------------------------------------------

def predictions_and_confidence(prob_ad, threshold):
    """AD=0 is predicted when P(AD) >= threshold; confidence = P(predicted class)."""
    prob_ad = np.asarray(prob_ad, dtype=float)
    predicted = np.where(prob_ad >= threshold, 0, 1)
    confidence = np.where(predicted == 0, prob_ad, 1.0 - prob_ad)
    return predicted, confidence


def expected_calibration_error(confidence, correct, n_bins=15):
    """Standard equal-width ECE over the confidence of the predicted class."""
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=float)
    edges = np.linspace(0.5, 1.0, n_bins + 1)  # binary: confidence >= 0.5 at t=0.5
    edges[0] = 0.0
    ece, bins = 0.0, []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence > low) & (confidence <= high)
        if low == 0.0:
            mask |= confidence == 0.0
        if mask.any():
            gap = abs(correct[mask].mean() - confidence[mask].mean())
            ece += mask.mean() * gap
            bins.append({"low": float(low), "high": float(high), "n": int(mask.sum()),
                         "accuracy": float(correct[mask].mean()),
                         "confidence": float(confidence[mask].mean())})
    return float(ece), bins


def brier_score(prob_ad, labels):
    """Mean squared error of P(AD) against the AD indicator (AD=0 label)."""
    target = (np.asarray(labels) == 0).astype(float)
    return float(np.mean((np.asarray(prob_ad, dtype=float) - target) ** 2))


def coverage_curve(confidence, correct):
    """Accuracy of the k most confident predictions, for every k."""
    order = np.argsort(-np.asarray(confidence), kind="stable")
    correct_sorted = np.asarray(correct, dtype=float)[order]
    k = np.arange(1, len(order) + 1)
    return k / len(order), np.cumsum(correct_sorted) / k, np.asarray(confidence)[order]


def choose_reject_cutoff(confidence, correct, target_accuracy):
    """Largest coverage whose accepted accuracy >= target; returns cut-off or None."""
    coverage, accuracy, sorted_conf = coverage_curve(confidence, correct)
    valid = np.where(accuracy >= target_accuracy)[0]
    if len(valid) == 0:
        return None
    k = valid[-1]
    # Ties: accept everything with confidence >= the cut-off value.
    return float(sorted_conf[k])


def apply_reject_rule(confidence, correct, labels, cutoff):
    confidence = np.asarray(confidence)
    correct = np.asarray(correct, dtype=bool)
    labels = np.asarray(labels)
    accepted = confidence >= cutoff
    ad = labels == 0
    result = {
        "cutoff_confidence": float(cutoff),
        "coverage": float(accepted.mean()),
        "referred_to_specialist": int((~accepted).sum()),
        "accepted": int(accepted.sum()),
        "accepted_accuracy": float(correct[accepted].mean()) if accepted.any() else None,
        "referred_accuracy": float(correct[~accepted].mean()) if (~accepted).any() else None,
        "ad_accepted_recall": (float(correct[accepted & ad].mean())
                               if (accepted & ad).any() else None),
        "ad_missed_and_not_referred": int((accepted & ad & ~correct).sum()),
    }
    return result


def roc_points(prob_ad, labels):
    """ROC curve with AD (label 0) positive."""
    prob_ad = np.asarray(prob_ad, dtype=float)
    positive = np.asarray(labels) == 0
    order = np.argsort(-prob_ad, kind="stable")
    tp = np.cumsum(positive[order])
    fp = np.cumsum(~positive[order])
    tpr = np.concatenate([[0.0], tp / max(positive.sum(), 1)])
    fpr = np.concatenate([[0.0], fp / max((~positive).sum(), 1)])
    return fpr, tpr


def parse_raw_name(raw_path):
    """Sequence family and scan year from an ADNI raw filename in the metadata."""
    name = str(raw_path).split("/")[-1]
    sequence = re.search(r"_MR_(.*?)_br_raw", name)
    year = re.search(r"br_raw_(\d{4})", name)
    sequence = sequence.group(1).strip("_") if sequence else "unknown"
    if "FSPGR" in sequence.upper():
        family = "IR-FSPGR"
    elif "MPRAGE" in sequence.upper():
        family = "MPRAGE"
    else:
        family = "other"
    year = int(year.group(1)) if year else None
    if year is None:
        year_bin = "unknown"
    elif year <= 2014:
        year_bin = "<=2014"
    elif year <= 2018:
        year_bin = "2015-2018"
    else:
        year_bin = ">=2019"
    return {"sequence": sequence, "sequence_family": family, "year": year,
            "year_bin": year_bin}


def group_accuracy(groups, correct, labels):
    """Accuracy, AD recall and size for each group value."""
    table = {}
    groups = np.asarray(groups, dtype=object)
    correct = np.asarray(correct, dtype=bool)
    ad = np.asarray(labels) == 0
    for value in sorted(set(groups.tolist()), key=str):
        mask = groups == value
        table[str(value)] = {
            "slices": int(mask.sum()),
            "accuracy": float(correct[mask].mean()),
            "ad_fraction": float(ad[mask].mean()),
            "ad_recall": float(correct[mask & ad].mean()) if (mask & ad).any() else None,
            "nc_recall": float(correct[mask & ~ad].mean()) if (mask & ~ad).any() else None,
        }
    return table


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_reliability(results, path):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot([0.5, 1], [0.5, 1], "--", color="grey", lw=1, label="Perfect calibration")
    for name, res in results.items():
        bins = res["calibration"]["bins"]
        ax.plot([b["confidence"] for b in bins], [b["accuracy"] for b in bins], marker="o",
                label=f"{name} (ECE={res['calibration']['ece']:.3f})")
    ax.set(xlabel="Mean confidence", ylabel="Observed accuracy", xlim=(0.5, 1), ylim=(0, 1),
           title="Reliability (slice level)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_roc(curves, path):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    for name, (fpr, tpr, auc) in curves.items():
        ax.plot(fpr, tpr, label=f"{name} (AUROC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set(xlabel="False positive rate (NC called AD)", ylabel="True positive rate (AD recall)",
           title="ROC, AD positive (slice level)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_coverage(curves, cutoff_info, path):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    for name, (coverage, accuracy) in curves.items():
        ax.plot(coverage, accuracy, label=name)
    if cutoff_info:
        for name, info in cutoff_info.items():
            ax.scatter([info["coverage"]], [info["accepted_accuracy"]], zorder=3)
            ax.annotate(f"{name}: {info['coverage']:.0%} kept", (info["coverage"],
                        info["accepted_accuracy"]), fontsize=7, xytext=(4, -10),
                        textcoords="offset points")
    ax.set(xlabel="Coverage (fraction of slices decided automatically)",
           ylabel="Accuracy of automatically decided slices", ylim=(0.4, 1.01),
           title="Accuracy vs coverage (reject option)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_confusion(tp, fn, fp, tn, title, path):
    plt = _plt()
    matrix = np.array([[tp, fn], [fp, tn]])
    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    ax.imshow(matrix, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{matrix[i, j]}\n({matrix[i, j] / matrix[i].sum():.0%})",
                    ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    ax.set_xticks([0, 1], ["Pred AD", "Pred NC"])
    ax.set_yticks([0, 1], ["True AD", "True NC"])
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_groups(group_tables, path):
    plt = _plt()
    names = list(group_tables)
    fig, axes = plt.subplots(1, len(names), figsize=(3.6 * len(names), 3.6), squeeze=False)
    for ax, name in zip(axes[0], names):
        table = group_tables[name]
        keys = list(table)
        values = [table[k]["accuracy"] for k in keys]
        bars = ax.bar(range(len(keys)), values, color="#4C78A8")
        for bar, key in zip(bars, keys):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"n={table[key]['slices']}", ha="center", fontsize=7)
        ax.set_xticks(range(len(keys)), keys, rotation=20, fontsize=8)
        ax.set(ylim=(0, 1.05), title=name.replace("_", " "), ylabel="Slice accuracy")
        ax.axhline(0.8, ls="--", color="grey", lw=1)
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------
# Model inference (PyTorch imported lazily so the functions above test alone)
# --------------------------------------------------------------------------

def run_inference(models, dataset, device, batch_size, num_workers):
    import torch
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    scores, labels, forward_seconds = [], [], 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for images, batch_labels in loader:
            images = images.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            probabilities = torch.stack([torch.softmax(m(images), dim=1)[:, 0]
                                         for m in models]).mean(dim=0)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            forward_seconds += time.perf_counter() - start
            scores.extend(probabilities.cpu().tolist())
            labels.extend(batch_labels.tolist())
    peak_mb = (torch.cuda.max_memory_allocated(device) / 1024 ** 2
               if device.type == "cuda" else None)
    return np.array(scores), np.array(labels), forward_seconds, peak_mb


def evaluate_split(name, prob_ad, labels, threshold):
    from predict import rank_auroc
    predicted, confidence = predictions_and_confidence(prob_ad, threshold)
    correct = predicted == labels
    ece, bins = expected_calibration_error(confidence, correct)
    tp = int(((labels == 0) & (predicted == 0)).sum())
    fn = int(((labels == 0) & (predicted == 1)).sum())
    fp = int(((labels == 1) & (predicted == 0)).sum())
    tn = int(((labels == 1) & (predicted == 1)).sum())
    wrong_conf = confidence[~correct]
    return {
        "split": name,
        "slices": int(len(labels)),
        "accuracy": float(correct.mean()),
        "auroc_ad": rank_auroc(prob_ad.tolist(), labels.tolist()),
        "tp_ad": tp, "fn_ad": fn, "fp_ad": fp, "tn_ad": tn,
        "calibration": {"ece": ece, "brier": brier_score(prob_ad, labels), "bins": bins},
        "mean_confidence_correct": float(confidence[correct].mean()),
        "mean_confidence_wrong": float(wrong_conf.mean()) if len(wrong_conf) else None,
        "wrong_with_confidence_ge_0.9": int((wrong_conf >= 0.9).sum()),
    }, predicted, confidence, correct


def failure_cases(samples, prob_ad, labels, threshold, k=3):
    """Most confidently wrong scans of each kind (scan = mean of its slices)."""
    from dataset import get_image_id
    scans = defaultdict(list)
    for (path, label), p in zip(samples, prob_ad):
        scans[get_image_id(path)].append((path, float(p), label))
    rows = []
    for scan_id, entries in scans.items():
        p = float(np.mean([e[1] for e in entries]))
        label = entries[0][2]
        predicted = 0 if p >= threshold else 1
        if predicted != label:
            confidence = p if predicted == 0 else 1 - p
            rows.append({"scan_id": scan_id, "true": "AD" if label == 0 else "NC",
                         "predicted": "AD" if predicted == 0 else "NC",
                         "scan_prob_ad": p, "confidence": confidence,
                         "paths": sorted((e[0] for e in entries),
                                         key=lambda q: int(Path(q).stem.rpartition("_")[2]))})
    rows.sort(key=lambda r: -r["confidence"])
    false_negatives = [r for r in rows if r["true"] == "AD"][:k]
    false_positives = [r for r in rows if r["true"] == "NC"][:k]
    return false_negatives + false_positives


def plot_failure_montage(cases, path, slices_per_case=5):
    from PIL import Image
    plt = _plt()
    if not cases:
        return
    fig, axes = plt.subplots(len(cases), slices_per_case,
                             figsize=(1.9 * slices_per_case, 2.0 * len(cases)), squeeze=False)
    for row, case in enumerate(cases):
        paths = case["paths"]
        picks = np.linspace(0, len(paths) - 1, slices_per_case).round().astype(int)
        for col, index in enumerate(picks):
            ax = axes[row][col]
            with Image.open(paths[index]) as image:
                ax.imshow(image.convert("L"), cmap="gray", vmin=0, vmax=255)
            ax.axis("off")
        axes[row][0].set_title(f"case {row + 1}: true {case['true']}, pred {case['predicted']}, "
                               f"P(AD)={case['scan_prob_ad']:.2f}", fontsize=7, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    parser.add_argument("--model", choices=["convnext", "cnn", "cnn_matched"], default="convnext")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-accuracy", type=float, default=0.90,
                        help="Reject rule: accepted VALIDATION accuracy to reach (fixed in advance)")
    parser.add_argument("--confirm-final-test", action="store_true",
                        help="Also analyse the official test with the frozen model/rule")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--failure-cases", type=int, default=3,
                        help="Most confident wrong scans per error type")
    args = parser.parse_args()

    import torch
    from types import SimpleNamespace
    from dataset import (load_datasets, load_patient_map, get_image_id, METADATA_PATH)
    from predict import (load_best_metadata, check_checkpoint_compatibility,
                         get_sample_info)
    from modules import ConvNeXt
    from baseline import build_baseline

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_meta = [load_best_metadata(c) for c in args.checkpoint]
    if any(m is None for m in all_meta):
        raise ValueError("Every checkpoint needs best_meta.json (new-format run)")
    for key in ("split_mode", "input_mode", "selection_threshold", "model"):
        if len({str(m.get(key)) for m in all_meta}) != 1:
            raise ValueError(f"Checkpoints disagree on {key}")
    meta = all_meta[0]
    if meta["model"] != args.model:
        raise ValueError(f"Checkpoints are {meta['model']!r}; pass --model {meta['model']}")
    split_mode = meta.get("split_mode", "strict")
    input_mode = meta.get("input_mode", "single")
    threshold = meta["selection_threshold"]
    print(f"Model {args.model} x{len(args.checkpoint)}; split {split_mode}; input {input_mode}; "
          f"class threshold {threshold} (from best_meta.json)")

    train_ds, val_ds, test_ds = load_datasets(split_mode, input_mode)
    patient_map = load_patient_map()
    check_args = SimpleNamespace(split="test" if args.confirm_final_test else "validation",
                                 model=args.model, input_mode=input_mode)
    for m in all_meta:
        check_checkpoint_compatibility(m, check_args, train_ds, val_ds, test_ds, patient_map)

    models = []
    for checkpoint in args.checkpoint:
        model = (ConvNeXt(num_classes=2, in_channels=3) if args.model == "convnext"
                 else build_baseline(args.model)).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device), strict=True)
        model.eval()
        models.append(model)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"model": args.model, "ensemble_size": len(models), "split_mode": split_mode,
              "input_mode": input_mode, "class_threshold": threshold,
              "checkpoints": [c.as_posix() for c in args.checkpoint]}

    # ---- validation: calibration and the reject rule (chosen here only) ----
    val_samples = get_sample_info(val_ds)
    val_prob, val_labels, _, _ = run_inference(models, val_ds, device, args.batch_size,
                                               args.num_workers)
    if val_labels.tolist() != [label for _, label in val_samples]:
        raise RuntimeError("Validation order changed")
    val_res, _, val_conf, val_correct = evaluate_split("validation", val_prob, val_labels,
                                                       threshold)
    cutoff = choose_reject_cutoff(val_conf, val_correct, args.target_accuracy)
    report["validation"] = val_res
    report["reject_rule"] = {"target_validation_accuracy": args.target_accuracy,
                             "cutoff_confidence": cutoff}
    if cutoff is not None:
        report["reject_rule"]["validation"] = apply_reject_rule(val_conf, val_correct,
                                                                val_labels, cutoff)
    results = {"validation": val_res}
    roc = {"validation": (*roc_points(val_prob, val_labels), val_res["auroc_ad"])}
    cov = {"validation": coverage_curve(val_conf, val_correct)[:2]}
    rule_points = ({"validation": report["reject_rule"]["validation"]}
                   if cutoff is not None else {})

    # ---- test: frozen model, frozen threshold, frozen reject rule ----------
    if args.confirm_final_test:
        test_samples = get_sample_info(test_ds)
        test_prob, test_labels, seconds, peak_mb = run_inference(
            models, test_ds, device, args.batch_size, args.num_workers)
        if test_labels.tolist() != [label for _, label in test_samples]:
            raise RuntimeError("Test order changed")
        test_res, test_pred, test_conf, test_correct = evaluate_split(
            "test", test_prob, test_labels, threshold)
        report["test"] = test_res
        results["test"] = test_res
        roc["test"] = (*roc_points(test_prob, test_labels), test_res["auroc_ad"])
        cov["test"] = coverage_curve(test_conf, test_correct)[:2]
        if cutoff is not None:
            report["reject_rule"]["test"] = apply_reject_rule(test_conf, test_correct,
                                                              test_labels, cutoff)
            rule_points["test"] = report["reject_rule"]["test"]

        # Groups from course metadata (scan year, sequence) and subject overlap.
        metadata = json.loads(Path(METADATA_PATH).read_text())
        seen = {patient_map[get_image_id(p)] for ds in (train_ds, val_ds)
                for p, _ in get_sample_info(ds)}
        info = [parse_raw_name(metadata[get_image_id(p)]["raw"]) for p, _ in test_samples]
        subjects = [patient_map[get_image_id(p)] for p, _ in test_samples]
        groups = {
            "true_class": ["AD" if y == 0 else "NC" for y in test_labels],
            "sequence_family": [i["sequence_family"] for i in info],
            "scan_year": [i["year_bin"] for i in info],
        }
        if split_mode == "official":
            groups["subject_in_training_folder"] = ["seen" if s in seen else "unseen"
                                                    for s in subjects]
        report["test_groups"] = {name: group_accuracy(values, test_correct, test_labels)
                                 for name, values in groups.items()}
        report["test_sequences"] = dict(Counter(i["sequence"] for i in info).most_common())
        plot_groups(report["test_groups"], args.output_dir / "test_accuracy_by_group.png")
        plot_confusion(test_res["tp_ad"], test_res["fn_ad"], test_res["fp_ad"],
                       test_res["tn_ad"], f"Test confusion (slice), threshold {threshold}",
                       args.output_dir / "test_confusion.png")

        # Failure-case candidates: CSV keeps raw IDs locally (output dir is
        # git-ignored); the montage shows only anonymous case numbers.
        cases = failure_cases(test_samples, test_prob, test_labels, threshold,
                              args.failure_cases)
        for case in cases:
            extra = parse_raw_name(metadata[case["scan_id"]]["raw"])
            case.update(sequence=extra["sequence"], year=extra["year"],
                        subject_seen=(patient_map[case["scan_id"]] in seen))
        with (args.output_dir / "failure_cases_LOCAL_ONLY.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["case", "scan_id", "true", "predicted", "scan_prob_ad",
                             "confidence", "sequence", "year", "subject_seen"])
            for n, c in enumerate(cases, 1):
                writer.writerow([n, c["scan_id"], c["true"], c["predicted"],
                                 f"{c['scan_prob_ad']:.3f}", f"{c['confidence']:.3f}",
                                 c["sequence"], c["year"], c["subject_seen"]])
        report["failure_cases"] = [{k: v for k, v in c.items() if k not in ("paths", "scan_id")}
                                   for c in cases]
        plot_failure_montage(cases, args.output_dir / "failure_cases_montage.png")

        report["resources"] = {
            "parameters": int(sum(p.numel() for p in models[0].parameters())),
            "inference_forward_ms_per_slice": 1000 * seconds / len(test_labels),
            "inference_batch_size": args.batch_size,
            "inference_peak_gpu_allocated_mb": peak_mb,
        }
        training_peaks = []
        for checkpoint in args.checkpoint:
            history = checkpoint.with_name("history.json")
            if history.is_file():
                rows = json.loads(history.read_text())
                training_peaks += [r["peak_gpu_memory_mb"] for r in rows
                                   if r.get("peak_gpu_memory_mb") is not None]
        report["resources"]["training_peak_gpu_allocated_mb"] = (max(training_peaks)
                                                                 if training_peaks else None)

    plot_reliability(results, args.output_dir / "reliability.png")
    plot_roc(roc, args.output_dir / "roc.png")
    plot_coverage(cov, rule_points, args.output_dir / "coverage_accuracy.png")
    (args.output_dir / "analysis.json").write_text(json.dumps(report, indent=2))

    # Short console summary.
    for name, res in results.items():
        print(f"{name.upper():10s} acc={res['accuracy']:.4f} AUROC={res['auroc_ad']:.4f} "
              f"ECE={res['calibration']['ece']:.4f} Brier={res['calibration']['brier']:.4f} "
              f"conf(correct)={res['mean_confidence_correct']:.3f} "
              f"conf(wrong)={res['mean_confidence_wrong']:.3f} "
              f"wrong@conf>=0.9={res['wrong_with_confidence_ge_0.9']}")
    rule = report["reject_rule"]
    print(f"Reject rule (chosen on validation for {args.target_accuracy:.0%} accepted accuracy): "
          f"confidence >= {rule['cutoff_confidence']}")
    for split in ("validation", "test"):
        if split in rule:
            r = rule[split]
            print(f"  {split:10s} coverage={r['coverage']:.3f} accepted_acc={r['accepted_accuracy']:.4f} "
                  f"referred={r['referred_to_specialist']} "
                  f"AD missed & not referred={r['ad_missed_and_not_referred']}")
    if "test_groups" in report:
        for name, table in report["test_groups"].items():
            print(f"Test by {name}: " + "; ".join(
                f"{k}: acc={v['accuracy']:.3f} (n={v['slices']})" for k, v in table.items()))
        print("Resources:", report["resources"])
        print("Failure-case candidates (details in failure_cases_LOCAL_ONLY.csv):")
        for n, c in enumerate(report["failure_cases"], 1):
            print(f"  case {n}: true {c['true']} -> pred {c['predicted']}, "
                  f"P(AD)={c['scan_prob_ad']:.3f}, {c['sequence']}, {c['year']}, "
                  f"subject seen={c['subject_seen']}")
    print("Saved:", args.output_dir)


if __name__ == "__main__":
    main()
