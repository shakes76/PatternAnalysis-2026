"""Summarise the completed runs of one or more plan directories into a comparison table.

Every number comes from a run's ``metrics.json``, ``config.json``,
``history.csv`` and ``replay_check.json``; nothing is re-computed from
predictions. Runs are grouped by case id with the ``_seed<N>`` and
``_fold<N>`` suffixes removed, so seeds or folds of one configuration are
averaged (mean and sample standard deviation).

With ``--data-root`` the scan AUROC and balanced accuracy are also computed on
the validation scans acquired in the period shared by both classes (2017 on,
see docs/ACQUISITION_CONFOUND.md) from each run's ``val_scan_predictions.csv``.

With ``--baseline`` and ``--candidate`` (two group names), runs of the two
groups that share the same fold and seed are paired and the per-pair
candidate - baseline differences are reported, which is the fair comparison
between ConvNeXt and the CNN baseline on identical patients.

    python slurm/summarize_runs.py runs/round3_cv --output runs/round3_cv/summary \\
        --baseline CV_cnn_context --candidate CV_tiny_context
"""

import argparse
import csv
import json
from pathlib import Path
import re
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adni_data.acquisition import SHARED_PERIOD_START, acquisition_info  # noqa: E402
from evaluation.metrics import binary_metrics  # noqa: E402

UNITS = ("slice", "scan", "patient")
SUFFIX = re.compile(r"_(seed|fold)\d+")


def shared_period_scores(run: Path, info: dict) -> dict:
    """Scan AUROC / balanced accuracy on validation scans acquired in the shared period."""
    with (run / "val_scan_predictions.csv").open(newline="") as handle:
        scans = [row for row in csv.DictReader(handle)
                 if info[row["image_id"]]["year"] >= SHARED_PERIOD_START]
    scores = binary_metrics([int(row["label"]) for row in scans], [float(row["probability"]) for row in scans])
    return {"shared_period_scans": len(scans), "shared_period_ad_scans": sum(row["label"] == "1" for row in scans),
            "shared_period_scan_auroc": scores["auroc"],
            "shared_period_scan_balanced_accuracy": scores["balanced_accuracy"]}


def read_run(run: Path, info: dict | None = None) -> dict:
    """One row of the per-run table."""
    metrics = json.loads((run / "metrics.json").read_text())
    config = json.loads((run / "config.json").read_text())
    with (run / "history.csv").open(newline="") as handle:
        history = list(csv.DictReader(handle))
    replay = run / "replay_check.json"
    row = {
        "plan": run.parent.name, "case": run.name, "group": SUFFIX.sub("", run.name),
        "model": metrics["model_name"], "fold": metrics["fold"], "seed": config["seed_base"],
        "context_slices": config["execution_controls"]["context_slices"],
        "checkpoint_selection": metrics["checkpoint_selection"],
        "checkpoint_epoch": metrics.get("best_epoch", metrics["epochs_completed"]),
        "epochs": metrics["epochs_completed"],
        "final_train_slice_accuracy": float(history[-1]["train_slice_accuracy"]),
    }
    for unit in UNITS:
        row[f"{unit}_accuracy"] = metrics["metrics"][unit]["accuracy"]
        row[f"{unit}_auroc"] = metrics["metrics"][unit]["auroc"]
    (tn, fp), (fn, tp) = metrics["metrics"]["patient"]["confusion_matrix"]
    resources = metrics["resources"]
    row.update({
        "patient_tn": tn, "patient_fp": fp, "patient_fn": fn, "patient_tp": tp,
        "parameters": resources["trainable_parameters"],
        "train_peak_mib": resources["training_peak_cuda_allocated_mib"],
        "seconds_per_epoch": resources["training_seconds"] / metrics["epochs_completed"],
        "replay_match": json.loads(replay.read_text())["match"] if replay.exists() else None,
    })
    if info is not None:
        row.update(shared_period_scores(run, info))
    return row


def mean_std(values: list[float]) -> str:
    """'mean ± sd' (sd omitted for a single value)."""
    if len(values) == 1:
        return f"{values[0]:.3f}"
    return f"{statistics.mean(values):.3f} ± {statistics.stdev(values):.3f}"


def group_table(rows: list[dict]) -> list[str]:
    """Markdown table of accuracy / AUROC averaged over the runs of each group."""
    lines = ["| Group | Runs | " + " | ".join(f"{u} acc | {u} AUROC" for u in UNITS) + " | Params | Peak MiB | s/epoch |",
             "|---|---:|" + "---:|" * (2 * len(UNITS) + 3)]
    for group in dict.fromkeys(row["group"] for row in rows):
        members = [row for row in rows if row["group"] == group]
        cells = [mean_std([row[f"{unit}_{name}"] for row in members]) for unit in UNITS for name in ("accuracy", "auroc")]
        peak = members[0]["train_peak_mib"]  # None for CPU runs.
        resources = [f"{members[0]['parameters']:,}", "-" if peak is None else f"{peak:.0f}",
                     f"{statistics.mean(row['seconds_per_epoch'] for row in members):.0f}"]
        lines.append(f"| {group} | {len(members)} | " + " | ".join(cells + resources) + " |")
    return lines


def paired_table(rows: list[dict], baseline: str, candidate: str) -> list[str]:
    """Candidate - baseline differences for runs sharing fold and seed."""
    key = lambda row: (row["fold"], row["seed"])
    base = {key(row): row for row in rows if row["group"] == baseline}
    pairs = [(base[key(row)], row) for row in rows if row["group"] == candidate and key(row) in base]
    if not pairs:
        raise ValueError(f"No runs of {candidate} share fold and seed with {baseline}.")
    lines = [f"Paired by fold and seed: {candidate} - {baseline} ({len(pairs)} pairs).", "",
             "| Fold | Seed | " + " | ".join(f"Δ {u} acc" for u in UNITS) + " |", "|---:|---:|" + "---:|" * len(UNITS)]
    for b, c in sorted(pairs, key=lambda pair: key(pair[0])):
        lines.append(f"| {b['fold']} | {b['seed']} | "
                     + " | ".join(f"{c[f'{u}_accuracy'] - b[f'{u}_accuracy']:+.3f}" for u in UNITS) + " |")
    lines.append("| mean | | " + " | ".join(
        f"{statistics.mean(c[f'{u}_accuracy'] - b[f'{u}_accuracy'] for b, c in pairs):+.3f}" for u in UNITS) + " |")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("plan_dirs", type=Path, nargs="+", help="Plan output directories (runs/<plan name>).")
    parser.add_argument("--output", type=Path, required=True, help="Prefix for <output>.csv and <output>.md.")
    parser.add_argument("--baseline", help="Group name of the baseline (e.g. CV_cnn_context).")
    parser.add_argument("--candidate", help="Group name compared against the baseline.")
    parser.add_argument("--data-root", type=Path,
                        help="ADNI root with meta_data_with_label.json; adds shared-period scan metrics.")
    args = parser.parse_args(argv)
    if bool(args.baseline) != bool(args.candidate):
        parser.error("--baseline and --candidate go together.")

    runs = sorted(run for plan in args.plan_dirs for run in plan.iterdir() if (run / "metrics.json").exists())
    if not runs:
        parser.error("No completed runs (metrics.json) found.")
    info = acquisition_info(args.data_root) if args.data_root else None
    rows = [read_run(run, info) for run in runs]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = ["# Run summary", "", f"Source: {', '.join(str(plan) for plan in args.plan_dirs)}", "",
             "## Per run (validation patients)", "",
             "| Case | Ckpt epoch | " + " | ".join(f"{u} acc / AUROC" for u in UNITS)
             + " | Patient FP / FN | Train acc | Replay |", "|---|---:|" + "---|" * len(UNITS) + "---|---:|---|"]
    for row in rows:
        lines.append(f"| {row['plan']}/{row['case']} | {row['checkpoint_epoch']}/{row['epochs']} | "
                     + " | ".join(f"{row[f'{u}_accuracy']:.3f} / {row[f'{u}_auroc']:.3f}" for u in UNITS)
                     + f" | {row['patient_fp']} / {row['patient_fn']} | {row['final_train_slice_accuracy']:.3f}"
                     + f" | {row['replay_match']} |")
    lines += ["", "## Per group (mean ± sd over seeds / folds)", "", *group_table(rows)]
    if info is not None:
        lines += ["", f"## Scans acquired {SHARED_PERIOD_START} on (both classes present)", "",
                  "| Group | Runs | Scans (AD) per run | Scan AUROC | Scan balanced acc. |", "|---|---:|---|---:|---:|"]
        for group in dict.fromkeys(row["group"] for row in rows):
            members = [row for row in rows if row["group"] == group]
            sizes = ", ".join(f"{r['shared_period_scans']} ({r['shared_period_ad_scans']})" for r in members)
            lines.append(f"| {group} | {len(members)} | {sizes} | "
                         f"{mean_std([r['shared_period_scan_auroc'] for r in members])} | "
                         f"{mean_std([r['shared_period_scan_balanced_accuracy'] for r in members])} |")
    if args.baseline:
        lines += ["", "## Candidate vs baseline", "", *paired_table(rows, args.baseline, args.candidate)]
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
