"""Reproduce report tables and selected SVGs without model inference.

Main runs are baseline/siamese for seed 42 and <model>_seed<seed> for
43, 44, 67 and 69; ablation runs are scratch and siamese_light_seed<seed>,
and profile_<model> holds the one-epoch memory profile. Stored decisions
define binary negatives. Run this module to print Markdown table rows, use
--json for named row groups, or pass --figures to also redraw selected SVGs.

Author: Alex Anderson (47472442), COMP3710 Pattern Recognition, 2026.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn.metrics import roc_auc_score

from .compare import read_predictions
from .dataset import DATA_DIR, read_samples
from .visualisation import plot_case_comparison, plot_curves

MODELS = ("baseline", "siamese")
SEEDS = (42, 43, 44, 67, 69)
ABLATIONS = {  # report name: run directories
    "Baseline, ImageNet start, λ = 0": [f"baseline_seed{s}" if s != 42 else "baseline" for s in (42, 43, 44)],
    "Siamese, ImageNet start, λ = 0.5": [f"siamese_seed{s}" if s != 42 else "siamese" for s in (42, 43, 44)],
    "Light Siamese, ImageNet start, λ = 0.25": [f"siamese_light_seed{s}" for s in (42, 43, 44)],
    "Baseline, random start, λ = 0, 15 epochs": ["scratch"],
}
HEADERS = {
    "benchmark_rows": "Final measure|Baseline|Siamese",
    "embedding_rows": "Final embedding pairs|Baseline|Siamese",
    "ablation_rows": "Final model|Seeds|AUROC|Melanoma AP|Accuracy|Melanoma recall|Workload (%)|Capture (%)",
    "confusion_rows": "Seed-42 model|TN|FP|FN|TP",
    "uncertainty_rows": "Error-ranking AUROC|Baseline|Siamese",
    "review_comparison_rows": "Calibration workload (%)|Probability-only review|Reference-only review",
    "final_policy_rows": "Model|Fixed final policy|Workload (%)|Melanoma capture (%)",
    "resource_rows": "Measurement|Baseline|Siamese",
}


def run_path(runs: Path, model: str, seed: int) -> Path:
    """Return the saved run directory for one model and seed.

    Parameters
    ----------
    runs : pathlib.Path
        Parent directory containing the saved runs.
    model : {'baseline', 'siamese'}
        Model name used in the run directory.
    seed : int
        Training seed. Seed 42 uses the bare model name; other seeds use
        the model name followed by _seed and the seed number.

    Returns
    -------
    pathlib.Path
        Run directory path, without checking that it exists.
    """
    return runs / (model if seed == 42 else f"{model}_seed{seed}")


def mean_sd(values: list[float], digits: int = 3, scale: float = 1) -> str:
    """Format a mean and sample standard deviation over trained seeds.

    Parameters
    ----------
    values : list of float
        Measurements from trained seeds. A single value is shown alone.
    digits : int, optional
        Decimal places for both values. Default is 3.
    scale : float, optional
        Multiplier applied before summarising. Default is 1; use 100 to
        express fractions as percentages.

    Returns
    -------
    str
        Formatted mean and sample standard deviation separated by ±.
    """
    array = np.asarray(values) * scale
    if len(array) == 1:
        return f"{array[0]:.{digits}f}"
    return f"{array.mean():.{digits}f} ± {array.std(ddof=1):.{digits}f}"


def table_row(cells: list[str]) -> str:
    """Join cells into one Markdown table row.

    Parameters
    ----------
    cells : list of str
        Cell text in column order.

    Returns
    -------
    str
        Pipe-delimited Markdown row.
    """
    return "| " + " | ".join(cells) + " |"


def read_json(path: Path) -> dict[str, Any]:
    """Read one saved JSON metrics file.

    Parameters
    ----------
    path : pathlib.Path
        JSON file written by training or prediction.

    Returns
    -------
    dict
        Parsed metrics.
    """
    return json.loads(path.read_text())


def report_tables(runs: Path) -> dict[str, str]:
    """Calculate report table rows from saved metrics and predictions.

    Parameters
    ----------
    runs : pathlib.Path
        Directory containing the ten named baseline and Siamese runs.

    Returns
    -------
    dict of str to str
        Named Markdown table rows from full-precision saved results.

    Notes
    -----
    All final prediction files must have matching image identifiers,
    patient identifiers and labels in the same order. Computational cost
    and confusion counts describe seed 42; aggregate rows use five seeds.
    """
    final, calibration = {}, {}
    for m in MODELS:
        final[m] = [read_json(run_path(runs, m, s) / "final/metrics.json") for s in SEEDS]
        calibration[m] = [read_json(run_path(runs, m, s) / "calibration/metrics.json") for s in SEEDS]
    training = {m: read_json(run_path(runs, m, 42) / "metrics.json") for m in MODELS}
    profile = {m: read_json(runs / f"profile_{m}" / "metrics.json") for m in MODELS}
    tables: dict[str, str] = {}
    rows = []
    for name, parent, key in (
        ("AUROC", "classifier", "auroc"),
        ("Melanoma AP (PR-AUC)", "classifier", "average_precision"),
        ("Decision accuracy", "decision", "accuracy"),
        ("Macro-F1", "decision", "macro_f1"),
        ("Benign precision", "decision", "benign_precision"),
        ("Benign recall", "decision", "benign_recall"),
        ("Benign F1", "decision", "benign_f1"),
        ("Melanoma precision", "decision", "melanoma_precision"),
        ("Melanoma recall", "decision", "melanoma_recall"),
        ("Melanoma F1", "decision", "melanoma_f1"),
        ("Reference-score AUROC", "reference", "auroc"),
        ("Brier score", "classifier", "brier"),
        ("Images sent to review, %", "triage", "workload"),
        ("Melanomas reaching review, %", "triage", "melanoma_capture"),
    ):
        cells = []
        for model in MODELS:
            values = []
            for report in final[model]:
                d = report["decision"]
                benign_f1 = 2 * d["tn"] / (2 * d["tn"] + d["fp"] + d["fn"])
                values.append(benign_f1 if key == "benign_f1" else report[parent][key])
            if parent == "triage":
                cells.append(mean_sd(values, 1, 100))  # percentages
            else:
                cells.append(mean_sd(values, 4 if key == "brier" else 3))
        rows.append(table_row([name, *cells]))
    tables["benchmark_rows"] = "\n".join(rows)
    rows = []
    for name, key in (
        ("Mean distance, benign–benign", "benign_benign"),
        ("Mean distance, melanoma–benign", "mixed"),
        ("Mean distance, melanoma–melanoma", "melanoma_melanoma"),
        ("Melanoma pair AUROC (same class closer)", "melanoma_pair_auroc"),
    ):
        cells = []
        for m in MODELS:
            cells.append(mean_sd([r["embedding_pairs"][key] for r in final[m]]))
        rows.append(table_row([name, *cells]))
    tables["embedding_rows"] = "\n".join(rows)
    rows = []
    for name, run_names in ABLATIONS.items():
        reports = [read_json(runs / n / "final/metrics.json") for n in run_names]
        cells = []
        for parent, key, digits, scale in (
            ("classifier", "auroc", 3, 1), ("classifier", "average_precision", 3, 1),
            ("decision", "accuracy", 3, 1), ("decision", "melanoma_recall", 3, 1),
            ("triage", "workload", 1, 100), ("triage", "melanoma_capture", 1, 100),
        ):
            cells.append(mean_sd([r[parent][key] for r in reports], digits, scale))
        rows.append(table_row([name, str(len(run_names)), *cells]))
    tables["ablation_rows"] = "\n".join(rows)
    rows = []
    for m in MODELS:
        counts = [str(final[m][0]["decision"][k]) for k in ("tn", "fp", "fn", "tp")]
        rows.append(table_row([m.capitalize(), *counts]))
    tables["confusion_rows"] = "\n".join(rows)

    predictions = {}
    for m in MODELS:
        predictions[m] = [read_predictions(run_path(runs, m, s) / "final/predictions.csv") for s in SEEDS]
    identity = predictions["baseline"][0]
    for group in predictions.values():
        for prediction in group:
            assert all(np.array_equal(prediction[k], identity[k]) for k in ("image_name", "patient_id", "label"))
    rows = []
    for name, key in (
        ("All binary errors: classifier uncertainty", "confidence"),
        ("All binary errors: reference disagreement", "reference_disagreement"),
    ):
        cells = []
        for m in MODELS:
            cells.append(mean_sd([r["error_detection_auroc"][key] for r in final[m]]))
        rows.append(table_row([name, *cells]))
    for name, key in (
        ("Missed melanomas among binary negatives: reference score", "reference_score"),
        ("Missed melanomas among binary negatives: probability", "probability"),
    ):
        cells = []
        for model in MODELS:
            values = []
            for p in predictions[model]:
                negative = p["decision"] != "melanoma"
                values.append(float(roc_auc_score(p["label"][negative], p[key][negative])))
            cells.append(mean_sd(values))
        rows.append(table_row([name, *cells]))
    tables["uncertainty_rows"] = "\n".join(rows)
    rows = []
    for m in MODELS:
        cells = []
        for policy in ("classifier_review", "reference_review"):
            cells.append(mean_sd([r[policy]["flagged_fraction"] for r in calibration[m]], 1, 100))
        rows.append(table_row([m.capitalize(), *cells]))
    tables["review_comparison_rows"] = "\n".join(rows)
    rows = []
    for model in MODELS:
        for name, parent, workload, capture in (
            ("Classifier flags only", "decision", "flagged_fraction", "melanoma_recall"),
            ("Probability-only review", "classifier_review", "flagged_fraction", "melanoma_recall"),
            ("Reference-only review", "reference_review", "flagged_fraction", "melanoma_recall"),
            ("Classifier flags + reference review", "triage", "workload", "melanoma_capture"),
        ):
            rows.append(table_row([
                model.capitalize(), name,
                mean_sd([r[parent][workload] for r in final[model]], 1, 100),
                mean_sd([r[parent][capture] for r in final[model]], 1, 100),
            ]))
    tables["final_policy_rows"] = "\n".join(rows)

    rows = []
    scoring = {m: final[m][0] for m in MODELS}
    for name, source, key, scale, digits in (
        ("Parameters", training, "parameters", 1, 0),
        ("Training, minutes", training, "training_hours", 60, 1),
        ("Peak MPS driver allocation, training (GiB)", profile, "peak_gpu_memory_gib", 1, 2),
        ("MPS driver allocation after inference benchmark (GiB)", scoring, "peak_gpu_memory_gib", 1, 2),
        ("Peak main-process RSS during training (GiB)", training, "peak_process_memory_gib", 1, 2),
        ("Forward latency, 32-image batch (ms)", scoring, "batch_latency_ms", 1, 1),
        ("Image-scoring throughput (images/s)", scoring, "images_per_second_end_to_end", 1, 1),
    ):
        values = [source[m]["profile"][key] for m in MODELS]
        cells = [f"{v:,}" if name == "Parameters" else f"{v * scale:.{digits}f}" for v in values]
        rows.append(table_row([name, *cells]))
    tables["resource_rows"] = "\n".join(rows)
    return tables


def redraw_figures(runs: Path, output: Path) -> None:
    """Replot seed-42 scores and matched crops without loading weights.

    Parameters
    ----------
    runs : pathlib.Path
        Parent directory containing the baseline and Siamese seed-42 runs.
    output : pathlib.Path
        Destination directory for report SVGs. Matching files are replaced;
        saved run files are unchanged.

    Notes
    -----
    Redraws score curves and five matched failure examples, including a
    melanoma labelled benign by both complete review policies. Image crops
    use the image size from configs/baseline.yaml and the original JPEGs.
    """
    predictions = {m: read_predictions(runs / m / "final/predictions.csv") for m in MODELS}
    training = {m: json.loads((runs / m / "metrics.json").read_text()) for m in MODELS}
    for model in MODELS:
        p = predictions[model]
        plot_curves(
            p["label"], p["probability"], p["reference_score"], training[model]["rule"],
            output / f"final_curves_{model}.svg", f"{model.capitalize()} · seed 42 · final",
        )
    baseline = predictions["baseline"]
    order = np.argsort(baseline["probability"])
    missed_by_both = [
        i for i in order
        if baseline["label"][i] == 1 and all(p["decision"][i] == "benign" for p in predictions.values())
    ]
    chosen = (
        [i for i in order if baseline["label"][i] == 1][:2]
        + [i for i in order[::-1] if baseline["label"][i] == 0][:2]
        + missed_by_both[:1]
    )
    samples = {sample.image_name: sample for sample in read_samples()}
    config = yaml.safe_load((Path(__file__).parent / "configs/baseline.yaml").read_text())
    for number, indices in enumerate((chosen[:2], chosen[2:4], chosen[4:]), 1):
        cases = [samples[str(baseline["image_name"][i])] for i in indices]
        values = {}
        for model, prediction in predictions.items():
            values[model] = {key: array[indices] for key, array in prediction.items()}
        plot_case_comparison(cases, values, config["image_size"], output / f"final_cases_comparison_{number}.svg")


def main() -> None:
    """Print saved-result tables and optionally redraw report figures.

    Read run and figure directories and the JSON output option from the
    command line. Default output is Markdown; no model weights are loaded.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=DATA_DIR / "runs")
    parser.add_argument("--figures", type=Path, help="also regenerate selected report SVGs from saved data")
    parser.add_argument("--json", action="store_true", help="print named Markdown table rows as JSON")
    args = parser.parse_args()
    tables = report_tables(args.runs)
    if args.json:
        print(json.dumps(tables, indent=2))
    else:
        for name, rows in tables.items():
            columns = HEADERS[name].split("|")
            print(f"\n## {name.removesuffix('_rows').replace('_', ' ').title()}\n")
            print("| " + " | ".join(columns) + " |")
            print("| " + " | ".join("---" for _ in columns) + " |")
            print(rows)
    if args.figures:
        redraw_figures(args.runs, args.figures)


if __name__ == "__main__":
    main()
