"""Run Step 9 on private saved validation predictions; never open MRI images.

All locations are restored from existing pointer files in --private-root, so a
fresh PowerShell session does not need remembered variables. Historical model
source, manifests, checkpoints and run folders are read-only. The saved Step 9
pointer is updated only after the result bundle has been generated successfully.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import traceback
from typing import Any
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np

from confidence_analysis import (CAUTION, ConfidenceError, analyse_patients,
                                 require, TEMPERATURE_BOUNDS, CONFIDENCE_GRID)
from dataset import DataError, load_finalized_splits, preprocessing_spec
from training_utils import aggregate_predictions, selection_key
from compare_step8 import validate_reports

NEW_SOURCE = ("confidence_analysis.py", "analyse_confidence.py", "CONFIDENCE_METHOD.md",
              "tests_confidence/test_confidence.py")
PLOTS = ("step9_cnn_reliability.png", "step9_convnext_reliability.png", "step9_risk_coverage.png")
BUNDLE_FILES = ("confidence_summary.json", "operating_rules.json", "README_confidence_results.md", *PLOTS)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError, UnicodeError) as exc:
        raise ConfidenceError(f"Cannot read required JSON file named {path.name}.") from exc
    require(isinstance(value, dict), "Expected a JSON object.")
    return value


def write_json(path: Path, value: Any) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def pointer(private: Path, name: str) -> Path:
    try:
        text = (private / name).read_text(encoding="utf-8-sig").strip()
    except (OSError, UnicodeError) as exc:
        raise ConfidenceError(f"Missing or unreadable saved pointer: {name}.") from exc
    require(bool(text), f"Empty saved pointer: {name}.")
    value = Path(text).resolve()
    require(value.is_dir(), f"Saved directory for {name} is missing.")
    return value


def same_numbers(left: Any, right: Any, context: str) -> None:
    """Reject changed or stale saved aggregates without echoing private records."""
    if isinstance(left, dict) and isinstance(right, dict):
        require(set(left) == set(right), f"Inconsistent saved {context} keys.")
        for key in left:
            same_numbers(left[key], right[key], context)
    elif isinstance(left, list) and isinstance(right, list):
        require(len(left) == len(right), f"Inconsistent saved {context} length.")
        for a, b in zip(left, right):
            same_numbers(a, b, context)
    elif isinstance(left, (int, float)) and isinstance(right, (int, float)):
        require(math.isfinite(float(left)) and math.isfinite(float(right))
                and math.isclose(left, right, abs_tol=1e-9, rel_tol=1e-7),
                f"Saved {context} values do not reproduce.")
    else:
        require(left == right, f"Saved {context} values disagree.")


def verify_cached_predictions(path: Path, records: tuple, report: dict[str, Any]) -> tuple[list, list, list]:
    cached = read_json(path)
    rows = cached.get("slices")
    require(isinstance(rows, list) and len(rows) == len(records), "Cached slice predictions are missing or incomplete.")
    indexed = {}
    for row in rows:
        require(isinstance(row, dict) and isinstance(row.get("image_path"), str), "Invalid cached slice record.")
        key = row["image_path"]
        require(key not in indexed, "Duplicate slice prediction.")
        indexed[key] = row
    require(set(indexed) == {r.image_path for r in records},
            "Cached predictions must contain exactly the finalized validation images, not training/test images.")
    probabilities = []
    for record in records:
        row = indexed[record.image_path]
        require(row.get("subject_id") == record.subject_id and row.get("scan_id") == record.scan_id
                and row.get("label") == record.label, "Cached identifiers/labels disagree with validation manifests.")
        p = row.get("p_ad")
        require(isinstance(p, (int, float)) and math.isfinite(p) and 0 <= p <= 1,
                "A cached probability is invalid.")
        probabilities.append(float(p))
    metrics, regrouped = aggregate_predictions(records, probabilities)
    same_numbers(metrics, report["selected_validation_metrics"], "validation metrics")
    for level, key in (("scans", "scan_id"), ("patients", "subject_id")):
        previous = cached.get(level)
        require(isinstance(previous, list) and all(isinstance(r, dict) and isinstance(r.get(key), str) for r in previous),
                "A saved prediction level is malformed.")
        require(len({r[key] for r in previous}) == len(previous), "Duplicate aggregate prediction.")
        same_numbers(sorted(previous, key=lambda r: r[key]), regrouped[level], "prediction aggregation")
    patients = regrouped["patients"]
    return ([r["subject_id"] for r in patients], [r["label"] for r in patients], [r["p_ad"] for r in patients])


def check_inputs(private: Path) -> dict[str, Any]:
    project = pointer(private, "project_path.txt")
    require(project == Path(__file__).resolve().parent,
            "Run the installed analyse_confidence.py from the saved assignment project, not the tools copy.")
    paths = {"initial": pointer(private, "latest_baseline_run.txt"),
             "baseline": pointer(private, "latest_step8_baseline.txt"),
             "convnext": pointer(private, "latest_step8_convnext.txt"),
             "comparison": pointer(private, "latest_step8_comparison.txt"),
             "splits": pointer(private, "latest_patient_splits.txt")}
    reports = {key: read_json(paths[key] / ("baseline_summary.json" if key == "initial" else "experiment_summary.json"))
               for key in ("initial", "baseline", "convnext")}
    validate_reports(reports["initial"], reports["baseline"], reports["convnext"])
    bundle = load_finalized_splits(private / "data" / "ADNI", paths["splits"])
    for report in reports.values():
        require(report["manifest_sha256"] == bundle.manifest_hashes
                and report["preprocessing"] == preprocessing_spec(0.0), "Run data or preprocessing differs from finalized inputs.")
    for filename, expected in reports["convnext"]["source_code_sha256"].items():
        require(Path(filename).name == filename and sha256(project / filename) == expected,
                "A historical model/training source file changed after Step 8. Stop; do not overwrite it or rerun training.")
    comparison = read_json(paths["comparison"] / "comparison_summary.json")
    require(comparison.get("matched_step8_configuration_verified") is True
            and comparison.get("manifest_sha256") == bundle.manifest_hashes,
            "The saved comparison has not passed for these manifests.")
    for key, name in (("initial", "Initial CNN"), ("baseline", "Regularized CNN"), ("convnext", "ConvNeXt-Tiny")):
        entry = comparison.get("models", {}).get(name, {})
        require(entry.get("selected_epoch") == reports[key]["selected_epoch"], "Saved comparison references a different selected epoch.")
        same_numbers(entry.get("patient_metrics"), reports[key]["selected_validation_metrics"]["patient"], "comparison")
    chosen = "baseline" if selection_key(reports["baseline"]["selected_validation_metrics"]) > selection_key(
        reports["initial"]["selected_validation_metrics"]) else "initial"
    chosen_name = "Regularized CNN" if chosen == "baseline" else "Initial CNN"
    require(comparison.get("preferred_cnn_by_prespecified_validation_rule") == chosen_name,
            "Baseline selection differs from the saved predefined comparison.")
    inputs = {}
    for alias, key in (("cnn", chosen), ("convnext", "convnext")):
        report = reports[key]
        require(report.get("selected_checkpoint_reloaded_and_checked") is True,
                "The selected checkpoint was not successfully verified in the original run.")
        checkpoint = paths[key] / ("PRIVATE_best_baseline.pt" if key == "initial" else "PRIVATE_best_model.pt")
        require(checkpoint.is_file() and sha256(checkpoint) == report.get("selected_checkpoint_sha256"),
                "Selected checkpoint is missing or its bytes differ. Do not restart training; preserve the error.")
        prediction_file = paths[key] / "PRIVATE_validation_predictions.json"
        subjects, labels, p = verify_cached_predictions(prediction_file, bundle.records["validation"], report)
        inputs[alias] = {"name": chosen_name if alias == "cnn" else "ConvNeXt-Tiny", "report": report,
            "subjects": subjects, "labels": labels, "p": p,
            "input_hashes": {"run_summary_sha256": sha256(paths[key] / ("baseline_summary.json" if key == "initial" else "experiment_summary.json")),
                             "cached_predictions_sha256": sha256(prediction_file), "selected_checkpoint_sha256": sha256(checkpoint)}}
    require(inputs["cnn"]["subjects"] == inputs["convnext"]["subjects"]
            and inputs["cnn"]["labels"] == inputs["convnext"]["labels"], "Paired model cohorts disagree.")
    return {"project": project, "manifest_sha256": bundle.manifest_hashes,
            "models": inputs, "comparison_sha256": sha256(paths["comparison"] / "comparison_summary.json")}


def plot_results(output: Path, summary: dict[str, Any]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for alias, report in summary["models"].items():
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
        for key, caption in (("raw_probability_scores", "Raw"), (None, "Cross-fitted temperature")):
            scores = report[key] if key else report["crossfit_diagnostic"]["pooled_probability_scores"]
            bins = [b for b in scores["reliability_bins"] if b["n"]]
            ax.plot([b["mean_predicted_ad_probability"] for b in bins],
                    [b["observed_ad_fraction"] for b in bins], marker="o", label=caption)
        ax.set(xlabel="Mean predicted P(AD) within bin", ylabel="Observed AD fraction",
               xlim=(0, 1), ylim=(0, 1), title=report["name"] + ": development calibration")
        ax.legend(); fig.tight_layout(); fig.savefig(output / f"step9_{alias}_reliability.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5))
    for report in summary["models"].values():
        rows = [r for r in report["raw_risk_coverage_curve"] if r["n_accepted"] > 0]
        rows.sort(key=lambda r: (r["coverage"], -r["threshold"]))
        ax.plot([r["coverage"] for r in rows], [r["accepted_error_rate"] for r in rows],
                marker=".", label=report["name"])
    ax.set(xlabel="Fraction receiving an automated prediction (coverage)",
           ylabel="Error fraction among accepted patients (risk)", xlim=(0, 1), ylim=(0, 1),
           title="Raw confidence ranking: validation development data")
    ax.legend(); fig.tight_layout(); fig.savefig(output / PLOTS[-1], dpi=160); plt.close(fig)


def format_value(value: Any, *, percentage: bool = False) -> str:
    if value is None:
        return "Not estimable"
    return f"{value:.2%}" if percentage else f"{value:.4f}"


def report_markdown(report: dict[str, Any]) -> str:
    lines = ["## Step 9: confidence calibration and referral analysis", "", CAUTION, "",
        "No further neural-network training or inference was performed. Saved validation predictions "
        "were checked against their original metrics, finalized manifests and selected checkpoint hashes. "
        "The patient probability is the mean of scan means of slice P(AD). No final-test images or predictions were used.", "",
        "The preferred CNN from the existing validation comparison was retained; it was not switched "
        "according to these new confidence results. Temperature scaling acts on clipped patient log-odds "
        "with T in [0.5,10]. The diagnostic class cutoff remains 0.5. A single temperature cannot change "
        "these class decisions or improve confidence ordering, apart from numerical clipping/ties. "
        "It does not make an inaccurate model accurate.", "",
        "### Probability calibration", "",
        "NLL and Brier score should be lower for better probability quality. Five-bin AD-probability "
        "ECE is bin-sensitive and imprecise for this small cohort. The full-validation fit is an "
        "in-sample calibration result, not an unbiased improvement estimate. Five-fold cross-fitting "
        "holds each patient's labels out of temperature and cutoff fitting only; checkpoint selection "
        "had already used the complete validation cohort.", "",
        "| Model | Raw NLL | Full-fit NLL | Cross-fit NLL | Raw Brier | Cross-fit Brier | T (full fit) |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for m in report["models"].values():
        raw=m["raw_probability_scores"]; fit=m["full_validation_fit_probability_scores"]
        cf=m["crossfit_diagnostic"]["pooled_probability_scores"]
        lines.append(f"| {m['name']} | {raw['nll']:.4f} | {fit['nll']:.4f} | {cf['nll']:.4f} | {raw['brier']:.4f} | {cf['brier']:.4f} | {m['full_validation_temperature_fit']['temperature']:.4f} |")
    lines += ["", "### Does confidence identify mistakes?", "",
        "Error-detection AUROC uses wrong prediction as the positive class and 1-confidence as its "
        "score. A value of 0.5 is chance ranking; it is different from AD classification AUROC.", "",
        "| Model | Wrong patients | Mean confidence: correct | Mean confidence: wrong | Errors at confidence >=0.90 | Error-detection AUROC |",
        "|---|---:|---:|---:|---:|---:|"]
    for m in report["models"].values():
        d=m["raw_confidence_diagnostics"]
        lines.append(f"| {m['name']} | {d['n_wrong']} | {format_value(d['mean_confidence_correct'])} | {format_value(d['mean_confidence_wrong'])} | {d['n_wrong_high_confidence']} | {format_value(d['error_detection_auroc'])} |")
    lines += ["", "### Proposed referral policy", "",
        "Accept the 0.5-threshold class prediction only when calibrated max(P(AD),1-P(AD)) >= tau; "
        "otherwise refer for review. From the predefined finite grid, select maximum development "
        "coverage subject to at least max(20,ceil(25% of fit patients)) accepted, <=10% observed error "
        "among accepted patients, and <=10% of all AD patients incorrectly auto-cleared as NC. "
        "Break coverage ties by the smaller cutoff. These are exploratory engineering constraints, "
        "NOT clinical safety standards. If no cutoff qualifies, the rule refers everyone.", "",
        "| Model | Full-fit tau | Full-fit coverage | Full-fit accepted error | Cross-fit coverage | Cross-fit accepted error | Cross-fit AD auto-cleared / all AD |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for m in report["models"].values():
        r=m["operating_rule"]; fit=r["development_fit_triage"]; cf=m["crossfit_diagnostic"]["pooled_triage"]
        tau="Refer all" if r["confidence_threshold"] is None else f"{r['confidence_threshold']:.3f}"
        lines.append(f"| {m['name']} | {tau} | {fit['coverage']:.2%} | {format_value(fit['accepted_error_rate'],percentage=True)} | {cf['coverage']:.2%} | {format_value(cf['accepted_error_rate'],percentage=True)} | {cf['n_ad_automatically_cleared_as_nc']}/{cf['n_ad_total']} |")
    lines += ["", "Referred cases are not counted as correct diagnoses. 'AD not automatically cleared' "
        "is a routing measure, not diagnostic sensitivity or an assumption of perfect human review. "
        "Accuracy within an accepted subset is NOT all-patient test accuracy and does not demonstrate "
        "the assessment's 0.80 target. Descriptive Wilson intervals are in the JSON summary; they "
        "do not account for selection or cross-fit dependence.", "",
        "### Status and next decision", "",
        "The full-validation temperatures and referral rules are saved for a subsequent, explicitly "
        "frozen evaluation plan. The final test has NOT been run. These diagnostics must be reviewed "
        "before any claim of an operational benefit or any final evaluation. No new training run was "
        "chosen automatically on the basis of these results.", "",
        "### Assistance and reproducibility", "",
        "ChatGPT assisted with implementation, synthetic tests and this generated report draft. "
        "Document the review and local checks actually performed in the Artificial Intelligence Usage "
        "Disclosure; do not claim clinical validation. Package/source hashes and input hashes are "
        "recorded in the aggregate summary. Method reference: Guo et al., On Calibration of Modern "
        "Neural Networks, ICML 2017, https://arxiv.org/abs/1706.04599 .", ""]
    for filename in PLOTS:
        lines.append(f"![Development confidence analysis](figures/{filename})\n")
    return "\n".join(lines)


def run(private: Path) -> Path:
    private = private.resolve()
    require(private.is_dir(), "Private root does not exist.")
    require(not any((p / ".git").exists() for p in (private, *private.parents)),
            "Private outputs must be outside a Git repository.")
    inputs = check_inputs(private)
    output = private / ("confidence-run-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f"))
    output.mkdir()
    report: dict[str, Any] = {"format_version": 1, "status": "RUNNING", "step9_completed": False,
        "created_utc": datetime.now(timezone.utc).isoformat(), "models": {}, "errors": [],
        "neural_network_training_performed": False, "additional_model_inference_performed": False,
        "mri_images_opened": 0, "final_test_model_evaluation_performed": False,
        "final_test_images_decoded": 0, "dataset_splits_checkpoints_or_training_sources_modified": False,
        "evaluation_scope": "Patient-level validation development only; prior checkpoint-selection reuse acknowledged",
        "manifest_sha256": inputs["manifest_sha256"], "comparison_sha256": inputs["comparison_sha256"],
        "source_code_sha256": {n: sha256(inputs["project"] / n) for n in NEW_SOURCE},
        "caution": CAUTION}
    try:
        private_rows = {}
        policies = {"format_version": 1, "status": "VALIDATION_FITTED_CANDIDATES_REVIEW_BEFORE_FINAL_TEST",
                    "manifest_sha256": inputs["manifest_sha256"], "models": {}, "clinical_validation": False,
                    "patient_probability_aggregation": "mean of scan means of slice P(AD)",
                    "input_probability_clip": [1e-7, 1 - 1e-7], "caution": CAUTION}
        for alias, values in inputs["models"].items():
            print(f"Analysing cached validation predictions: {values['name']}", flush=True)
            analyzed, individual = analyse_patients(values["subjects"], values["labels"], values["p"])
            analyzed.update(name=values["name"], selected_epoch=values["report"]["selected_epoch"],
                            checkpoint_sha256=values["input_hashes"]["selected_checkpoint_sha256"],
                            input_hashes=values["input_hashes"],
                            original_validation_patient_metrics=values["report"]["selected_validation_metrics"]["patient"])
            report["models"][alias] = analyzed
            private_rows[alias] = individual
            policies["models"][alias] = {"name": values["name"], "selected_epoch": analyzed["selected_epoch"],
                "checkpoint_sha256": analyzed["checkpoint_sha256"], "temperature": analyzed["full_validation_temperature_fit"]["temperature"],
                "diagnostic_threshold": 0.5, "confidence_threshold": analyzed["operating_rule"]["confidence_threshold"],
                "refer_all": analyzed["operating_rule"]["confidence_threshold"] is None,
                "fit_partition": "validation", "fit_patients": analyzed["patient_n"]}
        write_json(output / "PRIVATE_patient_diagnostics.json", private_rows)
        write_json(output / "operating_rules.json", policies)
        plot_results(output, report)
        (output / "README_confidence_results.md").write_text(report_markdown(report), encoding="utf-8")
        report.update(status="COMPLETED", step9_completed=True)
        write_json(output / "confidence_summary.json", report)
        with ZipFile(output / "COMP3710_Step9_Results.zip", "w", ZIP_DEFLATED) as archive:
            for name in BUNDLE_FILES:
                archive.write(output / name, arcname=name)
        pointer_file = private / "latest_step9_confidence.txt"
        temp = pointer_file.with_suffix(".tmp")
        temp.write_text(str(output) + "\n", encoding="utf-8")
        os.replace(temp, pointer_file)
    except BaseException as exc:
        report.update(status="FAILED", step9_completed=False,
                      errors=[{"type": type(exc).__name__, "message": str(exc) if isinstance(exc, (ConfidenceError, DataError)) else "See local PRIVATE_traceback.txt."}])
        write_json(output / "confidence_summary.json", report)
        (output / "PRIVATE_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print(f"Preserve output directory: {output}", file=sys.stderr)
        raise
    print("STEP9_CONFIDENCE_ANALYSIS_COMPLETED")
    print("VALIDATION_DIAGNOSTICS_ONLY_FINAL_TEST_UNTOUCHED")
    print(f"Results: {output}")
    print(f"Upload only: {output / 'COMP3710_Step9_Results.zip'}")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.private_root)
    except (Exception, KeyboardInterrupt) as exc:
        print(f"STEP9_STOPPED: {str(exc) if isinstance(exc, (ConfidenceError, DataError)) else type(exc).__name__}. No training restart required.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
