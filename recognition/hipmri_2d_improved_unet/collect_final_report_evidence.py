"""Collect report facts from saved JSON only; never execute a model or read images."""

import argparse
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics


PROJECT = Path(__file__).resolve().parent
EVIDENCE = PROJECT / "evidence"
MODELS = ("baseline_unet", "improved_unet")
EXPECTED_EPOCHS = {"baseline_unet": 5, "improved_unet": 17}
SOURCES = {
    "id5_plan": "id5_run_plan_20261026.json",
    "id5": "id5_comparison_20261026.json",
    "baseline_history": "id5_baseline_20261026.json",
    "improved_history": "id5_improved_20261026.json",
    "old_ce": "formal_comparison_20261026.json",
    "diagnosis": "selection_diagnosis_20261026.json",
    "test": "frozen_test_comparison_20261026.json",
    "test_plan": "frozen_test_plan_20261026.json",
    "recipe": "frozen_test_recipe_20261026.json",
    "attempt": "frozen_test_attempt_20261026.json",
    "baseline_test": "frozen_test_baseline_20261026.json",
    "improved_test": "frozen_test_improved_20261026.json",
    "failure": "frozen_test_failure_20261026.json",
    "io_fix": "frozen_test_io_fix_20261026.json",
    "approval": "frozen_test_recovery_approval_20261026.json",
    "recovery_plan": "frozen_test_recovery_plan_20261026.json",
    "recovery": "frozen_test_recovery_20261026.json",
    "baseline_profile": "id5_baseline_profile_20261026.json",
    "improved_profile": "id5_improved_profile_20261026.json",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    path = Path(path)
    require(path.suffix in (".json", ".py"), "Only saved JSON and this Python source may be hashed")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    def reject(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject)


def finite(value, label, minimum=None):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value), f"{label}: expected a finite number")
    require(minimum is None or value >= minimum, f"{label}: below allowed minimum")
    return value


def equal_score(actual, expected, label):
    if expected is None:
        require(actual is None, f"{label}: expected undefined score")
    else:
        finite(actual, label)
        require(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12),
                f"{label}: score differs from saved confusion counts")


def verify_metrics(metrics, label):
    matrix = metrics["confusion_matrix"]
    require(len(matrix) == 6 and all(len(row) == 6 for row in matrix), f"{label}: expected 6x6 confusion")
    require(all(isinstance(v, int) and not isinstance(v, bool) and v >= 0
                for row in matrix for v in row), f"{label}: invalid confusion count")
    total = sum(map(sum, matrix))
    require(total > 0 and total == metrics["valid_pixels"], f"{label}: invalid pixel total")
    rows = [sum(row) for row in matrix]
    columns = [sum(matrix[r][c] for r in range(6)) for c in range(6)]
    dice, iou = [], []
    for c in range(6):
        denominator = rows[c] + columns[c]
        dice.append(2 * matrix[c][c] / denominator if denominator else None)
        union = denominator - matrix[c][c]
        iou.append(matrix[c][c] / union if union else None)
        equal_score(metrics["dice_per_class"][c], dice[-1], f"{label} ID{c} Dice")
        equal_score(metrics["iou_per_class"][c], iou[-1], f"{label} ID{c} IoU")
    for key, values in (("mean_dice_excluding_class0", dice), ("mean_iou_excluding_class0", iou)):
        defined = [value for value in values[1:] if value is not None]
        equal_score(metrics[key], sum(defined) / len(defined) if defined else None, f"{label} {key}")
    equal_score(metrics["pixel_accuracy"], sum(matrix[c][c] for c in range(6)) / total, f"{label} accuracy")
    finite(metrics["cross_entropy"], f"{label} saved CE", 0)
    tp, reference, predicted = matrix[5][5], rows[5], columns[5]
    return {
        "dice_per_class": dice, "iou_per_class": iou,
        "macro_dice_excluding_id0": metrics["mean_dice_excluding_class0"],
        "macro_iou_excluding_id0": metrics["mean_iou_excluding_class0"],
        "pixel_accuracy": metrics["pixel_accuracy"], "valid_pixels": total,
        "cross_entropy_saved_shared_loss": metrics["cross_entropy"],
        "reference_support_per_class": rows,
        "id5": {"tp": tp, "fp": predicted - tp, "fn": reference - tp,
                "reference_pixels": reference, "predicted_pixels": predicted,
                "dice": dice[5], "iou": iou[5],
                "precision": tp / predicted if predicted else None,
                "recall": tp / reference if reference else None},
        "verification": "CM-derived overlap, macro and accuracy match within1e-12; CE only checked finite and retained from the saved complete shared-loss pass.",
    }


def comparison(baseline, improved):
    finite(baseline, "baseline comparison value", 0)
    finite(improved, "Improved comparison value", 0)
    return {"baseline": baseline, "improved": improved,
            "difference_improved_minus_baseline": improved - baseline,
            "ratio_improved_over_baseline": improved / baseline if baseline else None,
            "relative_change_percent": 100 * (improved - baseline) / baseline if baseline else None}


def score_comparison(baseline, improved):
    result = comparison(baseline, improved)
    result["difference_percentage_points"] = 100 * (improved - baseline)
    return result


def linear_percentile(values, quantile):
    values = sorted(values)
    index = (len(values) - 1) * quantile
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def profiles(documents, selected):
    all_profiles = [documents[tag + "_profile"] for tag in ("baseline", "improved")]
    require(all_profiles[0]["settings"] == all_profiles[1]["settings"], "Profile methods differ")
    rows = []
    for index, (batch, width) in enumerate((b, w) for b in (1, 4) for w in (128, 144)):
        values = {}
        for model, profile in zip(MODELS, all_profiles):
            require(profile["status"] == "passed" and profile["model"] == model, "Incomplete/wrong profile")
            require(profile["checkpoint_epoch"] == EXPECTED_EPOCHS[model]
                    and profile["checkpoint_sha256"] == selected[model]["checkpoint_sha256"], "Profile checkpoint binding differs")
            require(profile["parameters"] == selected[model]["parameters"], "Profile parameter count differs")
            require(len(profile["results"]) == 4, "Expected all four profile shapes")
            item = profile["results"][index]
            require(item["input_shape"] == [batch, 1, 256, width]
                    and item["output_shape"] == [batch, 6, 256, width], "Profile shape/order differs")
            require((item["warmup_iterations"], item["timed_iterations"]) == (10, 50), "Profile repetition count differs")
            raw = item["raw_batch_times_ms"]
            require(len(raw) == 50 and all(finite(t, "raw profile time", 0) > 0 for t in raw), "Invalid raw times")
            median, p95 = statistics.median(raw), linear_percentile(raw, .95)
            equal_score(item["median_batch_ms"], median, "Profile median")
            equal_score(item["p95_batch_ms"], p95, "Profile p95")
            equal_score(item["median_ms_per_slice"], median / batch, "Profile per-slice time")
            stage = item["stage_measurement"]
            require(stage["status"] == "passed", "Profile resource stage incomplete")
            values[model] = {"median_batch_ms": item["median_batch_ms"], "p95_batch_ms": item["p95_batch_ms"],
                             "median_ms_per_slice": item["median_ms_per_slice"],
                             "peak_allocated_bytes": stage["peak_cuda_allocated_bytes"],
                             "peak_reserved_bytes": stage["peak_cuda_reserved_bytes"],
                             "median_recomputed": median, "p95_linear_recomputed": p95}
        rows.append({"input_shape": [batch, 1, 256, width], "models": values,
                     **{key + "_comparison": comparison(values[MODELS[0]][key], values[MODELS[1]][key])
                        for key in ("median_batch_ms", "p95_batch_ms", "median_ms_per_slice",
                                    "peak_allocated_bytes", "peak_reserved_bytes")}})
    return {"settings": all_profiles[0]["settings"], "combinations": rows,
            "scope": "Same fixed synthetic method and shapes, 10 warmups/50 timed forwards; these are existing local snapshots, not isolated fixed-clock or statistical speed estimates. Batch milliseconds are not single-slice milliseconds."}


def verify_history(history, chosen, name):
    require(history["status"] == "passed" and history["completed_epochs"] == 20
            and history["optimizer_steps"] == 57300, f"{name}: incomplete training budget")
    records = history["epochs"]
    require([row["epoch"] for row in records] == list(range(1, 21)), "History epochs differ")
    for record in records:
        for split, samples, batches, updates in (("training", 11460, 2865, 2865), ("validation", 660, 165, 0)):
            metrics = record[split]
            require((metrics["samples"], metrics["batches"], metrics["optimizer_steps"])
                    == (samples, batches, updates), f"{name} {split}: budget differs")
            verify_metrics(metrics, f"{name} epoch{record['epoch']} {split}")
    by_id5 = min(records, key=lambda r: (-r["validation"]["dice_per_class"][5], r["validation"]["cross_entropy"], r["epoch"]))
    by_ce = min(records, key=lambda r: (r["validation"]["cross_entropy"], r["epoch"]))
    require(by_id5["epoch"] == EXPECTED_EPOCHS[name] == chosen["selected_epoch"], "Primary selection differs")
    require(chosen["selected_validation"] == by_id5["validation"], "Selected validation differs from full history")
    return {"epochs_verified": 20, "complete_updates": 57300, "id5_best_epoch": by_id5["epoch"],
            "ce_best_epoch": by_ce["epoch"], "last_epoch": records[-1]["epoch"],
            "all_training_and_validation_confusion_scores_verified": True}


def collect(documents, hashes):
    d, plan, test = documents["id5"], documents["id5_plan"], documents["test"]
    require(d["formal_plan_sha256"] == hashes["id5_plan"], "ID5 plan binding differs")
    require(plan["original_formal_comparison_sha256"] == hashes["old_ce"], "Old CE experiment binding differs")
    require(documents["diagnosis"]["formal_comparison_sha256"] == hashes["old_ce"], "Old diagnosis binding differs")
    require(test["recipe_sha256"] == hashes["recipe"]
            and test["recovery_authorization_sha256"] == hashes["approval"], "Test recipe/approval binding differs")
    require(test["inputs"]["original_plan"]["sha256"] == hashes["test_plan"]
            and test["inputs"]["recovery_plan"]["sha256"] == hashes["recovery_plan"]
            and test["inputs"]["recovery_report"]["sha256"] == hashes["recovery"], "Test input chain differs")
    require(test["status"] == "passed_verification" and test["test_scores_used_for_tuning"] is False, "Test not verified or used for tuning")
    histories_pair = [documents[tag + "_history"] for tag in ("baseline", "improved")]
    for key in ("model_config", "optimizer", "seed", "settings", "objective", "math_settings", "versions", "data"):
        require(histories_pair[0][key] == histories_pair[1][key], f"Paired training control differs: {key}")
    for history in histories_pair:
        protocol = plan["shared_protocol"]
        require(history["seed"] == protocol["seed"] == 3710
                and history["model_config"] == protocol["model_config"], "Training seed/model configuration differs from plan")
        require(history["optimizer"] == {"name": "Adam", "lr": protocol["optimizer"]["learning_rate"]},
                "Training optimizer differs from plan")
        require(history["settings"] == {"epochs": 20, "batch_size": 4, "num_workers": 0,
                                        "cudnn_deterministic": True, "cudnn_benchmark": False},
                "Training runtime controls differ from the fixed protocol")
        require(history["source_sha256"] == plan["source_sha256"], "Recorded training source hashes differ")
        for split in ("training", "validation"):
            require(history["data"][split + "_manifest_sha256"] == plan["data"][split]["manifest_sha256"],
                    f"Training {split} manifest binding differs")
    selected, validation, testing, histories = {}, {}, {}, {}
    for name, tag in zip(MODELS, ("baseline", "improved")):
        chosen, tested = d["results"][name], test["results"][name]
        checkpoint_sha = chosen["checkpoint_verification"]["id5_best"]["sha256"]
        require(chosen["selected_epoch"] == tested["checkpoint_epoch"] == EXPECTED_EPOCHS[name]
                and tested["checkpoint_sha256"] == checkpoint_sha, "Validation/test frozen checkpoint differs")
        require(tested["parameters"] == chosen["parameters"], "Validation/test parameter count differs")
        require(tested["state_sha256_before"] == tested["state_sha256_after"]
                and tested["optimizer_steps"] == 0, "Completed test changed state/parameters")
        selected[name] = {"epoch": chosen["selected_epoch"], "checkpoint_sha256": checkpoint_sha,
                          "parameters": chosen["parameters"]}
        validation[name] = verify_metrics(chosen["selected_validation"], f"{name} selected validation")
        testing[name] = verify_metrics(tested["metrics"], f"{name} completed test")
        require((tested["metrics"]["samples"], tested["metrics"]["batches"], tested["metrics"]["valid_pixels"])
                == (540, 135, 17694720), "Completed test counts differ")
        histories[name] = verify_history(documents[tag + "_history"], chosen, name)
    require(histories[MODELS[0]]["ce_best_epoch"] == 5 and histories[MODELS[1]]["ce_best_epoch"] == 3, "CE-best epochs differ")
    require(testing[MODELS[0]]["reference_support_per_class"] == testing[MODELS[1]]["reference_support_per_class"], "Models saw different test references")
    metric_changes = {}
    for split, values in (("validation", validation), ("test", testing)):
        baseline, improved = (values[name] for name in MODELS)
        metric_changes[split] = {"id5_dice": score_comparison(baseline["id5"]["dice"], improved["id5"]["dice"]),
                                 "id5_iou": score_comparison(baseline["id5"]["iou"], improved["id5"]["iou"]),
                                 "macro_dice": score_comparison(baseline["macro_dice_excluding_id0"], improved["macro_dice_excluding_id0"]),
                                 "cross_entropy": comparison(baseline["cross_entropy_saved_shared_loss"], improved["cross_entropy_saved_shared_loss"])}
    resources = {"parameters": comparison(*(selected[name]["parameters"] for name in MODELS)), "phases": {}}
    for phase in ("training", "validation", "checkpoint_restoration"):
        stages = [d["results"][name]["phase_resource_summary"][phase] for name in MODELS]
        resources["phases"][phase] = {
            key: {**comparison(*(stage[key] for stage in stages)),
                  "baseline_MiB": stages[0][key] / 1048576, "improved_MiB": stages[1][key] / 1048576}
            for key in ("peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes")}
    resources["inference_profiles"] = profiles(documents, selected)
    resources["observed_training_wall_seconds"] = comparison(*(d["results"][name]["run_elapsed_seconds"] for name in MODELS))
    resources["wall_clock_scope"] = "Run wall time includes I/O, system/cache/load and checkpoint work. Test stages additionally used different logging implementations. Neither run nor test wall-time ratios isolate architecture speed."
    diagnosis = documents["diagnosis"]
    require([diagnosis["models"][tag]["checkpoint_epoch"] for tag in ("baseline_best", "improved_best", "improved_last")] == [5, 3, 20], "Historical failure checkpoint scope differs")
    failures = []
    for case in diagnosis["failure_cases"]:
        require(case["status"] == "selected", "Missing historical case")
        per_model = {}
        for tag, epoch in (("baseline_best", 5), ("improved_best", 3), ("improved_last", 20)):
            item = case["models"][tag]
            tp, fp, fn = item["tp"], item["fp"], item["fn"]
            denominator = 2 * tp + fp + fn
            equal_score(item["dice"], 2 * tp / denominator if denominator else None, "Historical case Dice")
            per_model[tag] = {"epoch": epoch, **{key: item[key] for key in ("tp", "fp", "fn", "reference_pixels", "dice")}}
        failures.append({"case_label": case["case_label"], "filename": case["filename"],
                         "selection_rule": case["selection_rule"], "models": per_model,
                         "existing_figure_relative_to_repo": diagnosis["output_directory"] + "/" + case["figure"]["image_filename"]})
    require(len(failures) == 5, "Expected five historical validation examples")
    attempt, baseline, recovered = documents["attempt"], documents["baseline_test"], documents["recovery"]
    require(attempt["status"] == "incomplete" and attempt["models"]["baseline_unet"] == baseline
            and recovered["preserved_baseline_result"] == baseline, "Original baseline was not preserved")
    partial = attempt["models"]["improved_unet"]
    require((partial["forward_calls"], partial["forward_samples"]) == (111, 444)
            and "metrics" not in partial, "Original incomplete exposure differs")
    require(recovered["status"] == "passed" and set(recovered["models"]) == {"improved_unet"}
            and recovered["models"]["improved_unet"] == documents["improved_test"], "Recovery is not Improved-only complete")
    require(recovered["baseline_forward_calls_this_recovery"] == 0, "Baseline was rerun")
    require(documents["approval"]["user_answer"] == "允许，仅补跑 Improved"
            and documents["approval"]["additional_full_evaluations_authorized"] == {"baseline_unet": 0, "improved_unet": 1}, "Recovery authorization differs")
    exposure = test["test_exposure"]
    require((exposure["improved"]["cumulative_improved_forward_calls"], exposure["improved"]["cumulative_improved_forward_samples"],
             exposure["improved"]["repeated_prior_samples"], exposure["baseline"]["recovery_forward_calls"])
            == (246, 984, 444, 0), "Final test exposure count differs")
    old_ce = {name: {key: documents["old_ce"]["results"][name][key]
                    for key in ("selected_epoch", "validation_id5_dice_pixel_pooled", "validation_cross_entropy")}
              for name in MODELS}
    return {
        "selected_models": selected, "shared_training_protocol": deepcopy(plan["shared_protocol"]),
        "training_budget": deepcopy(plan["training_budget"]), "data": deepcopy(test["data"]),
        "preprocessing": deepcopy(test["preprocessing"]), "frozen_evaluation": deepcopy(test["evaluation"]),
        "verified_training_histories": histories, "validation": validation, "test": testing,
        "metric_changes_improved_minus_baseline": metric_changes, "resources": resources,
        "old_ce_selected_validation_experiment": old_ce,
        "historical_validation_failures": {"scope": "Existing original CE experiment baseline5/Improved3/last20; not per-case evidence for frozen Improved17 and not test failure autopsies.",
                                            "selection_policy": diagnosis["duplicate_policy"],
                                            "example_scope": diagnosis["example_scope"], "cases": failures},
        "interruption_and_authorized_recovery": {"failure": deepcopy(documents["failure"]["failure"]),
                                                 "old_attempt_status": "incomplete", "old_partial_score_available": False,
                                                 "io_fix_changes": deepcopy(documents["io_fix"]["changes"]),
                                                 "human_answer": documents["approval"]["user_answer"],
                                                 "exposure": deepcopy(exposure)},
        "limits": {"case_ids_are_identity_proxies": True, "spatial_units_unknown": True,
                   "id5_organ_mapping_is_accepted_derivative_assumption": True,
                   "single_seed": True, "test_case_proxy_groups": 3,
                   "no_statistical_significance_or_module_causality_claim": True,
                   "no_post_test_model_or_hyperparameter_selection": True},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=EVIDENCE / "final_report_summary_20261026.json")
    args = parser.parse_args()
    output = args.output.resolve()
    require(output.is_relative_to(PROJECT) and output.suffix == ".json", "New summary must be a project JSON file")
    require(not output.exists(), "Refusing to overwrite prior evidence")
    paths = {key: EVIDENCE / name for key, name in SOURCES.items()}
    hashes = {key: sha(path) for key, path in paths.items()}
    documents = {key: read(path) for key, path in paths.items()}
    facts = collect(documents, hashes)
    require(all(sha(path) == hashes[key] for key, path in paths.items()), "Saved input evidence changed during collection")
    result = {"analysis_kind": "final_report_evidence_collection_from_saved_json_only", "status": "passed_verification",
              "created_at": datetime.now(timezone(timedelta(hours=10))).isoformat(),
              "performed_by": "Codex stdlib JSON/arithmetic checks; not user verification",
              "experiment_parent_revision": "85aa4323fbe91ed7af3b33e8d156547309651623",
              "collector_sha256": sha(Path(__file__)),
              "input_sources": {key: {"file": SOURCES[key], "sha256": hashes[key]} for key in SOURCES},
              "execution_scope": {"saved_json_files_read": len(paths), "raw_mri_or_mask_reads": 0,
                                  "checkpoint_file_reads": 0, "new_model_calls": 0,
                                  "new_training": False, "new_test_evaluation": False,
                                  "existing_evidence_unchanged": True},
              **facts}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Verified saved report evidence without model/data execution: {output}", flush=True)


if __name__ == "__main__":
    main()
