"""Verify saved test JSON after an authorized Improved-only recovery; no inference."""

import argparse
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path

from summarize_formal_runs import finite_number, read_json, require, sha256_file, verify_metrics


PROJECT = Path(__file__).resolve().parent
REPO = PROJECT.parents[1]
MODELS = ("baseline_unet", "improved_unet")
EXPECTED_EPOCHS = {"baseline_unet": 5, "improved_unet": 17}
SAMPLES, BATCHES, PIXELS = 540, 135, 17694720


def inside(root, relative):
    path = Path(relative)
    require(not path.is_absolute(), "Linked evidence must be repository-relative")
    absolute = (root / path).resolve()
    require(absolute.is_relative_to(root.resolve()), "Linked evidence leaves the repository")
    return absolute


def linked_json(plan, key, root=REPO):
    path = inside(root, plan[key + "_file"])
    require(sha256_file(path) == plan[key + "_sha256"], f"Linked JSON changed: {key}")
    return read_json(path)


def validate_resource(stage, label):
    require(stage["status"] == "passed" and stage["device"] == "cuda", f"{label}: incomplete resource stage")
    require(isinstance(stage["resident_state"], str) and bool(stage["resident_state"]), f"{label}: missing resident-state scope")
    finite_number(stage["elapsed_seconds"], label + " resource elapsed", minimum=0)
    for kind in ("allocated", "reserved"):
        values = [stage[f"{point}_cuda_{kind}_bytes"] for point in ("start", "peak", "end")]
        require(all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in values),
                f"{label}: invalid allocator counters")
        require(values[1] >= max(values[0], values[2]), f"{label}: invalid allocator peak")


def verify_complete(entry, spec, label):
    require(entry["status"] == "passed", f"{label}: model result is incomplete")
    require(entry["checkpoint_epoch"] == spec["epoch"] == EXPECTED_EPOCHS[label]
            and entry["checkpoint_sha256"] == spec["checkpoint_sha256"], f"{label}: frozen checkpoint binding differs")
    require((entry["forward_calls"], entry["forward_samples"], entry["optimizer_steps"], entry["full_test_passes"])
            == (BATCHES, SAMPLES, 0, 1), f"{label}: full-test forward counts differ")
    require(entry["state_sha256_before"] == entry["state_sha256_after"]
            and isinstance(entry["state_sha256_before"], str) and len(entry["state_sha256_before"]) == 64
            and entry["parameters_have_no_gradients"] is True, f"{label}: state/gradient check failed")
    metrics = entry["metrics"]
    require((metrics["samples"], metrics["batches"], metrics["optimizer_steps"], metrics["valid_pixels"])
            == (SAMPLES, BATCHES, 0, PIXELS), f"{label}: full-test metric counts differ")
    # CE depends on probabilities. The shared checker only verifies CE is finite;
    # confusion counts reconstruct Dice/IoU/macro/accuracy, never CE.
    verify_metrics(metrics, label + " complete test", num_classes=6)
    validate_resource(entry["resource_measurement"], label)


def verify_exposure(attempt, recovery):
    require(attempt["status"] == "incomplete", "The original failed attempt must remain incomplete")
    original = attempt["models"]["improved_unet"]
    require(original["status"] == "incomplete" and "metrics" not in original
            and (original["forward_calls"], original["forward_samples"], original["optimizer_steps"])
            == (111, 444, 0), "The original partial pass must remain 111/444 with no recovered metrics")
    require(set(recovery["models"]) == {"improved_unet"}, "Recovery must not include a new baseline evaluation")
    require(recovery["baseline_forward_calls_this_recovery"] == 0
            and recovery["full_test_passes_this_recovery"] == 1, "Recovery includes an unauthorized extra pass")
    expected = {"prior_improved_forward_calls": 111, "prior_improved_forward_samples": 444,
                "new_improved_forward_calls": 135, "new_improved_forward_samples": 540,
                "planned_repeated_prior_samples": 444, "repeated_prior_samples": 444,
                "cumulative_improved_forward_calls": 246,
                "cumulative_improved_forward_samples": 984, "partial_metrics_recovered": False,
                "baseline_forward_calls_this_recovery": 0}
    require(recovery["exposure"] == expected, "Recovery exposure accounting differs from the original and full rerun")
    return {"unique_test_slices": SAMPLES,
            "baseline": {"original_forward_calls": 135, "original_forward_samples": 540,
                         "recovery_forward_calls": 0, "cumulative_forward_calls": 135,
                         "cumulative_forward_sample_events": 540, "completed_full_passes": 1,
                         "score_origin": "The complete baseline result from the original attempt; not rerun."},
            "improved": {**deepcopy(expected), "completed_full_passes": 1,
                         "cumulative_forward_sample_events": 984,
                         "score_origin": "Only the complete authorized 135-batch rerun; no partial metrics were recovered or merged.",
                         "order_note": "Unchanged deterministic manifest order repeats the first 444 slices forwarded by the incomplete attempt."},
            "all_models_cumulative_forward_calls": 381,
            "all_models_cumulative_forward_sample_events": 1524,
            "once_only_boundary": "Baseline had one full pass. Improved had an incomplete 111-batch pass plus one explicitly authorized full rerun; it was not evaluated only once."}


def verify_inputs(original_plan, attempt, baseline, recovery_plan, recovery, paths):
    require(original_plan["status"] == "approved_for_execution" and original_plan["user_confirmation_received"] is True,
            "The original test plan lacks authorization")
    require(recovery_plan["status"] == "approved_for_execution" and recovery_plan["user_recovery_confirmation_received"] is True,
            "The recovery plan lacks additional authorization")
    require(recovery["status"] == "passed" and recovery["analysis_kind"] == "authorized_improved_only_test_recovery",
            "Improved recovery must be complete before test scores can be compared")
    require(recovery_plan["recovery_model"] == "improved_unet" and recovery_plan["checkpoint_epoch"] == 17,
            "Recovery must preserve Improved epoch17")
    for key in ("models", "evaluation", "data"):
        require(recovery_plan[key] == original_plan[key], f"Recovery changed frozen {key}")
    require(original_plan["evaluation"]["batch_size"] == 4
            and original_plan["evaluation"]["num_classes"] == 6
            and original_plan["evaluation"]["post_test_tuning"] is False,
            "Frozen shared test rules differ")
    test = original_plan["data"]["test"]
    require((test["pairs"], test["valid_pixels"]) == (SAMPLES, PIXELS), "Frozen test size differs")
    specs = {spec["model"]: spec for spec in original_plan["models"]}
    require(set(specs) == set(MODELS) and all(specs[name]["epoch"] == epoch for name, epoch in EXPECTED_EPOCHS.items()),
            "Frozen selected model inventory changed")
    for key, argument in (("original_plan", "original_plan"), ("original_attempt", "original_attempt"), ("baseline_result", "baseline")):
        digest = sha256_file(paths[argument])
        require(digest == recovery_plan[key + "_sha256"] == recovery[key + "_sha256"], f"{key}: input provenance differs")
    require(recovery["plan_sha256"] == sha256_file(paths["recovery_plan"]), "Recovery report is not bound to its execution plan")
    require(attempt["plan_sha256"] == sha256_file(paths["original_plan"])
            and attempt["source_revision"] == original_plan["source_revision"]
            and attempt["source_sha256"] == original_plan["source_sha256"], "Original attempt source/plan binding differs")
    require(recovery["source_revision"] == recovery_plan["source_revision"]
            and recovery["source_sha256"] == recovery_plan["source_sha256"], "Recovery source binding differs")
    for name, digest in original_plan["source_sha256"].items():
        if name != "evaluate_frozen_test.py":
            require(recovery_plan["source_sha256"].get(name) == digest, f"A non-persistence original source changed: {name}")
    require(recovery["original_evaluation_source_sha256"] == original_plan["source_sha256"]["evaluate_frozen_test.py"]
            and recovery["repaired_evaluation_source_sha256"] == recovery_plan["source_sha256"]["evaluate_frozen_test.py"],
            "Original/repaired execution source binding differs")
    for name, digest in recovery_plan["source_sha256"].items():
        require(sha256_file(inside(PROJECT, name)) == digest, f"Frozen recovery source changed: {name}")

    recipe = linked_json(recovery_plan, "original_recipe")
    audit = linked_json(recovery_plan, "audit")
    approval = linked_json(recovery_plan, "recovery_authorization")
    original_guard = linked_json(recovery_plan, "original_guard")
    require(original_guard == attempt, "Original consumed guard and attempt differ")
    require(recipe["models"] == original_plan["models"] and recipe["evaluation"] == original_plan["evaluation"]
            and recipe["frozen_before_test_manifest_or_pixel_read"] is True, "Pre-test recipe differs")
    for key in ("original_recipe_sha256", "audit_sha256", "original_guard_sha256", "recovery_authorization_sha256"):
        require(recovery[key] == recovery_plan[key], f"Recovery linked provenance differs: {key}")
    require(original_plan["recipe_sha256"] == attempt["recipe_sha256"] == recovery_plan["original_recipe_sha256"]
            and original_plan["audit_sha256"] == attempt["audit_sha256"] == recovery_plan["audit_sha256"],
            "Original recipe/audit chain differs")
    require(audit["status"] == "passed_full" and audit["frozen_plan_sha256"] == original_plan["recipe_sha256"],
            "Test input audit is not bound to the frozen recipe")
    for report in (attempt, recovery):
        require(report["test_manifest_sha256"] == test["manifest_sha256"]
                and report["test_content_manifest_sha256"] == test["test_content_manifest_sha256"]
                == audit["test_content_manifest_sha256"], "Frozen test content binding differs")
        require(report["evaluation"] == original_plan["evaluation"] and report["new_training"] is False
                and report["post_test_tuning"] is False, "Recorded evaluation rules or no-tuning policy differ")
    require(recovery["runtime"] == attempt["runtime"], "Recovery runtime differs from the original attempt")
    require(approval["user_recovery_confirmation_received"] is True
            and approval["additional_full_evaluations_authorized"] == {"baseline_unet": 0, "improved_unet": 1}
            and approval["recovery_model"] == "improved_unet" and approval["epoch"] == 17
            and approval["checkpoint_sha256"] == specs["improved_unet"]["checkpoint_sha256"]
            and approval["repeated_prior_test_forward_samples_disclosed"] == 444
            and approval["post_test_model_or_hyperparameter_tuning_authorized"] is False,
            "Human approval does not cover this Improved-only repeated exposure")
    for approval_key, plan_key in (("original_plan_sha256", "original_plan_sha256"),
                                   ("original_attempt_sha256", "original_attempt_sha256"),
                                   ("original_consumed_guard_sha256", "original_guard_sha256"),
                                   ("original_baseline_sha256", "baseline_result_sha256")):
        require(approval[approval_key] == recovery_plan[plan_key], "Recovery approval provenance changed")
    require(recovery["preserved_baseline_result"] == baseline == attempt["models"]["baseline_unet"],
            "The original baseline was replaced or edited")
    require(attempt["frozen_checkpoint_sha256"] == {name: spec["checkpoint_sha256"] for name, spec in specs.items()},
            "Original model file bindings differ")
    exposure = verify_exposure(attempt, recovery)
    improved = recovery["models"]["improved_unet"]
    require(improved["state_sha256_before"] == attempt["models"]["improved_unet"]["state_sha256_before"],
            "Recovered model state differs from the original partial attempt")
    verify_complete(baseline, specs["baseline_unet"], "baseline_unet")
    verify_complete(improved, specs["improved_unet"], "improved_unet")
    require(recovery["original_attempt_and_baseline_unchanged"] is True
            and recovery["frozen_sources_and_weights_unchanged"] is True, "Recovery did not verify preserved inputs")
    for key in ("content_verification_before", "content_verification_after"):
        verification = recovery[key]
        require(verification["all_compressed_files_match"] is True
                and (verification["pairs"], verification["compressed_files_verified"], verification["model_forward_calls"])
                == (SAMPLES, SAMPLES * 2, 0)
                and verification["voxel_arrays_materialized"] is False
                and verification["content_manifest_sha256"] == test["test_content_manifest_sha256"],
                f"Recovery compressed-byte verification differs: {key}")
    guard = inside(REPO, recovery["once_guard"])
    require(read_json(guard) == recovery, "Recovery guard and final report differ")
    return specs, exposure


def save_aggregate_figure(results, output):
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(output.parent / ".matplotlib"))
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6.4))
    fig.subplots_adjust(left=.08, right=.98, top=.90, bottom=.22)
    xs = list(range(6))
    for name, offset, color, label in (
        ("baseline_unet", -.18, "tab:blue", "Baseline epoch5: original full pass"),
        ("improved_unet", .18, "tab:orange", "Improved epoch17: authorized full rerun"),
    ):
        scores = results[name]["metrics"]["dice_per_class"]
        bars = ax.bar([x + offset for x in xs], [0 if value is None else value for value in scores],
                      width=.36, color=color, label=label)
        for bar, value in zip(bars, scores):
            if value is None:
                ax.annotate("undefined", (bar.get_x() + bar.get_width() / 2, .015),
                            ha="center", rotation=90, fontsize=8)
    ax.set(xticks=xs, xticklabels=[f"ID{value}" for value in xs], ylim=(0, 1.04),
           ylabel="Pixel-pooled complete-test Dice", xlabel="Raw class ID",
           title="Frozen test comparison: 540 slices; unchanged model and evaluation rules")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(axis="y", alpha=.2)
    fig.text(.5, .035,
             "Baseline: 135 forwards once. Improved: 111 incomplete + 135 authorized forwards; only the full rerun supplies its score.\n"
             "ID5 organ interpretation retains the accepted derivative-ID assumption; no test-based tuning.",
             ha="center", fontsize=8)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-plan", type=Path, required=True)
    parser.add_argument("--original-attempt", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--recovery-plan", type=Path, required=True)
    parser.add_argument("--recovery-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output_dir.exists(), "Use a new postprocessing directory; prior evidence is preserved")
    paths = {name: getattr(args, name) for name in ("original_plan", "original_attempt", "baseline", "recovery_plan", "recovery_report")}
    input_hashes = {name: sha256_file(path) for name, path in paths.items()}
    original, attempt, baseline, plan, recovery = [read_json(paths[name]) for name in paths]
    specs, exposure = verify_inputs(original, attempt, baseline, plan, recovery, paths)
    entries = {"baseline_unet": baseline, "improved_unet": recovery["models"]["improved_unet"]}
    results = {name: {"checkpoint_epoch": entry["checkpoint_epoch"], "checkpoint_sha256": entry["checkpoint_sha256"],
                      "parameters": specs[name]["parameters"], "model_config": specs[name]["model_config"],
                      "metrics": deepcopy(entry["metrics"]), "resource_measurement": deepcopy(entry["resource_measurement"]),
                      "forward_calls_for_complete_score": entry["forward_calls"],
                      "forward_samples_for_complete_score": entry["forward_samples"], "optimizer_steps": 0,
                      "state_sha256_before": entry["state_sha256_before"], "state_sha256_after": entry["state_sha256_after"],
                      "parameters_have_no_gradients": True,
                      "complete_score_origin": "original_attempt" if name == "baseline_unet" else "authorized_improved_only_recovery"}
               for name, entry in entries.items()}
    output = {"analysis_kind": "verified_frozen_test_comparison_after_authorized_improved_recovery",
              "status": "passed_verification", "created_at": datetime.now(timezone(timedelta(hours=10))).isoformat(),
              "performed_by": "Codex JSON/SHA postprocessing; not user verification",
              "postprocessing_source_sha256": sha256_file(Path(__file__)),
              "inputs": {name: {"path": str(path), "sha256": input_hashes[name]} for name, path in paths.items()},
              "original_source_revision": original["source_revision"], "recovery_source_revision": plan["source_revision"],
              "recipe_sha256": original["recipe_sha256"], "audit_sha256": original["audit_sha256"],
              "recovery_authorization_sha256": plan["recovery_authorization_sha256"],
              "data": deepcopy(original["data"]), "preprocessing": deepcopy(original["preprocessing"]),
              "evaluation": deepcopy(original["evaluation"]), "runtime": deepcopy(recovery["runtime"]),
              "results": results, "test_exposure": exposure,
              "original_attempt_preserved": {"status": attempt["status"], "sha256": input_hashes["original_attempt"],
                                             "improved_partial_has_no_metrics": "metrics" not in attempt["models"]["improved_unet"],
                                             "improved_partial_resource_measurement": deepcopy(attempt["models"]["improved_unet"]["resource_measurement"]),
                                             "error": deepcopy(attempt.get("error"))},
              "metric_verification": {"all_confusion_derived_overlap_macro_accuracy_recomputed": True,
                                      "cross_entropy": "Finite shared-loss value retained from the complete pass; not derivable from the confusion matrix.",
                                      "checkpoint_verification_scope": "Stored SHA/epoch/state bindings verified across frozen recipe/plans and execution records; this helper does not load or hash model files."},
              "new_model_forward_calls": 0, "new_test_raw_file_reads": 0, "new_training": False,
              "test_scores_used_for_tuning": False, "aggregate_figure": "aggregate_test_dice.png",
              "resource_scope": "Retain each original resident-state/allocator scope. Complete baseline and Improved stages come from their distinct actual passes; the original failed Improved stage remains separate. Timing includes input I/O and is not a pure-forward or statistically isolated hardware comparison.",
              "claim_boundary": "Only the preselected baseline5 and Improved17 are compared. The explicitly authorized repeated Improved exposure is disclosed, no partial scores are reconstructed, and no test-driven selection/tuning occurs. Three filename case proxies do not establish clinical validity, statistical significance, physical units or independently confirmed patient identity; ID5 retains the accepted derivative-ID assumption."}
    require(all(sha256_file(path) == input_hashes[name] for name, path in paths.items()), "An input JSON changed during postprocessing")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    save_aggregate_figure(results, args.output_dir / output["aggregate_figure"])
    import json
    with (args.output_dir / "comparison.json").open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(f"Verified saved complete-test scores without any forward: {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
