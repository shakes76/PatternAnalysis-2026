"""Metadata-only R1 audit and VAL-only candidate choice; never open MRI or weights.

Reuse the separately hashed, standard-library A1 count reaggregator. Checkpoint
identities are compared across frozen plans and evaluator receipts; checkpoint
bytes and the evaluator's tensor-based parent-state test are not repeated here.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from experiments.summarize_appearance import equal, mean, read, recompute, require, safe, sha

EXPERIMENT = "appearance_recovery_v1"
ARMS = ("clean_adapt", "polarity_rehearsal")
REFERENCES = ("seed42_reference", "gamma_only", "gamma_polarity")
MODELS = (*REFERENCES, *ARMS)
CONDITIONS = ("clean", "gamma_0.7", "gamma_1.5", "polarity", "spatial_gain")
HELPER_SHA = "3dc7e71c2b31dd181a2938a70b01cb0c27a08a6e5fea4cc1e9902d2640193942"
GATE = {"new_clean_prostate_Dice_minus_seed42_reference_min": 0.0,
        "new_clean_foreground_Dice_minus_seed42_reference_min": -0.005,
        "new_polarity_prostate_Dice_minus_gamma_polarity_min": -0.02, "all_three_required": True}


def digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def candidate_choice(results, same_platform):
    """Use simultaneous same-platform references, never historical or M015 scores."""
    gates = {}
    for name in ARMS:
        terms = (("clean_prostate", name, "clean", "prostate_dice", "seed42_reference", 0.0),
                 ("clean_foreground", name, "clean", "foreground_mean_dice", "seed42_reference", -0.005),
                 ("polarity_prostate", name, "polarity", "prostate_dice", "gamma_polarity", -0.02))
        deltas, checks = {}, {}
        for label, model, condition, metric, reference, minimum in terms:
            a, b = results[model][condition]["aggregate"][metric], results[reference][condition]["aggregate"][metric]
            delta = a - b if a is not None and b is not None else None
            deltas[label] = delta
            checks[label] = delta is not None and delta >= minimum
        gates[name] = {"deltas": deltas, "numeric_checks": checks, "numeric_gate_passes": all(checks.values()),
                       "eligible_candidate": same_platform and all(checks.values())}
    eligible = [n for n in ARMS if gates[n]["eligible_candidate"]]
    eligible.sort(key=lambda n: (-results[n]["clean"]["aggregate"]["prostate_dice"],
                                -results[n]["clean"]["aggregate"]["foreground_mean_dice"], n))
    return gates, {"selected_model_id": eligible[0] if eligible else None, "selection_basis": "validation_only",
                   "eligible_model_ids": eligible, "same_platform_verified": same_platform,
                   "tie_break_order": ["clean_prostate_dice_desc", "clean_foreground_dice_desc", "model_id_asc"]}


def signature(evaluation):
    return ({(c["patient"], c["visit"]): {k: c[k] for k in ("per_class", "foreground_mean_dice", "foreground_mean_iou")}
             for c in evaluation["cases"]}, evaluation["aggregate"])


def summarize(protocol_path, evaluation_path, project=PROJECT):
    project = Path(project).resolve()
    protocol_path, evaluation_path = Path(protocol_path).resolve(strict=True), Path(evaluation_path).resolve(strict=True)
    require(protocol_path.is_relative_to(project) and evaluation_path.is_relative_to(project), "Metadata inputs must be project-relative.")
    require(sha(PROJECT / "experiments/summarize_appearance.py") == HELPER_SHA, "Count helper SHA changed.")
    plan, index, protocol_sha, files = read(protocol_path), read(evaluation_path), sha(protocol_path), {}
    equal(plan, {"experiment_id": EXPERIMENT, "allowed_splits": ["train", "validation"], "training_seed": 42,
                 "training_epochs_per_model": 8, "conditions": list(CONDITIONS),
                 "inference": {"split": "validation", "device": "cuda", "amp": True, "saved_arrays": False}}, "recovery protocol")
    equal(plan["engineering_gate"], GATE, "fixed engineering gate")
    require(set(plan["models"]) == set(ARMS) and set(plan["references"]) == set(REFERENCES), "Five-model protocol scope differs.")
    require(plan["warm_start"]["checkpoint_sha256"] == plan["references"]["gamma_polarity"]["checkpoint_sha256"], "Warm-start identity differs from frozen B.")
    for name, probability in zip(ARMS, (0.0, 0.1)):
        equal(plan["models"][name], {"kind": "recovery_arm", "polarity_probability": probability}, "fixed policy")
    for relative, expected in plan["required_file_sha256"].items():
        require(sha(safe(project, relative)) == expected, "Protected source/input SHA differs: " + relative)
    def remember(path):
        require(path.is_relative_to(project), "Receipt escapes project.")
        files[str(path.relative_to(project))] = sha(path)
        return files[str(path.relative_to(project))]
    for p in (protocol_path, evaluation_path): remember(p)
    manifest_path, identity_path = (safe(project, plan[k]) for k in ("manifest", "cache_identity"))
    require(remember(manifest_path) == plan["manifest_sha256"] and remember(identity_path) == plan["cache_identity_sha256"], "Frozen data metadata SHA differs.")
    appearance_path = safe(project, plan["appearance_protocol"])
    require(remember(appearance_path) == plan["appearance_protocol_sha256"], "Frozen A1 protocol SHA differs.")
    require(remember(safe(project, "training_sources.json")) == plan["training_sources_sha256"], "Original sourcebook SHA differs.")
    appearance, manifest = read(appearance_path), read(manifest_path)
    require(plan["training_conditions"] == dict(appearance["training_conditions"], lr=1e-4), "Recovery changed training settings beyond lr.")
    require(manifest["num_classes"] == 6 and manifest["label_ids"] == list(range(6))
            and not manifest.get("partial") and not manifest.get("development_only"), "Expected full six-class data metadata.")
    records, partitions = manifest["records"], manifest["patient_splits"]
    require(set(partitions) == {"train", "validation", "test"} and len(set(sum(partitions.values(), []))) == sum(map(len, partitions.values())), "Patient partitions overlap.")
    require(Counter(r["split"] for r in records) == {"train": 143, "validation": 38, "test": 30}
            and all(r["patient"] in partitions[r["split"]] for r in records), "Manifest scope/partition mismatch.")
    keys = {(r["patient"], r["visit"]) for r in records if r["split"] == "validation"}
    require(len(keys) == 38 and len(partitions["validation"]) == 6 and {p for p, _ in keys} == set(partitions["validation"]), "Expected38 VAL scans/six patients.")
    common = {"schema_version": 1, "experiment_id": EXPERIMENT, "split": "validation", "protocol_sha256": protocol_sha,
              "manifest_sha256": plan["manifest_sha256"], "saved_arrays": False}
    equal(index, {**common, "status": "completed", "appearance_protocol_sha256": plan["appearance_protocol_sha256"],
                 "cache_identity_sha256": plan["cache_identity_sha256"], "source_file_sha256": plan["required_file_sha256"],
                 "conditions": list(CONDITIONS), "n_scans_per_condition": 38, "n_unique_patients": 6, "test_arrays_opened": 0}, "evaluation index")
    require(set(index["models"]) == set(index["results"]) == set(MODELS), "Index model scope differs.")
    equal(index["runtime"], {"device": "cuda", "amp": True}, "evaluation precision")
    runtime = {k: index["runtime"][k] for k in ("gpu_name", "python", "torch", "cuda")}
    require(all(isinstance(v, str) and v and "/" not in v and "\\" not in v for v in runtime.values()), "Invalid public runtime metadata.")
    results, training, models, selected_history, target_signature = {}, {}, {}, {}, None
    for name in MODELS:
        receipt = index["models"][name]
        require(digest(receipt["checkpoint_sha256"]) and digest(receipt["state_before_sha256"])
                and receipt["state_unchanged"] is True and type(receipt["parameters"]) is int and receipt["parameters"] > 0, "Model identity/state receipt differs.")
        if name in REFERENCES:
            spec = plan["references"][name]
            equal(receipt, {"checkpoint_sha256": spec["checkpoint_sha256"], "checkpoint_epoch": spec["selected_epoch"]}, "frozen reference")
        else:
            root = (project / plan["models"][name]["checkpoint"]).resolve().parent
            require(root.is_relative_to(project), "Arm metadata path escapes project.")
            run_path, history_path, provenance_path = (root / f for f in ("run.json", "history.json", "provenance.json"))
            run, history, provenance = (read(p) for p in (run_path, history_path, provenance_path))
            require(set(run["runtime"]) == set(runtime) and all(isinstance(v, str) and v and "/" not in v and "\\" not in v
                    for v in run["runtime"].values()), "Invalid public training runtime metadata.")
            experiment = {"experiment_id": EXPERIMENT, "arm_tag": name, "protocol_sha256": protocol_sha}
            parent = {"parent_checkpoint_sha256": plan["warm_start"]["checkpoint_sha256"], "parent_epoch": 29,
                      "warm_start_weights_only": True, "optimizer_reset": True, "scaler_reset": True}
            equal(run, {"experiment": experiment, "config": plan["training_conditions"], "status": "completed", "completed_epochs": 8,
                        "requested_total_epochs": 8, "development_only": False, "source_file_sha256": plan["required_file_sha256"], **parent}, "arm run")
            require(run["config"] == plan["training_conditions"], "Arm complete config differs.")
            equal(provenance, {"experiment": experiment, "protocol_sha256": protocol_sha, "manifest_sha256": plan["manifest_sha256"],
                  "source_file_sha256": plan["required_file_sha256"], "checkpoint_sha256": receipt["checkpoint_sha256"], **parent}, "arm provenance")
            protected = {}
            for path, field in ((run_path, "run_sha256"), (history_path, "history_sha256"), (provenance_path, "provenance_sha256")):
                actual = remember(path); protected[str(path.relative_to(project))] = actual
                require(actual == receipt[field] and (field == "provenance_sha256" or actual == provenance[field]), "Arm receipt SHA differs.")
            verification_path = safe(project, provenance["cache_verification_path"])
            actual, verification = remember(verification_path), read(verification_path)
            protected[str(verification_path.relative_to(project))] = actual
            require(actual == receipt["cache_verification_sha256"] == provenance["cache_verification_sha256"] == run["cache_verification_sha256"], "Cache receipt SHA differs.")
            equal(receipt["protected_file_sha256"], protected, "protected metadata files")
            equal(verification, {"status": "verified", "manifest_sha256": plan["manifest_sha256"], "counts": {"train": 143, "validation": 38}, "test_arrays_opened": 0}, "cache verification")
            rows = verification["records"]
            expected = {(r["patient"], r["visit"], r["split"]) for r in records if r["split"] in ("train", "validation")}
            require(len(rows) == 181 and {(r["patient"], r["visit"], r["split"]) for r in rows} == expected
                    and all(r["decoded_identity_verified"] is True for r in rows), "Incomplete181-case cache receipt.")
            require(len(history) == 8 and [r["epoch"] for r in history] == list(range(1, 9)), "Incomplete eight-epoch history.")
            curve = []
            for row in history:
                equal(row, {"training_examples": 143, "training_batches": 143, "development_only": False}, "epoch coverage")
                require(all(type(row[k]) is int and 0 <= row[k] <= 143 for k in ("optimizer_steps", "amp_overflow_skipped_optimizer_steps"))
                        and row["optimizer_steps"] > 0 and row["optimizer_steps"] + row["amp_overflow_skipped_optimizer_steps"] == 143, "Optimizer accounting differs.")
                counts = row["policy_counts"]
                require(set(counts) == {"noninverted", "inverted", "gamma_applied"} and all(type(v) is int and 0 <= v <= 143 for v in counts.values())
                        and counts["inverted"] + counts["noninverted"] == 143 and (name != "clean_adapt" or counts["inverted"] == 0), "Policy exposure counts differ.")
                aggregate, _ = recompute(row["validation"]["cases"], keys, manifest["target_shape_dhw"])
                equal(row["validation"]["aggregate"], aggregate, "epoch metrics"); equal(row["validation_foreground_dice"], aggregate["foreground_mean_dice"], "epoch selector")
                equal(row["validation_foreground_iou"], aggregate["foreground_mean_iou"], "epoch IoU")
                require(all(type(row[k]) in (int, float) and math.isfinite(row[k]) for k in ("train_loss", "train_seconds"))
                        and row["train_seconds"] > 0 and type(row["peak_allocated_cuda_bytes_training"]) is int and row["peak_allocated_cuda_bytes_training"] > 0, "Invalid training resource/loss.")
                curve.append({k: row[k] for k in ("epoch", "train_loss", "train_seconds", "validation_foreground_dice", "optimizer_steps", "amp_overflow_skipped_optimizer_steps", "policy_counts", "peak_allocated_cuda_bytes_training")})
            best = max(history, key=lambda r: r["validation_foreground_dice"])
            require(best["epoch"] == run["best_epoch"] == receipt["checkpoint_epoch"], "Earliest clean selector differs.")
            equal(run["best_validation_foreground_dice"], best["validation_foreground_dice"], "best score")
            totals = {k: sum(row["policy_counts"][k] for row in history) for k in ("noninverted", "inverted", "gamma_applied")}
            equal(run["training_policy_counts"], totals, "total exposure counts")
            require(digest(run["initial_parameter_state_sha256"]) and receipt["initial_parameter_state_sha256"] == run["initial_parameter_state_sha256"]
                    and receipt["training_runtime"] == run["runtime"], "Initial state/runtime receipt differs.")
            selected_history[name] = best["validation"]
            training[name] = {"selected_epoch": best["epoch"], "initial_parameter_state_sha256": run["initial_parameter_state_sha256"], "runtime": run["runtime"],
                "epochs": curve, "total_train_seconds": sum(r["train_seconds"] for r in history), "total_optimizer_steps": sum(r["optimizer_steps"] for r in history),
                "total_amp_skipped_steps": sum(r["amp_overflow_skipped_optimizer_steps"] for r in history), "policy_counts": totals,
                "peak_allocated_cuda_bytes_training": max(r["peak_allocated_cuda_bytes_training"] for r in history)}
        models[name] = {"checkpoint_sha256": receipt["checkpoint_sha256"], "checkpoint_epoch": receipt["checkpoint_epoch"], "state_unchanged": True}
        require(set(index["results"][name]) == set(CONDITIONS), "Missing/extra evaluation conditions.")
        results[name] = {}
        for condition in CONDITIONS:
            entry = index["results"][name][condition]
            equal(entry["evaluation_file"], f"{name}/{condition}/evaluation.json", "evaluation filename")
            path = safe(evaluation_path.parent, entry["evaluation_file"])
            require(remember(path) == entry["evaluation_sha256"], "Evaluation file SHA differs.")
            raw = read(path)
            equal(raw, {**common, "model_id": name, "condition": condition, "device": "cuda", "amp": True,
                        "checkpoint_sha256": receipt["checkpoint_sha256"]}, "evaluation metadata")
            aggregate, compact = recompute(raw["cases"], keys, manifest["target_shape_dhw"])
            equal(raw["aggregate"], aggregate, "evaluation aggregate"); equal(entry["aggregate"], aggregate, "index aggregate")
            targets = {(c["patient"], c["visit"]): [v["target_voxels"] for v in c["per_class"]] for c in raw["cases"]}
            require(target_signature is None or targets == target_signature, "GT counts differ across models/conditions.")
            target_signature = targets
            if name in ARMS and condition == "clean":
                require(signature(raw) == signature(selected_history[name]), "Standalone clean differs from selected history.")
            for case in raw["cases"]:
                require(type(case["inference_ms"]) in (int, float) and math.isfinite(case["inference_ms"]) and case["inference_ms"] >= 0
                        and type(case["peak_allocated_cuda_bytes"]) is int and case["peak_allocated_cuda_bytes"] > 0, "Invalid inference resources.")
            latency, memory = mean(c["inference_ms"] for c in raw["cases"]), max(c["peak_allocated_cuda_bytes"] for c in raw["cases"])
            for data in (raw, entry): equal(data, {"mean_inference_ms": latency, "peak_allocated_cuda_bytes": memory}, "inference resources")
            totals = {k: v for k, v in aggregate.items() if k != "patients"}
            totals.update(prostate_dice=aggregate["per_class"][5]["dice"], prostate_iou=aggregate["per_class"][5]["iou"])
            results[name][condition] = {"aggregate": totals, "patients": aggregate["patients"], "cases_prostate": compact,
                                        "mean_inference_ms": latency, "peak_allocated_cuda_bytes": memory}
    paired = {"initial_parameter_state_sha256_equal": training[ARMS[0]]["initial_parameter_state_sha256"] == training[ARMS[1]]["initial_parameter_state_sha256"],
              "training_runtime_exactly_equal": training[ARMS[0]]["runtime"] == training[ARMS[1]]["runtime"],
              "training_and_evaluation_runtime_equal": all(training[n]["runtime"] == runtime for n in ARMS),
              "gamma_exposure_counts_equal": [r["policy_counts"]["gamma_applied"] for r in training[ARMS[0]]["epochs"]]
              == [r["policy_counts"]["gamma_applied"] for r in training[ARMS[1]]["epochs"]]}
    same_platform = all(paired.values())
    gates, recommendation = candidate_choice(results, same_platform)
    return {"schema_version": 1, "status": "completed", "experiment_id": EXPERIMENT, "split": "validation", "protocol_sha256": protocol_sha,
            "manifest_sha256": plan["manifest_sha256"], "evaluation_index_sha256": sha(evaluation_path), "input_file_sha256": files,
            "summarizer_sha256": sha(__file__), "count_helper_sha256": HELPER_SHA, "models": models,
            "scope": {"n_models": 5, "n_conditions": 5, "n_scans": 38, "n_patients": 6, "n_case_evaluations": 950,
                      "test_arrays_read_by_summarizer": 0, "arrays_loaded": False, "checkpoint_files_read": False},
            "runtime": runtime, "paired_execution_checks": paired, "confounds": [] if same_platform else ["Initial states, runtimes or shared-gamma exposure counts differ; no candidate recommended."],
            "engineering_gates": gates, "recommendation": recommendation, "training": training, "results": results,
            "limits": ["Exploratory post-test-informed development; repeated VAL selection/analysis on the same six patients, not thirty independent patients.",
                       "Candidate uses only prespecified same-platform clean/polarity VAL gates; M015 or any TEST result cannot choose it.",
                       "No independent confirmation, clinical validity, population superiority, unique root cause, or replacement of frozen v1/A1/M1 results.",
                       "Checkpoint SHA and parent-state checks here bind evaluator metadata; no checkpoint bytes/tensors or MRI arrays are reread. Guarded evaluator performed tensor-based warm-start validation.",
                       "Allocated CUDA memory and warmup-excluded cache-volume latency are descriptive; training seconds exclude validation/saving. CUDA warn-only nondeterminism remains.",
                       "All five models, five conditions, 950 case summaries and negative outcomes retained; no M015 retention result is included."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "evaluation", "output"): parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    require(not os.path.lexists(args.output), "Output already exists; refusing overwrite.")
    data = json.dumps(summarize(args.protocol, args.evaluation), indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=args.output.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(data); stream.flush(); os.fsync(stream.fileno()); os.link(temporary, args.output)
        finally: temporary.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
