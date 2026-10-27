"""One explicitly authorized Improved-only test recovery; preserve the first attempt."""

import argparse
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path

import torch

import evaluate_frozen_test as frozen
from audit_test_manifest import verify_test_content
from dataset import collect_pairs, patient_ids
from metrics import SegmentationMetrics
from train import manifest_hash, seed_everything


PROJECT = Path(__file__).resolve().parent
REPO = PROJECT.parents[1]
RECOVERY_GUARD = "outputs/frozen_test_recovery_once_20261026.guard.json"


def linked(plan, key, root):
    path = frozen.inside(root, plan[key + "_file"])
    digest = plan[key + "_sha256"]
    frozen.require(frozen.sha(path) == digest, f"Recovery source file changed: {key}")
    frozen.require(plan["protected_original_files_sha256"].get(plan[key + "_file"]) == digest,
                   f"Recovery source must remain protected: {key}")
    return frozen.read(path)


def verify_baseline(entry, spec, pairs, batches, valid_pixels):
    """Recalculate already saved confusion counts on CPU; never build a model."""
    frozen.require(entry["status"] == "passed" and entry["checkpoint_epoch"] == 5
                   and entry["checkpoint_sha256"] == spec["checkpoint_sha256"], "Original baseline is not the frozen completed model")
    frozen.require((entry["forward_calls"], entry["forward_samples"], entry["optimizer_steps"], entry["full_test_passes"])
                   == (batches, pairs, 0, 1), "Original baseline pass counts differ")
    frozen.require(entry["state_sha256_before"] == entry["state_sha256_after"]
                   and entry["parameters_have_no_gradients"] is True, "Original baseline state check failed")
    metrics = entry["metrics"]
    frozen.require((metrics["samples"], metrics["batches"], metrics["optimizer_steps"], metrics["valid_pixels"])
                   == (pairs, batches, 0, valid_pixels), "Original baseline metric counts differ")
    matrix = metrics["confusion_matrix"]
    frozen.require(len(matrix) == 6 and all(len(row) == 6 for row in matrix)
                   and all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for row in matrix for v in row),
                   "Original baseline confusion counts are invalid")
    accumulator = SegmentationMetrics(6)
    accumulator.confusion_matrix = torch.tensor(matrix, dtype=torch.int64)
    frozen.require(all(metrics[key] == value for key, value in accumulator.compute().items()),
                   "Original baseline scores do not match its saved confusion counts")
    frozen.require(math.isfinite(metrics["cross_entropy"]) and metrics["cross_entropy"] >= 0,
                   "Original baseline CE is invalid")


def validate(plan, plan_path, data_root, root, project, synthetic=False):
    frozen.require(plan["status"] == "approved_for_execution" and plan["user_recovery_confirmation_received"] is True,
                   "An additional explicit recovery approval is required")
    frozen.require(plan["recovery_model"] == "improved_unet" and plan["checkpoint_epoch"] == 17,
                   "Only Improved epoch17 is authorized for recovery")
    frozen.require(plan["recovery_guard"] == RECOVERY_GUARD, "The fixed recovery guard cannot be changed")
    frozen.require(plan["original_guard_file"] == frozen.ONCE_GUARD, "The original consumed guard must be preserved")
    frozen.require((root / frozen.ONCE_GUARD).is_file() and not (root / RECOVERY_GUARD).exists(),
                   "Original attempt must exist and recovery must not already be consumed")
    required = frozen.REQUIRED_SOURCES | {"recover_frozen_test.py"}
    frozen.require(required.issubset(plan["source_sha256"]), "Missing frozen recovery dependencies")
    frozen.verify_hashes(plan["source_sha256"], project, "recovery source")
    frozen.verify_hashes(plan["protected_original_files_sha256"], root, "original record")
    old = linked(plan, "original_plan", root)
    attempt = linked(plan, "original_attempt", root)
    guard = linked(plan, "original_guard", root)
    baseline = linked(plan, "baseline_result", root)
    recipe = linked(plan, "original_recipe", root)
    audit = linked(plan, "audit", root)
    approval = linked(plan, "recovery_authorization", root)
    frozen.require(old["status"] == "approved_for_execution" and old["user_confirmation_received"] is True,
                   "Original test plan lacks approval")
    frozen.require(old["recipe_sha256"] == plan["original_recipe_sha256"]
                   and old["audit_sha256"] == plan["audit_sha256"], "Original test plan recipe/audit binding changed")
    frozen.require(plan["models"] == old["models"] == recipe["models"]
                   and plan["evaluation"] == old["evaluation"] == recipe["evaluation"]
                   and plan["data"] == old["data"], "Recovery changed the original models, rules or data")
    frozen.require([(m["model"], m["epoch"]) for m in plan["models"]] == list(frozen.MODEL_EPOCHS),
                   "Frozen model inventory changed")
    frozen.require(all(plan["source_sha256"].get(name) == digest for name, digest in old["source_sha256"].items()
                       if name != "evaluate_frozen_test.py"), "A non-persistence original dependency changed")
    frozen.require(plan["original_evaluation_source_sha256"] == old["source_sha256"]["evaluate_frozen_test.py"],
                   "Original execution source SHA must be retained")
    frozen.require(recipe["frozen_before_test_manifest_or_pixel_read"] is True
                   and all(plan["protected_original_files_sha256"].get(n) == h
                           for n, h in recipe["protected_original_files_sha256"].items()),
                   "The original pre-test freeze or protected records changed")
    frozen.require(audit["status"] == "passed_full" and audit["frozen_plan_sha256"] == plan["original_recipe_sha256"]
                   and audit["prior_plan_sha256"] == recipe["prior_id5_plan_sha256"]
                   and all(plan["source_sha256"].get(n) == h for n, h in audit["source_sha256"].items()),
                   "Audit source/binding differs from the original frozen experiment")
    frozen.require(attempt == guard and attempt["status"] == "incomplete"
                   and attempt["plan_sha256"] == plan["original_plan_sha256"]
                   and attempt["source_revision"] == old["source_revision"]
                   and attempt["recipe_sha256"] == plan["original_recipe_sha256"]
                   and attempt["audit_sha256"] == plan["audit_sha256"]
                   and attempt["source_sha256"] == old["source_sha256"]
                   and attempt["evaluation"] == plan["evaluation"], "Original incomplete attempt/guard provenance differs")
    specs = {item["model"]: item for item in plan["models"]}
    frozen.require(attempt["frozen_checkpoint_sha256"] == {n: s["checkpoint_sha256"] for n, s in specs.items()},
                   "Original checkpoint binding changed")
    frozen.require(baseline == attempt["models"]["baseline_unet"], "Original baseline result differs from its attempt")
    failed = attempt["models"]["improved_unet"]
    count = plan["data"]["test"]["pairs"]
    batches, pixels = math.ceil(count / 4), plan["data"]["test"]["valid_pixels"]
    prior_calls, prior_samples = failed["forward_calls"], failed["forward_samples"]
    if not synthetic:
        frozen.require((count, batches, pixels, prior_calls, prior_samples) == (540, 135, 17694720, 111, 444),
                       "Recovery does not match the audited original incomplete attempt")
    frozen.require(failed["status"] == "incomplete" and failed["checkpoint_epoch"] == 17
                   and failed["checkpoint_sha256"] == specs["improved_unet"]["checkpoint_sha256"]
                   and failed["optimizer_steps"] == 0 and "metrics" not in failed
                   and 0 < prior_calls < batches and prior_samples == 4 * prior_calls,
                   "Original Improved must remain an incomplete partial pass without recovered metrics")
    verify_baseline(baseline, specs["baseline_unet"], count, batches, pixels)
    frozen.require(approval["user_recovery_confirmation_received"] is True
                   and approval["user_answer"] == "允许，仅补跑 Improved"
                   and approval["additional_full_evaluations_authorized"] == {"baseline_unet": 0, "improved_unet": 1}
                   and approval["recovery_model"] == "improved_unet" and approval["epoch"] == 17
                   and approval["checkpoint_sha256"] == specs["improved_unet"]["checkpoint_sha256"]
                   and approval["same_original_model_preprocessing_batch_and_evaluation_rules_required"] is True
                   and approval["repeated_prior_test_forward_samples_disclosed"] == prior_samples
                   and approval["original_failure_and_baseline_must_be_preserved"] is True
                   and approval["post_test_model_or_hyperparameter_tuning_authorized"] is False,
                   "Human approval does not authorize this exact Improved-only repeated exposure")
    for approval_key, plan_key in (("original_plan_sha256", "original_plan_sha256"),
                                   ("original_attempt_sha256", "original_attempt_sha256"),
                                   ("original_consumed_guard_sha256", "original_guard_sha256"),
                                   ("original_baseline_sha256", "baseline_result_sha256")):
        frozen.require(approval[approval_key] == plan[plan_key], "Recovery approval provenance changed")
    pairs_by_split, ids_by_split = {}, {}
    for split, folders in frozen.SPLIT_FOLDERS.items():
        pairs = collect_pairs(data_root / folders[0], data_root / folders[1])
        ids = sorted(patient_ids(pairs))
        declared, audited = plan["data"][split], audit["splits"][split]
        frozen.require((len(pairs), manifest_hash(pairs), ids) ==
                       (declared["pairs"], declared["manifest_sha256"], declared.get("case_ids", declared.get("case_proxy_ids")))
                       and (len(pairs), manifest_hash(pairs), ids) ==
                       (audited["pairs"], audited["manifest_sha256"], audited["case_proxy_ids"]),
                       f"Recovery {split} manifest changed")
        pairs_by_split[split], ids_by_split[split] = pairs, set(ids)
    frozen.require(all(not ids_by_split[a] & ids_by_split[b] for a, b in
                       (("training", "validation"), ("training", "test"), ("validation", "test"))), "Case leakage detected")
    frozen.require(attempt["test_manifest_sha256"] == plan["data"]["test"]["manifest_sha256"]
                   and attempt["test_content_manifest_sha256"] == audit["test_content_manifest_sha256"]
                   == plan["data"]["test"]["test_content_manifest_sha256"], "Recovery test content binding changed")
    content = verify_test_content(pairs_by_split["test"], audit)
    for spec in plan["models"]:
        frozen.verify_checkpoint(spec, root)  # CPU metadata only; baseline is never constructed.
    frozen.require(frozen.sha(plan_path) == plan["_plan_sha256"], "Recovery plan changed during setup")
    return pairs_by_split["test"], audit, attempt, baseline, specs["improved_unet"], content


def execute(args, root=None, project=None, allow_cpu_for_synthetic=False):
    root, project = Path(root or REPO).resolve(), Path(project or PROJECT).resolve()
    out = Path(args.output_dir).resolve()
    frozen.require(out.is_relative_to(root) and not out.exists(), "Use a new repository recovery output directory")
    guard = root / RECOVERY_GUARD
    frozen.require(not guard.exists(), "Recovery attempt already exists; do not rerun")
    plan_path = Path(args.plan).resolve()
    plan = frozen.read(plan_path)
    plan["_plan_sha256"] = frozen.sha(plan_path)
    device = torch.device(args.device)
    frozen.require(str(device) == plan["evaluation"]["device"] and
                   (device.type == "cuda" or (allow_cpu_for_synthetic and plan.get("synthetic_test_only") is True)),
                   "Recovery device differs from the unchanged original controls")
    frozen.require(device.type != "cuda" or torch.cuda.is_available(), "CUDA is unavailable")
    pairs, audit, attempt, baseline, spec, content = validate(
        plan, plan_path, Path(args.data_root), root, project, allow_cpu_for_synthetic)
    torch.set_num_threads(1)
    seed_everything(3710)
    runtime = {"torch": str(torch.__version__), "cuda": torch.version.cuda,
               "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
               "math_settings": {"matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                                 "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                                 "float32_matmul_precision": torch.get_float32_matmul_precision()}}
    frozen.require(runtime == attempt["runtime"], "Recovery runtime/math settings differ from original attempt")
    prior = attempt["models"]["improved_unet"]
    report = {"analysis_kind": "authorized_improved_only_test_recovery", "status": "running",
              "created_at": datetime.now(timezone(timedelta(hours=10))).isoformat(),
              "performed_by": "Codex automation after explicit additional human authorization",
              "source_revision": plan["source_revision"], "source_sha256": plan["source_sha256"],
              "plan_sha256": plan["_plan_sha256"],
              **{key: plan[key] for key in ("original_plan_sha256", "original_recipe_sha256", "original_attempt_sha256",
                                            "original_guard_sha256", "baseline_result_sha256", "audit_sha256",
                                            "recovery_authorization_sha256", "original_evaluation_source_sha256")},
              "repaired_evaluation_source_sha256": plan["source_sha256"]["evaluate_frozen_test.py"],
              "evaluation": plan["evaluation"], "runtime": runtime, "models": {},
              "preserved_baseline_result": deepcopy(baseline), "baseline_forward_calls_this_recovery": 0,
              "test_manifest_sha256": plan["data"]["test"]["manifest_sha256"],
              "test_content_manifest_sha256": audit["test_content_manifest_sha256"],
              "content_verification_before": content, "once_guard": RECOVERY_GUARD,
              "output_directory": str(out.relative_to(root)), "test_split_read": True,
              "new_training": False, "post_test_tuning": False,
              "exposure": {"prior_improved_forward_calls": prior["forward_calls"],
                           "prior_improved_forward_samples": prior["forward_samples"],
                           "new_improved_forward_calls": 0, "new_improved_forward_samples": 0,
                           "planned_repeated_prior_samples": prior["forward_samples"], "repeated_prior_samples": 0,
                           "partial_metrics_recovered": False, "baseline_forward_calls_this_recovery": 0},
              "claim_boundary": "Authorized full Improved-only rerun after an incomplete I/O-failed attempt; prior forwarded test samples are repeated, not unseen. Baseline is preserved, not rerun. No test-driven tuning."}
    guard.parent.mkdir(parents=True, exist_ok=True)
    with guard.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    primary_error = None
    try:
        out.mkdir(parents=True)
        frozen.persist(out / "evaluation.json", report)
        actual = frozen.evaluate_model(spec, pairs, device, out, report, guard, root)
        recovered = report["models"]["improved_unet"]
        frozen.require(set(report["models"]) == {"improved_unet"}, "Recovery must not evaluate baseline")
        frozen.require(actual["valid_pixels"] == plan["data"]["test"]["valid_pixels"], "Recovery valid pixels differ")
        frozen.require(recovered["state_sha256_before"] == prior["state_sha256_before"],
                       "Recovered model state differs from original failed attempt")
        report["content_verification_after"] = verify_test_content(pairs, audit)
        frozen.verify_hashes(plan["source_sha256"], project, "recovery source")
        frozen.verify_hashes(plan["protected_original_files_sha256"], root, "original record")
        for chosen in plan["models"]:
            frozen.require(frozen.sha(frozen.inside(root, chosen["checkpoint"])) == chosen["checkpoint_sha256"],
                           "A frozen checkpoint copy changed during recovery")
        frozen.require(frozen.sha(plan_path) == plan["_plan_sha256"], "Recovery plan changed")
        report.update(status="passed", full_test_passes_this_recovery=1, original_attempt_and_baseline_unchanged=True,
                      frozen_sources_and_weights_unchanged=True)
    except BaseException as error:
        primary_error = error
        report.update(status="incomplete", error={"type": type(error).__name__, "message": str(error)})
        for entry in report["models"].values():
            if entry["status"] == "running":
                entry["status"] = "incomplete"
        raise
    finally:
        new = report["models"].get("improved_unet", {})
        report["exposure"].update(new_improved_forward_calls=new.get("forward_calls", 0),
                                  new_improved_forward_samples=new.get("forward_samples", 0),
                                  repeated_prior_samples=min(prior["forward_samples"], new.get("forward_samples", 0)),
                                  cumulative_improved_forward_calls=prior["forward_calls"] + new.get("forward_calls", 0),
                                  cumulative_improved_forward_samples=prior["forward_samples"] + new.get("forward_samples", 0))
        frozen.finalize_report(report, out, guard, primary_error)
    print(f"Completed authorized Improved-only recovery; baseline was not rerun: {out}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=PROJECT.parents[3] / "Final Report/HipMRI_2D_Improved_UNet_学习/data/keras_slices_data")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    execute(parser.parse_args())


if __name__ == "__main__":
    main()
