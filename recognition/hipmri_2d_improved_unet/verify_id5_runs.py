"""Verify new ID5-selected 20-epoch runs and replay validation, without training."""

import argparse
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import HipMRIPairDataset, collate_padded, collect_pairs
from modules import build_model
from resource_monitor import StageMeasurement
from summarize_formal_runs import (
    EPOCHS, MODEL_NAMES, TOTAL_UPDATES, TRAIN_BATCHES, TRAIN_SAMPLES,
    VALIDATION_BATCHES, VALIDATION_SAMPLES, TRAINING_WALL_CLOCK_NOTE,
    finite_number, read_json, require, sha256_file, verify_checkpoint,
    verify_metrics, verify_source_hashes, verify_stages,
)
from train import manifest_hash, run_epoch
from verify_selected_validation import state_sha256


PROJECT = Path(__file__).resolve().parent
REPO = PROJECT.parents[1]
ID5_RULE = "Maximum full-validation pixel-pooled ID5 Dice; exact ties choose lower CE, then earlier epoch"
CE_RULE = "Minimum full-validation valid-pixel CE; exact ties choose earlier epoch"
FILES = {"primary": "best.pt", "id5_best": "id5_best.pt", "ce_best": "ce_best.pt", "last": "last.pt"}


def candidate(record):
    """Independent ranking input; no dependency on the training selector."""
    epoch = record["epoch"]
    require(isinstance(epoch, int) and not isinstance(epoch, bool) and epoch > 0, "Epoch must be a positive integer")
    validation = record["validation"]
    ce = finite_number(validation["cross_entropy"], "validation CE", minimum=0)
    values = validation["dice_per_class"]
    require(len(values) > 5 and values[5] is not None, "A defined validation ID5 Dice is required")
    dice = finite_number(values[5], "validation ID5 Dice", minimum=0)
    require(dice <= 1, "Validation ID5 Dice exceeds one")
    return {"epoch": epoch, "validation_cross_entropy": ce, "validation_id5_dice": dice}


def independent_selections(records):
    """Exact numeric ties use CE and then epoch, without a tolerance band."""
    require(bool(records), "Cannot rank an empty history")
    values = [candidate(record) for record in records]
    require(len({value["epoch"] for value in values}) == len(values), "Duplicate history epochs")
    by_id5 = min(values, key=lambda value: (-value["validation_id5_dice"], value["validation_cross_entropy"], value["epoch"]))
    by_ce = min(values, key=lambda value: (value["validation_cross_entropy"], value["epoch"]))
    last = max(values, key=lambda value: value["epoch"])
    return {"primary": by_id5, "id5_best": by_id5, "ce_best": by_ce, "last": last}


def protected_paths(plan, root=REPO):
    """Resolve only declared old experiment files inside the current repository."""
    protected = plan.get("protected_original_files_sha256")
    require(isinstance(protected, dict) and bool(protected), "Plan must freeze original experiment file hashes")
    root = Path(root).resolve()
    paths = {}
    for name, digest in protected.items():
        relative = Path(name)
        require(not relative.is_absolute(), "Protected original path must be repository-relative")
        path = (root / relative).resolve()
        require(path.is_relative_to(root), "Protected original path leaves the repository")
        require(isinstance(digest, str) and len(digest) == 64, "Invalid protected SHA256")
        require(path.is_file() and sha256_file(path) == digest, f"Protected original file changed: {name}")
        paths[name] = digest
    return paths


def tensor_state_digest(weights):
    digest = hashlib.sha256()
    for name, value in sorted(weights.items()):
        cpu = value.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(cpu.dtype), list(cpu.shape)]).encode("utf-8"))
        digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


def verify_run(report, run_dir, model_name, plan, plan_hash):
    """Verify budgets, all history metrics and the three independently chosen roles."""
    protocol = plan["shared_protocol"]
    require(report["status"] == "passed" and report["formal_run"] is True, f"{model_name}: incomplete formal run")
    require(report["model"] == model_name, f"{model_name}: wrong model")
    require(report["formal_plan_sha256"] == plan_hash and report["formal_source_revision"] == plan["source_revision"], f"{model_name}: plan/source revision differs")
    require(report["limited_training_smoke"] is False and report["test_split_read"] is False, f"{model_name}: wrong data scope")
    require(report["model_config"] == protocol["model_config"] and report["seed"] == protocol["seed"], f"{model_name}: config/seed differs")
    require(report["optimizer"] == {"name": "Adam", "lr": protocol["optimizer"]["learning_rate"]}, f"{model_name}: optimizer differs")
    require(report["settings"] == {"epochs": EPOCHS, "batch_size": 4, "num_workers": 0, "cudnn_deterministic": True, "cudnn_benchmark": False}, f"{model_name}: settings differ")
    require(report["completed_epochs"] == EPOCHS and report["optimizer_steps"] == TOTAL_UPDATES, f"{model_name}: budget incomplete")
    data = report["data"]
    expected_data = {
        "full_training_pairs": TRAIN_SAMPLES, "selected_training_pairs": TRAIN_SAMPLES,
        "full_validation_pairs": VALIDATION_SAMPLES, "selected_training_case_groups": 31,
        "training_case_groups": 31, "validation_case_groups": 4, "shared_case_ids": [],
        "training_manifest_sha256": plan["data"]["training"]["manifest_sha256"],
        "validation_manifest_sha256": plan["data"]["validation"]["manifest_sha256"],
        "selection_policy": "all training pairs",
    }
    for key, expected in expected_data.items():
        require(data[key] == expected, f"{model_name}: wrong {key}")
    names = data["selected_training_filenames"]
    require(len(names) == len(set(names)) == TRAIN_SAMPLES, f"{model_name}: training filenames differ")
    require(all(isinstance(name, str) and Path(name).name == name and name.startswith("case_") for name in names), f"{model_name}: invalid filename")
    text = "\n".join(f"{name}|{name.replace('case_', 'seg_', 1)}" for name in names)
    require(hashlib.sha256(text.encode("utf-8")).hexdigest() == data["training_manifest_sha256"], f"{model_name}: filenames do not match manifest")
    history = report["epochs"]
    require([item["epoch"] for item in history] == list(range(1, EPOCHS + 1)), f"{model_name}: incomplete epochs")
    for record in history:
        for split, samples, batches, steps in (
            ("training", TRAIN_SAMPLES, TRAIN_BATCHES, TRAIN_BATCHES),
            ("validation", VALIDATION_SAMPLES, VALIDATION_BATCHES, 0),
        ):
            item = record[split]
            label = f"{model_name} epoch{record['epoch']} {split}"
            require((item["samples"], item["batches"], item["optimizer_steps"]) == (samples, batches, steps), f"{label}: counts differ")
            verify_metrics(item, label, report["model_config"]["num_classes"])
            require(item["valid_pixels"] == history[0][split]["valid_pixels"], f"{label}: valid pixels changed")
        norms = record["training"]["gradient_norm_range"]
        require(len(norms) == 2 and 0 < finite_number(norms[0], "gradient minimum") <= finite_number(norms[1], "gradient maximum"), f"{model_name}: invalid gradients")
    require(sum(record["training"]["optimizer_steps"] for record in history) == TOTAL_UPDATES, f"{model_name}: updates differ")
    selections = independent_selections(history)
    declared = report["checkpoint_selection"]
    require(declared["primary_metric"] == "id5_dice" and declared["files"] == FILES, f"{model_name}: wrong selection mode/files")
    for role, expected in selections.items():
        require(declared[role] == expected, f"{model_name}: declared {role} differs from independent ranking")
    reload = report["checkpoint_reload"]
    require(reload["passed"] is True and reload["selection_metric"] == "id5_dice", f"{model_name}: restoration did not pass")
    require(reload["best_epoch"] == selections["primary"]["epoch"], f"{model_name}: wrong restored epoch")
    require(reload["best_validation_cross_entropy"] == selections["primary"]["validation_cross_entropy"], f"{model_name}: wrong restored CE")
    require(reload["best_validation_id5_dice"] == selections["primary"]["validation_id5_dice"], f"{model_name}: wrong restored Dice")
    finite_number(reload["reference_logit_max_abs_difference"], "restored reference difference", minimum=0)
    finite_number(report["elapsed_seconds"], "run elapsed", minimum=0)
    checkpoint_checks = {}
    for role, filename in FILES.items():
        chosen = selections[role]
        path = run_dir / filename
        check = verify_checkpoint(path, report, chosen["epoch"], chosen["validation_cross_entropy"], chosen["epoch"] * TRAIN_BATCHES, plan)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        require(checkpoint["selection_metric"] == "id5_dice", f"{model_name}/{filename}: wrong selection mode")
        require(checkpoint["validation_id5_dice"] == chosen["validation_id5_dice"], f"{model_name}/{filename}: wrong Dice metadata")
        snapshot = checkpoint["checkpoint_selection"]
        for key, value in independent_selections(history[:chosen["epoch"]]).items():
            require(snapshot[key] == value, f"{model_name}/{filename}: stale selection snapshot {key}")
        check["validation_id5_dice"] = checkpoint["validation_id5_dice"]
        check["model_state_sha256"] = tensor_state_digest(checkpoint["model_state"])
        checkpoint_checks[role] = check
        del checkpoint
    require(checkpoint_checks["primary"]["sha256"] == checkpoint_checks["id5_best"]["sha256"], f"{model_name}: best.pt is not a byte-identical ID5-best alias")
    require(checkpoint_checks["primary"]["model_state_sha256"] == checkpoint_checks["id5_best"]["model_state_sha256"], f"{model_name}: primary and ID5-best tensors differ")
    resources = verify_stages(report)
    return selections, checkpoint_checks, resources


def compare_original_trajectory(report, original):
    """Report actual deterministic agreement without assuming it from equal seeds."""
    differences = []
    max_ce_difference = 0.0
    require(len(report["epochs"]) == len(original["epochs"]) == EPOCHS, "Original/new histories have different budgets")
    keys = ("confusion_matrix", "dice_per_class", "iou_per_class", "mean_dice_excluding_class0",
            "mean_iou_excluding_class0", "pixel_accuracy", "valid_pixels", "samples", "batches", "optimizer_steps")
    for new, old in zip(report["epochs"], original["epochs"]):
        for split in ("training", "validation"):
            for key in keys:
                if new[split][key] != old[split][key]:
                    differences.append({"epoch": new["epoch"], "split": split, "field": key})
            delta = abs(new[split]["cross_entropy"] - old[split]["cross_entropy"])
            max_ce_difference = max(max_ce_difference, delta)
            if delta != 0:
                differences.append({"epoch": new["epoch"], "split": split, "field": "cross_entropy", "absolute_difference": delta})
            if split == "training" and new[split].get("gradient_norm_range") != old[split].get("gradient_norm_range"):
                differences.append({"epoch": new["epoch"], "split": split, "field": "gradient_norm_range"})
    return {"all_epoch_metrics_equal_exactly": not differences, "maximum_ce_absolute_difference": max_ce_difference,
            "differences": differences, "scope": "Actual per-epoch metrics, excluding wall time/resources; not an assumption from matching seeds or a cross-environment reproducibility guarantee."}


def compare_state_trees(new, old, path="state"):
    """Compare all model/Adam tensors exactly while retaining explicit mismatches."""
    differences, tensor_count = [], 0
    if isinstance(new, torch.Tensor) or isinstance(old, torch.Tensor):
        tensor_count = 1
        equal = (isinstance(new, torch.Tensor) and isinstance(old, torch.Tensor)
                 and new.dtype == old.dtype and new.shape == old.shape
                 and torch.equal(new.detach().cpu(), old.detach().cpu()))
        if not equal:
            differences.append(path)
    elif isinstance(new, dict) and isinstance(old, dict):
        if set(new) != set(old):
            differences.append(path + ".keys")
        for key in new.keys() & old.keys():
            delta, count = compare_state_trees(new[key], old[key], f"{path}.{key}")
            differences.extend(delta)
            tensor_count += count
    elif isinstance(new, (list, tuple)) and isinstance(old, (list, tuple)):
        if type(new) is not type(old) or len(new) != len(old):
            differences.append(path + ".structure")
        for index, (left, right) in enumerate(zip(new, old)):
            delta, count = compare_state_trees(left, right, f"{path}[{index}]")
            differences.extend(delta)
            tensor_count += count
    elif type(new) is not type(old) or new != old:
        differences.append(path)
    return sorted(differences), tensor_count


def compare_original_checkpoints(new_dir, original_dir):
    result = {}
    for new_name, old_name in (("ce_best.pt", "best.pt"), ("last.pt", "last.pt")):
        new_path, old_path = new_dir / new_name, original_dir / old_name
        new = torch.load(new_path, map_location="cpu", weights_only=True)
        old = torch.load(old_path, map_location="cpu", weights_only=True)
        entry = {"new_file": new_name, "original_file": old_name,
                 "new_file_sha256": sha256_file(new_path), "original_file_sha256": sha256_file(old_path),
                 "new_epoch": new["epoch"], "original_epoch": old["epoch"]}
        for field in ("model_state", "optimizer_state"):
            differences, count = compare_state_trees(new[field], old[field], field)
            entry[field] = {"equal_exactly": not differences, "compared_tensor_count": count, "differing_paths": differences}
        entry["all_model_and_optimizer_state_equal_exactly"] = all(entry[field]["equal_exactly"] for field in ("model_state", "optimizer_state"))
        entry["scope"] = "Actual CPU per-tensor and non-tensor model/Adam-state comparison; new checkpoint selection metadata/file bytes may differ. A mismatch is reported rather than replaced by an assumed equal-seed result."
        result[new_name.removesuffix(".pt")] = entry
        del new, old
    return result


def replay_id5(name, report, run_dir, pairs, device):
    chosen = independent_selections(report["epochs"])["id5_best"]
    expected = report["epochs"][chosen["epoch"] - 1]["validation"]
    path = run_dir / "id5_best.pt"
    before_file = sha256_file(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(name, **checkpoint["model_config"]).to(device).eval()
    model.load_state_dict(checkpoint["model_state"], strict=True)
    del checkpoint
    before_state = state_sha256(model)
    forwards = 0

    def check_forward(module, inputs, output):
        nonlocal forwards
        require(not module.training and not torch.is_grad_enabled(), f"{name}: validation must disable training/gradients")
        forwards += 1

    hook = model.register_forward_hook(check_forward)
    loader = DataLoader(HipMRIPairDataset(pairs), batch_size=4, shuffle=False, num_workers=0, collate_fn=collate_padded)
    measurement = StageMeasurement(device, f"post_training_id5_best_validation_replay_{name}", "One restored ID5-selected model, validation inputs/logits; no optimizer or training gradients; allocator cache retained. Post-training verification, separate from training-stage resource records.")
    try:
        with measurement:
            actual = run_epoch(model, loader, device, num_classes=6)
    finally:
        hook.remove()
    require((actual["samples"], actual["batches"], actual["optimizer_steps"]) == (VALIDATION_SAMPLES, VALIDATION_BATCHES, 0), f"{name}: replay budget differs")
    verify_metrics(actual, f"{name} restored validation", 6)
    for key in ("confusion_matrix", "dice_per_class", "iou_per_class", "mean_dice_excluding_class0", "mean_iou_excluding_class0", "pixel_accuracy", "valid_pixels"):
        require(actual[key] == expected[key], f"{name}: restored {key} differs from selected epoch")
    ce_delta = abs(actual["cross_entropy"] - expected["cross_entropy"])
    require(ce_delta <= 1e-6, f"{name}: restored CE differs beyond 1e-6")
    after_state = state_sha256(model)
    require(before_state == after_state and forwards == VALIDATION_BATCHES, f"{name}: model state/forward count changed")
    require(all(parameter.grad is None for parameter in model.parameters()), f"{name}: replay created gradients")
    require(sha256_file(path) == before_file, f"{name}: replay modified checkpoint")
    return {"selected_epoch": chosen["epoch"], "checkpoint_sha256": before_file,
            "expected_selected_epoch_validation": expected, "replayed_validation": actual,
            "ce_absolute_difference": ce_delta, "state_sha256_before": before_state,
            "state_sha256_after": after_state, "forward_calls": forwards, "optimizer_steps": 0,
            "parameters_have_no_gradients": True, "resource_measurement": measurement.result,
            "status": "passed", "test_split_read": False}


def plot_curves(reports, selections, output_dir):
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    for name, color in zip(MODEL_NAMES, ("tab:blue", "tab:orange")):
        label = "Baseline" if name == "baseline_unet" else "Improved"
        history = reports[name]["epochs"]
        epochs = [row["epoch"] for row in history]
        axes[0].plot(epochs, [row["validation"]["cross_entropy"] for row in history], color=color, label=f"{label} validation CE")
        axes[1].plot(epochs, [row["validation"]["dice_per_class"][5] for row in history], color=color, label=f"{label} validation ID5 Dice")
        for role, marker in (("id5_best", "*"), ("ce_best", "s")):
            chosen = selections[name][role]
            for axis, key in ((axes[0], "validation_cross_entropy"), (axes[1], "validation_id5_dice")):
                axis.scatter(chosen["epoch"], chosen[key], color=color, marker=marker, s=130 if marker == "*" else 45, edgecolors="black", linewidths=.6, label=f"{label} {role}, epoch {chosen['epoch']}", zorder=4)
    axes[0].set(title="Validation probability loss", ylabel="Valid-pixel cross-entropy")
    axes[1].set(title="Primary selection objective", ylabel="Pixel-pooled validation ID5 Dice", ylim=(0, 1))
    for axis in axes:
        axis.set(xlabel="Complete training epoch", xticks=list(range(1, EPOCHS + 1)), xlim=(.5, EPOCHS + .5))
        axis.tick_params(axis="x", labelsize=8)
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    fig.suptitle("New fixed-20-epoch runs: ID5 Dice selected; ties lower CE, then earlier epoch\nID5 = prostate under accepted derivative-ID-preservation assumption; validation only", fontsize=10)
    fig.savefig(output_dir / "id5_selection_curves.png", dpi=160)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--original-plan", type=Path, default=PROJECT / "evidence/formal_run_plan_20261026.json")
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--improved-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=PROJECT.parents[3] / "Final Report/HipMRI_2D_Improved_UNet_学习/data/keras_slices_data")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    require(not args.output_dir.exists(), "Use a new verification output directory")
    plan, original_plan = read_json(args.plan), read_json(args.original_plan)
    plan_hash, original_plan_hash = sha256_file(args.plan), sha256_file(args.original_plan)
    require(plan["status"] == "approved_for_execution" and plan["user_confirmation_received"] is True, "New plan lacks explicit approval")
    require(plan["epochs_per_model"] == EPOCHS and plan["training_budget"]["planned_optimizer_updates_per_model_if_complete"] == TOTAL_UPDATES, "New plan budget differs")
    require(plan["label_mapping_assumption_accepted"] is True and plan["test_split_read"] is False, "New plan label assumption/data scope differs")
    protected = protected_paths(plan)
    require(str(args.original_plan.resolve().relative_to(REPO)).replace("\\", "/") in {str(Path(name)).replace("\\", "/") for name in protected}, "Protected inventory must include the original formal plan")
    source_hashes = verify_source_hashes(plan)
    for name, digest in original_plan["source_sha256"].items():
        if name != "train.py":
            require(source_hashes.get(name) == digest, f"Unapproved non-selection core change: {name}")
    run_dirs = {"baseline_unet": args.baseline_dir, "improved_unet": args.improved_dir}
    planned_runs = {row["model"]: (REPO / row["output_dir"]).resolve() for row in plan["runs_in_order"]}
    require(all(run_dirs[name].resolve() == planned_runs[name] for name in MODEL_NAMES), "Run directories differ from plan")
    summaries = {name: path / "run_summary.json" for name, path in run_dirs.items()}
    summary_hashes = {name: sha256_file(path) for name, path in summaries.items()}
    reports = {name: read_json(path) for name, path in summaries.items()}
    torch.set_num_threads(1)
    selections, checks, resources = {}, {}, {}
    for name in MODEL_NAMES:
        print(f"Verifying completed {name} history and all checkpoint roles", flush=True)
        selections[name], checks[name], resources[name] = verify_run(reports[name], run_dirs[name], name, plan, plan_hash)
    for key in ("model_config", "optimizer", "seed", "settings", "objective", "selection_metric", "device", "gpu_name", "math_settings", "versions", "data"):
        require(reports[MODEL_NAMES[0]][key] == reports[MODEL_NAMES[1]][key], f"Paired shared control differs: {key}")
    require(checks[MODEL_NAMES[0]]["primary"]["optimizer_group_settings"] == checks[MODEL_NAMES[1]]["primary"]["optimizer_group_settings"], "Paired Adam controls differ")
    device = torch.device(args.device)
    require(device.type != "cuda" or torch.cuda.is_available(), "CUDA is unavailable")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    math_settings = {"matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32, "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32, "float32_matmul_precision": torch.get_float32_matmul_precision()}
    require(all(report["math_settings"] == math_settings for report in reports.values()), "Replay backend math settings differ")
    pairs = collect_pairs(args.data_root / "keras_slices_validate", args.data_root / "keras_slices_seg_validate")
    require(len(pairs) == VALIDATION_SAMPLES and manifest_hash(pairs) == plan["data"]["validation"]["manifest_sha256"], "Validation manifest differs")
    replay, trajectories, original_checkpoint_comparisons = {}, {}, {}
    original_runs = {row["model"]: REPO / row["output_dir"] for row in original_plan["runs_in_order"]}
    for name in MODEL_NAMES:
        original_summary = original_runs[name] / "run_summary.json"
        relative_summary = str(original_summary.relative_to(REPO)).replace("\\", "/")
        require(relative_summary in {str(Path(path)).replace("\\", "/") for path in protected}, "Protected inventory must include original run summaries")
        trajectories[name] = compare_original_trajectory(reports[name], read_json(original_summary))
        original_checkpoint_comparisons[name] = compare_original_checkpoints(run_dirs[name], original_runs[name])
        print(f"Replaying {name} ID5-best epoch {selections[name]['id5_best']['epoch']} on all 660 validation pairs", flush=True)
        replay[name] = replay_id5(name, reports[name], run_dirs[name], pairs, device)
    require(protected_paths(plan) == protected, "Original artifacts changed during verification")
    require(sha256_file(args.plan) == plan_hash and sha256_file(args.original_plan) == original_plan_hash, "Plan changed during verification")
    require(verify_source_hashes(plan) == source_hashes, "Frozen source changed during verification")
    require(all(sha256_file(path) == summary_hashes[name] for name, path in summaries.items()), "A new run summary changed during verification")
    output = {"analysis_kind": "verified_id5_selected_fixed_budget_train_validation_comparison",
              "performed_by": "Codex automation; not user-performed verification",
              "created_at": datetime.now(timezone(timedelta(hours=10))).isoformat(), "status": "passed_verification",
              "formal_plan_sha256": plan_hash, "formal_source_revision": plan["source_revision"],
              "original_formal_plan_sha256": original_plan_hash, "source_sha256": source_hashes,
              "verification_script_sha256": sha256_file(Path(__file__)), "selection_rule": ID5_RULE,
              "ce_retention_rule": CE_RULE, "checkpoint_files": FILES,
              "budget": {"epochs_per_model": EPOCHS, "optimizer_updates_per_model": TOTAL_UPDATES},
              "data": deepcopy(plan["data"]), "protected_original_files_sha256": protected,
              "original_artifacts_unchanged": True, "new_training_by_verifier": False, "test_split_read": False,
              "label_mapping_boundary": "ID5=prostate remains the user-accepted unverified derivative-ID-preservation assumption.",
              "aggregation": "Full-validation valid-pixel pooled confusion counts; not mean batch or case-balanced Dice.",
              "training_wall_clock_note": TRAINING_WALL_CLOCK_NOTE,
              "replay_runtime": {"device": str(device), "math_settings": math_settings, "torch": str(torch.__version__), "cuda": torch.version.cuda},
              "results": {}, "claim_boundary": "New validation-selected comparison, one seed and fixed budget. Original CE-selected experiment preserved. No test evaluation, statistical-significance or module-causality claim."}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name in MODEL_NAMES:
        selected = selections[name]["id5_best"]
        validation = reports[name]["epochs"][selected["epoch"] - 1]["validation"]
        output["results"][name] = {"selected_epoch": selected["epoch"], "parameters": reports[name]["parameters"],
                                  "validation_id5_dice_pixel_pooled": validation["dice_per_class"][5],
                                  "validation_id5_iou_pixel_pooled": validation["iou_per_class"][5],
                                  "validation_cross_entropy": validation["cross_entropy"],
                                  "selected_validation": validation, "selections": selections[name],
                                  "checkpoint_verification": checks[name], "phase_resource_summary": resources[name],
                                  "run_elapsed_seconds": reports[name]["elapsed_seconds"],
                                  "run_summary_sha256": summary_hashes[name],
                                  "prior_ce_run_summary_sha256": sha256_file(original_runs[name] / "run_summary.json"),
                                  "trajectory_vs_original_ce_run": trajectories[name],
                                  "checkpoint_states_vs_original_ce_run": original_checkpoint_comparisons[name],
                                  "selected_validation_replay": replay[name]}
        compact = deepcopy(reports[name])
        omitted = compact["data"].pop("selected_training_filenames")
        compact["data"]["selected_training_filenames_omitted_count"] = len(omitted)
        compact["source_sha256"] = source_hashes
        compact["analysis_verification"] = {"checkpoint_verification": checks[name], "all_history_metrics_recomputed": True,
                                             "source_and_manifest_hashes_match": True, "summary_sha256": summary_hashes[name],
                                             "selected_validation_replay": replay[name]}
        (args.output_dir / f"{name.removesuffix('_unet')}_report.json").write_text(json.dumps(compact, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    plot_curves(reports, selections, args.output_dir)
    output["curves_file"] = "id5_selection_curves.png"
    (args.output_dir / "comparison.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Verified new ID5-selected runs and both restored validation replays: {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
