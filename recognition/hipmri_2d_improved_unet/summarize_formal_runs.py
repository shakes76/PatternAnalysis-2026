"""Verify and summarize completed formal train/validation runs, without inference."""

import argparse
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path

import torch


PROJECT = Path(__file__).resolve().parent
REPO = PROJECT.parents[1]
MODEL_NAMES = ("baseline_unet", "improved_unet")
EPOCHS = 20
TRAIN_SAMPLES, VALIDATION_SAMPLES = 11460, 660
TRAIN_BATCHES, VALIDATION_BATCHES = 2865, 165
TOTAL_UPDATES = EPOCHS * TRAIN_BATCHES
TRAINING_WALL_CLOCK_NOTE = (
    "Training-stage wall time includes input I/O and metric computation. "
    "Model-run order and filesystem/cache state were not isolated, so differences "
    "in total training time cannot be attributed to architecture-only compute speed. "
    "Use the separate warmed-up synthetic forward profile for pure forward comparisons."
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reject_json_constant(value):
    raise ValueError(f"Non-finite JSON constant is not accepted: {value}")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject_json_constant)


def finite_number(value, label, minimum=None):
    require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{label}: expected a number")
    value = float(value)
    require(math.isfinite(value), f"{label}: non-finite value")
    if minimum is not None:
        require(value >= minimum, f"{label}: value below {minimum}")
    return value


def equal_score(actual, expected, label):
    if expected is None:
        require(actual is None, f"{label}: expected an undefined score")
    else:
        value = finite_number(actual, label)
        require(math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-12), f"{label}: score differs from confusion counts")


def verify_metrics(record, label, num_classes):
    matrix = record["confusion_matrix"]
    require(isinstance(matrix, list) and len(matrix) == num_classes, f"{label}: invalid confusion shape")
    require(all(isinstance(row, list) and len(row) == num_classes for row in matrix), f"{label}: invalid confusion rows")
    require(all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for row in matrix for value in row), f"{label}: confusion counts must be nonnegative integers")
    total = sum(sum(row) for row in matrix)
    require(total == record["valid_pixels"] and total > 0, f"{label}: confusion sum differs from valid pixels")
    require(len(record["dice_per_class"]) == len(record["iou_per_class"]) == num_classes, f"{label}: wrong class-score count")
    dice, iou = [], []
    for class_id in range(num_classes):
        true_positive = matrix[class_id][class_id]
        reference_count = sum(matrix[class_id])
        prediction_count = sum(row[class_id] for row in matrix)
        denominator = reference_count + prediction_count
        dice.append(2 * true_positive / denominator if denominator else None)
        union = denominator - true_positive
        iou.append(true_positive / union if union else None)
        equal_score(record["dice_per_class"][class_id], dice[-1], f"{label}: ID{class_id} Dice")
        equal_score(record["iou_per_class"][class_id], iou[-1], f"{label}: ID{class_id} IoU")
    for key, values in (("mean_dice_excluding_class0", dice), ("mean_iou_excluding_class0", iou)):
        defined = [value for value in values[1:] if value is not None]
        equal_score(record[key], sum(defined) / len(defined) if defined else None, f"{label}: {key}")
    equal_score(record["pixel_accuracy"], sum(matrix[i][i] for i in range(num_classes)) / total, f"{label}: accuracy")
    finite_number(record["cross_entropy"], f"{label}: CE", minimum=0)
    finite_number(record["elapsed_seconds"], f"{label}: elapsed", minimum=0)


def verify_source_hashes(plan):
    hashes = {}
    for name, expected in plan["source_sha256"].items():
        path = (PROJECT / name).resolve()
        require(path.is_relative_to(PROJECT), f"Source path leaves the project: {name}")
        hashes[name] = sha256_file(path)
        require(hashes[name] == expected, f"Frozen source hash changed: {name}")
    return hashes


def verify_checkpoint(path, report, epoch, validation_ce, expected_updates, plan):
    """Read tensors on CPU only; do not construct a model or run a forward."""
    before_hash = sha256_file(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("model", "model_config", "seed"):
        require(checkpoint[key] == report[key], f"{path.name}: {key} differs from run report")
    require(checkpoint["epoch"] == epoch, f"{path.name}: wrong checkpoint epoch")
    require(checkpoint["limited_training_smoke"] is False, f"{path.name}: checkpoint is marked smoke")
    require(checkpoint["validation_cross_entropy"] == validation_ce, f"{path.name}: wrong validation CE")
    weights = checkpoint["model_state"]
    require(isinstance(weights, dict) and bool(weights), f"{path.name}: missing model state")
    for name, tensor in weights.items():
        require(isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu", f"{path.name}: invalid CPU weight {name}")
        require(bool(torch.isfinite(tensor).all()), f"{path.name}: non-finite model weight {name}")
    # Both frozen architectures have parameter-only state_dicts: their instance
    # normalization tracks no running statistics, and neither has batch norm.
    require(sum(tensor.numel() for tensor in weights.values()) == report["parameters"], f"{path.name}: parameter count disagrees with frozen architecture")
    optimizer = checkpoint["optimizer_state"]
    states, groups = optimizer["state"], optimizer["param_groups"]
    require(bool(states) and bool(groups), f"{path.name}: missing Adam state")
    parameter_ids = [parameter_id for group in groups for parameter_id in group["params"]]
    require(len(parameter_ids) == len(set(parameter_ids)) == len(weights), f"{path.name}: optimizer parameter inventory differs from model")
    require(set(parameter_ids) == set(states), f"{path.name}: missing optimizer states")
    group_settings = []
    approved_optimizer = plan["shared_protocol"]["optimizer"]
    for group in groups:
        require(group["lr"] == approved_optimizer["learning_rate"], f"{path.name}: learning rate changed")
        require(group["weight_decay"] == approved_optimizer["weight_decay"], f"{path.name}: weight decay changed")
        group_settings.append({key: value for key, value in group.items() if key != "params"})
    step_values = []
    for parameter_id, state in states.items():
        require("step" in state and "exp_avg" in state and "exp_avg_sq" in state, f"{path.name}: incomplete Adam state {parameter_id}")
        step = state["step"]
        if isinstance(step, torch.Tensor):
            require(step.device.type == "cpu" and step.numel() == 1, f"{path.name}: invalid CPU Adam step")
            step = step.item()
        step = finite_number(step, f"{path.name}: Adam step", minimum=0)
        require(step == expected_updates, f"{path.name}: Adam step does not equal epoch x {TRAIN_BATCHES}")
        step_values.append(int(step))
        for field, value in state.items():
            if isinstance(value, torch.Tensor):
                require(value.device.type == "cpu" and bool(torch.isfinite(value).all()), f"{path.name}: non-finite or non-CPU optimizer tensor {field}")
    require(sha256_file(path) == before_hash, f"Checkpoint changed while being verified: {path.name}")
    result = {
        "file": path.name, "sha256": before_hash, "epoch": epoch,
        "model_metadata_match": True, "model_weights_finite": True,
        "model_state_tensor_count": len(weights), "optimizer_state_count": len(states),
        "all_optimizer_steps_equal": expected_updates,
        "optimizer_step_range": [min(step_values), max(step_values)],
        "optimizer_group_settings": group_settings,
        "verification_scope": "CPU torch.load and tensor/metadata checks only; no new forward, inference or training",
    }
    del checkpoint, weights, optimizer, states
    return result


def verify_stages(report):
    stages = report["resource_stages"]
    expected_names = {"initial_reference_probe", "checkpoint_restoration"}
    expected_names.update(f"{split}_epoch_{epoch}" for split in ("training", "validation") for epoch in range(1, EPOCHS + 1))
    names = [stage["stage"] for stage in stages]
    require(len(names) == len(set(names)) and set(names) == expected_names, f"{report['model']}: incomplete or duplicate resource stages")
    cuda = str(report["device"]).startswith("cuda")
    for stage in stages:
        require(stage["status"] == "passed", f"{stage['stage']}: resource stage failed")
        require(stage["device"] == report["device"], f"{stage['stage']}: stage device changed")
        require(bool(stage["resident_state"]), f"{stage['stage']}: missing resident-state scope")
        finite_number(stage["elapsed_seconds"], f"{stage['stage']}: elapsed", minimum=0)
        for kind in ("allocated", "reserved"):
            keys = [f"{point}_cuda_{kind}_bytes" for point in ("start", "peak", "end")]
            if cuda:
                require(all(isinstance(stage[key], int) and stage[key] >= 0 for key in keys), f"{stage['stage']}: missing CUDA counters")
                require(stage[keys[1]] >= max(stage[keys[0]], stage[keys[2]]), f"{stage['stage']}: invalid peak counter")
            else:
                require(all(stage[key] is None for key in keys), f"{stage['stage']}: CPU memory counter must be null")
    summary = {}
    for group in ("training", "validation", "checkpoint_restoration"):
        selected = [stage for stage in stages if stage["stage"].startswith(group + "_epoch_") or stage["stage"] == group]
        entry = {
            "record_count": len(selected),
            "elapsed_seconds_sum": sum(stage["elapsed_seconds"] for stage in selected),
            "resident_state_descriptions": list(dict.fromkeys(stage["resident_state"] for stage in selected)),
        }
        for kind in ("allocated", "reserved"):
            key = f"peak_cuda_{kind}_bytes"
            maximum = max(selected, key=lambda stage: stage[key]) if cuda else None
            entry[key] = maximum[key] if maximum else None
            entry[f"{kind}_peak_record"] = maximum["stage"] if maximum else None
        entry["scope"] = "Maximum absolute PyTorch allocator peak within this phase; includes stated resident objects and cached reserved memory. Not a delta, total GPU VRAM or an isolated optimizer-free model measurement."
        if group == "training":
            entry["wall_clock_note"] = TRAINING_WALL_CLOCK_NOTE
        summary[group] = entry
    return summary


def verify_run(report, run_dir, model_name, plan, plan_hash):
    require(report["status"] == "passed", f"{model_name}: wait until the run reports passed")
    require(report["model"] == model_name and report["formal_run"] is True, f"{model_name}: wrong run identity")
    require(report["formal_plan_sha256"] == plan_hash, f"{model_name}: formal plan hash differs")
    require(report["formal_source_revision"] == plan["source_revision"], f"{model_name}: wrong frozen source revision")
    require(report["limited_training_smoke"] is False and report["test_split_read"] is False, f"{model_name}: invalid data scope")
    require(report["model_config"] == plan["shared_protocol"]["model_config"], f"{model_name}: model configuration differs")
    require(report["seed"] == plan["shared_protocol"]["seed"], f"{model_name}: seed differs")
    require(report["optimizer"] == {"name": "Adam", "lr": plan["shared_protocol"]["optimizer"]["learning_rate"]}, f"{model_name}: optimizer differs")
    require(report["settings"] == {"epochs": EPOCHS, "batch_size": 4, "num_workers": 0, "cudnn_deterministic": True, "cudnn_benchmark": False}, f"{model_name}: shared settings differ")
    require(report["completed_epochs"] == EPOCHS and report["optimizer_steps"] == TOTAL_UPDATES, f"{model_name}: incomplete training budget")
    data = report["data"]
    for key, expected in {
        "full_training_pairs": TRAIN_SAMPLES, "selected_training_pairs": TRAIN_SAMPLES,
        "full_validation_pairs": VALIDATION_SAMPLES, "selected_training_case_groups": 31,
        "training_case_groups": 31, "validation_case_groups": 4, "shared_case_ids": [],
        "training_manifest_sha256": plan["data"]["training"]["manifest_sha256"],
        "validation_manifest_sha256": plan["data"]["validation"]["manifest_sha256"],
        "selection_policy": "all training pairs",
    }.items():
        require(data[key] == expected, f"{model_name}: wrong {key}")
    filenames = data["selected_training_filenames"]
    require(len(filenames) == len(set(filenames)) == TRAIN_SAMPLES, f"{model_name}: training filename count differs")
    require(all(isinstance(name, str) and Path(name).name == name and name.startswith("case_") for name in filenames), f"{model_name}: invalid selected filename")
    text = "\n".join(f"{name}|{name.replace('case_', 'seg_', 1)}" for name in filenames)
    require(hashlib.sha256(text.encode("utf-8")).hexdigest() == data["training_manifest_sha256"], f"{model_name}: selected filenames do not match manifest hash")
    epochs = report["epochs"]
    require([record["epoch"] for record in epochs] == list(range(1, EPOCHS + 1)), f"{model_name}: incomplete epoch sequence")
    for record in epochs:
        for split, samples, batches, updates in (
            ("training", TRAIN_SAMPLES, TRAIN_BATCHES, TRAIN_BATCHES),
            ("validation", VALIDATION_SAMPLES, VALIDATION_BATCHES, 0),
        ):
            item = record[split]
            label = f"{model_name} epoch{record['epoch']} {split}"
            require((item["samples"], item["batches"], item["optimizer_steps"]) == (samples, batches, updates), f"{label}: incomplete samples, batches or updates")
            verify_metrics(item, label, report["model_config"]["num_classes"])
            require(item["valid_pixels"] == epochs[0][split]["valid_pixels"], f"{label}: valid-pixel count changed")
        norms = record["training"]["gradient_norm_range"]
        require(len(norms) == 2 and 0 < finite_number(norms[0], "minimum gradient norm") <= finite_number(norms[1], "maximum gradient norm"), f"{model_name}: invalid gradient range")
    require(sum(record["training"]["optimizer_steps"] for record in epochs) == TOTAL_UPDATES, f"{model_name}: total steps differ")
    selected = min(epochs, key=lambda record: record["validation"]["cross_entropy"])
    epoch, ce = selected["epoch"], selected["validation"]["cross_entropy"]
    reload_result = report["checkpoint_reload"]
    require(reload_result["passed"] is True, f"{model_name}: recorded checkpoint restoration did not pass")
    require(reload_result["best_epoch"] == epoch and reload_result["best_validation_cross_entropy"] == ce, f"{model_name}: best epoch is not earliest minimum-validation-CE epoch")
    finite_number(reload_result["reference_logit_max_abs_difference"], f"{model_name}: recorded reload difference", minimum=0)
    finite_number(report["elapsed_seconds"], f"{model_name}: run elapsed", minimum=0)
    checkpoints = {
        "best": verify_checkpoint(run_dir / "best.pt", report, epoch, ce, epoch * TRAIN_BATCHES, plan),
        "last": verify_checkpoint(run_dir / "last.pt", report, EPOCHS, epochs[-1]["validation"]["cross_entropy"], TOTAL_UPDATES, plan),
    }
    resources = verify_stages(report)
    return selected, checkpoints, resources


def plot_comparison(reports, selections, output_dir):
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    colors = {"baseline_unet": "tab:blue", "improved_unet": "tab:orange"}
    names = {"baseline_unet": "Baseline", "improved_unet": "Improved"}
    for model_name in MODEL_NAMES:
        history = reports[model_name]["epochs"]
        epochs = [record["epoch"] for record in history]
        color, name = colors[model_name], names[model_name]
        for split, style in (("training", "--"), ("validation", "-")):
            axes[0].plot(epochs, [record[split]["cross_entropy"] for record in history], linestyle=style, color=color, label=f"{name} {split}" + (" (online)" if split == "training" else ""))
        dice = [record["validation"]["dice_per_class"][5] for record in history]
        axes[1].plot(epochs, [float("nan") if value is None else value for value in dice], color=color, label=f"{name} validation ID5")
        selected = selections[model_name]
        best_epoch = selected["epoch"]
        axes[0].scatter(best_epoch, selected["validation"]["cross_entropy"], marker="*", s=130, color=color, edgecolors="black", linewidths=0.5, zorder=4, label=f"{name} CE-selected epoch {best_epoch}")
        best_dice = selected["validation"]["dice_per_class"][5]
        if best_dice is not None:
            axes[1].scatter(best_epoch, best_dice, marker="*", s=130, color=color, edgecolors="black", linewidths=0.5, zorder=4, label=f"{name} CE-selected epoch {best_epoch}")
        for axis in axes:
            axis.axvline(best_epoch, color=color, linestyle=":", alpha=0.45)
    axes[0].set(title="Shared cross-entropy objective", ylabel="Valid-pixel CE")
    axes[1].set(title="Pixel-pooled validation ID5 Dice\nID5 = prostate under accepted ID-preservation assumption", ylabel="Full-validation pixel-pooled ID5 Dice", ylim=(0, 1))
    for axis in axes:
        axis.set(xlabel="Complete training epoch", xlim=(0.5, EPOCHS + 0.5), xticks=list(range(1, EPOCHS + 1)))
        axis.tick_params(axis="x", labelsize=8)
        axis.legend(fontsize=8)
        axis.grid(alpha=0.2)
    fig.suptitle("Fixed 20-epoch train/validation runs; stars mark lowest-validation-CE checkpoints", fontsize=11)
    fig.savefig(output_dir / "comparison_curves.png", dpi=160)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=PROJECT / "evidence/formal_run_plan_20261026.json")
    parser.add_argument("--baseline-dir", type=Path, default=REPO / "outputs/baseline_formal_20261026")
    parser.add_argument("--improved-dir", type=Path, default=REPO / "outputs/improved_formal_20261026")
    parser.add_argument("--output-dir", type=Path, default=REPO / "outputs/formal_analysis_20261026")
    return parser.parse_args()


def main():
    args = parse_args()
    require(not args.output_dir.exists(), f"Refusing to overwrite analysis evidence: {args.output_dir}")
    plan_hash = sha256_file(args.plan)
    plan = read_json(args.plan)
    require(plan["status"] == "approved_for_execution" and plan["user_confirmation_received"] is True, "The plan lacks recorded approval")
    require(plan["epochs_per_model"] == EPOCHS and plan["training_budget"]["planned_optimizer_updates_per_model_if_complete"] == TOTAL_UPDATES, "The fixed formal budget differs")
    require(plan["label_mapping_assumption_accepted"] is True and plan["test_split_read"] is False, "The label assumption or data scope differs")
    source_hashes = verify_source_hashes(plan)
    run_dirs = {"baseline_unet": args.baseline_dir, "improved_unet": args.improved_dir}
    summary_paths = {name: directory / "run_summary.json" for name, directory in run_dirs.items()}
    reports = {name: read_json(path) for name, path in summary_paths.items()}
    require(all(report.get("status") == "passed" for report in reports.values()), "Both runs must report passed before formal analysis; no output was created")
    summary_hashes = {name: sha256_file(path) for name, path in summary_paths.items()}
    torch.set_num_threads(1)
    selected, checkpoint_checks, resources = {}, {}, {}
    for name in MODEL_NAMES:
        selected[name], checkpoint_checks[name], resources[name] = verify_run(reports[name], run_dirs[name], name, plan, plan_hash)
    baseline, improved = (reports[name] for name in MODEL_NAMES)
    for key in ("model_config", "optimizer", "seed", "settings", "objective", "selection_metric", "device", "gpu_name", "math_settings", "versions", "data"):
        require(baseline[key] == improved[key], f"Paired run control differs: {key}")
    require(checkpoint_checks["baseline_unet"]["best"]["optimizer_group_settings"] == checkpoint_checks["improved_unet"]["best"]["optimizer_group_settings"], "Paired Adam settings differ")
    for split in ("training", "validation"):
        require(baseline["epochs"][0][split]["valid_pixels"] == improved["epochs"][0][split]["valid_pixels"], f"Paired {split} valid-pixel counts differ")
    require(sha256_file(args.plan) == plan_hash and verify_source_hashes(plan) == source_hashes, "Plan or frozen sources changed during analysis")
    require(all(sha256_file(summary_paths[name]) == summary_hashes[name] for name in MODEL_NAMES), "A run report changed during analysis")
    comparison = {
        "analysis_kind": "verified_fixed_budget_train_validation_comparison",
        "performed_by": "Codex automation; not user-performed verification",
        "created_at": datetime.now(timezone(timedelta(hours=10))).isoformat(),
        "status": "passed_verification",
        "formal_plan_sha256": plan_hash, "formal_source_revision": plan["source_revision"],
        "source_sha256": source_hashes, "analysis_script_sha256": sha256_file(Path(__file__)),
        "budget": {"complete_epochs_per_model": EPOCHS, "optimizer_updates_per_model": TOTAL_UPDATES},
        "data": deepcopy(plan["data"]),
        "selection_rule": "Minimum full-validation valid-pixel CE; exact ties choose earliest epoch",
        "primary_metric": "Selected-checkpoint full-validation pixel-pooled ID5 Dice",
        "organ_interpretation": "ID5 is interpreted as Prostate under the user-accepted assumption that the course derivative preserves source label IDs; derivative no-remapping is not independently verified",
        "aggregation": "Pool valid-pixel confusion counts over the entire validation split; not per-case/volume balanced or average batch Dice",
        "resource_scope": "Training, validation and checkpoint restoration are reported separately. Peaks include each stage's documented resident state and allocator cache, not total GPU VRAM. Run elapsed also contains inter-stage checkpoint/plot/report work.",
        "training_wall_clock_note": TRAINING_WALL_CLOCK_NOTE,
        "results": {}, "test_split_read": False,
        "new_inference_or_training_performed_by_analysis": False,
        "inference_profile": "Not read or merged by this helper; profile results can be added separately after profiling",
        "claim_boundary": "One seed and one fixed 20-epoch budget on the available train/validation splits. Not test performance, case-balanced performance, statistical significance, an ablation or a substitute for new results using old project scores.",
    }
    for name in MODEL_NAMES:
        report, chosen = reports[name], selected[name]
        validation = chosen["validation"]
        comparison["results"][name] = {
            "selected_epoch": chosen["epoch"], "parameters": report["parameters"],
            "validation_id5_dice_pixel_pooled": validation["dice_per_class"][5],
            "validation_id5_iou_pixel_pooled": validation["iou_per_class"][5],
            "validation_dice_per_class": validation["dice_per_class"],
            "validation_iou_per_class": validation["iou_per_class"],
            "validation_macro_dice_excluding_class0": validation["mean_dice_excluding_class0"],
            "validation_macro_iou_excluding_class0": validation["mean_iou_excluding_class0"],
            "validation_cross_entropy": validation["cross_entropy"],
            "validation_valid_pixels": validation["valid_pixels"],
            "run_elapsed_seconds": report["elapsed_seconds"],
            "phase_resource_summary": resources[name], "checkpoint_verification": checkpoint_checks[name],
            "original_summary_sha256": summary_hashes[name],
        }
    # Every completion/source/checkpoint assertion precedes output creation.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name in MODEL_NAMES:
        compact = deepcopy(reports[name])
        omitted = compact["data"].pop("selected_training_filenames")
        compact["data"]["selected_training_filenames_omitted_count"] = len(omitted)
        compact["source_sha256"] = source_hashes
        compact["source_hash_basis"] = plan["source_hash_basis"]
        compact["best_checkpoint_sha256"] = checkpoint_checks[name]["best"]["sha256"]
        compact["last_checkpoint_sha256"] = checkpoint_checks[name]["last"]["sha256"]
        compact["analysis_verification"] = {
            "performed_by": "Codex automation", "original_summary_sha256": summary_hashes[name],
            "frozen_source_and_plan_hashes_match": True, "full_budget_and_epoch_counts_match": True,
            "confusion_counts_and_reported_metrics_match": True,
            "checkpoint_verification": checkpoint_checks[name],
            "training_wall_clock_note": TRAINING_WALL_CLOCK_NOTE,
            "new_forward_or_training": False,
        }
        with (args.output_dir / f"{name.removesuffix('_unet')}_report.json").open("x", encoding="utf-8") as handle:
            json.dump(compact, handle, indent=2, allow_nan=False)
            handle.write("\n")
    plot_comparison(reports, selected, args.output_dir)
    comparison["curves_file"] = "comparison_curves.png"
    with (args.output_dir / "comparison.json").open("x", encoding="utf-8") as handle:
        json.dump(comparison, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Verified both complete formal runs; analysis saved: {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
