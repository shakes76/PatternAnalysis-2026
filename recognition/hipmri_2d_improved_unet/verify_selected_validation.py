"""Replay both selected formal checkpoints on validation only, without training.

The example is chosen from reference labels before either model is restored:
smallest numeric case, its smallest numeric week, then the upper middle slice
among that group's numerically sorted slices containing reference ID5.
"""

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

import torch
from torch.utils.data import DataLoader

from dataset import HipMRIPairDataset, collate_padded, collect_pairs, crop_to_original, load_pair
from modules import build_model
from resource_monitor import StageMeasurement
from train import manifest_hash, run_epoch


MODELS = ("baseline_unet", "improved_unet")
CE_ATOL = 1e-6
NAME_PATTERN = re.compile(r"case_(\d+)_week_(\d+)_slice_(\d+)\.nii\.gz")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def state_sha256(model):
    """Hash CPU copies of every parameter/persistent buffer, including metadata."""
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        cpu = value.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(cpu.dtype), list(cpu.shape)]).encode("utf-8"))
        digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


def select_example(pairs):
    metadata = []
    for index, (image_path, mask_path) in enumerate(pairs):
        match = NAME_PATTERN.fullmatch(image_path.name)
        require(match is not None, f"Unexpected validation MRI name: {image_path.name}")
        case, week, slice_id = (int(part) for part in match.groups())
        metadata.append((case, week, slice_id, image_path.name, index, image_path, mask_path))
    require(bool(metadata), "Validation split is empty.")
    case = min(item[0] for item in metadata)
    week = min(item[1] for item in metadata if item[0] == case)
    group = sorted((item for item in metadata if item[:2] == (case, week)), key=lambda item: (item[2], item[3]))
    eligible = []
    for item in group:
        sample = load_pair(item[5], item[6])
        if bool((sample["target"][sample["valid_mask"]] == 5).any()):
            eligible.append(item)
    require(bool(eligible), "The fixed smallest-case/smallest-week group contains no reference ID5; no fallback selection is made.")
    selected_rank = len(eligible) // 2
    selected = eligible[selected_rank]
    sample = load_pair(selected[5], selected[6])
    return selected[4], sample, {
        "split": "validation",
        "rule": "Smallest numeric case, then that case's smallest numeric week; reference-ID5-positive slices sorted by numeric slice ID, basename as tie-break; take zero-based rank count//2 (upper middle when even). No model output enters selection.",
        "case_id": case, "week": week, "slice_id": selected[2],
        "group_slice_count": len(group), "id5_positive_slice_count": len(eligible),
        "eligible_numeric_slice_ids": [item[2] for item in eligible],
        "selected_eligible_rank_zero_based": selected_rank,
        "validation_manifest_index_zero_based": selected[4],
        "filename": selected[5].name, "mask_filename": selected[6].name,
        "reference_id5_pixels": int(((sample["target"] == 5) & sample["valid_mask"]).sum()),
        "original_size": list(sample["original_size"]),
        "selection_reads": "Only MRI/mask pairs in the metadata-selected validation case/week are materialized; no training or test data.",
    }


def selected_record(summary, plan, model_name, plan_digest):
    require(summary.get("status") == "passed", f"{model_name}: the formal run is not passed.")
    require(summary.get("model") == model_name and summary.get("formal_run") is True, f"{model_name}: not the expected formal run.")
    require(summary.get("formal_plan_sha256") == plan_digest, f"{model_name}: formal-plan hash differs.")
    require(summary.get("formal_source_revision") == plan["source_revision"], f"{model_name}: source revision differs.")
    require(summary.get("completed_epochs") == 20 and len(summary["epochs"]) == 20, f"{model_name}: expected 20 completed epochs.")
    require([item["epoch"] for item in summary["epochs"]] == list(range(1, 21)), f"{model_name}: epoch sequence differs.")
    require(summary.get("optimizer_steps") == 57300 and not summary.get("limited_training_smoke", True), f"{model_name}: incomplete budget or smoke subset.")
    require(summary.get("test_split_read") is False, f"{model_name}: unexpected test scope.")
    protocol = plan["shared_protocol"]
    require(summary["model_config"] == protocol["model_config"], f"{model_name}: model config differs.")
    require(summary["settings"]["batch_size"] == 4 and summary["settings"]["num_workers"] == 0, f"{model_name}: validation batching differs.")
    for split in ("training", "validation"):
        require(summary["data"][f"{split}_manifest_sha256"] == plan["data"][split]["manifest_sha256"], f"{model_name}: {split} manifest differs.")
    require(summary["data"]["selected_training_pairs"] == 11460 and summary["data"]["full_validation_pairs"] == 660, f"{model_name}: split counts differ.")
    for item in summary["epochs"]:
        require(item["training"]["samples"] == 11460 and item["training"]["optimizer_steps"] == 2865, f"{model_name}: incomplete training epoch.")
        validation = item["validation"]
        require(validation["samples"] == 660 and validation["batches"] == 165 and validation["optimizer_steps"] == 0, f"{model_name}: incomplete or updating validation.")
        require(math.isfinite(validation["cross_entropy"]), f"{model_name}: non-finite recorded validation CE.")
    selected = min(summary["epochs"], key=lambda item: (item["validation"]["cross_entropy"], item["epoch"]))
    reload_record = summary["checkpoint_reload"]
    require(reload_record["passed"] is True and reload_record["best_epoch"] == selected["epoch"], f"{model_name}: best checkpoint is not earliest minimum-CE epoch.")
    require(reload_record["best_validation_cross_entropy"] == selected["validation"]["cross_entropy"], f"{model_name}: best validation CE differs from the epoch record.")
    return selected


def save_figure(output_path, sample, predictions, example, model_reports):
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(output_path.parent.resolve() / ".matplotlib"))
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    size = sample["original_size"]
    image = crop_to_original(sample["image"], size)[0].numpy()
    target = crop_to_original(sample["target"], size).numpy()
    colors = ListedColormap(["black", "gold", "deepskyblue", "magenta", "limegreen", "orange"])
    fig, axes = plt.subplots(1, 4, figsize=(14, 7), layout="constrained")
    axes[0].imshow(image, cmap="gray", interpolation="nearest")
    axes[0].set_title("Validation MRI\nper-slice min-max")
    panels = [(target, "Reference mask")]
    for name in MODELS:
        title = "Baseline" if name == "baseline_unet" else "Improved"
        panels.append((crop_to_original(predictions[name], size).numpy(), f"{title} best\nepoch {model_reports[name]['selected_epoch']}"))
    for axis, (values, title) in zip(axes[1:], panels):
        shown = axis.imshow(values, cmap=colors, vmin=-0.5, vmax=5.5, interpolation="nearest")
        axis.set_title(title)
    fig.colorbar(shown, ax=list(axes[1:]), ticks=list(range(6)), label="Raw class ID")
    fig.suptitle(
        f"Validation: {example['filename']} | reference-only fixed selection\n"
        "Raw IDs; provider tissue mapping assumes derivative IDs preserved"
    )
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def verify(args, report):
    project = Path(__file__).resolve().parent
    repo = project.parents[1]
    plan_digest = sha256(args.formal_plan)
    plan = read_json(args.formal_plan)
    require(plan["status"] == "approved_for_execution" and plan["user_confirmation_received"], "The formal plan lacks recorded approval.")
    require(plan["epochs_per_model"] == 20, "This verification expects the approved 20-epoch protocol.")
    require(plan["shared_protocol"]["model_config"] == {"in_channels": 1, "num_classes": 6, "base_channels": 16}, "Unexpected model protocol.")
    require(plan["shared_protocol"]["batch_size"] == 4 and plan["shared_protocol"]["num_workers"] == 0, "Expected batch4 and num_workers0.")
    for name, digest in plan["source_sha256"].items():
        require(sha256(project / name) == digest, f"Frozen source changed: {name}")
    report["formal_plan_sha256"] = plan_digest
    report["formal_source_revision"] = plan["source_revision"]
    report["frozen_source_sha256"] = plan["source_sha256"]
    report["verification_source_sha256"] = sha256(Path(__file__))
    report["source_hash_basis"] = "Current working-tree file bytes before Git line-ending normalization."
    runs = {item["model"]: repo / item["output_dir"] for item in plan["runs_in_order"]}
    require(set(runs) == set(MODELS), "Expected exactly baseline and Improved runs.")
    summaries, selected, summary_hashes = {}, {}, {}
    for name in MODELS:
        summary_path = runs[name] / "run_summary.json"
        summary_hashes[name] = sha256(summary_path)
        summaries[name] = read_json(summary_path)
        selected[name] = selected_record(summaries[name], plan, name, plan_digest)

    device = torch.device(args.device)
    require(device.type != "cuda" or torch.cuda.is_available(), "CUDA requested but unavailable.")
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    math_settings = {
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }
    require(all(summary["math_settings"] == math_settings for summary in summaries.values()), "Backend math settings differ from formal training; do not accept a changed replay.")
    report["device"] = str(device)
    report["math_settings"] = math_settings
    report["runtime"] = {"pytorch": str(torch.__version__), "cuda": torch.version.cuda,
                         "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None}
    pairs = collect_pairs(args.data_root / "keras_slices_validate", args.data_root / "keras_slices_seg_validate")
    require(len(pairs) == 660 and manifest_hash(pairs) == plan["data"]["validation"]["manifest_sha256"], "Validation manifest differs from the frozen 660-pair split.")
    example_index, sample, example = select_example(pairs)
    example["batch_index_zero_based"], example["index_within_batch"] = divmod(example_index, 4)
    example["validation_batch_size"] = 4
    report["example"] = example
    report["validation_manifest_sha256"] = manifest_hash(pairs)
    predictions = {}
    for name in MODELS:
        checkpoint_path = runs[name] / "best.pt"
        checkpoint_digest = sha256(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        record = selected[name]
        expected = record["validation"]
        require(checkpoint["model"] == name and checkpoint["model_config"] == plan["shared_protocol"]["model_config"], f"{name}: checkpoint model/config differs.")
        require(checkpoint["epoch"] == record["epoch"] and checkpoint["validation_cross_entropy"] == expected["cross_entropy"], f"{name}: checkpoint is not the selected epoch.")
        require(checkpoint["seed"] == plan["shared_protocol"]["seed"] and checkpoint.get("limited_training_smoke") is False, f"{name}: not the formal checkpoint.")
        model = build_model(name, **checkpoint["model_config"]).to(device).eval()
        model.load_state_dict(checkpoint["model_state"], strict=True)
        del checkpoint
        before = state_sha256(model)
        seen_batches = 0

        def capture_example(module, inputs, output):
            nonlocal seen_batches
            require(not module.training and not torch.is_grad_enabled(), f"{name}: replay forward must be eval and without gradients.")
            if seen_batches == example["batch_index_zero_based"]:
                predictions[name] = output[example["index_within_batch"]].detach().argmax(dim=0).cpu()
            seen_batches += 1

        hook = model.register_forward_hook(capture_example)
        loader = DataLoader(HipMRIPairDataset(pairs), batch_size=4, shuffle=False,
                            num_workers=0, collate_fn=collate_padded)
        measurement = StageMeasurement(
            device, f"post_training_selected_validation_replay_{name}",
            "One selected restored model; no optimizer or training gradients. Validation inputs/logits are present during forward; unused allocator cache is retained. This is post-training verification, separate from formal training/validation resource stages.",
        )
        try:
            with measurement:
                actual = run_epoch(model, loader, device, num_classes=6)
        finally:
            hook.remove()
        after = state_sha256(model)
        entry = {
            "selected_epoch": record["epoch"], "selection_rule": "earliest epoch with minimum full-validation CE",
            "checkpoint_sha256_before": checkpoint_digest, "summary_sha256": summary_hashes[name],
            "expected_selected_epoch_validation": expected, "replayed_validation": actual,
            "state_sha256_before": before, "state_sha256_after": after,
            "resource_measurement": measurement.result,
            "ce_absolute_difference": abs(actual["cross_entropy"] - expected["cross_entropy"]),
        }
        report["models"][name] = entry
        require(actual["samples"] == 660 and actual["batches"] == 165 and actual["optimizer_steps"] == 0, f"{name}: replay is incomplete or performed updates.")
        require(seen_batches == 165 and name in predictions, f"{name}: failed to capture exactly the example's validation batch.")
        require(before == after, f"{name}: model state changed during replay.")
        require(math.isfinite(actual["cross_entropy"]) and entry["ce_absolute_difference"] <= CE_ATOL, f"{name}: validation CE differs by more than {CE_ATOL}.")
        for key in ("confusion_matrix", "dice_per_class", "iou_per_class", "mean_dice_excluding_class0", "mean_iou_excluding_class0", "pixel_accuracy", "valid_pixels"):
            require(actual[key] == expected[key], f"{name}: selected-epoch {key} differs on full validation replay.")
        entry["checkpoint_sha256_after"] = sha256(checkpoint_path)
        require(entry["checkpoint_sha256_after"] == checkpoint_digest, f"{name}: checkpoint changed during replay.")
        entry["passed"] = True
        entry["all_forward_calls_eval_without_gradients"] = True
        entry["validation_forward_calls"] = seen_batches
        entry["optimizer_constructed"] = False
        entry["example_prediction_source"] = "Argmax captured from this example's original batch4 validation forward; no separate batch1 prediction."
        del model, loader
        print(f"{name}: selected epoch {record['epoch']}; full validation matches; CE difference={entry['ce_absolute_difference']:.3g}", flush=True)

    for name in MODELS:
        require(sha256(runs[name] / "run_summary.json") == summary_hashes[name], f"{name}: raw run summary changed during verification.")
    require(sha256(args.formal_plan) == plan_digest, "Formal plan changed during verification.")
    for name, digest in plan["source_sha256"].items():
        require(sha256(project / name) == digest, f"Frozen source changed during verification: {name}")
    save_figure(args.output_dir / "validation_example.png", sample, predictions, example, report["models"])
    report["figure"] = "validation_example.png"
    report["status"] = "passed"


def main():
    project = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-plan", type=Path, default=project / "evidence/formal_run_plan_20261026.json")
    parser.add_argument("--data-root", type=Path, default=project.parents[3] / "Final Report/HipMRI_2D_Improved_UNet_学习/data/keras_slices_data")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/selected_validation_verification_20261026"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "verification_kind": "selected_formal_checkpoints_full_validation_replay",
        "created_at": datetime.now(timezone(timedelta(hours=10))).isoformat(),
        "performed_by": "Codex automation; not user verification", "status": "running",
        "scope": ["validation only", "existing completed formal checkpoints"],
        "training_data_read": False, "test_split_read": False, "new_training_performed": False,
        "training_data_scope_note": "No training MRI/mask files or training-directory scan; completed training-run JSON metadata is read.",
        "ce_comparison_atol": CE_ATOL, "confusion_comparison": "strict exact equality",
        "metric_scope": "Full 660-slice validation replay against the recorded selected epoch; the example is illustrative and not a replacement for ID5 validation Dice.",
        "label_mapping_boundary": "Provider tissue names apply to the course derivative only under the user-accepted unverified assumption that original IDs were retained.",
        "resource_scope": "Post-training validation verification only; no optimizer. Do not merge these measurements into the original formal-run stage records or call them training cost.",
        "claim_boundary": "Consistency verification on already-used validation data, not independent test evaluation or additional training.",
        "models": {},
    }
    started = time.perf_counter()
    try:
        verify(args, report)
    except (Exception, KeyboardInterrupt) as error:
        report["status"] = "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        with (args.output_dir / "verification.json").open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
            handle.write("\n")


if __name__ == "__main__":
    main()
