"""Train/validate baseline or Improved U-Net with the same data and evaluation flow."""

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import HipMRIPairDataset, collate_padded, collect_pairs, load_pair, patient_ids
from checkpoint_selection import SELECTION_RULES, checkpoint_candidate, checkpoint_key, is_better_checkpoint
from losses import segmentation_cross_entropy
from metrics import SegmentationMetrics
from modules import build_model
from resource_monitor import StageMeasurement


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def manifest_hash(pairs):
    text = "\n".join(f"{image.name}|{mask.name}" for image, mask in pairs)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def select_training_pairs(pairs, limit, seed, reference_name):
    """Fixed seeded subset for a bounded smoke run; retain the learning reference."""
    if limit is None or limit >= len(pairs):
        return list(pairs)
    if limit <= 0:
        raise ValueError("train-limit must be positive.")
    reference = next((i for i, pair in enumerate(pairs) if pair[0].name == reference_name), None)
    if reference is None:
        raise ValueError(f"Training reference not found: {reference_name}")
    other_indices = [i for i in range(len(pairs)) if i != reference]
    selected = [reference] + random.Random(seed).sample(other_indices, limit - 1)
    return [pairs[i] for i in sorted(selected)]


def move_batch(batch, device):
    return {key: value.to(device) for key, value in batch.items() if key in ("image", "target", "valid_mask")}


def run_epoch(model, loader, device, num_classes, optimizer=None, progress_every=0, progress_label="epoch"):
    """Update only during training; validation disables gradients and updates."""
    training = optimizer is not None
    model.train(training)
    metrics = SegmentationMetrics(num_classes)
    loss_sum, pixels, samples, steps = 0.0, 0, 0, 0
    gradient_norms = []
    started = time.perf_counter()
    with torch.set_grad_enabled(training):
        for batch in loader:
            batch = move_batch(batch, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(batch["image"])
            loss = segmentation_cross_entropy(logits, batch["target"], batch["valid_mask"])
            if not torch.isfinite(loss):
                raise ValueError("Non-finite loss; stopping instead of saving a misleading run.")
            if training:
                loss.backward()
                norms = [parameter.grad.detach().float().norm() for parameter in model.parameters() if parameter.grad is not None]
                gradient_norm = torch.stack(norms).norm()
                if not torch.isfinite(gradient_norm) or gradient_norm.item() == 0:
                    raise ValueError("Expected finite, nonzero gradients before the optimizer update.")
                gradient_norms.append(gradient_norm.item())
                optimizer.step()
            count = int(batch["valid_mask"].sum().item())
            loss_sum += loss.detach().item() * count
            pixels += count
            samples += batch["image"].shape[0]
            steps += 1
            metrics.update(logits.detach().argmax(dim=1), batch["target"], batch["valid_mask"])
            if progress_every and steps % progress_every == 0:
                print(f"{progress_label}: batches={steps}; samples={samples}; online CE={loss_sum / pixels:.6f}", flush=True)
    result = metrics.compute()
    result.update({
        "cross_entropy": loss_sum / pixels if pixels else None,
        "samples": samples, "batches": steps, "elapsed_seconds": round(time.perf_counter() - started, 3),
        "optimizer_steps": steps if training else 0,
    })
    if gradient_norms:
        result["gradient_norm_range"] = [min(gradient_norms), max(gradient_norms)]
    return result


def reference_probe(model, reference_sample, device):
    batch = move_batch(collate_padded([reference_sample]), device)
    model.eval()
    with torch.inference_mode():
        logits = model(batch["image"])
        loss = segmentation_cross_entropy(logits, batch["target"], batch["valid_mask"])
    return float(loss.item()), logits.detach().cpu()


def measured_call(report, device, stage, resident_state, operation):
    measurement = StageMeasurement(device, stage, resident_state)
    try:
        with measurement:
            return operation()
    finally:
        report["resource_stages"].append(measurement.result)


def validate_formal_plan(args, training_pairs, validation_pairs):
    """Reject drift from the approved protocol before starting a formal run."""
    if args.formal_plan is None:
        return None
    plan = json.loads(args.formal_plan.read_text(encoding="utf-8"))
    if plan["status"] != "approved_for_execution" or not plan["user_confirmation_received"]:
        raise ValueError("Formal training requires a recorded explicit user confirmation.")
    protocol = plan["shared_protocol"]
    expected = {
        "epochs": plan["epochs_per_model"], "batch_size": protocol["batch_size"],
        "base_channels": protocol["model_config"]["base_channels"],
        "num_classes": protocol["model_config"]["num_classes"],
        "lr": protocol["optimizer"]["learning_rate"], "seed": protocol["seed"],
        "device": "cuda", "train_limit": None,
    }
    if any(getattr(args, key) != value for key, value in expected.items()):
        raise ValueError(f"CLI differs from the approved formal protocol: {expected}")
    if args.model not in protocol["models"]:
        raise ValueError("Model not listed in the approved plan.")
    planned_selection = plan.get("metrics_and_model_selection", {}).get("checkpoint_selection", {}).get("selection_metric", "ce")
    if getattr(args, "selection_metric", "ce") != planned_selection:
        raise ValueError("CLI checkpoint selection differs from the approved formal plan.")
    for split, pairs in (("training", training_pairs), ("validation", validation_pairs)):
        if manifest_hash(pairs) != plan["data"][split]["manifest_sha256"]:
            raise ValueError(f"{split} manifest differs from the frozen plan.")
    for name, digest in plan["source_sha256"].items():
        if hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Source changed after the formal plan was recorded: {name}")
    return plan


def plot_history(history, output_dir, model_name="baseline_unet"):
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir.resolve() / ".matplotlib"))
    import matplotlib.pyplot as plt

    epochs = [record["epoch"] for record in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for split in ("training", "validation"):
        axes[0].plot(epochs, [record[split]["cross_entropy"] for record in history], marker="o", label=split)
    axes[0].set(xlabel="Epoch on this run's training set", ylabel="Valid-pixel CE", title=f"{model_name}: loss")
    axes[0].legend()
    for class_id in range(len(history[0]["validation"]["dice_per_class"])):
        values = [record["validation"]["dice_per_class"][class_id] for record in history]
        axes[1].plot(epochs, [float("nan") if value is None else value for value in values], marker="o", label=f"ID {class_id}")
    axes[1].set(xlabel="Epoch on this run's training set", ylabel="Full-validation Dice", ylim=(0, 1), title="Class IDs; source mapping assumes derivative IDs preserved")
    axes[1].legend(ncol=2)
    fig.savefig(output_dir / "training_curves.png", dpi=150)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    workspace = Path(__file__).resolve().parents[4]
    parser.add_argument("--data-root", type=Path, default=workspace / "Final Report/HipMRI_2D_Improved_UNet_学习/data/keras_slices_data")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/baseline"))
    parser.add_argument("--model", choices=("baseline_unet", "improved_unet"), default="baseline_unet")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--num-classes", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=3710)
    parser.add_argument("--train-limit", type=int, help="Fixed training subset for smoke runs; omitted means all training pairs.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--reference-name", default="case_004_week_0_slice_12.nii.gz")
    parser.add_argument("--formal-plan", type=Path, help="Approved frozen protocol and source/manifest hashes; optional for smoke checks.")
    parser.add_argument("--selection-metric", choices=tuple(SELECTION_RULES), default="ce",
                        help="Primary checkpoint rule; ID5 uses exact ties by lower CE, then earlier epoch.")
    parser.add_argument("--progress-every", type=int, default=500, help="Training batch progress interval; zero disables it.")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.base_channels) <= 0 or args.num_classes < 2 or args.lr <= 0:
        parser.error("Epochs, batch size, width and learning rate must be positive; classes >=2.")
    if args.train_limit is not None and args.train_limit <= 0:
        parser.error("train-limit must be positive")
    if args.progress_every < 0:
        parser.error("progress-every must be nonnegative")
    if args.selection_metric == "id5_dice" and args.num_classes <= 5:
        parser.error("ID5 selection requires at least six classes")
    return args


def execute_training(args):
    selection_metric = getattr(args, "selection_metric", "ce")
    if selection_metric not in SELECTION_RULES or (selection_metric == "id5_dice" and args.num_classes <= 5):
        raise ValueError("ID5 selection requires at least six classes and a supported selection metric.")
    if args.output_dir.exists():
        raise FileExistsError(f"Use a new output directory to preserve prior run evidence: {args.output_dir}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    torch.set_num_threads(1)
    seed_everything(args.seed)
    training_pairs = collect_pairs(args.data_root / "keras_slices_train", args.data_root / "keras_slices_seg_train")
    validation_pairs = collect_pairs(args.data_root / "keras_slices_validate", args.data_root / "keras_slices_seg_validate")
    train_ids, validation_ids = patient_ids(training_pairs), patient_ids(validation_pairs)
    overlap = train_ids & validation_ids
    if overlap:
        raise ValueError(f"Train/validation case-group overlap: {sorted(overlap)}")
    formal_plan = validate_formal_plan(args, training_pairs, validation_pairs)
    selected = select_training_pairs(training_pairs, args.train_limit, args.seed, args.reference_name)
    reference_pair = next((pair for pair in training_pairs if pair[0].name == args.reference_name), None)
    if reference_pair is None:
        raise ValueError("Fixed training reference not found.")
    reference_sample = load_pair(*reference_pair)
    generator = torch.Generator().manual_seed(args.seed)
    training_loader = DataLoader(HipMRIPairDataset(selected), batch_size=args.batch_size, shuffle=True,
                                 num_workers=0, collate_fn=collate_padded, generator=generator)
    validation_loader = DataLoader(HipMRIPairDataset(validation_pairs), batch_size=args.batch_size, shuffle=False,
                                   num_workers=0, collate_fn=collate_padded)
    config = {"in_channels": 1, "num_classes": args.num_classes, "base_channels": args.base_channels}
    model = build_model(args.model, **config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    initial_parameter = next(model.parameters()).detach().cpu().clone()
    args.output_dir.mkdir(parents=True)
    report = {
        "milestone": f"{args.model}_{'formal_training' if formal_plan else 'minimal_training'}", "performed_by": "Codex automation; not user-performed verification",
        "created_at": datetime.now(timezone(timedelta(hours=10))).isoformat(),
        "status": "running", "model": args.model, "model_config": config,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "optimizer": {"name": "Adam", "lr": args.lr}, "objective": "valid-region one-hot cross-entropy",
        "selection_metric": SELECTION_RULES[selection_metric], "seed": args.seed,
        "checkpoint_selection": {
            "primary_metric": selection_metric, "primary_rule": SELECTION_RULES[selection_metric],
            "ce_rule": SELECTION_RULES["ce"], "id5_rule": SELECTION_RULES["id5_dice"],
            "files": {"primary": "best.pt", "id5_best": "id5_best.pt", "ce_best": "ce_best.pt", "last": "last.pt"},
            "primary": None, "id5_best": None, "ce_best": None, "last": None,
        },
        "device": str(device), "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "versions": {"python": sys.version.split()[0], "python_torch": str(torch.__version__), "numpy": np.__version__,
                     "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version()},
        "settings": {"epochs": args.epochs, "batch_size": args.batch_size, "num_workers": 0, "cudnn_deterministic": True, "cudnn_benchmark": False},
        "data": {"full_training_pairs": len(training_pairs), "selected_training_pairs": len(selected),
                 "full_validation_pairs": len(validation_pairs), "selected_training_case_groups": len(patient_ids(selected)),
                 "training_case_groups": len(train_ids), "validation_case_groups": len(validation_ids), "shared_case_ids": sorted(overlap),
                 "training_manifest_sha256": manifest_hash(selected), "validation_manifest_sha256": manifest_hash(validation_pairs),
                 "selection_policy": "all training pairs" if len(selected) == len(training_pairs) else "fixed seeded subset including the learning reference; indices sorted before seeded epoch shuffling",
                 "selected_training_filenames": [pair[0].name for pair in selected]},
        "limited_training_smoke": len(selected) < len(training_pairs), "test_split_read": False,
        "improved_unet_implemented": True, "is_improved_run": args.model == "improved_unet",
        "reference_name": args.reference_name,
        "reference_ce_before_training": None, "epochs": [],
        "formal_run": formal_plan is not None,
        "formal_plan_sha256": hashlib.sha256(args.formal_plan.read_bytes()).hexdigest() if formal_plan else None,
        "formal_source_revision": formal_plan["source_revision"] if formal_plan else None,
        "resource_stages": [],
        "resource_policy": "Per-stage synchronized absolute PyTorch allocated/reserved peaks. Cached reserved memory and resident models/optimizer/gradients remain included; no empty_cache. Maximum of recorded stages is not total GPU VRAM or an unmeasured full-process peak.",
        "math_settings": {"matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                          "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                          "float32_matmul_precision": torch.get_float32_matmul_precision()},
        "claim_boundary": "A limited smoke run demonstrates the pipeline, not final model quality, fair model comparison or unseen-test performance.",
        "case_identifier_note": "Numeric case field is a grouping proxy; original patient identity mapping is unverified.",
    }
    if not report["limited_training_smoke"]:
        report["claim_boundary"] = "Training/validation run only; not a final fair model comparison or unseen-test evaluation."
    if formal_plan:
        report["claim_boundary"] = "Approved fixed-budget training/validation run. A paired comparison requires both runs complete and verified. No test evaluation."
        report["primary_report_metric"] = "full-validation pixel-pooled dice_per_class[5]"
        report["label_mapping_note"] = "The user accepted the working assumption that the course derivative preserves provider IDs (ID5=Prostate); this is not independent provenance proof."
    best_logits = None
    started = time.perf_counter()
    try:
        report["reference_ce_before_training"], _ = measured_call(
            report, device, "initial_reference_probe", "One model and empty Adam state; reference input/output during inference.",
            lambda: reference_probe(model, reference_sample, device),
        )
        for epoch in range(1, args.epochs + 1):
            training = measured_call(
                report, device, f"training_epoch_{epoch}",
                "Training model, Adam state (created on first updates), gradients, batch inputs/outputs/activations. Prior allocations and cached reserved blocks remain resident.",
                lambda: run_epoch(model, training_loader, device, args.num_classes, optimizer,
                                  progress_every=args.progress_every, progress_label=f"{args.model} epoch {epoch}/{args.epochs}"),
            )
            validation = measured_call(
                report, device, f"validation_epoch_{epoch}",
                "Same model plus resident Adam state and last training gradients; validation inputs/outputs; eval and no optimizer updates. Cached reserved blocks remain included.",
                lambda: run_epoch(model, validation_loader, device, args.num_classes),
            )
            record = {"epoch": epoch, "training": training, "validation": validation}
            report["epochs"].append(record)
            candidate = checkpoint_candidate(epoch, validation)
            # Fail rather than silently fall back to CE if the primary ID5
            # metric is undefined. CE-only legacy smoke runs may omit ID5.
            checkpoint_key(candidate, selection_metric)
            selection = report["checkpoint_selection"]
            ce_improved = is_better_checkpoint(candidate, selection["ce_best"], "ce")
            id5_improved = candidate["validation_id5_dice"] is not None and is_better_checkpoint(candidate, selection["id5_best"], "id5_dice")
            primary_improved = is_better_checkpoint(candidate, selection["primary"], selection_metric)
            report["checkpoint_selection"] = {
                **selection, "last": candidate,
                "ce_best": candidate if ce_improved else selection["ce_best"],
                "id5_best": candidate if id5_improved else selection["id5_best"],
                "primary": candidate if primary_improved else selection["primary"],
            }
            checkpoint = {"model": args.model, "model_config": config, "model_state": model.state_dict(),
                          "optimizer_state": optimizer.state_dict(), "epoch": epoch, "seed": args.seed,
                          "validation_cross_entropy": validation["cross_entropy"],
                          "validation_id5_dice": candidate["validation_id5_dice"],
                          "selection_metric": selection_metric, "selection_rule": SELECTION_RULES[selection_metric],
                          "checkpoint_selection": report["checkpoint_selection"],
                          "limited_training_smoke": report["limited_training_smoke"]}
            torch.save(checkpoint, args.output_dir / "last.pt")
            if ce_improved:
                torch.save(checkpoint, args.output_dir / "ce_best.pt")
            if id5_improved:
                torch.save(checkpoint, args.output_dir / "id5_best.pt")
            if primary_improved:
                _, best_logits = reference_probe(model, reference_sample, device)
                primary_file = "id5_best.pt" if selection_metric == "id5_dice" else "ce_best.pt"
                shutil.copyfile(args.output_dir / primary_file, args.output_dir / "best.pt")
            print(f"epoch {epoch}/{args.epochs}: train CE={training['cross_entropy']:.6f}; validation CE={validation['cross_entropy']:.6f}; validation ID5 Dice={validation['dice_per_class'][5] if args.num_classes > 5 else None}; valid pixels={validation['valid_pixels']}", flush=True)
            (args.output_dir / "run_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        report["first_parameter_max_abs_change"] = float((next(model.parameters()).detach().cpu() - initial_parameter).abs().max())
        report["reference_ce_after_training"], _ = reference_probe(model, reference_sample, device)
        restoration = StageMeasurement(device, "checkpoint_restoration",
            "Final training model, Adam state and last gradients remain resident; a second restored model and reference input/output are added. Checkpoint tensors load on CPU. Cached reserved blocks remain included.")
        try:
            with restoration:
                loaded = torch.load(args.output_dir / "best.pt", map_location="cpu", weights_only=True)
                restored = build_model(loaded["model"], **loaded["model_config"]).to(device)
                restored.load_state_dict(loaded["model_state"])
                restored_ce, restored_logits = reference_probe(restored, reference_sample, device)
                torch.testing.assert_close(restored_logits, best_logits, rtol=1e-5, atol=1e-6)
        finally:
            report["resource_stages"].append(restoration.result)
        primary = report["checkpoint_selection"]["primary"]
        report["checkpoint_reload"] = {"passed": True, "best_epoch": primary["epoch"],
                                       "best_validation_cross_entropy": primary["validation_cross_entropy"],
                                       "best_validation_id5_dice": primary["validation_id5_dice"], "selection_metric": selection_metric,
                                       "reference_ce": restored_ce, "reference_logit_max_abs_difference": float((restored_logits - best_logits).abs().max())}
        report["optimizer_steps"] = sum(record["training"]["optimizer_steps"] for record in report["epochs"])
        report["peak_cuda_allocated_bytes"] = max(stage["peak_cuda_allocated_bytes"] for stage in report["resource_stages"]) if device.type == "cuda" else None
        report["peak_cuda_reserved_bytes"] = max(stage["peak_cuda_reserved_bytes"] for stage in report["resource_stages"]) if device.type == "cuda" else None
        report["completed_epochs"] = len(report["epochs"])
        if formal_plan and (report["completed_epochs"] != 20 or report["optimizer_steps"] != 57300):
            raise ValueError("Formal run did not complete the approved 20 epochs / 57300 updates.")
        plot_history(report["epochs"], args.output_dir, args.model)
        report["curves_file"] = "training_curves.png"
        report["status"] = "passed"
    except (Exception, KeyboardInterrupt) as error:
        report["status"] = "incomplete" if formal_plan else "failed"
        report["completed_epochs"] = len(report["epochs"])
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        (args.output_dir / "run_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Completed {args.model} run: {args.output_dir.resolve()}; optimizer steps={report['optimizer_steps']}; primary best epoch={report['checkpoint_selection']['primary']['epoch']}", flush=True)


def main():
    args = parse_args()
    output_existed = args.output_dir.exists()
    try:
        execute_training(args)
    except (Exception, KeyboardInterrupt) as error:
        summary_path = args.output_dir / "run_summary.json"
        if args.formal_plan is not None and not output_existed and not summary_path.exists():
            setup_failure = {
                "status": "incomplete", "failure_phase": "setup",
                "created_at": datetime.now(timezone(timedelta(hours=10))).isoformat(),
                "performed_by": "Codex automation; not user verification",
                "model": args.model, "formal_run_requested": True,
                "formal_training_started": False, "completed_epochs": 0,
                "optimizer_steps": 0, "test_split_read": False,
                "error": {"type": type(error).__name__, "message": str(error)},
                "claim_boundary": "Setup failed before the recorded training loop. No model quality or completed-experiment claim.",
            }
            try:
                args.output_dir.mkdir(parents=True, exist_ok=True)
                with summary_path.open("x", encoding="utf-8") as handle:
                    json.dump(setup_failure, handle, indent=2)
                    handle.write("\n")
            except Exception as persistence_error:
                print(f"Could not save setup failure: {persistence_error}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()
