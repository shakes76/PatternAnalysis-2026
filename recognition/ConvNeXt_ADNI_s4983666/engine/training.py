"""Train one randomly initialized model on one frozen development fold.

Data roles (see adni_data/manifests.py; every patient is in exactly one role):

1. ``train`` slices update the weights.
2. ``val`` patients are scored after every epoch so the learning curves can be
   monitored and configurations compared. They never choose the checkpoint:
   the model is trained for a fixed number of epochs and the weights after the
   last epoch (the EMA copy when enabled) are saved as ``final.pt``.

Calibration and test patients are never loaded here; predict.py uses them
only after a checkpoint is frozen.
"""

import argparse
from collections import Counter
import math
from pathlib import Path
import sys
import time
from typing import Any

import torch
from torch import nn

from dataset import load_fold, make_loader
from adni_data.augmentation import AUGMENTATION_NAMES, make_augmentation
from adni_data.preprocessing import (add_preprocessing_arguments, fit_training_crop, prepare_scans,
                                   preprocessing_from_args)
from adni_data.sampling import SAMPLING_NAMES, sampling_weights
from engine.ema import ModelEMA
from engine.scheduling import learning_rate
from evaluation.inference import evaluate
from evaluation.metrics import binary_metrics
from evaluation.reporting import prediction_report
from evaluation.resources import cuda_peak_mib, profile_inference
from models.pretrained import load_imagenet_weights
from modules import MODEL_CHOICES, PRETRAINED_MODELS, count_parameters, create_model, model_minimum_size
from utils.artifacts import code_fingerprints, plot_history, validate_output, write_csv, write_json
from utils.evaluation_artifacts import export_evaluation_artifacts
from utils.preprocessing_artifacts import export_preprocessing_audit
from utils.runtime import environment_info, seed_everything, select_device, sync_device
from utils.training_controls import (ExecutionControls, add_execution_arguments, autocast_context,
                                     controls_from_args, make_criterion, mixup_batch, validate_device)

CHECKPOINT_FORMAT = 6


def train_epoch(model: nn.Module, loader, optimizer: torch.optim.Optimizer, criterion: nn.Module,
                device: torch.device, controls: ExecutionControls, ema: ModelEMA | None) -> dict:
    """One pass over the training slices; returns the mean loss and online train metrics.

    With Mixup the online accuracy is measured against each image's dominant
    label (the one with weight >= 0.5), which is only an approximate signal.
    """
    model.train()
    total_loss, count = 0.0, 0
    labels_seen, probabilities = [], []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=device.type == "cuda")
        labels = batch["label"].to(device, non_blocking=device.type == "cuda")
        mixed, labels_a, labels_b, lam = mixup_batch(images, labels, controls.mixup_alpha)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(controls, device):
            logits = model(mixed).float()
        loss = criterion(logits, labels_a)
        if lam < 1.0:
            loss = lam * loss + (1 - lam) * criterion(logits, labels_b)
        if not torch.isfinite(loss):
            raise ValueError("Training produced a non-finite loss.")
        loss.backward()
        optimizer.step()
        if ema is not None:
            ema.update(model)
        total_loss += loss.item() * labels.numel()
        count += labels.numel()
        dominant = labels_a if lam >= 0.5 else labels_b
        labels_seen.extend(int(v) for v in dominant.cpu().tolist())
        probabilities.extend(torch.sigmoid(logits.detach()).cpu().tolist())
    return {"loss": total_loss / count, **binary_metrics(labels_seen, probabilities)}


def build_config(args: argparse.Namespace, model_name: str, model: nn.Module, controls: ExecutionControls,
                 augmentation, preprocessing, image_size, data: dict, pos_weight: float,
                 sampling_config: dict, crop_fit: dict, device: torch.device, pretrained: dict | None) -> dict:
    """Everything needed to rebuild and audit the run; stored in config.json and final.pt."""
    return {
        "checkpoint_format_version": CHECKPOINT_FORMAT, "model_name": model_name,
        "model_architecture": ({"depths": list(model.depths), "channels": list(model.channels)}
                               if hasattr(model, "depths") else {"name": model_name}),
        "initialization": "imagenet_1k" if pretrained else "random", "pretrained_weights": pretrained,
        "execution_controls": controls.to_dict(),
        "augmentation_config": augmentation.to_dict(),
        "preprocessing_config": preprocessing.to_dict() if preprocessing.name != "none" else None,
        "preprocessing_crop_fit": crop_fit,
        "normalization": "(uint8 / 255 - 0.5) / 0.5",
        "image_size": list(image_size), "fold": args.fold, "seed": args.seed + args.fold, "seed_base": args.seed,
        "expected_slices": int(data["report"]["config"]["expected_slices"]),
        "manifest_sha256": data["manifest_sha256"],
        "train_manifests": ["train.csv", "early_stop.csv"],
        "evaluation_mode": "val_monitored_each_epoch",
        "checkpoint_selection": "final_epoch",
        "ema_decay": args.ema_decay, "threshold": 0.5,
        "train_slice_class_counts": {str(k): v for k, v in sorted(Counter(
            int(row["label"]) for row in data["train"]).items())},
        "train_pos_weight": pos_weight, "training_sampling": sampling_config,
        "epochs": args.epochs,
        "lr": args.lr, "weight_decay": args.weight_decay, "batch_size": args.batch_size,
        "lr_schedule": {"name": args.lr_schedule, "warmup_epochs": args.warmup_epochs,
                        "min_lr_ratio": args.min_lr_ratio},
        "workers": args.workers, "threads": args.threads,
        "data_root": str(args.data_root.resolve()), "splits_dir": str(args.splits_dir.resolve()),
        "code_sha256": code_fingerprints(), "environment": environment_info(device),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Train for a fixed number of epochs, monitoring val, and save the final weights."""
    model_name = MODEL_CHOICES[args.model]
    controls = controls_from_args(args)
    if model_name == "small_cnn_v1" and controls.drop_path is not None:
        raise ValueError("DropPath applies to ConvNeXt only.")
    if (model_name in PRETRAINED_MODELS) != (args.pretrained_weights is not None):
        raise ValueError("--pretrained-weights is required for, and only allowed with, convnext_tiny_in1k.")
    if min(args.epochs, args.batch_size, args.threads) < 1 or args.workers < 0:
        raise ValueError("Epochs, batch size and threads must be positive.")
    if args.lr <= 0 or args.weight_decay < 0:
        raise ValueError("Learning rate must be positive and weight decay non-negative.")
    if args.ema_decay and not 0 < args.ema_decay < 1:
        raise ValueError("EMA decay must be 0 (off) or in (0, 1).")
    minimum_size = model_minimum_size(model_name)
    if min(args.image_height, args.image_width) < minimum_size:
        raise ValueError(f"Image sides must be at least {minimum_size} for {args.model}.")
    learning_rate(1, args.epochs, args.lr, args.lr_schedule, args.warmup_epochs, args.min_lr_ratio)

    augmentation = make_augmentation(args.augmentation)
    preprocessing = preprocessing_from_args(args)
    device = select_device(args.device)
    validate_device(controls, device)
    output = validate_output(args.output, args.data_root, args.splits_dir)
    data = load_fold(args.data_root, args.splits_dir, args.fold)
    expected_slices = int(data["report"]["config"]["expected_slices"])
    seed = args.seed + args.fold
    seed_everything(seed)
    torch.set_num_threads(args.threads)

    # Optional scan preprocessing: crop size is fitted on training scans only.
    image_size = (args.image_height, args.image_width)
    train_scan_parameters, crop_fit = None, {"status": "not_required"}
    if preprocessing.name != "none":
        train_scan_parameters = prepare_scans(data["train"], args.data_root, preprocessing)
        preprocessing, crop_fit = fit_training_crop(train_scan_parameters, preprocessing,
                                                    role="train", minimum_size=minimum_size)
        if preprocessing.crops:
            image_size = (preprocessing.crop_height, preprocessing.crop_width)

    model = create_model(model_name, input_channels=controls.context_slices, drop_path=controls.drop_path)
    # Optional ImageNet-1k backbone (cited in models/pretrained.py); the AD classifier stays random.
    pretrained = load_imagenet_weights(model, args.pretrained_weights) if args.pretrained_weights else None
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ema = ModelEMA(model, args.ema_decay) if args.ema_decay else None
    final_model = ema.module if ema is not None else model  # Weights that are scored and saved.

    # Class weight from training slices only: pos_weight = NC count / AD count.
    class_counts = Counter(int(row["label"]) for row in data["train"])
    if min(class_counts[0], class_counts[1]) == 0:
        raise ValueError("Training must contain both AD and NC slices.")
    _, sampling_config = sampling_weights(data["train"], args.sampling)
    # Balanced sampling already equalises the classes; weighting the loss as well would double-count.
    pos_weight = class_counts[0] / class_counts[1] if args.sampling == "slice_uniform" else 1.0
    criterion = make_criterion(controls, pos_weight, device)

    loader_args = (args.data_root, image_size, args.batch_size, args.workers, seed)
    train_loader = make_loader(data["train"], *loader_args, shuffle=True, device=device, role="train",
                               augmentation=augmentation, sampling=args.sampling, preprocessing=preprocessing,
                               scan_parameters=train_scan_parameters, context_slices=controls.context_slices)
    val_loader = make_loader(data["val"], *loader_args, shuffle=False, device=device, role="val",
                             preprocessing=preprocessing, context_slices=controls.context_slices)

    config = build_config(args, model_name, model, controls, augmentation, preprocessing, image_size,
                          data, pos_weight, sampling_config, crop_fit, device, pretrained)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", config)
    preprocessing_audits = {}
    if preprocessing.name != "none":
        preprocessing_audits = {role: export_preprocessing_audit(output, loader.dataset, role)
                                for role, loader in (("train", train_loader), ("val", val_loader))}

    print(f"Training {model_name} on fold {args.fold} ({device}); augmentation={augmentation.name}, "
          f"context_slices={controls.context_slices}, ema={args.ema_decay}, "
          f"{args.epochs} epochs, checkpoint=final epoch.", flush=True)
    history, training_peak, evaluation_peak = [], 0.0, 0.0
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_lr = learning_rate(epoch, args.epochs, args.lr, args.lr_schedule,
                                 args.warmup_epochs, args.min_lr_ratio)
        for group in optimizer.param_groups:
            group["lr"] = epoch_lr
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        sync_device(device)
        epoch_started = time.perf_counter()
        train = train_epoch(model, train_loader, optimizer, criterion, device, controls, ema)
        sync_device(device)
        train_seconds = time.perf_counter() - epoch_started
        epoch_peak = cuda_peak_mib(device)  # None on CPU.
        if device.type == "cuda":
            training_peak = max(training_peak, epoch_peak)
            torch.cuda.reset_peak_memory_stats(device)

        # Monitoring only: val scores are logged but never decide which weights are kept.
        scores, slices, scans = evaluate(final_model, val_loader, device, expected_slices, controls)
        if device.type == "cuda":
            evaluation_peak = max(evaluation_peak, cuda_peak_mib(device))
        history.append({
            "epoch": epoch, "learning_rate": epoch_lr, "train_slice_loss": train["loss"],
            "train_slice_accuracy": train["accuracy"], "train_slice_auroc": train["auroc"],
            "train_seconds": train_seconds, "train_peak_cuda_mib": epoch_peak,
            "val_slice_loss": scores["slice"]["log_loss"], "val_slice_accuracy": scores["slice"]["accuracy"],
            "val_slice_auroc": scores["slice"]["auroc"], "val_scan_loss": scores["scan"]["log_loss"],
            "val_scan_accuracy": scores["scan"]["accuracy"], "val_scan_auroc": scores["scan"]["auroc"],
            "epoch_seconds": time.perf_counter() - epoch_started,
        })
        write_csv(output / "history.csv", history)
        print(f"Epoch {epoch:03d}: lr={epoch_lr:.2e} train_loss={train['loss']:.4f} "
              f"train_acc={train['accuracy']:.3f} val_slice_acc={scores['slice']['accuracy']:.3f} "
              f"val_scan_acc={scores['scan']['accuracy']:.3f} val_scan_auroc={scores['scan']['auroc']}", flush=True)
    plot_history(output, history)

    # The checkpoint is the model after the last epoch; its val scores are the last row above.
    del optimizer
    checkpoint = {"config": config, "epoch": args.epochs,
                  "model_state": {k: v.detach().cpu().clone() for k, v in final_model.state_dict().items()}}
    torch.save(checkpoint, output / "final.tmp")
    (output / "final.tmp").replace(output / "final.pt")
    report, patients = prediction_report(slices, scans)
    scores["patient"] = report["patient_metrics"]
    profile = profile_inference(final_model, image_size, device, controls,
                                enabled=not args.skip_inference_profile)

    result = {
        "status": "complete",
        "evaluation_role": "development_validation",
        "model_name": model_name, "fold": args.fold, "epochs_completed": len(history),
        "checkpoint_selection": config["checkpoint_selection"], "metrics": scores,
        "coursework_report": report, "manifest_sha256": data["manifest_sha256"],
        "resources": {
            "trainable_parameters": count_parameters(model),
            "training_seconds": math.fsum(row["train_seconds"] for row in history),
            "training_and_evaluation_seconds": time.perf_counter() - started,
            "training_peak_cuda_allocated_mib": training_peak if device.type == "cuda" else None,
            "evaluation_peak_cuda_allocated_mib": evaluation_peak if device.type == "cuda" else None,
            "device": config["environment"]["device_name"], "inference_profile": profile,
        },
    }
    if preprocessing.name != "none":
        result["preprocessing_audits"] = preprocessing_audits
    write_csv(output / "val_slice_predictions.csv", slices)
    write_csv(output / "val_scan_predictions.csv", scans)
    result["failure_examples"] = export_evaluation_artifacts(
        output, report, slices, patients, data["val"], args.data_root, image_size, "val",
        preprocessing=preprocessing, scan_parameters=val_loader.dataset.scan_parameters)
    write_json(output / "metrics.json", result)  # Written last: marks a completed run.
    for unit in ("slice", "scan", "patient"):
        print(f"val {unit}: accuracy={scores[unit]['accuracy']:.4f} AUROC={scores[unit]['auroc']}", flush=True)
    print(f"Results: {output / 'metrics.json'}", flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    """Command-line options; defaults are the ConvNeXt recipe used for the report."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=Path("/home/groups/comp3710/ADNI"))
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=tuple(MODEL_CHOICES), default="convnext_lite")
    parser.add_argument("--pretrained-weights", type=Path, default=None,
                        help="Official convnext_tiny_1k_224_ema.pth (Liu et al. 2022); only for convnext_tiny_in1k.")
    parser.add_argument("--fold", type=int, default=1)
    add_execution_arguments(parser)
    add_preprocessing_arguments(parser)
    parser.add_argument("--augmentation", choices=AUGMENTATION_NAMES, default="strong",
                        help="Training-only augmentation profile (default: strong).")
    parser.add_argument("--sampling", choices=SAMPLING_NAMES, default="slice_uniform")
    parser.add_argument("--epochs", type=int, default=80,
                        help="Fixed training length; the weights after the last epoch are saved.")
    parser.add_argument("--ema-decay", type=float, default=0.999,
                        help="Per-step EMA of the weights; 0 disables it (default: 0.999).")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--lr-schedule", choices=("constant", "warmup_cosine", "cosine"), default="warmup_cosine")
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--min-lr-ratio", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=3710)
    parser.add_argument("--image-height", type=int, default=240)
    parser.add_argument("--image-width", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--skip-inference-profile", action="store_true",
                        help="Skip the separate latency benchmark (useful for smoke tests).")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run(args)
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
