"""Train a fresh CNN on one frozen development fold and evaluate outer val once.

The calibration and final-test sets are used only by the source integrity audit;
their images/labels never enter training, checkpoint selection, or model scoring.
"""

import argparse
from collections import Counter
import math
from pathlib import Path
import sys
import time

import torch
from torch import nn

from dataset import load_fold
from modules import SmallCNN, count_parameters
from training_utils import (
    code_fingerprints, environment_info, evaluate, make_loader, plot_history,
    seed_everything, select_device, sync_device, validate_output, write_csv, write_json,
)


def train_epoch(model, loader, optimizer, criterion, device):
    """Update weights using only training slices and fail on non-finite loss."""
    model.train()
    total_loss, count = 0.0, 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=device.type == "cuda")
        labels = batch["label"].to(device, non_blocking=device.type == "cuda")
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), labels)
        if not torch.isfinite(loss).item():
            raise ValueError("Training produced non-finite loss; no evaluation will be published.")
        loss.backward()
        optimizer.step()
        total_loss += loss.detach().item() * labels.numel()
        count += labels.numel()
    return total_loss / count


def run(args):
    """Keep checkpoint selection and outer-fold evaluation in separate phases."""
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1 or args.workers < 0 or args.threads < 1:
        raise ValueError("Epochs, patience, batch size, and threads must be positive; workers cannot be negative.")
    if min(args.image_height, args.image_width) < 16:
        raise ValueError("Image height and width must be at least 16.")
    if not all(math.isfinite(v) for v in (args.lr, args.weight_decay, args.min_delta)):
        raise ValueError("Optimizer and early-stopping settings must be finite.")
    if args.lr <= 0 or args.weight_decay < 0 or args.min_delta < 0:
        raise ValueError("Learning rate must be positive; weight decay and min delta cannot be negative.")

    output = validate_output(args.output, args.data_root, args.splits_dir)
    data = load_fold(args.data_root, args.splits_dir, args.fold)
    expected_slices = int(data["report"]["config"]["expected_slices"])
    seed = args.seed + args.fold
    seed_everything(seed)
    torch.set_num_threads(args.threads)
    device = select_device(args.device)
    image_size = (args.image_height, args.image_width)

    # No resume option: every run/fold creates an independent model and optimizer.
    model = SmallCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    class_counts = Counter(int(row["label"]) for row in data["train"])
    if class_counts[0] == 0 or class_counts[1] == 0:
        raise ValueError("Training must contain both AD and NC slices.")
    pos_weight = class_counts[0] / class_counts[1]
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    loader_args = (args.data_root, image_size, args.batch_size, args.workers, seed)
    train_loader = make_loader(data["train"], *loader_args, shuffle=True, device=device)
    early_loader = make_loader(data["early_stop"], *loader_args, shuffle=False, device=device)

    config = {
        "checkpoint_format_version": 1, "model_name": "small_cnn_v1",
        "fold": args.fold, "seed": seed, "seed_base": args.seed,
        "image_size": list(image_size), "expected_slices": expected_slices,
        "normalization": "(grayscale_uint8 / 255 - 0.5) / 0.5",
        "augmentation": "none", "aggregation": "mean_slice_AD_probability",
        "threshold": 0.5, "calibration": "not_fitted",
        "checkpoint_selection": "minimum_early_stop_scan_log_loss",
        "manifest_sha256": data["manifest_sha256"],
        "split_config": data["report"]["config"],
        "train_slice_class_counts": {str(k): v for k, v in class_counts.items()},
        "train_pos_weight": pos_weight,
        "epochs_limit": args.epochs, "patience": args.patience, "min_delta": args.min_delta,
        "lr": args.lr, "weight_decay": args.weight_decay,
        "batch_size": args.batch_size, "workers": args.workers, "threads": args.threads,
        "data_root": str(args.data_root.resolve()), "splits_dir": str(args.splits_dir.resolve()),
        "code_sha256": code_fingerprints(), "environment": environment_info(device),
    }
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", config)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    history, best_loss, patience_loss, stale_epochs, best_epoch = [], math.inf, math.inf, 0, 0
    print(f"Training fold {args.fold} on {device}; selection uses early-stop scans only.", flush=True)

    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        early_scores, _, _ = evaluate(model, early_loader, device, expected_slices)
        scan_scores = early_scores["scan"]
        selection_loss = scan_scores["log_loss"]
        improved = selection_loss < best_loss
        if improved:
            best_loss, best_epoch = selection_loss, epoch
            # Checkpoints contain tensors and plain data, compatible with weights_only.
            checkpoint = {
                "config": config, "epoch": epoch, "early_stop_scan_loss": selection_loss,
                "model_state": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
            }
            temporary = output / "best.tmp"
            torch.save(checkpoint, temporary)
            temporary.replace(output / "best.pt")
        # Save every strict minimum; min_delta affects patience, not which
        # checkpoint is reported as the minimum observed early-stop loss.
        if selection_loss < patience_loss - args.min_delta:
            patience_loss, stale_epochs = selection_loss, 0
        else:
            stale_epochs += 1
        history.append({
            "epoch": epoch, "train_slice_loss": train_loss,
            "early_stop_scan_loss": selection_loss,
            "early_stop_scan_accuracy": scan_scores["accuracy"],
            "early_stop_scan_macro_f1": scan_scores["macro_f1"],
            "early_stop_scan_auroc": scan_scores["auroc"],
            "selected_checkpoint": improved, "epoch_seconds": time.perf_counter() - epoch_started,
        })
        write_csv(output / "history.csv", history)
        print(f"Epoch {epoch:03d}: train_loss={train_loss:.4f}, "
              f"early_stop_scan_loss={selection_loss:.4f}, "
              f"early_stop_scan_f1={scan_scores['macro_f1']:.4f}", flush=True)
        if stale_epochs >= args.patience:
            print(f"Early stopping after {epoch} epochs; selected epoch {best_epoch}.", flush=True)
            break

    # Outer validation is constructed/scored only after the checkpoint is fixed.
    selected = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(selected["model_state"])
    validation_loader = make_loader(data["val"], *loader_args, shuffle=False, device=device)
    scores, slices, scans = evaluate(model, validation_loader, device, expected_slices)
    sync_device(device)
    result = {
        "status": "complete", "evaluation_role": "development_outer_validation",
        "fold": args.fold, "best_epoch": best_epoch, "epochs_completed": len(history),
        "best_early_stop_scan_loss": best_loss, "manifest_sha256": data["manifest_sha256"],
        "aggregation": config["aggregation"], "threshold": 0.5, "calibration": "not_fitted",
        "metrics": scores,
        "resources": {
            "trainable_parameters": count_parameters(model),
            "training_and_evaluation_seconds": time.perf_counter() - started,
            "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
            "timing_note": "Forward timing excludes data loading and transfer; mean per slice at configured batch size.",
        },
    }
    write_csv(output / "val_slice_predictions.csv", slices)
    write_csv(output / "val_scan_predictions.csv", scans)
    plot_history(output, history)
    # This marker is written last; partial/failed runs have no completed result.
    write_json(output / "metrics.json", result)
    print(f"Outer validation: accuracy={scores['scan']['accuracy']:.4f}, "
          f"macro_F1={scores['scan']['macro_f1']:.4f}, AUROC={scores['scan']['auroc']}", flush=True)
    print(f"Completed. Results: {output / 'metrics.json'}", flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=3710)
    parser.add_argument("--image-height", type=int, default=240)
    parser.add_argument("--image-width", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    try:
        run(args)
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
