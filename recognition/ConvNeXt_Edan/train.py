"""
Training pipeline for ADNI Alzheimer's MRI classification.

Supports:
    - Baseline CNN
    - ConvNeXt-Tiny
    - Patient-independent train/validation splits
    - Early stopping
    - Mixed-precision GPU training
    - Model checkpointing
    - Training history and plots

Classes:
    NC = 0
    AD = 1
"""

import argparse
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset import (
    collect_images,
    create_patient_splits,
    create_dataloaders,
    load_splits,
    save_splits,
)

from modules import create_model, count_parameters


SEED = 42


def set_seed(seed=SEED):
    """Make experiments as reproducible as practical."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_splits(split_path):
    """
    Load existing patient-independent splits.

    Create them only if they do not already exist.
    """

    split_path = Path(split_path)

    if split_path.exists():
        print(f"Loading existing splits from {split_path}")
        splits = load_splits(split_path)
    else:
        print("Creating patient-independent splits...")
        records = collect_images()
        splits = create_patient_splits(records)
        save_splits(splits, split_path)
        print(f"Saved splits to {split_path}")

    # Validate patient independence.
    patient_sets = {
        name: {record["patient_id"] for record in records}
        for name, records in splits.items()
    }

    assert patient_sets["train"].isdisjoint(patient_sets["val"])
    assert patient_sets["train"].isdisjoint(patient_sets["test"])
    assert patient_sets["val"].isdisjoint(patient_sets["test"])

    print("Patient-independent splits verified.")

    return splits


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    """
    Run one training or validation epoch.

    Returns:
        Average loss and classification accuracy.
    """

    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):

            with torch.autocast(
                device_type=device.type,
                enabled=(device.type == "cuda"),
            ):
                outputs = model(images)
                loss = criterion(outputs, labels)

            if training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += batch_size

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
    }


def plot_history(history, output_dir):
    """Save training and validation curves."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Training")
    plt.plot(epochs, history["val_loss"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_accuracy"], label="Training")
    plt.plot(epochs, history["val_accuracy"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_curve.png", dpi=150)
    plt.close()


def train(args):

    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError(
            "No compatible GPU detected. "
            "Request a Slurm GPU allocation or use --allow-cpu "
            "for a small debugging run."
        )

    print("=" * 60)
    print("ADNI MRI CLASSIFICATION")
    print("=" * 60)
    print("Model:", args.model)
    print("Device:", device)
    print("Pretrained:", args.pretrained)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        torch.cuda.reset_peak_memory_stats()

    splits = get_splits(args.splits)

    loaders = create_dataloaders(
        splits,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.workers,
    )

    model = create_model(
        model_name=args.model,
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    print("Total parameters:", f"{count_parameters(model):,}")
    print(
        "Trainable parameters:",
        f"{count_parameters(model, trainable_only=True):,}",
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda"),
    )

    output_dir = Path(args.output_dir) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "epoch_seconds": [],
    }

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):

        start_time = time.perf_counter()

        train_metrics = run_epoch(
            model,
            loaders["train"],
            criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
        )

        val_metrics = run_epoch(
            model,
            loaders["val"],
            criterion,
            device,
        )

        scheduler.step(val_metrics["loss"])

        elapsed = time.perf_counter() - start_time

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["epoch_seconds"].append(elapsed)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Train loss: {train_metrics['loss']:.4f} | "
            f"Train acc: {train_metrics['accuracy']:.4f} | "
            f"Val loss: {val_metrics['loss']:.4f} | "
            f"Val acc: {val_metrics['accuracy']:.4f} | "
            f"Time: {elapsed:.1f}s",
            flush=True,
        )

        if val_metrics["loss"] < best_val_loss:

            best_val_loss = val_metrics["loss"]
            epochs_without_improvement = 0

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "model_name": args.model,
                "pretrained": args.pretrained,
                "image_size": args.image_size,
                "epoch": epoch,
                "best_val_loss": best_val_loss,
            }

            torch.save(
                checkpoint,
                output_dir / "best_model.pth",
            )

            print("  Saved improved checkpoint.", flush=True)

        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print("Early stopping triggered.")
            break

    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    plot_history(history, output_dir)

    if device.type == "cuda":
        peak_vram_gb = (
            torch.cuda.max_memory_allocated() / (1024 ** 3)
        )
        print(f"Peak allocated GPU memory: {peak_vram_gb:.2f} GiB")

    print("\nTraining complete.")
    print("Best validation loss:", f"{best_val_loss:.4f}")
    print("Results saved to:", output_dir)


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=["baseline", "convnext"],
        required=True,
    )

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=1)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=SEED)

    parser.add_argument(
        "--splits",
        type=str,
        default="splits.json",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
    )

    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
