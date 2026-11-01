"""Train the CNN baseline or GFNet-inspired ADNI classifier."""

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn

from dataset import build_dataloaders
from modules import CNNBaseline, GFNet


def parse_args():
    """Read common paths and optional training-budget overrides."""
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--model", choices=["cnn", "gfnet"], required=True)
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto"
    )

    parser.add_argument('--skip-test', action='store_true',
                        help='Train/validate only; do not evaluate test data after training.')
    return parser.parse_args()


def set_seed(seed):
    """Seed CPU/GPU generators; this does not promise bitwise GPU determinism."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, device, optimizer=None):
    """Train or evaluate for one epoch using sample-weighted metrics."""
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            if training:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if training:
                # Backpropagate only through training batches.
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            # Weight batch means by sample count, including the shorter last batch.
            total_loss += loss.item() * batch_size
            total_correct += (
                outputs.argmax(dim=1) == labels
            ).sum().item()
            total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Cannot evaluate an empty DataLoader.")

    return (
        total_loss / total_samples,
        total_correct / total_samples,
    )


def save_curves(history, best_epoch, output_path):
    """Save loss and accuracy traces, marking the validation-selected epoch."""
    epochs = [row["epoch"] for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(
        epochs, [row["train_loss"] for row in history], label="Train"
    )
    axes[0].plot(
        epochs, [row["val_loss"] for row in history], label="Validation"
    )
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].set_title("Training and Validation Loss")

    axes[1].plot(
        epochs,
        [row["train_accuracy"] * 100 for row in history],
        label="Train",
    )
    axes[1].plot(
        epochs,
        [row["val_accuracy"] * 100 for row in history],
        label="Validation",
    )
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Training and Validation Accuracy")

    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.axvline(
            best_epoch,
            color="gray",
            linestyle="--",
            label=f"Best epoch: {best_epoch}",
        )
        ax.legend()
        ax.grid(True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    """Train from scratch, reload the best weights and optionally test them."""
    args = parse_args()

    epochs = args.epochs
    if epochs is None:
        epochs = 10 if args.model == "cnn" else 30

    if epochs < 1 or args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("Invalid epochs, batch size, or worker count.")

    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    output_dir = Path(args.output_dir)

    # Require a fresh run directory to protect existing checkpoints.
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"{output_dir} is not empty. Choose a new output directory."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, _ = build_dataloaders(
        split_folder=args.split_dir,
        source_root=args.source_root,
        image_root=args.image_root,
        model_name=args.model,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Seed immediately before creating the model.
    set_seed(args.seed)

    if args.model == "cnn":
        model = CNNBaseline().to(device)
        learning_rate = 0.001
        weight_decay = 0.0

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
        )
        scheduler = None
    else:
        model = GFNet(
            image_size=256,
            patch_size=16,
            dim=128,
            depth=8,
            dropout=0.1,
        ).to(device)

        learning_rate = 0.0003
        weight_decay = 0.0005

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=1e-6,
        )

    # Cross entropy consumes logits directly; softmax belongs in evaluation.
    criterion = nn.CrossEntropyLoss()
    checkpoint_path = output_dir / f"{args.model}_best.pth"

    from predict import environment_info
    config = {
        "environment": environment_info(device),
        **vars(args),
        "epochs": epochs,
        "resolved_device": str(device),
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "optimizer": "Adam" if args.model == "cnn" else "AdamW",
        "scheduler": None if scheduler is None else "CosineAnnealingLR",
        "image_size": 256,
        "label_map": {"NC": 0, "AD": 1},
        "parameters": sum(p.numel() for p in model.parameters()),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "gpu": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),
    }

    if args.model == "gfnet":
        config["architecture"] = {
            "patch_size": 16,
            "dim": 128,
            "depth": 8,
            "dropout": 0.1,
        }

    with (output_dir / "config.json").open("w") as file:
        json.dump(config, file, indent=2)

    print("Model:", args.model)
    print("Device:", device)
    print("Epochs:", epochs)
    print("Parameters:", config["parameters"])
    print("Checkpoint:", checkpoint_path)

    history = []
    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        learning_rate_used = optimizer.param_groups[0]["lr"]

        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_loss, val_accuracy = run_epoch(
            model, val_loader, criterion, device
        )

        if not np.isfinite(train_loss) or not np.isfinite(val_loss):
            raise RuntimeError("Non-finite loss detected.")

        history.append({
            "epoch": epoch,
            "learning_rate": learning_rate_used,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        })

        # The test set is never used to choose an epoch.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch

            # Plain state_dict format matches the original notebooks.
            torch.save(model.state_dict(), checkpoint_path)

            with (output_dir / "best_metrics.json").open("w") as file:
                json.dump({
                    "best_epoch": best_epoch,
                    "validation_loss": val_loss,
                    "validation_accuracy": val_accuracy,
                }, file, indent=2)

            print(f"Best model saved at epoch {epoch}")

        with (output_dir / "history.json").open("w") as file:
            json.dump(history, file, indent=2)

        print(
            f"Epoch [{epoch}/{epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Acc: {train_accuracy:.2%} | "
            f"Val Loss: {val_loss:.4f} "
            f"Val Acc: {val_accuracy:.2%} | "
            f"LR: {learning_rate_used:.8f}",
            flush=True,
        )

        if scheduler is not None:
            # One cosine update per completed epoch, matching the GFNet notebook.
            scheduler.step()

    save_curves(
        history,
        best_epoch,
        output_dir / "training_curves.png",
    )

    # Verify the saved best checkpoint on validation data.
    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
    )
    validation_loss, validation_accuracy = run_epoch(
        model, val_loader, criterion, device
    )

    print("\nTraining completed.")
    print("Best epoch:", best_epoch)
    print(f"Reloaded validation loss: {validation_loss:.4f}")
    print(f"Reloaded validation accuracy: {validation_accuracy:.2%}")
    print("Checkpoint:", checkpoint_path)

    if not args.skip_test:
        # Evaluate the selected weights in a separate process. This uses the
        # exact same evaluation entry point as a later checkpoint-only run.
        # Release training state first so it does not occupy the GPU during test.
        del optimizer, scheduler, model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        command = [sys.executable, str(Path(__file__).with_name('predict.py')),
                   '--model', args.model, '--checkpoint', str(checkpoint_path.resolve()),
                   '--split-dir', str(Path(args.split_dir).resolve()),
                   '--source-root', str(Path(args.source_root).resolve()),
                   '--image-root', str(Path(args.image_root).resolve()),
                   '--output-dir', str((output_dir / 'evaluation').resolve()),
                   '--batch-size', str(args.batch_size), '--num-workers', str(args.num_workers),
                   '--device', device.type]
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()