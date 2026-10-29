
"""
COMP3710 - ADNI Alzheimer's Disease Classification

Training script for:
    1. BaselineCNN
    2. ConvNeXt-Tiny

Features:
    - Patient-independent train/validation/test splits
    - GPU training
    - ImageNet-pretrained ConvNeXt
    - Optional frozen backbone
    - Early stopping
    - Best and last checkpoints
    - Training history CSV
    - Resume training
    - Smoke testing

The test set is NOT evaluated during training.
"""

import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

from dataset import DATA_DIR, SPLITS_CSV, get_dataloaders
from modules import build_model


# ============================================================
# 1. Reproducibility
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 2. Training / Validation Loop
# ============================================================

def run_epoch(
    model,
    loader,
    loss_fn,
    device,
    optimizer=None,
    max_batches=None,
):
    is_train = optimizer is not None

    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_images = 0

    start = time.perf_counter()

    with torch.set_grad_enabled(is_train):

        for step, (images, labels) in enumerate(
            loader, start=1
        ):
            if (
                max_batches is not None
                and step > max_batches
            ):
                break

            images = images.to(
                device,
                non_blocking=device.type == "cuda",
            )

            labels = labels.to(
                device,
                dtype=torch.long,
                non_blocking=device.type == "cuda",
            )

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = loss_fn(logits, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            n = labels.size(0)

            total_images += n
            total_loss += loss.item() * n

            total_correct += (
                logits.argmax(1) == labels
            ).sum().item()

            if is_train and (
                step == 1 or step % 100 == 0
            ):
                print(
                    f"  batch {step}/"
                    f"{min(len(loader), max_batches or len(loader))} "
                    f"loss={loss.item():.4f}",
                    flush=True,
                )

    if total_images == 0:
        raise RuntimeError(
            "No data processed; check dataset and batch limits."
        )

    return {
        "loss": total_loss / total_images,
        "accuracy": total_correct / total_images,
        "images": total_images,
        "seconds": time.perf_counter() - start,
    }


# ============================================================
# 3. Save Training History
# ============================================================

def write_history(path, records):

    fieldnames = [
        "epoch",
        "train_loss",
        "train_accuracy",
        "val_loss",
        "val_accuracy",
        "train_seconds",
        "val_seconds",
    ]

    with path.open("w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(records)


# ============================================================
# 4. Main Training Function
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="ADNI model training"
    )

    # --------------------------------------------------------
    # Training configuration
    # --------------------------------------------------------

    parser.add_argument(
        "--model",
        choices=["baseline", "convnext"],
        default="convnext",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    # --------------------------------------------------------
    # Dataset paths
    # --------------------------------------------------------

    parser.add_argument(
        "--csv-path",
        type=Path,
        default=SPLITS_CSV,
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
    )

    # --------------------------------------------------------
    # Checkpoint configuration
    # --------------------------------------------------------

    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints"),
    )

    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from a last.pth checkpoint",
    )

    # --------------------------------------------------------
    # Model configuration
    # --------------------------------------------------------

    parser.add_argument(
        "--no-pretrained",
        action="store_true",
    )

    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help=(
            "Freeze ConvNeXt backbone and train "
            "only the final classification layer"
        ),
    )

    parser.add_argument(
        "--partial-finetune",
        action="store_true",
        help="Train final ConvNeXt feature stage and classifier",
    )

    # --------------------------------------------------------
    # Early stopping
    # --------------------------------------------------------

    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help=(
            "Stop after N epochs without validation "
            "loss improvement; 0 disables"
        ),
    )

    # --------------------------------------------------------
    # Testing / hardware
    # --------------------------------------------------------

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run 10 batches for training and validation",
    )

    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail if CUDA GPU is unavailable",
    )

    args = parser.parse_args()

    # ========================================================
    # 5. Validate Arguments
    # ========================================================

    if (
        args.epochs < 1
        or args.batch_size < 1
        or args.num_workers < 0
    ):
        parser.error(
            "Epochs and batch size must be positive; "
            "num-workers must be nonnegative."
        )

    if args.freeze_backbone and args.model != "convnext":
        parser.error(
            "--freeze-backbone only applies "
            "to --model convnext"
        )

    if args.early_stopping_patience < 0:
        parser.error(
            "--early-stopping-patience must be >= 0"
        )

    if args.smoke_test and args.resume:
        parser.error(
            "Smoke tests cannot be resumed."
        )

    if args.resume and args.no_pretrained:
        print(
            "Note: --no-pretrained is irrelevant "
            "when resuming a checkpoint."
        )

    # ========================================================
    # 6. Initialize Random Seed and Device
    # ========================================================

    set_seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")

    elif torch.backends.mps.is_available():
        device = torch.device("mps")

    else:
        device = torch.device("cpu")

    if args.require_cuda and device.type != "cuda":
        raise RuntimeError(
            "CUDA GPU unavailable. "
            "Run inside a GPU allocation."
        )

    print(f"Device: {device}", flush=True)

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
            flush=True,
        )

        torch.cuda.reset_peak_memory_stats()

    # ========================================================
    # 7. Load Dataset
    # ========================================================

    loaders = get_dataloaders(
        csv_path=args.csv_path,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=224,
    )

    print(
        "Dataset sizes:",
        {
            name: len(loader.dataset)
            for name, loader in loaders.items()
        },
        flush=True,
    )

    print("Class mapping: NC=0, AD=1")
    print("TEST SET IS NOT EVALUATED DURING TRAINING.")

    # ========================================================
    # 8. Initialize Model
    # ========================================================

    model = build_model(
        model_name=args.model,
        num_classes=2,
        pretrained=(
            not args.no_pretrained
            and args.resume is None
        ),
        freeze_backbone=args.freeze_backbone,
        partial_finetune=args.partial_finetune,
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    # Optimize only parameters that are trainable.
    optimizer = AdamW(
        (
            p for p in model.parameters()
            if p.requires_grad
        ),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    total_params = sum(
        p.numel() for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Total parameters: {total_params:,}",
        flush=True,
    )

    print(
        f"Trainable parameters: {trainable_params:,}",
        flush=True,
    )

    # ========================================================
    # 9. Prepare Checkpoint Directory
    # ========================================================

    output_dir = (
        args.checkpoint_dir
        / (
            f"{args.model}_smoke"
            if args.smoke_test
            else args.model
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    start_epoch = 1
    best_val_loss = float("inf")
    history = []
    epochs_without_improvement = 0

    # ========================================================
    # 10. Resume From Checkpoint (Optional)
    # ========================================================

    if args.resume:

        if not args.resume.is_file():
            raise FileNotFoundError(args.resume)

        # Resume only from checkpoint files you trust.
        checkpoint = torch.load(
            args.resume,
            map_location="cpu",
            weights_only=False,
        )

        if checkpoint.get("model_name") != args.model:
            raise ValueError(
                "Checkpoint model does not match --model"
            )

        if (
            checkpoint.get("freeze_backbone", False)
            != args.freeze_backbone
        ):
            raise ValueError(
                "Checkpoint freeze_backbone setting "
                "differs from --freeze-backbone"
            )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        # Transfer optimizer state to active device.
        for state in optimizer.state.values():

            for key, value in state.items():

                if torch.is_tensor(value):
                    state[key] = value.to(device)

        start_epoch = checkpoint["epoch"] + 1

        best_val_loss = checkpoint.get(
            "best_val_loss",
            float("inf"),
        )

        history = checkpoint.get(
            "history",
            [],
        )

        epochs_without_improvement = checkpoint.get(
            "epochs_without_improvement",
            0,
        )

        print(
            f"Resuming from epoch {start_epoch}",
            flush=True,
        )

    # ========================================================
    # 11. Run Training
    # ========================================================

    max_batches = (
        10 if args.smoke_test else None
    )

    epochs_to_run = (
        1 if args.smoke_test else args.epochs
    )

    if start_epoch > epochs_to_run:
        print(
            "Nothing to train: checkpoint already "
            f"completed epoch {start_epoch - 1}."
        )
        return

    for epoch in range(
        start_epoch,
        epochs_to_run + 1,
    ):

        print(
            f"\nEpoch {epoch}/{epochs_to_run}",
            flush=True,
        )

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        train = run_epoch(
            model=model,
            loader=loaders["train"],
            loss_fn=criterion,
            device=device,
            optimizer=optimizer,
            max_batches=max_batches,
        )

        # ----------------------------------------------------
        # Validate
        # ----------------------------------------------------

        val = run_epoch(
            model=model,
            loader=loaders["val"],
            loss_fn=criterion,
            device=device,
            max_batches=max_batches,
        )

        print(
            f" Train: loss={train['loss']:.4f} "
            f"acc={train['accuracy']:.2%} "
            f"({train['images']} images, "
            f"{train['seconds']:.1f}s)",
            flush=True,
        )

        print(
            f" Val:   loss={val['loss']:.4f} "
            f"acc={val['accuracy']:.2%} "
            f"({val['images']} images, "
            f"{val['seconds']:.1f}s)",
            flush=True,
        )

        # ====================================================
        # 12. Record Training History
        # ====================================================

        history.append({
            "epoch": epoch,
            "train_loss": train["loss"],
            "train_accuracy": train["accuracy"],
            "val_loss": val["loss"],
            "val_accuracy": val["accuracy"],
            "train_seconds": train["seconds"],
            "val_seconds": val["seconds"],
        })

        # ====================================================
        # 13. Check Validation Improvement
        # ====================================================

        if val["loss"] < best_val_loss:

            best_val_loss = val["loss"]
            improved = True

        else:
            improved = False

        if improved:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # ====================================================
        # 14. Save Checkpoints
        # ====================================================

        payload = {
            "epoch": epoch,
            "model_name": args.model,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "history": history,
            "class_mapping": {
                "NC": 0,
                "AD": 1,
            },
            "image_size": 224,
            "freeze_backbone": args.freeze_backbone,
            "epochs_without_improvement": (
                epochs_without_improvement
            ),
        }

        torch.save(
            payload,
            output_dir / "last.pth",
        )

        if improved:

            torch.save(
                payload,
                output_dir / "best.pth",
            )

            print(
                " Saved new best checkpoint",
                flush=True,
            )

        write_history(
            output_dir / "history.csv",
            history,
        )

        # ====================================================
        # 15. Early Stopping
        # ====================================================

        if (
            args.early_stopping_patience
            and epochs_without_improvement
            >= args.early_stopping_patience
        ):

            print(
                "Early stopping: no validation "
                "loss improvement in "
                f"{epochs_without_improvement} epochs.",
                flush=True,
            )

            break

    # ========================================================
    # 16. Final Statistics
    # ========================================================

    if device.type == "cuda":

        peak = (
            torch.cuda.max_memory_allocated()
            / (1024 ** 3)
        )

        print(
            f"Peak allocated GPU memory: "
            f"{peak:.2f} GiB",
            flush=True,
        )

    print(
        f"Checkpoints/history: "
        f"{output_dir.resolve()}",
        flush=True,
    )

    if args.smoke_test:

        print(
            "Smoke test complete: limited batches, "
            "not valid final metrics.",
            flush=True,
        )


if __name__ == "__main__":
    main()

