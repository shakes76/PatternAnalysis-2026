"""Local cosine-learning-rate comparison against the fixed-LR control."""

import argparse
import math
import csv
import json
import hashlib
from itertools import islice
import time
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import ADNISliceDataset
from dataset_aug import AugmentedADNISliceDataset
from modules import SimpleCNN, ConvNeXtTiny


def run_epoch(model, loader, criterion, device, optimizer=None, max_batches=None):
    training = optimizer is not None
    model.train(training)

    loss_sum = 0.0
    count = 0
    # Rows: true labels; columns: predicted labels. NC=0, AD=1.
    confusion = torch.zeros(2, 2, dtype=torch.long)

    with torch.set_grad_enabled(training):
        for images, labels in islice(loader, max_batches):
            images = images.to(device)
            labels = labels.to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)

            if not torch.isfinite(loss).item():
                raise RuntimeError("Non-finite loss")

            if training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            loss_sum += loss.item() * batch_size
            count += batch_size

            predictions = logits.argmax(dim=1)
            indices = (labels * 2 + predictions).detach().cpu()
            confusion += torch.bincount(
                indices, minlength=4
            ).reshape(2, 2)

    if count == 0:
        raise RuntimeError("No samples processed")

    tn, fp, fn, tp = confusion.flatten().tolist()

    return {
        "loss": loss_sum / count,
        "accuracy": (tn + tp) / count,
        "nc_recall": tn / (tn + fp) if tn + fp else 0.0,
        "ad_recall": tp / (tp + fn) if tp + fn else 0.0,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def epoch_learning_rate(epoch, epochs, peak=3e-5, minimum=1e-6):
    """Epoch 1 uses peak; the last epoch uses minimum (no warmup)."""
    if epochs < 2 or not 1 <= epoch <= epochs:
        raise ValueError("Cosine schedule requires >=2 epochs and a valid epoch")
    progress = (epoch - 1) / (epochs - 1)
    return minimum + 0.5 * (peak - minimum) * (1 + math.cos(math.pi * progress))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--model",
        choices=["baseline", "convnext"],
        default="baseline",
    )
    parser.add_argument("--smoke", action="store_true", help="Two training and two validation batches only")
    args = parser.parse_args()
    if args.model != "convnext":
        raise ValueError("This experiment requires --model convnext")

    if args.epochs < 2 or args.batch_size < 1:
        raise ValueError("Epochs must be >=2 and batch size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: install CUDA-enabled PyTorch in the local environment.")

    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)

    base = Path(__file__).resolve().parent
    manifests = base.parents[1] / "local_data" / "splits_local"
    train_data = AugmentedADNISliceDataset(manifests / "train.csv")
    val_data = ADNISliceDataset(manifests / "val.csv")

    train_patients = {r["patient_id"] for r in train_data.rows}
    val_patients = {r["patient_id"] for r in val_data.rows}
    if train_patients & val_patients:
        raise ValueError("Patient overlap between train and validation")

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device("cuda")
    if args.model == "baseline":
        model = SimpleCNN().to(device)
        learning_rate = 1e-3
        weight_decay = 0.0
        optimizer_name = "Adam"
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    else:
        model = ConvNeXtTiny().to(device)
        learning_rate = 3e-5
        weight_decay = 0.05
        optimizer_name = "AdamW"
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    criterion = nn.CrossEntropyLoss()

    run_dir = base / "runs" / (
            ("local_cosine_smoke_" if args.smoke else "local_convnext_aug_cosine_") + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    run_dir.mkdir(parents=True, exist_ok=False)

    config = {
        "experiment": "local_convnext_train_affine_cosine",
        "lr_schedule": {"type": "epoch_cosine", "peak": 3e-5, "minimum": 1e-6,
                        "warmup_epochs": 0, "first_epoch": 1, "last_epoch": args.epochs},
        "diagnostic_only": args.smoke,
        "python_note": "Local environment; compare separately from remote runs",
        "manifest_sha256": {
            name: hashlib.sha256((manifests / f"{name}.csv").read_bytes()).hexdigest()
            for name in ("train", "val")
        },
        "train_augmentation": {
            "rotation_degrees": 5,
            "translation_fraction": [0.03, 0.03],
            "interpolation": "bilinear",
            "fill": 0,
        },
        "validation_augmentation": None,
        "model": type(model).__name__,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": learning_rate,
        "optimizer": optimizer_name,
        "weight_decay": weight_decay,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "seed": seed,
        "train_images": len(train_data),
        "val_images": len(val_data),
        "train_patients": len(train_patients),
        "val_patients": len(val_patients),
        "torch_version": str(torch.__version__),
        "gpu": torch.cuda.get_device_name(0),
        "selection_metric": "validation_loss",
        "evaluation_unit": "slice",
    }
    with (run_dir / "config.json").open("w") as file:
        json.dump(config, file, indent=2)

    print("Run directory:", run_dir, flush=True)
    print("GPU:", config["gpu"], flush=True)
    print("Train / validation images:",
          len(train_data), len(val_data), flush=True)

    fields = [
        "epoch", "learning_rate", "train_loss", "train_accuracy",
        "val_loss", "val_accuracy", "val_nc_recall", "val_ad_recall",
        "val_tn", "val_fp", "val_fn", "val_tp",
        "epoch_seconds", "peak_allocated_mib",
    ]
    best_val_loss = float("inf")

    with (run_dir / "training_log.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for epoch in range(1, (1 if args.smoke else args.epochs) + 1):
            current_lr = epoch_learning_rate(epoch, args.epochs)
            for group in optimizer.param_groups:
                group["lr"] = current_lr
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()

            train = run_epoch(
                model, train_loader, criterion, device, optimizer,
                max_batches=2 if args.smoke else None
            )
            val = run_epoch(model, val_loader, criterion, device,
                            max_batches=2 if args.smoke else None)

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            peak = torch.cuda.max_memory_allocated() / 1024**2

            if not args.smoke and val["loss"] < best_val_loss:
                best_val_loss = val["loss"]
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch,
                        "val_loss": val["loss"],
                        "config": config,
                    },
                    run_dir / f"best_{args.model}.pth",
                )

            writer.writerow({
                "epoch": epoch,
                "learning_rate": current_lr,
                "train_loss": train["loss"],
                "train_accuracy": train["accuracy"],
                "val_loss": val["loss"],
                "val_accuracy": val["accuracy"],
                "val_nc_recall": val["nc_recall"],
                "val_ad_recall": val["ad_recall"],
                "val_tn": val["tn"],
                "val_fp": val["fp"],
                "val_fn": val["fn"],
                "val_tp": val["tp"],
                "epoch_seconds": elapsed,
                "peak_allocated_mib": peak,
            })
            file.flush()

            print(
                f"Epoch {epoch}/{args.epochs} | lr {current_lr:.3e} | "
                f"train loss {train['loss']:.4f}, "
                f"acc {train['accuracy']:.1%} | "
                f"val loss {val['loss']:.4f}, "
                f"acc {val['accuracy']:.1%} | "
                f"NC recall {val['nc_recall']:.1%}, "
                f"AD recall {val['ad_recall']:.1%} | "
                f"{elapsed:.1f}s, {peak:.1f} MiB",
                flush=True,
            )

    if args.smoke:
        print("SMOKE PASS: partial batches only; no checkpoint saved. Do not report as performance.", flush=True)
    else:
        print("Finished. Best validation loss:", best_val_loss, flush=True)


if __name__ == "__main__":
    main()