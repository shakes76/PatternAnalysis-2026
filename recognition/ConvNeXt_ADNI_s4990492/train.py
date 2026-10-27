"""Train the CNN baseline and evaluate a held-out validation split."""

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import ADNISliceDataset
from modules import SimpleCNN, ConvNeXtTiny


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)

    loss_sum = 0.0
    count = 0
    # Rows: true labels; columns: predicted labels. NC=0, AD=1.
    confusion = torch.zeros(2, 2, dtype=torch.long)

    with torch.set_grad_enabled(training):
        for images, labels in loader:
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--model",
        choices=["baseline", "convnext"],
        default="baseline",
    )
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("Epochs and batch size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("Run this script in a Slurm GPU job.")

    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)

    base = Path(__file__).resolve().parent
    train_data = ADNISliceDataset(base / "splits" / "train.csv")
    val_data = ADNISliceDataset(base / "splits" / "val.csv")

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
        learning_rate = 1e-4
        weight_decay = 0.05
        optimizer_name = "AdamW"
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    criterion = nn.CrossEntropyLoss()

    run_dir = base / "runs" / (
            args.model + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    run_dir.mkdir(parents=True, exist_ok=False)

    config = {
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
        "epoch", "train_loss", "train_accuracy",
        "val_loss", "val_accuracy", "val_nc_recall", "val_ad_recall",
        "val_tn", "val_fp", "val_fn", "val_tp",
        "epoch_seconds", "peak_allocated_mib",
    ]
    best_val_loss = float("inf")

    with (run_dir / "training_log.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()

            train = run_epoch(
                model, train_loader, criterion, device, optimizer
            )
            val = run_epoch(model, val_loader, criterion, device)

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            peak = torch.cuda.max_memory_allocated() / 1024**2

            if val["loss"] < best_val_loss:
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
                f"Epoch {epoch}/{args.epochs} | "
                f"train loss {train['loss']:.4f}, "
                f"acc {train['accuracy']:.1%} | "
                f"val loss {val['loss']:.4f}, "
                f"acc {val['accuracy']:.1%} | "
                f"NC recall {val['nc_recall']:.1%}, "
                f"AD recall {val['ad_recall']:.1%} | "
                f"{elapsed:.1f}s, {peak:.1f} MiB",
                flush=True,
            )

    print("Finished. Best validation loss:", best_val_loss, flush=True)


def dispatch():
    """Expose final workflows while preserving historical CLI behavior."""
    import sys
    routes = {
        'final-train': 'train_convnext_local_imagenet_lr3e6',
        'rgb224-comparison': 'train_convnext_local_initialization',
        'final-validate': 'evaluate_rgb224_validation',
        'final-test': 'evaluate_rgb224_test',
    }
    if len(sys.argv) > 1 and sys.argv[1] in routes:
        import importlib
        module_name = routes[sys.argv.pop(1)]
        importlib.import_module(module_name).main()
    else:
        main()


if __name__ == "__main__":
    dispatch()