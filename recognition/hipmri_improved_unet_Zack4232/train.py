"""Train and validate a standard or improved HipMRI 2D U-Net."""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import DEFAULT_DATA_ROOT, HipMRIDataset, LABEL_NAMES
from metrics import positive_slice_stats, summarize_stats
from modules import UNet2D, ImprovedUNet2D
from sampling import ShapeBatchSampler
from losses import CrossEntropyDiceLoss


def synchronize():
    """Finish queued GPU work before reading wall-clock time."""
    torch.cuda.synchronize()


def save_checkpoint(path, model, optimizer, epoch, args, val_dice):
    """Replace a checkpoint only after its temporary file is complete."""
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "model_name": args.model,
            "dropout": 0.3 if args.model == "ImprovedUNet2D" else 0.0,
            "base_channels": 32,
            "num_classes": 6,
            "validation_prostate_dice": val_dice,
            "config": vars(args) | {
                "data_root": str(args.data_root),
                "output": str(args.output),
            },
        },
        temporary,
    )
    temporary.replace(path)


class LossMeter:
    """Track CE by slice and soft Dice by eligible slice-class pair."""

    def __init__(self):
        self.ce_sum = 0.0
        self.dice_sum = 0.0
        self.total_sum = 0.0
        self.slices = 0
        self.pairs = 0

    def update(self, ce, dice, total, targets):
        count = targets.shape[0]
        self.ce_sum += ce.item() * count
        self.total_sum += total.item() * count
        self.slices += count
        if dice is not None:
            pairs = sum(
                (targets == label).flatten(1).any(dim=1).sum().item()
                for label in range(1, 6)
            )
            self.dice_sum += dice.item() * pairs
            self.pairs += pairs

    def summary(self):
        if self.slices == 0:
            raise RuntimeError("Empty data loader.")
        return {
            "ce": self.ce_sum / self.slices,
            "soft_dice_loss": self.dice_sum / self.pairs if self.pairs else None,
            "total_batch_mean": self.total_sum / self.slices,
        }


def loss_components(criterion, logits, targets):
    if isinstance(criterion, CrossEntropyDiceLoss):
        ce, dice = criterion.components(logits, targets)
        return ce, dice, ce + criterion.dice_weight * dice
    ce = criterion(logits, targets)
    return ce, None, ce


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    meter = LossMeter()
    for batch_index, batch in enumerate(loader, start=1):
        images = batch["image"].cuda(non_blocking=True)
        targets = batch["mask"].cuda(non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        ce, dice, loss = loss_components(criterion, model(images), targets)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss at batch {batch_index}")
        loss.backward()
        optimizer.step()
        meter.update(ce, dice, loss, targets)
        if batch_index % 250 == 0 or batch_index == len(loader):
            values = meter.summary()
            print(
                f"  Batch {batch_index}/{len(loader)}: "
                f"running CE={values['ce']:.6f}; "
                f"total batch mean={values['total_batch_mean']:.6f}",
                flush=True,
            )
    return meter.summary()


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    meter = LossMeter()
    totals = [None for _ in LABEL_NAMES]
    for batch in loader:
        images = batch["image"].cuda(non_blocking=True)
        targets = batch["mask"].cuda(non_blocking=True)
        logits = model(images)
        ce, dice, loss = loss_components(criterion, logits, targets)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite validation loss.")
        predictions = logits.argmax(dim=1)
        meter.update(ce, dice, loss, targets)
        for class_id in range(len(LABEL_NAMES)):
            current = positive_slice_stats(predictions, targets, class_id=class_id)
            if totals[class_id] is None:
                totals[class_id] = current
            else:
                for key, value in current.items():
                    totals[class_id][key] += value
    values = meter.summary()
    scores = {
        name: summarize_stats(stats)
        for name, stats in zip(LABEL_NAMES, totals)
    }
    return values, scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--loss", choices=("ce", "ce_dice"), default="ce")
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--model", choices=("UNet2D", "ImprovedUNet2D"), default="UNet2D")
    args = parser.parse_args()
    if not np.isfinite(args.dice_weight) or args.dice_weight < 0:
        raise ValueError("Dice weight must be finite and non-negative.")

    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        raise ValueError("Epochs, batch size, and learning rate must be positive.")
    if not torch.cuda.is_available():
        raise RuntimeError("Run this script inside a GPU job.")

    # Keep CPU use within the requested two CPUs.
    torch.set_num_threads(2)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    args.output.mkdir(parents=True, exist_ok=False)
    train_data = HipMRIDataset(args.data_root, split="train")
    val_data = HipMRIDataset(args.data_root, split="validate")

    train_cases = {record.case_id for record in train_data.samples}
    val_cases = {record.case_id for record in val_data.samples}
    if train_cases & val_cases:
        raise RuntimeError("Training and validation cases overlap.")

    train_sampler = ShapeBatchSampler(
        train_data, batch_size=args.batch_size, seed=args.seed
    )
    print(
        "Training shape groups:",
        {str(shape): len(indices)
         for shape, indices in train_sampler.groups.items()},
        flush=True,
    )
    train_loader = DataLoader(
        train_data,
        batch_sampler=train_sampler,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    model = (
        UNet2D(base_channels=32) if args.model == "UNet2D"
        else ImprovedUNet2D(base_channels=32, dropout=0.3)
    ).cuda()
    parameter_count = sum(p.numel() for p in model.parameters())
    expected_parameters = {"UNet2D": 7759686, "ImprovedUNet2D": 12738610}
    if parameter_count != expected_parameters[args.model]:
        raise RuntimeError("Model parameter count differs from the checked model.")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate
    )
    criterion = (
        nn.CrossEntropyLoss() if args.loss == "ce"
        else CrossEntropyDiceLoss(dice_weight=args.dice_weight)
    )
    config = {
        **vars(args),
        "data_root": str(args.data_root),
        "output": str(args.output),
        "model": args.model,
        "dropout": 0.3 if args.model == "ImprovedUNet2D" else 0.0,
        "base_channels": 32,
        "parameter_count": parameter_count,
        "loss": args.loss,
        "loss_reporting": {
            "ce": "Slice-weighted mean of batch CE; same as the original baseline",
            "soft_dice_loss": "Mean over GT-positive slice-class pairs, labels 1-5; null for CE mode or no pairs",
            "total_batch_mean": "Slice-weighted mean of batch objectives, not CE plus the epoch pair-weighted Dice mean",
        },
        "augmentation": False,
        "mixed_precision": False,
        "num_workers": 0,
        "train_slices": len(train_data),
        "validation_slices": len(val_data),
        "train_case_ids": sorted(train_cases),
        "validation_case_ids": sorted(val_cases),
        "torch_version": str(torch.__version__),
        "cuda_build": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "metric_protocol": "Equal mean across ground-truth-positive slices",
    }
    (args.output / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    print(json.dumps(config, indent=2), flush=True)

    best_dice = -1.0
    fields = [
        "epoch", "train_ce", "validation_ce",
        "train_soft_dice_loss", "validation_soft_dice_loss",
        "train_total_batch_mean", "validation_total_batch_mean",
        "validation_prostate_dice", "validation_prostate_iou",
        "eligible_prostate_slices", "excluded_prostate_slices",
        "negative_slices_with_prostate_fp",
        "train_seconds", "validation_seconds",
        "peak_allocated_gib", "peak_reserved_gib",
    ]

    with (args.output / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, lineterminator="\n"
        )
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            train_sampler.set_epoch(epoch - 1)
            print(f"\nEpoch {epoch}/{args.epochs}", flush=True)
            torch.cuda.reset_peak_memory_stats()
            synchronize()
            start = time.perf_counter()

            train_loss = train_epoch(
                model, train_loader, optimizer, criterion
            )
            synchronize()
            train_seconds = time.perf_counter() - start

            start = time.perf_counter()
            val_loss, scores = validate(model, val_loader, criterion)
            synchronize()
            val_seconds = time.perf_counter() - start

            prostate = scores["Prostate"]
            val_dice = prostate["mean_dice"]
            if val_dice is None:
                raise RuntimeError("No eligible prostate validation slices.")

            row = {
                "epoch": epoch,
                "train_ce": train_loss["ce"],
                "validation_ce": val_loss["ce"],
                "train_soft_dice_loss": train_loss["soft_dice_loss"],
                "validation_soft_dice_loss": val_loss["soft_dice_loss"],
                "train_total_batch_mean": train_loss["total_batch_mean"],
                "validation_total_batch_mean": val_loss["total_batch_mean"],
                "validation_prostate_dice": val_dice,
                "validation_prostate_iou": prostate["mean_iou"],
                "eligible_prostate_slices": prostate["eligible_slices"],
                "excluded_prostate_slices": prostate["excluded_slices"],
                "negative_slices_with_prostate_fp":
                    prostate["negative_slices_with_fp"],
                "train_seconds": train_seconds,
                "validation_seconds": val_seconds,
                "peak_allocated_gib":
                    torch.cuda.max_memory_allocated() / 1024**3,
                "peak_reserved_gib":
                    torch.cuda.max_memory_reserved() / 1024**3,
            }
            writer.writerow(row)
            handle.flush()

            (args.output / f"validation_epoch_{epoch:03d}.json").write_text(
                json.dumps(scores, indent=2), encoding="utf-8"
            )
            save_checkpoint(
                args.output / "last.pt",
                model, optimizer, epoch, args, val_dice,
            )
            if val_dice > best_dice:
                best_dice = val_dice
                save_checkpoint(
                    args.output / "best.pt",
                    model, optimizer, epoch, args, val_dice,
                )

            print(json.dumps(row, indent=2), flush=True)
            print("Best validation prostate Dice:", best_dice, flush=True)

    print("Training complete. No test data were evaluated.", flush=True)


if __name__ == "__main__":
    main()
