"""Fit three fixed training slices to diagnose the learning pipeline."""

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import torch

from dataset import HipMRIDataset, LABEL_NAMES
from metrics import positive_slice_stats, summarize_stats
from modules import ImprovedUNet2D
from losses import CrossEntropyDiceLoss


@torch.no_grad()
def evaluate(model, images, targets, criterion, training=False):
    """Inspect current weights without changing the training RNG stream."""
    previous_mode = model.training
    try:
        model.train(training)
        # A training-mode check samples dropout but must not affect later updates.
        with torch.random.fork_rng(devices=[images.device.index]):
            logits = model(images)
            ce, soft_dice = criterion.components(logits, targets)
            total = ce + criterion.dice_weight * soft_dice
            if not torch.isfinite(total):
                raise RuntimeError("Non-finite diagnostic loss.")
            predictions = logits.argmax(dim=1)
            scores = [
                summarize_stats(positive_slice_stats(predictions, targets, c))
                for c in range(6)
            ]
            losses = {"ce": ce.item(), "soft_dice": soft_dice.item(),
                      "total": total.item()}
            return losses, predictions, scores
    finally:
        model.train(previous_mode)


def save_preview(images, targets, predictions, slice_ids, output):
    cmap = ListedColormap(
        ["black", "lightgray", "gold", "deepskyblue", "limegreen", "magenta"]
    )
    norm = BoundaryNorm(list(range(7)), 6)
    fig, axes = plt.subplots(
        len(slice_ids), 4, figsize=(10, 10), constrained_layout=True
    )
    for row, section in enumerate(slice_ids):
        x = images[row, 0].cpu().numpy()
        y = targets[row].cpu().numpy()
        p = predictions[row].cpu().numpy()
        axes[row, 0].imshow(x, cmap="gray", interpolation="nearest")
        label_plot = axes[row, 1].imshow(
            y, cmap=cmap, norm=norm, interpolation="nearest"
        )
        axes[row, 2].imshow(
            p, cmap=cmap, norm=norm, interpolation="nearest"
        )
        axes[row, 3].imshow(
            p != y, cmap="gray", vmin=0, vmax=1, interpolation="nearest"
        )
        titles = [
            f"Slice {section}: MRI", "Ground truth",
            "Prediction", "Errors (white)",
        ]
        for axis, title in zip(axes[row], titles):
            axis.set_title(title)
            axis.set_axis_off()

    bar = fig.colorbar(
        label_plot, ax=axes[:, 1:3].ravel().tolist(),
        ticks=[i + 0.5 for i in range(6)], shrink=0.6,
    )
    bar.ax.set_yticklabels(LABEL_NAMES)
    fig.suptitle("Improved U-Net overfit check (eval mode) - not test performance")
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 1:
        raise ValueError("steps must be positive.")
    if not torch.cuda.is_available():
        raise RuntimeError("Run inside a GPU job.")

    # Refuse to overwrite a previous experiment.
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    dataset = HipMRIDataset(split="train")
    slice_ids = [30, 37, 44]
    lookup = {
        (r.case_id, r.week, r.slice_id): i
        for i, r in enumerate(dataset.samples)
    }
    samples = [dataset[lookup[(4, 0, s)]] for s in slice_ids]
    images = torch.stack([s["image"] for s in samples]).cuda()
    targets = torch.stack([s["mask"] for s in samples]).cuda()

    print("Training case: 004; week: 0; slices:", slice_ids, flush=True)
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print(
        "Target pixels per class:",
        torch.bincount(targets.flatten(), minlength=6).tolist(),
        flush=True,
    )

    model = ImprovedUNet2D(base_channels=32, dropout=0.3).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = CrossEntropyDiceLoss(dice_weight=1.0)
    config = {
        "model_name": "ImprovedUNet2D", "base_channels": 32,
        "num_classes": 6, "dropout": 0.3, "seed": 42,
        "steps": args.steps, "learning_rate": 0.001,
        "loss": "ce_dice", "dice_weight": 1.0,
        "case_id": 4, "week": 0, "slice_ids": slice_ids,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "torch_version": str(torch.__version__),
        "cuda_build": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "metric_protocol": "Equal mean across GT-positive slices for each class",
        "train_mode_check": "One dropout sample at current weights; no optimizer update",
    }
    (args.output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print("Parameters:", config["parameter_count"], flush=True)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()

    with (args.output / "history.csv").open("w", newline="") as handle:
        fields = ["step"] + [
            f"{mode}_{field}" for mode in ("train_mode", "eval")
            for field in ("ce", "soft_dice", "total")
        ] + [f"{mode}_dice_class_{c}" for mode in ("train_mode", "eval")
             for c in range(6)]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()

        for step in range(args.steps + 1):
            if step > 0:
                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(images), targets)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss at step {step}")
                loss.backward()
                optimizer.step()

            if step % 50 == 0 or step == args.steps:
                train_losses, _, train_scores = evaluate(
                    model, images, targets, criterion, training=True
                )
                eval_losses, predictions, scores = evaluate(
                    model, images, targets, criterion, training=False
                )
                row = {"step": step}
                for mode, values, class_scores in (
                    ("train_mode", train_losses, train_scores),
                    ("eval", eval_losses, scores),
                ):
                    row.update({f"{mode}_{k}": v for k, v in values.items()})
                    row.update({f"{mode}_dice_class_{c}": score["mean_dice"]
                                for c, score in enumerate(class_scores)})
                writer.writerow(row)
                handle.flush()
                print(json.dumps(row), flush=True)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print("Training/check time (seconds):", round(elapsed, 2), flush=True)
    print(
        "Peak PyTorch allocated GPU memory (GiB):",
        round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        flush=True,
    )
    for name, score in zip(LABEL_NAMES, scores):
        print(name, score, flush=True)

    summary = {
        "config": config,
        "train_mode_losses": train_losses,
        "eval_losses": eval_losses,
        "train_mode_scores": dict(zip(LABEL_NAMES, train_scores)),
        "eval_scores": dict(zip(LABEL_NAMES, scores)),
        "training_and_check_seconds": elapsed,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "scope": "Three training slices only; not validation or test performance",
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "model_name": "ImprovedUNet2D",
        "base_channels": 32,
        "num_classes": 6,
        "dropout": 0.3,
        "steps": args.steps,
    }, args.output / "model.pt")

    save_preview(
        images, targets, predictions, slice_ids,
        args.output / "predictions.png",
    )
    print("Saved results to:", args.output.resolve(), flush=True)
    print("Diagnostic run complete; inspect metrics and predictions.", flush=True)


if __name__ == "__main__":
    main()
