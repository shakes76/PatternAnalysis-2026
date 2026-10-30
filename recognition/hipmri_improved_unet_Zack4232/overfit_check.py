"""Fit three fixed training slices to diagnose the learning pipeline."""

import argparse
import csv
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import torch
from torch import nn

from dataset import HipMRIDataset, LABEL_NAMES
from metrics import positive_slice_stats, summarize_stats
from modules import UNet2D


@torch.no_grad()
def evaluate(model, images, targets, criterion):
    model.eval()
    logits = model(images)
    predictions = logits.argmax(dim=1)
    scores = [
        summarize_stats(positive_slice_stats(predictions, targets, c))
        for c in range(6)
    ]
    return criterion(logits, targets).item(), predictions, scores


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
    fig.suptitle("Training subset overfit check - not test performance")
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
    torch.manual_seed(42)
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

    model = UNet2D().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()

    with (args.output / "history.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["step", "cross_entropy"]
            + [f"dice_class_{c}" for c in range(6)]
        )

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
                loss_value, predictions, scores = evaluate(
                    model, images, targets, criterion
                )
                dice = [s["mean_dice"] for s in scores]
                writer.writerow([step, loss_value] + dice)
                handle.flush()
                print(
                    f"Step {step}: CE={loss_value:.6f}; Dice={dice}",
                    flush=True,
                )

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

    torch.save({
        "model_state_dict": model.state_dict(),
        "steps": args.steps,
        "case_id": 4,
        "week": 0,
        "slice_ids": slice_ids,
        "seed": 42,
        "loss": "CrossEntropyLoss",
        "learning_rate": 1e-3,
        "base_channels": 32,
    }, args.output / "model.pt")

    save_preview(
        images, targets, predictions, slice_ids,
        args.output / "predictions.png",
    )
    print("Saved results to:", args.output.resolve(), flush=True)
    print("Diagnostic run complete; inspect metrics and predictions.", flush=True)


if __name__ == "__main__":
    main()
