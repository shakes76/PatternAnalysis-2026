"""predict.py: score trained segmentation models on the OASIS test set.

Usage:
    python predict.py --root ~/oasis_data --unet-weights runs/unet_b16/best.pt --improved-weights runs/improved_unet_b16/best.pt
"""

import argparse

import torch
import torch.nn.functional as F

from dataset import get_dataloaders
from train import MODELS, build_model, get_device, evaluate

CLASS_NAMES = ["background", "CSF", "grey matter", "white matter"]

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # draw to files only, no window
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

CLASS_COLOURS = ["black", "#4575b4", "#fdae61", "#ffffbf"]  # background, CSF, grey, white
CLASS_CMAP = ListedColormap(CLASS_COLOURS)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained model on the OASIS test set.")
    parser.add_argument("--root", required=True,
                        help="folder holding the six keras_png_slices_* folders")
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--unet-weights", help="best.pt of the plain U-Net")
    parser.add_argument("--improved-weights", help="best.pt of the Improved U-Net")
    parser.add_argument("--out-dir", default="figures", help="where figures are saved")
    parser.add_argument("--n-samples", type=int, default=6)
    parser.add_argument("--n-worst", type=int, default=5)
    args = parser.parse_args()
    if args.unet_weights is None and args.improved_weights is None:
        parser.error("give at least one of --unet-weights and --improved-weights")
    return args


def load_model(name, base_channels, weights, device):
    """Build a model, load trained weights into it and switch to evaluation mode."""
    model = build_model(name, base_channels).to(device)
    state = torch.load(weights, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model

def print_comparison(results):
    """Print per-class test Dice for each model side by side."""
    names = list(results)
    two = len(names) == 2

    def row(label, values):
        line = f"{label:14s}" + "".join(f"{v:16.4f}" for v in values)
        if two:
            line += f"{values[1] - values[0]:+16.4f}"
        print(line)

    print(f"{'':14s}" + "".join(f"{n:>16s}" for n in names)
          + (f"{'improved-unet':>16s}" if two else ""))
    for i, class_name in enumerate(CLASS_NAMES):
        row(class_name, [results[n][i].item() for n in names])
    row("mean", [results[n].mean().item() for n in names])
    row("lowest label", [results[n].min().item() for n in names])
    for n in names:
        print(f"{n}: all labels >= 0.9: {bool((results[n] >= 0.9).all())}")

@torch.no_grad()
def predict_slice(model, image, device):
    """Predicted class map (H, W) for one image of shape (1, H, W)."""
    logits = model(image.unsqueeze(0).to(device))
    return logits.argmax(dim=1)[0].cpu()


def save_samples(models, dataset, indices, device, path):
    """One row per test slice: input, ground truth, then one prediction per model."""
    columns = ["input", "truth"] + list(models)
    fig, axes = plt.subplots(len(indices), len(columns), squeeze=False,
                             figsize=(3 * len(columns), 3 * len(indices)))
    for r, idx in enumerate(indices):
        image, mask = dataset[int(idx)]
        panels = [image[0], mask.argmax(dim=0)]
        panels += [predict_slice(m, image, device) for m in models.values()]
        for c, panel in enumerate(panels):
            ax = axes[r][c]
            if c == 0:
                ax.imshow(panel.numpy(), cmap="gray")
            else:
                ax.imshow(panel.numpy(), cmap=CLASS_CMAP, vmin=0, vmax=3,
                          interpolation="nearest")
            ax.axis("off")
            if r == 0:
                ax.set_title(columns[c])
    fig.legend(handles=[Patch(color=CLASS_COLOURS[i], label=CLASS_NAMES[i]) for i in range(4)],
               loc="lower center", ncol=4)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)

@torch.no_grad()
def per_slice_dice(model, loader, device, smooth=1e-5):
    """Dice for every slice and class separately.

    Returns (dice, present): two (num_slices, num_classes) tensors, in dataset order.
    present marks the classes that appear in a slice's ground truth.
    """
    model.eval()
    dice, present = [], []
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        pred = F.one_hot(model(images).argmax(dim=1), masks.shape[1]).permute(0, 3, 1, 2).float()
        dims = (2, 3)  # sum over the pixels of each slice only
        inter = (pred * masks).sum(dim=dims)
        total = pred.sum(dim=dims) + masks.sum(dim=dims)
        dice.append(((2 * inter + smooth) / (total + smooth)).cpu())
        present.append((masks.sum(dim=dims) > 0).cpu())
    return torch.cat(dice), torch.cat(present)


def slice_scores(dice, present):
    """One score per slice: mean Dice over the tissue classes present in its truth."""
    tissue = present[:, 1:].float()
    count = tissue.sum(dim=1)
    scores = (dice[:, 1:] * tissue).sum(dim=1) / count.clamp(min=1)
    return torch.where(count > 0, scores, torch.ones_like(scores))


def print_worst_slices(worst, dataset, dice_by_model, scores, focus):
    """List the lowest-scoring slices with each model's score and per-class Dice."""
    print(f"\n{len(worst)} lowest-scoring test slices for {focus} "
          f"(median slice score {scores[focus].median().item():.3f}):")
    for rank, idx in enumerate(worst.tolist(), start=1):
        name = dataset.pairs[idx][0].name
        by_model = " | ".join(f"{m} {scores[m][idx]:.3f}" for m in scores)
        per_class = " ".join(f"{c} {d:.2f}" for c, d in
                             zip(CLASS_NAMES[1:], dice_by_model[focus][idx, 1:].tolist()))
        print(f"{rank}. [{idx}] {name} | {by_model} | {per_class}")

def save_failure_figures(model, dataset, worst, scores, device, out_dir):
    """One four-panel figure per slice: input, truth, prediction and the pixels that differ."""
    for rank, idx in enumerate(worst.tolist(), start=1):
        image, mask = dataset[idx]
        truth = mask.argmax(dim=0)
        pred = predict_slice(model, image, device)
        wrong = (pred != truth).numpy()

        fig, axes = plt.subplots(1, 4, figsize=(12, 3.6))
        axes[0].imshow(image[0].numpy(), cmap="gray")
        axes[1].imshow(truth.numpy(), cmap=CLASS_CMAP, vmin=0, vmax=3, interpolation="nearest")
        axes[2].imshow(pred.numpy(), cmap=CLASS_CMAP, vmin=0, vmax=3, interpolation="nearest")
        axes[3].imshow(image[0].numpy(), cmap="gray")
        axes[3].imshow(np.ma.masked_where(~wrong, wrong), cmap=ListedColormap(["red"]),
                       vmin=0, vmax=1, interpolation="nearest")
        for ax, title in zip(axes, ["input", "truth", "prediction", "errors (red)"]):
            ax.set_title(title)
            ax.axis("off")
        fig.suptitle(f"#{rank} {dataset.pairs[idx][0].name} | slice score {scores[idx]:.3f}")
        fig.tight_layout()
        path = out_dir / f"failure_{rank}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"saved {path}")

def main():
    args = parse_args()
    device = get_device()
    _, _, test_loader = get_dataloaders(args.root, batch_size=args.batch_size,
                                        num_workers=args.num_workers)
    print(f"device: {device} | test set: {len(test_loader.dataset)} slices")

    results, models = {}, {}
    for name, weights in (("unet", args.unet_weights),
                          ("improved_unet", args.improved_weights)):
        if weights is None:
            continue
        models[name] = load_model(name, args.base_channels, weights, device)
        results[name] = evaluate(models[name], test_loader, device)
    print_comparison(results)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    n = len(test_loader.dataset)
    indices = np.linspace(0, n - 1, args.n_samples + 2)[1:-1].astype(int)
    save_samples(models, test_loader.dataset, indices, device, out_dir / "predictions.png")
    print(f"saved {out_dir / 'predictions.png'}")

    dice_by_model, present = {}, None
    for name, model in models.items():
        dice_by_model[name], present = per_slice_dice(model, test_loader, device)
    scores = {name: slice_scores(d, present) for name, d in dice_by_model.items()}
    focus = "improved_unet" if "improved_unet" in models else "unet"
    worst = scores[focus].argsort()[:args.n_worst]
    print_worst_slices(worst, test_loader.dataset, dice_by_model, scores, focus)
    save_failure_figures(models[focus], test_loader.dataset, worst, scores[focus], device, out_dir)

if __name__ == "__main__":
    main()