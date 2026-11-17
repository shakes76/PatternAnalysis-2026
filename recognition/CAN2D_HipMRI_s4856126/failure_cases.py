"""Visualise difficult HipMRI prostate segmentation slices: Basic U-Net vs CAN.

Loads the same checkpoints and model architectures as evaluate.py.  Selects
5 illustrative cases without changing the models or tuning the test set.

Run on a Rangpur GPU compute node using failure_cases_can.slurm, or use:
    python failure_cases.py --split test --label 5 --top-k 5 --device cuda

Output: outputs/failure_cases_can/ (kept separate from older Improved U-Net work).
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import HipMRIDataset
from evaluate import load_model  # Same verified loader as the quantitative evaluation.


CHECKPOINTS = {
    "baseline": Path("checkpoints/baseline_unet.pt"),
    "can": Path("checkpoints/can_best.pt"),
}


def class_stats(gt, pred, label):
    """Single-slice Dice and FP/FN pixel counts for a verified class label."""
    target = gt == label
    predicted = pred == label
    tp = int(np.logical_and(target, predicted).sum())
    fp = int(np.logical_and(~target, predicted).sum())
    fn = int(np.logical_and(target, ~predicted).sum())
    denominator = 2 * tp + fp + fn
    # We only rank slices containing GT target; denominator is then nonzero.
    dice = (2 * tp / denominator) if denominator > 0 else None
    return dice, fp, fn


def plot_mask(ax, mask, title):
    ax.imshow(mask.astype(np.uint8), cmap="gray", vmin=0, vmax=1)
    ax.set_title(title, fontsize=11)
    ax.axis("off")


def plot_contours(ax, image, gt, prediction, title):
    """Ground-truth boundary in red, model prediction boundary in lime."""
    ax.imshow(image, cmap="gray")
    for mask, colour in ((gt, "red"), (prediction, "lime")):
        if np.any(mask) and not np.all(mask):
            ax.contour(mask.astype(float), levels=[0.5], colors=colour,
                       linewidths=1.4)
    ax.set_title(title, fontsize=11)
    ax.axis("off")


def plot_error(ax, image, pixels, title, cmap):
    ax.imshow(image, cmap="gray")
    overlay = np.ma.masked_where(~pixels, pixels.astype(float))
    ax.imshow(overlay, cmap=cmap, vmin=0, vmax=1, alpha=0.75)
    ax.set_title(title, fontsize=11)
    ax.axis("off")


def create_figure(image, gt_mask, baseline_mask, can_mask, label, record, path):
    """Show the raw MRI, GT, both predictions, boundary overlays and FP/FN."""
    gt = gt_mask == label
    baseline = baseline_mask == label
    can = can_mask == label

    fig, axes = plt.subplots(2, 5, figsize=(20, 9), constrained_layout=True)

    axes[0, 0].imshow(image, cmap="gray")
    axes[0, 0].set_title("Input MRI", fontsize=11)
    axes[0, 0].axis("off")
    plot_mask(axes[0, 1], gt, f"Ground truth (Label {label})")
    plot_mask(axes[0, 2], baseline, "Basic U-Net prediction")
    plot_mask(axes[0, 3], can, "CAN prediction")
    plot_contours(axes[0, 4], image, gt, baseline,
                  f"Basic contours | Dice {record['baseline_dice']:.4f}")

    plot_contours(axes[1, 0], image, gt, can,
                  f"CAN contours | Dice {record['can_dice']:.4f}")
    plot_error(axes[1, 1], image, baseline & ~gt,
               f"Basic FP ({record['baseline_fp']})", "Reds")
    plot_error(axes[1, 2], image, gt & ~baseline,
               f"Basic FN ({record['baseline_fn']})", "Blues")
    plot_error(axes[1, 3], image, can & ~gt,
               f"CAN FP ({record['can_fp']})", "Reds")
    plot_error(axes[1, 4], image, gt & ~can,
               f"CAN FN ({record['can_fn']})", "Blues")

    fig.suptitle(
        f"{record['filename']} | {record['selection_reason']} | "
        f"GT prostate pixels: {record['gt_pixels']} | "
        f"CAN minus Basic Dice: {record['dice_difference']:+.4f}\n"
        "Contours: red = ground truth, green = prediction.  "
        "FP = predicted extra; FN = missed target.",
        fontsize=12,
    )
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def select_cases(records, top_k):
    """Pick unique, contrasting examples; never claim anatomical base/apex from index."""
    if not records:
        return []

    gain = [r for r in records if r["dice_difference"] > 0]
    regression = [r for r in records if r["dice_difference"] < 0]
    larger = [r for r in records if r["gt_pixels"] >= 100]

    ranked = [
        ("CAN lowest Dice", sorted(records,
             key=lambda r: (r["can_dice"], -r["gt_pixels"]))),
        ("CAN largest improvement", sorted(gain,
             key=lambda r: (-r["dice_difference"], -r["gt_pixels"]))),
        ("CAN largest regression", sorted(regression,
             key=lambda r: (r["dice_difference"], -r["gt_pixels"]))),
        ("Basic U-Net lowest Dice", sorted(records,
             key=lambda r: (r["baseline_dice"], -r["gt_pixels"]))),
        ("Larger prostate, difficult slice", sorted(larger,
             key=lambda r: ((r["baseline_dice"] + r["can_dice"]) / 2,
                            -r["gt_pixels"]))),
    ]

    chosen = []
    seen = set()
    for reason, candidates in ranked:
        for row in candidates:
            if row["index"] not in seen:
                chosen.append({**row, "selection_reason": reason})
                seen.add(row["index"])
                break
        if len(chosen) >= top_k:
            return chosen

    # Fill remaining slots with difficult cases not already selected.
    for row in sorted(records, key=lambda r:
                      ((r["baseline_dice"] + r["can_dice"]) / 2,
                       -r["gt_pixels"])):
        if row["index"] not in seen:
            chosen.append({**row, "selection_reason": "Low Dice in both models"})
            seen.add(row["index"])
        if len(chosen) >= top_k:
            break
    return chosen


def write_csv(path, records):
    if not records:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["validate", "test"], default="test")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--label", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    if not 0 <= args.label < 6:
        parser.error("--label must be between 0 and 5")
    if args.top_k < 1 or args.batch_size < 1:
        parser.error("--top-k and --batch-size must be positive")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; run in a Slurm A100 GPU job")

    print("Loading Basic U-Net and CAN checkpoints...", flush=True)
    baseline_model = load_model("baseline", CHECKPOINTS["baseline"], device)
    can_model = load_model("can", CHECKPOINTS["can"], device)
    baseline_model.eval()
    can_model.eval()

    dataset = HipMRIDataset(split=args.split)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, pin_memory=device.type == "cuda")
    records = []
    offset = 0

    print(f"Comparing {len(dataset)} {args.split} slices...", flush=True)
    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device)
            baseline_preds = baseline_model(images).argmax(dim=1).cpu().numpy()
            can_preds = can_model(images).argmax(dim=1).cpu().numpy()
            targets = targets.numpy()

            for j, gt in enumerate(targets):
                gt_count = int((gt == args.label).sum())
                if gt_count == 0:
                    # These visualisations focus on boundary/undersegmentation
                    # where the selected anatomical class is actually present.
                    continue

                bdice, bfp, bfn = class_stats(gt, baseline_preds[j], args.label)
                cdice, cfp, cfn = class_stats(gt, can_preds[j], args.label)
                index = offset + j
                # Dataset implementations may not expose a public `pairs` list.
                filename = f"{args.split}_slice_{index:04d}"
                pairs = getattr(dataset, "pairs", None)
                if pairs is not None:
                    try:
                        item = pairs[index][0]
                        filename = Path(str(item)).name
                    except (TypeError, IndexError, KeyError, AttributeError):
                        pass
                records.append({
                    "index": index,
                    "filename": filename,
                    "gt_pixels": gt_count,
                    "baseline_dice": bdice,
                    "can_dice": cdice,
                    "dice_difference": cdice - bdice,
                    "baseline_fp": bfp,
                    "baseline_fn": bfn,
                    "can_fp": cfp,
                    "can_fn": cfn,
                })
            offset += len(targets)

    if not records:
        print(f"No {args.split} slices contain label {args.label}.")
        return

    output_dir = Path("outputs/failure_cases_can")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / f"{args.split}_all_target_slices.csv", records)

    selected = select_cases(records, args.top_k)
    write_csv(output_dir / f"{args.split}_selected_cases.csv", selected)

    for row in selected:
        idx = row["index"]
        image, gt = dataset[idx]
        with torch.inference_mode():
            inp = image.unsqueeze(0).to(device)
            baseline_pred = baseline_model(inp).argmax(dim=1)[0].cpu().numpy()
            can_pred = can_model(inp).argmax(dim=1)[0].cpu().numpy()

        out_path = output_dir / f"{args.split}_case_{idx:04d}_baseline_vs_can.png"
        create_figure(image[0].numpy(), gt.numpy(), baseline_pred,
                      can_pred, args.label, row, out_path)
        print(
            f"Case {idx}: GT={row['gt_pixels']}px, "
            f"Basic={row['baseline_dice']:.4f}, CAN={row['can_dice']:.4f}, "
            f"difference={row['dice_difference']:+.4f}, "
            f"reason={row['selection_reason']}",
            flush=True,
        )

    print("Saved figures and per-slice CSVs to:", output_dir, flush=True)
    print("These cases are illustrative, not a statistical significance test.",
          flush=True)
    print("Do not infer prostate base/apex or clinical outcomes from slice index alone.",
          flush=True)
    print("Failure-case analysis complete.", flush=True)


if __name__ == "__main__":
    main()
