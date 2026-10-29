"""Evaluate the trained baseline U-Net and improved U-NET on the test data set. Visualise HipMRI test predictions, failure cases and determine quantitative markers.

+Outputs:
- Console: per-class Dice / IoU, side-by-side comparison, inference profiling,
  and per-slice prostate error counts for five automatically selected cases.
- PNG: up to ten full-class MRI/ground-truth/prediction examples PER model,
  plus five six-panel prostate failure-case comparisons across both models.
"""


import math
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import numpy as np
import torch
import argparse
import time

from matplotlib.colors import ListedColormap
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from dataset import HipMRIDataset
from modules import UNet2D, ImprovedUNet2D


# Change DATA_ROOT if running on Windows instead of Rangpur.
DATA_ROOT = r"C:\Users\willf\OneDrive\Desktop\Comp3710\keras_slices_data"
MODELS = {
    "baseline": UNet2D,
    "improved": ImprovedUNet2D
}

# Defaults for the final 30-epoch experiment; change with command-line options.
BATCH_SIZE = 2
BASE_CHANNELS = 32       
NUM_CLASSES = 6
PROSTATE_CLASS = 5      
MAX_VISUALISATIONS = 10
FAILURE_CASES = 5
LATENCY_WARMUP = 10
LATENCY_RUNS = 30
RUNS_DIR = Path("runs")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_COLORS = ListedColormap(["black", "red", "lime", "blue", "yellow", "magenta"])

def dice_and_iou(intersections, totals):
    """Calculate pooled Dice and IoU for each class over the complete test set.

    `totals` is predicted pixel count + ground-truth pixel count per class.
    Both-empty classes are undefined (NaN), not perfect predictions.
    """
    dice, iou = [], []
    for intersection, total in zip(intersections, totals):
        intersection, total = int(intersection), int(total)
        if total == 0:
            dice.append(float("nan"))
            iou.append(float("nan"))
        else:
            dice.append(2 * intersection / total)
            iou.append(intersection / (total - intersection))
    return dice, iou


def prostate_slice_metrics(prediction, truth):
    """Return prostate Dice and TP/FP/FN counts for an individual slice."""
    predicted = prediction == PROSTATE_CLASS
    actual = truth == PROSTATE_CLASS
    tp = int(np.count_nonzero(predicted & actual))
    fp = int(np.count_nonzero(predicted & ~actual))
    fn = int(np.count_nonzero(~predicted & actual))
    denominator = 2 * tp + fp + fn
    return {
        "dice": (2 * tp / denominator if denominator else float("nan")),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "gt_pixels": int(np.count_nonzero(actual)),
    }


def full_class_overlay(ax, labels):
    """Overlay visible classes 1--5 on top of a grayscale MRI."""
    ax.imshow(
        np.ma.masked_equal(labels, 0),
        cmap=CLASS_COLORS,
        vmin=0,
        vmax=NUM_CLASSES - 1,
        alpha=0.5,
        interpolation="nearest",
    )


def display_arrays(rotate, *arrays):
    """Optionally rotate EVERY image/mask together for visualisation ONLY."""
    if rotate:
        return tuple(np.rot90(array) for array in arrays)
    return arrays


def save_comparison(image, ground_truth, prediction, filename, rotate=False):
    """Save the ORIGINAL three-panel full-class comparison for one model."""
    image, ground_truth, prediction = display_arrays(
        rotate, image, ground_truth, prediction
    )
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    try:
        for ax, title in zip(axes, ["MRI", "Ground truth", "U-Net prediction"]):
            ax.imshow(image, cmap="gray")
            ax.set_title(title)
            ax.axis("off")
        full_class_overlay(axes[1], ground_truth)
        full_class_overlay(axes[2], prediction)
        fig.tight_layout()
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(filename, dpi=150)
    finally:
        plt.close(fig)


def profile_model(model, sample):
    """Profile one-slice model forward passes, excluding dataset loading or training.
    """
    with torch.inference_mode():
        for _ in range(LATENCY_WARMUP):
            output = model(sample)
            del output

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        output = model(sample)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
            allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
            reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
        else:
            allocated = reserved = None
        del output

        times_ms = []
        for _ in range(LATENCY_RUNS):
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            output = model(sample)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            times_ms.append((time.perf_counter() - start) * 1000)
            del output

    return {
        "params": sum(p.numel() for p in model.parameters()),
        "median_ms": float(np.median(times_ms)),
        "peak_allocated_mib": allocated,
        "peak_reserved_mib": reserved,
    }


def evaluate_model(model, test_loader, save_dir, rotate=False):
    """Evaluate all test slices and save up to 10 original comparison PNGs.

    Retain only small uint8 predicted class maps for selecting matched errors.
    Artificial width padding is excluded from ALL Dice/IoU/error calculations.
    """
    intersections = torch.zeros(NUM_CLASSES, dtype=torch.int64)
    totals = torch.zeros(NUM_CLASSES, dtype=torch.int64)
    by_filename = {}
    saved_images = 0

    with torch.inference_mode():
        for batch in tqdm(test_loader, desc="Testing"):
            images = batch["image"].to(DEVICE)
            outputs = model(images)
            predictions = outputs.argmax(dim=1).to("cpu", dtype=torch.uint8)
            masks = batch["mask"]

            for index in range(len(predictions)):
                width = int(batch["original_width"][index])
                left = (predictions.shape[-1] - width) // 2
                right = left + width
                truth = masks[index, :, left:right]
                predicted = predictions[index, :, left:right]

                for cls in range(NUM_CLASSES):
                    pred_cls = predicted == cls
                    true_cls = truth == cls
                    intersections[cls] += (pred_cls & true_cls).sum()
                    totals[cls] += pred_cls.sum() + true_cls.sum()

                name = batch["filename"][index]
                pred_np = predicted.numpy().copy()
                truth_np = truth.numpy()
                by_filename[name] = {
                    "prediction": pred_np,
                    **prostate_slice_metrics(pred_np, truth_np),
                }

                # Preserve the original 'two initial images, then prostate slices' rule.
                if saved_images < MAX_VISUALISATIONS and (
                    saved_images < 2 or (truth == PROSTATE_CLASS).any()
                ):
                    image = batch["image"][index, 0, :, left:right].numpy()
                    filename = name.replace(".nii.gz", ".png")
                    save_comparison(
                        image, truth_np, pred_np, save_dir / filename, rotate=rotate
                    )
                    saved_images += 1

    print(f"Saved {saved_images} full-class comparison PNGs to {save_dir}")
    dice, iou = dice_and_iou(intersections, totals)
    return {"dice": dice, "iou": iou, "slices": by_filename}


def choose_failure_cases(baseline, improved, limit=5):
    """Select distinct prostate-positive error examples using fixed rankings.

    Selection is intentionally biased toward difficult slices; the cases are
    examples to inspect, NOT a representative sample or an anatomical diagnosis.
    """
    candidates = []
    for name, b in baseline["slices"].items():
        i = improved["slices"].get(name)
        if i is None:
            raise ValueError(f"Missing improved prediction for {name}")
        if b["gt_pixels"] == 0:
            continue  # Both models must be compared on a true prostate slice.
        if all(b[key] + i[key] == 0 for key in ("fp", "fn")):
            continue  # Exclude slices where both are already perfect.
        candidates.append((name, b, i))

    strategies = [
        ("Low Dice in both models", lambda b, i: -(b["dice"] + i["dice"]) / 2,
         lambda b, i: True),
        ("Baseline worse than improved", lambda b, i: i["dice"] - b["dice"],
         lambda b, i: i["dice"] > b["dice"]),
        ("Improved worse than baseline", lambda b, i: b["dice"] - i["dice"],
         lambda b, i: b["dice"] > i["dice"]),
        ("Most false positives", lambda b, i: max(b["fp"], i["fp"]),
         lambda b, i: max(b["fp"], i["fp"]) > 0),
        ("Most false negatives", lambda b, i: max(b["fn"], i["fn"]),
         lambda b, i: max(b["fn"], i["fn"]) > 0),
    ]

    selected = []
    used = set()
    for label, rank, eligible in strategies:
        if len(selected) >= limit:
            break
        available = [item for item in candidates
                     if item[0] not in used and eligible(item[1], item[2])]
        if not available:
            continue
        best = max(available, key=lambda item: rank(item[1], item[2]))
        selected.append((label, *best))
        used.add(best[0])

    # In small unusual datasets, fill any remaining slots with low-Dice slices.
    for name, b, i in sorted(candidates,
                             key=lambda item: item[1]["dice"] + item[2]["dice"]):
        if len(selected) >= limit:
            break
        if name not in used:
            selected.append(("Additional low-Dice case", name, b, i))
            used.add(name)
    return selected


def failure_plot(image, truth, baseline_pred, improved_pred, title,
                 filename, rotate=False):
    """Save six panels: four full-class views + two zoomed prostate error maps.

    The white contour in the baseline/improved panels shows TRUE prostate extent.
    Error panels use green TP, red FP, blue FN; all arrays stay aligned.
    """
    image, truth, baseline_pred, improved_pred = display_arrays(
        rotate, image, truth, baseline_pred, improved_pred
    )
    gt = truth == PROSTATE_CLASS
    bp = baseline_pred == PROSTATE_CLASS
    ip = improved_pred == PROSTATE_CLASS
    height, width = image.shape

    union = gt | bp | ip
    yy, xx = np.where(union)
    if len(xx):
        margin = 22
        xmin, xmax = max(0, int(xx.min()) - margin), min(width - 1, int(xx.max()) + margin)
        ymin, ymax = max(0, int(yy.min()) - margin), min(height - 1, int(yy.max()) + margin)
    else:
        xmin, xmax, ymin, ymax = 0, width - 1, 0, height - 1

    bmetrics = prostate_slice_metrics(baseline_pred, truth)
    imetrics = prostate_slice_metrics(improved_pred, truth)
    panels = [
        ("MRI (full slice)", None, None),
        ("Ground truth (classes 0–5)", truth, None),
        (f"Baseline prediction | Dice {bmetrics['dice']:.3f}", baseline_pred, None),
        (f"Improved prediction | Dice {imetrics['dice']:.3f}", improved_pred, None),
        (f"Baseline errors | FP {bmetrics['fp']}, FN {bmetrics['fn']}", None, bp),
        (f"Improved errors | FP {imetrics['fp']}, FN {imetrics['fn']}", None, ip),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    try:
        fig.suptitle(title, fontsize=13)
        for index, (ax, (subtitle, labels, prostate_prediction)) in enumerate(
            zip(axes.flat, panels)
        ):
            ax.imshow(image, cmap="gray")
            ax.set_title(subtitle, fontsize=10)
            ax.axis("off")

            if labels is not None:
                full_class_overlay(ax, labels)
                if index in (2, 3) and gt.any():
                    ax.contour(gt.astype(float), levels=[0.5],
                               colors="white", linewidths=0.8)

            if prostate_prediction is not None:
                tp = prostate_prediction & gt
                fp = prostate_prediction & ~gt
                fn = ~prostate_prediction & gt
                rgba = np.zeros((height, width, 4), dtype=np.float32)
                rgba[tp] = [0.15, 0.9, 0.2, 0.58]  # Green: correctly predicted.
                rgba[fp] = [1.0, 0.12, 0.12, 0.90]  # Red: false positives.
                rgba[fn] = [0.1, 0.5, 1.0, 0.90]   # Blue: false negatives.
                ax.imshow(rgba, interpolation="nearest")

            # Full images allow all-class inspection; zoom ONLY the two error maps.
            if index in (4, 5):
                ax.set_xlim(xmin - 0.5, xmax + 0.5)
                ax.set_ylim(ymax + 0.5, ymin - 0.5)

        fig.text(
            0.5, 0.02,
            "Classes 1–5 = coloured regions | White outline = true prostate | "
            "Green = correct, Red = false positive, Blue = false negative",
            ha="center", fontsize=9,
        )
        fig.tight_layout(rect=(0, 0.04, 1, 0.95))
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(filename, dpi=160)
    finally:
        plt.close(fig)


def print_model_metrics(model_name, result):
    """Print the original Dice report plus IoU in the console."""
    print(f"\n=== {model_name.upper()} — Test segmentation ===")
    print(f"{'Class':<19}{'Dice':>10}{'IoU':>12}")
    for cls, (dice, iou) in enumerate(zip(result["dice"], result["iou"])):
        label = f"Class {cls}" + (" prostate" if cls == PROSTATE_CLASS else "")
        print(f"{label:<19}{dice:>10.4f}{iou:>12.4f}")
    foreground = [d for d in result["dice"][1:] if not math.isnan(d)]
    if foreground:
        print(f"Mean foreground Dice: {np.mean(foreground):.4f}")
    print(f"Prostate Dice: {result['dice'][PROSTATE_CLASS]:.4f}")


def format_mib(value):
    """Format GPU memory in MiB or return N/A when CUDA is unavailable."""
    return f"{value:.1f}" if value is not None else "N/A (CPU)"


def main():
    """
    Evaluate both trained U-Nets on the same HipMRI test split.

    Prints Dice, IoU and computational profiling results, and
    generates multiclass predictions and representative failure plots.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=DATA_ROOT,
                        help="Path containing the six keras_slices_* folders")
    parser.add_argument("--runs-dir", default=str(RUNS_DIR),
                        help="Directory containing baseline/ and improved/ checkpoints")
    parser.add_argument("--rotate-display", action="store_true",
                        help="Rotate the PNG displays 90 degrees; metrics are unchanged")
    args = parser.parse_args()
    runs_dir = Path(args.runs_dir)

    print("Prediction device:", DEVICE)
    test_dataset = HipMRIDataset(args.data_root, split="test", augment=False)
    if not len(test_dataset):
        raise FileNotFoundError(f"No MRI test slices found in {test_dataset.image_dir}")
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print("Test slices:", len(test_dataset))

    # Identical sample and device for the two single-slice profiling runs.
    sample = test_dataset[0]["image"].unsqueeze(0).to(DEVICE)
    results, profiles = {}, {}

    for model_name, model_class in MODELS.items():
        model_path = runs_dir / model_name / "best_model.pt"
        save_dir = runs_dir / model_name / "predictions"
        if not model_path.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {model_path}")
        print(f"\nEvaluating {model_name}: {model_path}")
        model = model_class(
            in_channels=1,
            num_classes=NUM_CLASSES,
            base_channels=BASE_CHANNELS,
        ).to(DEVICE)
        model.load_state_dict(torch.load(
            model_path, map_location=DEVICE, weights_only=True
        ))
        model.eval()

        profiles[model_name] = profile_model(model, sample)
        results[model_name] = evaluate_model(
            model, test_loader, save_dir, rotate=args.rotate_display
        )
        print_model_metrics(model_name, results[model_name])

        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    print("\n=== Same-test-set quantitative comparison ===")
    print(f"{'Class':<16}{'Baseline Dice':>15}{'Improved Dice':>15}"
          f"{'Baseline IoU':>15}{'Improved IoU':>15}")
    for cls in range(NUM_CLASSES):
        b, i = results["baseline"], results["improved"]
        print(f"{('Class ' + str(cls)):<16}{b['dice'][cls]:>15.4f}"
              f"{i['dice'][cls]:>15.4f}{b['iou'][cls]:>15.4f}"
              f"{i['iou'][cls]:>15.4f}")

    print("\n=== Resource profiling (single-slice inference, same device) ===")
    print(f"{'Model':<12}{'Parameters':>15}{'Latency (ms)':>16}"
          f"{'Allocated (MiB)':>20}{'Reserved (MiB)':>20}")
    for name, profile in profiles.items():
        print(f"{name:<12}{profile['params']:>15,}"
              f"{profile['median_ms']:>16.3f}"
              f"{format_mib(profile['peak_allocated_mib']):>20}"
              f"{format_mib(profile['peak_reserved_mib']):>20}")
    print("Latency = median single-slice forward pass; data loading and saving excluded.")
    print("GPU memory = peak PyTorch allocated/reserved during inference, NOT training VRAM.")

    # Rank the SAME test slices using per-slice prostate Dice, FP, and FN.
    selected = choose_failure_cases(
        results["baseline"], results["improved"], FAILURE_CASES
    )
    output_dir = runs_dir / "analysis" / "failure_cases"
    output_dir.mkdir(parents=True, exist_ok=True)
    indices = {path.name: index for index, path in enumerate(test_dataset.files)}

    print("\n=== Suggested prostate segmentation failure cases ===")
    for case_no, (reason, name, b, i) in enumerate(selected, start=1):
        data = test_dataset[indices[name]]
        width = int(data["original_width"])
        left = (data["image"].shape[-1] - width) // 2
        right = left + width
        image = data["image"][0, :, left:right].numpy()
        truth = data["mask"][:, left:right].numpy()
        baseline_pred = b["prediction"]
        improved_pred = i["prediction"]

        png_name = f"case_{case_no}_{Path(name).name.replace('.nii.gz', '')}.png"
        png_path = output_dir / png_name
        failure_plot(
            image, truth, baseline_pred, improved_pred,
            f"Case {case_no}: {reason} | {name}", png_path,
            rotate=args.rotate_display,
        )
        print(f"Case {case_no} ({reason}): {name}")
        print(f"  Baseline: Dice {b['dice']:.4f}; FP {b['fp']}; FN {b['fn']}")
        print(f"  Improved: Dice {i['dice']:.4f}; FP {i['fp']}; FN {i['fn']}")
        print(f"  PNG: {png_path}")

    print(f"\nSaved {len(selected)} failure-case PNGs to {output_dir}")
    print("No CSV or numerical-results text files were generated.")


if __name__ == "__main__":
    main()
