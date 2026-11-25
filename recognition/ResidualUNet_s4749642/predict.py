"""Load a model trained by train.py and run prostate segmentation predictions.

Example:
    python predict.py --model-name residual_unet
"""

import argparse
import re
import sys
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import time
import matplotlib

import os, shutil, subprocess
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.ndimage import binary_erosion, distance_transform_edt
from tqdm.auto import tqdm

from dataset import get_dataloaders
from modules import get_model

PROSTATE_LABEL = 5  # Same label as train.py; also stored in the checkpoint
NUM_CLASSES = 6
LABEL_NAMES = {
    0: "Outside body",
    1: "Remaining tissue",
    2: "Bone",
    3: "Bladder",
    4: "Rectum",
    5: "Prostate",
}

matplotlib.use("Agg")

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Load a model saved by train.py and use it to predict prostate\n"
            "segmentations on the HipMRI 2D slices.\n"
            "\n"
            "The model is loaded from <checkpoint-dir>/<model-name>.pt."
        ),
        epilog=(
            "examples:\n"
            "  python predict.py --model-name residual_unet\n"
            "  python predict.py --model-name baseline_unet --checkpoint-dir ./checkpoints"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "--model-name",
        type=str,
        required=True,
        help=(
            "Name of the model to load, as given to train.py --model-name,\n"
            "without extension. e.g. 'residual_unet' loads\n"
            "<checkpoint-dir>/residual_unet.pt."
        ),
    )

    paths = parser.add_argument_group("paths")
    paths.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("./checkpoints"),
        help=(
            "Directory containing models saved by train.py.\n"
            "(default: %(default)s)"
        ),
    )
    paths.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./data/keras_slices_data"),
        help=(
            "Directory containing the dataset split folders\n"
            "(keras_slices_test, keras_slices_seg_test, etc.).\n"
            "(default: %(default)s)"
        ),
    )
    paths.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help=(
            "Directory for prediction outputs.\n"
            "Created if it does not exist.\n"
            "(default: %(default)s)"
        ),
    )

    options = parser.add_argument_group("prediction options")
    options.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help=(
            "Number of slices per batch.\n"
            "(default: %(default)s)"
        ),
    )

    preview = parser.add_argument_group("preview options")
    preview.add_argument(
        "--patient",
        type=str,
        default=None,
        help=(
            "Test patient ID to preview, as three digits (040, 041, 042).\n"
            "If omitted, a random test patient is chosen.\n"
            "(default: random)"
        ),
    )
    preview.add_argument(
        "--week",
        type=int,
        default=None,
        help=(
            "Scan week to preview. Patient 040 has weeks 0-5 and 7;\n"
            "041 and 042 have week 0 only.\n"
            "If omitted, a random week for the patient is chosen.\n"
            "(default: random)"
        ),
    )
    preview.add_argument(
        "--num-slices",
        type=int,
        default=5,
        help=(
            "Number of slices containing prostate to show, evenly\n"
            "spaced through the scan.\n"
            "(default: %(default)s)"
        ),
    )
    preview.add_argument(
        "--compare-with",
        type=str,
        default=None,
        help=(
            "Name of another model whose predict.py results are in\n"
            "<output-dir>/<name>_test_slice_metrics.csv. Adds the slices\n"
            "with the largest prostate Dice gain and loss against it\n"
            "to the failure cases.\n"
            "(default: none)"
        ),
    )
    preview.add_argument(
        "--open",
        action="store_true",
        help=(
            "Open the saved preview and failure-case images with the\n"
            "system's default application (VS Code when run from its\n"
            "terminal).\n"
            "(default: off)"
        ),
    )

    return parser.parse_args()


def load_model(checkpoint_dir, model_name, device):
    """Rebuild a model from a checkpoint saved by train.py."""
    path = checkpoint_dir / f"{model_name}.pt"
    if not path.is_file():
        sys.exit(f"Model not found: {path}")

    # The checkpoint stores settings as well as tensors, so weights_only=False.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_type" not in checkpoint:
        sys.exit(f"{path} was not saved by train.py (no model_type found).")

    model = get_model(checkpoint["model_type"])
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval(), checkpoint


def build_test_loader(data_dir, batch_size, device):
    """Return the test DataLoader, built the same way as in train.py."""
    _, _, test_loader = get_dataloaders(data_dir, batch_size, device)
    return test_loader

def synchronise(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)

def select_preview_slices(dataset, patient_id=None, week=None, count=5):
    """Pick `count` evenly spaced slices containing prostate from one patient's scan."""
    slice_number = lambda i: int(re.search(r"_slice_(\d+)", dataset.pairs[i][0].name).group(1))

    scans = sorted({
        re.match(r"case_(\d+)_week_(\d+)_", image_path.name).groups()
        for image_path, _ in dataset.pairs
    })                 

    if patient_id is None or week is None:
        options = [(p, w) for p, w in scans
                if (patient_id is None or p == patient_id)
                and (week is None or w == str(week))]
        patient_id, week = options[np.random.randint(len(options))]

    prefix = f"case_{patient_id}_week_{week}_slice_"
    scan = sorted((i for i, (image_path, _) in enumerate(dataset.pairs)
                if image_path.name.startswith(prefix)), key=slice_number)

    if not scan:
        sys.exit(f"No test slices for patient {patient_id}, week {week}.")

    with_prostate = [i for i in scan if (dataset[i][1] == PROSTATE_LABEL).any()]
    candidates = with_prostate or scan          # fall back if no prostate is visible
    picks = np.linspace(0, len(candidates) - 1, num=min(count, len(candidates)), dtype=int)
    # Return the patient and week, which may have been chosen at random.
    return [candidates[p] for p in picks], patient_id, week

def save_preview(model, dataset, indices, output_dir, model_name, device, title):
    fig, axes = plt.subplots(len(indices), 4, figsize=(12, 3 * len(indices)), squeeze=False)
    colours = dict(cmap="tab10", vmin=0, vmax=9, interpolation="nearest")
    for row, i in enumerate(indices):
        image, mask = dataset[i]
        with torch.inference_mode():
            prediction = model(image.unsqueeze(0).to(device)).argmax(1)[0].cpu().numpy()
        image, mask = image[0].numpy(), mask.numpy()

        # Crop the -100 padding so only the real slice is shown.
        rows, cols = np.where(mask != -100)
        region = (slice(rows.min(), rows.max() + 1), slice(cols.min(), cols.max() + 1))
        image, mask, prediction = image[region], mask[region], prediction[region]

        slice_number = re.search(r"_slice_(\d+)", dataset.pairs[i][0].name).group(1)
        axes[row, 0].imshow(image, cmap="gray")
        axes[row, 1].imshow(mask, **colours)
        axes[row, 2].imshow(prediction, **colours)
        axes[row, 3].imshow(image, cmap="gray")
        axes[row, 3].imshow(np.ma.masked_where(prediction == 0, prediction), alpha=0.4, **colours)
        for ax, name in zip(axes[row], ["MRI", "Ground truth", "Prediction", "Prediction overlay"]):
            ax.set_title(f"Slice {slice_number}: {name}")
            ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{model_name}_preview.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path

def open_file(path):
    """Open a file with the system's default application, if one is available."""
    if sys.platform == "win32":
        os.startfile(path)
    elif sys.platform == "darwin":
        subprocess.run(["open", path], check=False)
    elif shutil.which("code"):                      # VS Code, including remote sessions
        subprocess.run(["code", path], check=False)
    elif shutil.which("xdg-open") and os.environ.get("DISPLAY"):
        subprocess.run(["xdg-open", path], check=False)
    else:
        print("No viewer found; open the file manually:", path)


def predict(model, test_loader, output_dir, device, checkpoint):
    """Run the model on the test data and save the predictions."""

    if device.type == "cuda":
        torch.cuda.empty_cache()
    
    # Load inputs and their ground-truth masks
    indices = np.linspace(0, len(test_loader.dataset) - 1,
                        num=min(100, len(test_loader.dataset)), dtype=int)
    samples = [test_loader.dataset[int(i)] for i in indices]
    inputs = [image.unsqueeze(0) for image, _ in samples]
    masks = [mask for _, mask in samples]

    predictions = []
    times_ms = []
    with torch.inference_mode():
        # Warmup the model, not timed
        warmup = inputs[0].to(device)
        for _ in range(10):
            output = model(warmup).argmax(dim=1)
        del output, warmup
        
        # Reset time and memory stats
        synchronise(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        # Iterate through each input and predict, saving prediction
        for cpu_image in inputs:
            image = cpu_image.to(device)
            synchronise(device)
            start = time.perf_counter()
            prediction = model(image).argmax(dim=1)
            synchronise(device)
            times_ms.append((time.perf_counter()-start)*1000)
            predictions.append(prediction.cpu())
            del prediction, image

    # Gather performance metrics
    results = {"Model": checkpoint['model_type'], "Checkpoint epoch": checkpoint['epoch'],
           "Validation prostate Dice": checkpoint['val_prostate_dice'],
           "Parameters": sum(p.numel() for p in model.parameters()),
           "Inference batch size": 1, "Slices measured": len(times_ms),
           "Mean inference time (ms/slice)": np.mean(times_ms),
           "Median inference time (ms/slice)": np.median(times_ms),
           "95th percentile time (ms/slice)": np.percentile(times_ms, 95)}
    if device.type == "cuda":
        results.update({"GPU": torch.cuda.get_device_name(device),
                        "Peak inference allocated VRAM (GiB)": torch.cuda.max_memory_allocated(device)/1024**3,
                        "Peak inference reserved VRAM (GiB)": torch.cuda.max_memory_reserved(device)/1024**3})
    
    profile = pd.Series(results, name="Resource profile").to_frame()
    print(profile)


def calculate_hd95(predicted, reference):
    """Symmetric 95th percentile Hausdorff distance between two binary masks, in pixels."""
    if not predicted.any() and not reference.any():
        return np.nan  # No structure present: boundary distance is undefined.

    if not predicted.any() or not reference.any():
        return np.inf  # A structure was completely missed or falsely predicted.

    predicted_boundary = predicted & ~binary_erosion(predicted, border_value=0)
    reference_boundary = reference & ~binary_erosion(reference, border_value=0)

    # Distance from every pixel to the closest boundary pixel.
    distance_to_reference = distance_transform_edt(~reference_boundary)
    distance_to_prediction = distance_transform_edt(~predicted_boundary)

    distances = np.concatenate([
        distance_to_reference[predicted_boundary],
        distance_to_prediction[reference_boundary],
    ])
    return float(np.percentile(distances, 95))


def evaluate_test_set(model, test_loader, device):
    """Return per-class Dice, IoU and HD95 over the whole test set, as in train_isensse,
    and the same metrics for every slice and label."""
    # Accumulate overlap counts across all test slices.
    intersections = np.zeros(NUM_CLASSES, dtype=np.int64)
    predicted_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    reference_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    boundary_distances = [[] for _ in range(NUM_CLASSES)]
    slice_records = []
    sample_index = 0

    with torch.inference_mode():
        for images, masks in tqdm(test_loader, desc="Test metrics", leave=False, mininterval=5):
            predictions = model(images.to(device)).argmax(dim=1).cpu().numpy()
            masks = masks.numpy()

            for prediction, reference in zip(predictions, masks):
                # The test loader is not shuffled, so slices follow dataset.pairs.
                image_path = test_loader.dataset.pairs[sample_index][0]
                sample_index += 1

                # Remove padding before extracting boundaries.
                valid = reference != -100
                rows, columns = np.where(valid)
                region = (
                    slice(rows.min(), rows.max() + 1),
                    slice(columns.min(), columns.max() + 1),
                )
                prediction = prediction[region]
                reference = reference[region]
                valid = valid[region]

                for label in range(NUM_CLASSES):
                    predicted = (prediction == label) & valid
                    actual = (reference == label) & valid

                    intersection = int((predicted & actual).sum())
                    predicted_count = int(predicted.sum())
                    reference_count = int(actual.sum())
                    hd95 = calculate_hd95(predicted, actual)

                    intersections[label] += intersection
                    predicted_counts[label] += predicted_count
                    reference_counts[label] += reference_count
                    boundary_distances[label].append(hd95)

                    denominator = predicted_count + reference_count
                    union = denominator - intersection
                    slice_records.append({
                        "file": image_path.name,
                        "patient": image_path.name.split("_")[1],
                        "label": label,
                        "structure": LABEL_NAMES[label],
                        "dice": 2 * intersection / denominator if denominator else np.nan,
                        "iou": intersection / union if union else np.nan,
                        "hd95_pixels": hd95,
                        "false_negative_pixels": int((actual & ~predicted).sum()),
                        "false_positive_pixels": int((predicted & ~actual).sum()),
                        "missed_structure": reference_count > 0 and predicted_count == 0,
                        "false_structure": reference_count == 0 and predicted_count > 0,
                        "both_empty": denominator == 0,
                    })

    results = []
    for label in range(NUM_CLASSES):
        intersection = intersections[label]
        denominator = predicted_counts[label] + reference_counts[label]
        union = denominator - intersection

        distances = np.asarray(boundary_distances[label])
        finite_distances = distances[np.isfinite(distances)]

        results.append({
            "Label": label,
            "Structure": LABEL_NAMES[label],
            "Dice": 2 * intersection / denominator if denominator else np.nan,
            "IoU": intersection / union if union else np.nan,
            "Mean HD95, finite slices (pixels)": (
                finite_distances.mean() if finite_distances.size else np.nan
            ),
            "Finite HD95 slices": int(np.isfinite(distances).sum()),
            "One mask empty (HD95 infinite)": int(np.isinf(distances).sum()),
            "Both masks empty (HD95 undefined)": int(np.isnan(distances).sum()),
        })

    return pd.DataFrame(results).set_index("Label"), pd.DataFrame(slice_records)


def select_failure_cases(slice_metrics, compare_metrics=None, compare_name=None):
    """Pick one prostate slice per category, as in train_isensse's failure candidates cell."""
    prostate = slice_metrics[slice_metrics["label"] == PROSTATE_LABEL]

    # These rankings suggest examples; visual inspection determines the failure.
    rankings = [
        ("Most missed tissue",
         prostate.sort_values("false_negative_pixels", ascending=False)),
        ("Most incorrectly included tissue",
         prostate.sort_values("false_positive_pixels", ascending=False)),
    ]

    # Dice gain and loss need a second model's per-slice results on the same slices.
    if compare_metrics is not None:
        comparison = prostate.merge(
            compare_metrics[compare_metrics["label"] == PROSTATE_LABEL],
            on=["file", "patient", "label"],
            suffixes=("", "_other"),
            validate="one_to_one",
        )
        comparison["dice_change"] = comparison["dice"] - comparison["dice_other"]
        changed = comparison.dropna(subset=["dice_change"])
        rankings += [
            (f"Largest Dice improvement over {compare_name}",
             changed.sort_values("dice_change", ascending=False)),
            (f"Largest Dice reduction against {compare_name}",
             changed.sort_values("dice_change")),
        ]

    # Select a different slice for each category.
    selected, used = [], set()
    for category, ranked in rankings:
        for filename in ranked["file"]:
            if filename not in used:
                selected.append((category, filename))
                used.add(filename)
                break
    return selected


def print_failure_cases(selected, slice_metrics):
    """Print the prostate metrics of each selected slice, as in train_isensse."""
    prostate = slice_metrics[slice_metrics["label"] == PROSTATE_LABEL].set_index("file")
    rows = []
    for category, filename in selected:
        result = prostate.loc[filename]
        rows.append({
            "Selection": category,
            "Slice": filename,
            "Dice": result["dice"],
            "IoU": result["iou"],
            "HD95 (pixels)": result["hd95_pixels"],
            "Missed pixels": result["false_negative_pixels"],
            "Extra pixels": result["false_positive_pixels"],
        })
    print("\nFailure-case candidates (prostate)")
    print(pd.DataFrame(rows).set_index("Selection").round(4).to_string())


def save_failure_cases(model, dataset, selected, output_dir, model_name, device):
    """Plot MRI, reference and predicted prostate, and an error overlay for each case."""
    index_by_name = {image_path.name: i for i, (image_path, _) in enumerate(dataset.pairs)}

    fig, axes = plt.subplots(len(selected), 4, figsize=(14, 3.8 * len(selected)), squeeze=False)
    for row, (category, filename) in enumerate(selected):
        image, mask = dataset[index_by_name[filename]]
        with torch.inference_mode():
            prediction = model(image.unsqueeze(0).to(device)).argmax(dim=1)[0].cpu().numpy()
        image, mask = image[0].numpy(), mask.numpy()

        # Crop away padding so it does not appear in the figures.
        rows, columns = np.where(mask != -100)
        region = (slice(rows.min(), rows.max() + 1), slice(columns.min(), columns.max() + 1))
        image, mask, prediction = image[region], mask[region], prediction[region]

        actual = mask == PROSTATE_LABEL
        predicted = prediction == PROSTATE_LABEL

        # Show prostate only, keeping the MRI visible underneath.
        axes[row, 0].imshow(image, cmap="gray")
        for column, binary_mask in [(1, actual), (2, predicted)]:
            axes[row, column].imshow(image, cmap="gray")
            overlay = np.zeros((*binary_mask.shape, 4))
            overlay[binary_mask] = [1, 0.65, 0, 0.55]
            axes[row, column].imshow(overlay)

        # Green: correct overlap; red: extra tissue; blue: missed tissue.
        error_overlay = np.zeros((*actual.shape, 4))
        error_overlay[actual & predicted] = [0, 1, 0, 0.45]
        error_overlay[predicted & ~actual] = [1, 0, 0, 0.7]
        error_overlay[actual & ~predicted] = [0, 0.45, 1, 0.7]
        axes[row, 3].imshow(image, cmap="gray")
        axes[row, 3].imshow(error_overlay)

        for ax, title in zip(axes[row], ["MRI", "Reference prostate", "Predicted prostate", "Error overlay"]):
            ax.set_title(title)
            ax.axis("off")
        # Identify the exact case and the reason it was selected.
        axes[row, 0].set_title(f"{category}\n{filename}", fontsize=9)

    legend = [
        Patch(color="green", label="Correct overlap"),
        Patch(color="red", label="False positive: extra tissue"),
        Patch(color="dodgerblue", label="False negative: missed tissue"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3)
    fig.suptitle(f"{model_name}: candidate test failure cases", fontsize=14)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{model_name}_failure_cases.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path

def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Dataset:", args.data_dir)

    test_loader = build_test_loader(args.data_dir, args.batch_size, device)
    print(f"Test loader: {len(test_loader.dataset)} slices, {len(test_loader)} batches")

    model, checkpoint = load_model(args.checkpoint_dir, args.model_name, device)
    print(
        f"Loaded {args.model_name} ({checkpoint['model_type']}, "
        f"epoch {checkpoint['epoch']}, "
        f"val prostate Dice {checkpoint['val_prostate_dice']:.4f})"
    )

    predict(model, test_loader, args.output_dir, device, checkpoint)

    metrics, slice_metrics = evaluate_test_set(model, test_loader, device)
    print("\nTest metrics")
    print(metrics.round(4).to_string())
    print("\nProstate:")
    print(metrics.loc[[PROSTATE_LABEL]].round(4).to_string())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / f"{args.model_name}_test_metrics.csv"
    metrics.to_csv(metrics_path)
    print("Saved test metrics to", metrics_path)
    # Save individual slices for failure analysis and comparison with other models.
    slice_metrics_path = args.output_dir / f"{args.model_name}_test_slice_metrics.csv"
    slice_metrics.to_csv(slice_metrics_path, index=False)
    print("Saved per-slice test metrics to", slice_metrics_path)

    compare_metrics = None
    if args.compare_with:
        compare_path = args.output_dir / f"{args.compare_with}_test_slice_metrics.csv"
        if not compare_path.is_file():
            sys.exit(f"Comparison results not found: {compare_path}\n"
                     f"Run predict.py --model-name {args.compare_with} first.")
        # Read patient IDs as strings to keep the leading zeros, e.g. "040".
        compare_metrics = pd.read_csv(compare_path, dtype={"patient": str})

    selected = select_failure_cases(slice_metrics, compare_metrics, args.compare_with)
    print_failure_cases(selected, slice_metrics)
    failure_path = save_failure_cases(
        model, test_loader.dataset, selected, args.output_dir, args.model_name, device
    )
    print("Saved failure cases to", failure_path)

    indices, patient_id, week = select_preview_slices(
        test_loader.dataset, args.patient, args.week, args.num_slices
    )
    print(f"Preview: patient {patient_id}, week {week}, {len(indices)} slices")
    path = save_preview(model, test_loader.dataset, indices, args.output_dir, args.model_name, device,
                        f"{checkpoint['model_type']} — patient {patient_id}, week {week}")
    print("Saved preview to", path)
    if args.open:
        # The images are already saved, so a viewer failure should not stop the run.
        try:
            open_file(path)
            open_file(failure_path)
        except OSError as error:
            print(f"Could not open the images ({error}); they are saved in {args.output_dir}")


if __name__ == "__main__":
    main()
