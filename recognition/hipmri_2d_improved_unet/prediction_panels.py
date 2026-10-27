"""Array-only class-ID prediction panels; never load data, weights or models.

The supplied arrays must already be cropped to original image support. Error
colors are TP/FN/FP display categories for a chosen class, not new mask IDs.
"""

from pathlib import Path
import os

import numpy as np

PALETTE = ("black", "gold", "deepskyblue", "magenta", "limegreen", "orange")
ERROR_COLORS = {"tp": "#35a66f", "fn": "#d63c9b", "fp": "#24c4dd"}


def overlap_statistics(reference, prediction, num_classes=6):
    """Whole-original-image counts; both-empty class Dice/IoU remain None."""
    reference = np.asarray(reference)
    prediction = np.asarray(prediction)
    if (reference.ndim != 2 or prediction.shape != reference.shape
            or reference.dtype.kind not in "iu" or prediction.dtype.kind not in "iu"):
        raise ValueError("Reference/prediction must be matching 2D integer class IDs.")
    if not isinstance(num_classes, int) or isinstance(num_classes, bool) or num_classes < 2:
        raise ValueError("num_classes must be an integer >=2.")
    if not reference.size or (reference < 0).any() or (prediction < 0).any():
        raise ValueError("Cropped original arrays must be nonempty and have no padding IDs.")
    if (reference >= num_classes).any() or (prediction >= num_classes).any():
        raise ValueError("Class IDs exceed num_classes.")
    result = []
    for class_id in range(num_classes):
        actual, predicted = reference == class_id, prediction == class_id
        tp = int(np.count_nonzero(actual & predicted))
        fp = int(np.count_nonzero(~actual & predicted))
        fn = int(np.count_nonzero(actual & ~predicted))
        denominator = 2 * tp + fp + fn
        union = tp + fp + fn
        result.append({"id": class_id, "tp": tp, "fp": fp, "fn": fn,
                       "reference_pixels": tp + fn, "predicted_pixels": tp + fp,
                       "dice": 2 * tp / denominator if denominator else None,
                       "iou": tp / union if union else None})
    return {"valid_pixels": int(reference.size), "per_class": result}


def _roi(reference, predictions, target_id, margin=8):
    union = reference == target_id
    for prediction in predictions.values():
        union |= prediction == target_id
    if not union.any():
        return (0, reference.shape[0], 0, reference.shape[1])
    rows, columns = np.nonzero(union)
    return (max(0, int(rows.min()) - margin),
            min(reference.shape[0], int(rows.max()) + 1 + margin),
            max(0, int(columns.min()) - margin),
            min(reference.shape[1], int(columns.max()) + 1 + margin))


def save_prediction_panels(output_path, image, reference, predictions, metadata,
                           num_classes=6, target_id=5, zoom=True):
    """Save independent GT/predictions plus target-class FP/FN overlays.

    predictions maps display labels (including model/epoch) to original-size
    integer arrays. Full-image counts are recomputed, never supplied by a plot
    crop. The optional second row uses one union-of-targets ROI for all models.
    """
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite prediction panels: {output_path}")
    if output_path.suffix.lower() != ".png":
        raise ValueError("Panel output must be PNG.")
    image, reference = np.asarray(image), np.asarray(reference)
    if (image.ndim != 2 or image.shape != reference.shape or not image.size
            or not np.isfinite(image).all() or image.min() < 0 or image.max() > 1):
        raise ValueError("Image must be finite normalized [0,1] and match the reference.")
    if not predictions or not all(isinstance(name, str) and name for name in predictions):
        raise ValueError("At least one named prediction is required.")
    if not isinstance(target_id, int) or not 0 <= target_id < num_classes:
        raise ValueError("target_id must be within num_classes.")
    if metadata.get("split") not in {"training", "validation", "test"} or not metadata.get("filename"):
        raise ValueError("metadata requires split and filename.")
    predictions = {name: np.asarray(values) for name, values in predictions.items()}
    scores = {name: overlap_statistics(reference, values, num_classes)
              for name, values in predictions.items()}
    bounds = _roi(reference, predictions, target_id)
    regions = [(0, image.shape[0], 0, image.shape[1])]
    if zoom:
        regions.append(bounds)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(output_path.parent / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch, Rectangle

    columns = 2 + 2 * len(predictions)
    fig, axes = plt.subplots(len(regions), columns, squeeze=False,
                             figsize=(3.0 * columns, 5.0 * len(regions) + 2.1))
    fig.subplots_adjust(left=0.035, right=0.985, top=0.87, bottom=0.18,
                        wspace=0.20, hspace=0.33)
    palette = ListedColormap(PALETTE[:num_classes]) if num_classes <= 6 else plt.get_cmap("tab20", num_classes)
    for row, (r0, r1, c0, c1) in enumerate(regions):
        region = np.s_[r0:r1, c0:c1]
        extent = (c0 - 0.5, c1 - 0.5, r1 - 0.5, r0 - 0.5)
        image_part = image[region]
        for axis in axes[row]:
            axis.set_xlabel("Column (pixels)", fontsize=9)
            axis.set_ylabel("Row (pixels)", fontsize=9)
            axis.tick_params(labelsize=8)
        axes[row, 0].imshow(image_part, cmap="gray", vmin=0, vmax=1,
                            origin="upper", extent=extent, interpolation="nearest")
        axes[row, 0].set_title("MRI input" + (" (same ROI)" if row else " (full slice)"), fontsize=10)
        if row == 0 and zoom:
            zr0, zr1, zc0, zc1 = bounds
            axes[row, 0].add_patch(Rectangle((zc0 - 0.5, zr0 - 0.5), zc1-zc0, zr1-zr0,
                                             fill=False, edgecolor="white", linestyle="--", linewidth=1))
        axes[row, 1].imshow(reference[region], cmap=palette, vmin=-0.5, vmax=num_classes-0.5,
                            origin="upper", extent=extent, interpolation="nearest")
        axes[row, 1].set_title(f"Independent GT: class IDs\nID{target_id} reference {int((reference == target_id).sum())} pixels", fontsize=10)
        for model_index, (name, prediction) in enumerate(predictions.items()):
            column = 2 + 2 * model_index
            score = scores[name]["per_class"][target_id]
            dice = "NA (both empty)" if score["dice"] is None else f"{score['dice']:.4f}"
            axes[row, column].imshow(prediction[region], cmap=palette, vmin=-0.5, vmax=num_classes-0.5,
                                     origin="upper", extent=extent, interpolation="nearest")
            axes[row, column].set_title(f"{name}\nIndependent prediction; ID{target_id} Dice {dice}", fontsize=10)
            axis = axes[row, column + 1]
            axis.imshow(image_part, cmap="gray", vmin=0, vmax=1, origin="upper", extent=extent,
                        interpolation="nearest")
            actual, predicted = reference[region] == target_id, prediction[region] == target_id
            rgba = np.zeros((*actual.shape, 4), dtype=np.float32)
            from matplotlib.colors import to_rgba
            for mask, category in ((actual & predicted, "tp"), (actual & ~predicted, "fn"), (~actual & predicted, "fp")):
                rgba[mask] = to_rgba(ERROR_COLORS[category], alpha=0.78)
            axis.imshow(rgba, origin="upper", extent=extent, interpolation="nearest")
            axis.set_title(f"{name}: ID{target_id} overlay\nTP {score['tp']} | FN {score['fn']} | FP {score['fp']}", fontsize=10)
    suffix = " | " + metadata["case_label"] if metadata.get("case_label") else ""
    fig.suptitle(f"{metadata['split']}: {metadata['filename']}{suffix}\n"
                 "Independent class-ID GT/predictions; full-slice statistics, common display ROI", fontsize=13, y=0.98)
    legend = [Patch(color=PALETTE[index], label=f"Raw ID{index}") for index in range(min(num_classes,6))]
    legend += [Patch(color=ERROR_COLORS[category], label=f"ID{target_id} {label}")
               for category, label in (("tp","TP"),("fn","FN"),("fp","FP"))]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5,0.115), ncol=min(9,len(legend)), fontsize=9)
    note = ("Raw class IDs are unchanged. Error colors are display categories only. "
            "ID5 organ naming assumes the course derivative preserves provider IDs.\n"
            "Axes and ROI are pixels; physical units and base/apex position are unverified. "
            "ROI = union of reference/predicted target pixels + 8 pixels; full image if union is empty.\n"
            + metadata.get("claim_boundary", "Single-sample display; not a new full-split evaluation."))
    fig.text(0.035,0.035,note,fontsize=9,va="bottom")
    try:
        with output_path.open("xb") as stream:
            fig.savefig(stream,format="png",dpi=150)
    finally:
        plt.close(fig)
    return {"original_shape":list(image.shape),"target_id":target_id,
            "roi_bounds":[int(value) for value in bounds],"roi_stop_exclusive":True,
            "statistics_support":"full original image, never the display crop",
            "scores":scores,"columns":columns,"rows":len(regions)}
