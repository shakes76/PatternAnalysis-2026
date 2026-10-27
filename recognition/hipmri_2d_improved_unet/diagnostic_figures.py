"""Scientific ID5 error-mode figures from caller-supplied arrays only.

This module never reads MRI files, restores models, or changes class IDs. Error
colors are derived TP/FN/FP categories, not a replacement segmentation palette.
"""

import os
from pathlib import Path
import textwrap

import numpy as np


MODEL_LABELS = {
    "baseline_best": "Baseline best: epoch 5",
    "improved_best": "Improved best: epoch 3",
    "improved_last": "Improved last: epoch 20",
}
ERROR_COLORS = ("#24282f", "#35a66f", "#d63c9b", "#24c4dd")
ROI_MARGIN_PIXELS = 8


def _boolean_mask(values, shape, name):
    mask = np.asarray(values)
    if mask.shape != shape or mask.dtype != np.bool_:
        raise ValueError(f"{name} must be a boolean array with shape {shape}.")
    return mask


def _checked_statistics(reference, prediction, supplied, name):
    counts = {
        "tp": int(np.count_nonzero(reference & prediction)),
        "fn": int(np.count_nonzero(reference & ~prediction)),
        "fp": int(np.count_nonzero(~reference & prediction)),
    }
    for key, actual in counts.items():
        if key not in supplied or supplied[key] != actual:
            raise ValueError(f"{name}: supplied {key} does not match the full-image masks.")
    denominator = 2 * counts["tp"] + counts["fn"] + counts["fp"]
    expected = 2 * counts["tp"] / denominator if denominator else None
    supplied_dice = supplied.get("dice")
    if expected is None:
        if supplied_dice is not None:
            raise ValueError(f"{name}: both-empty Dice must be None.")
    elif supplied_dice is None or not np.isclose(supplied_dice, expected, rtol=0, atol=1e-9):
        raise ValueError(f"{name}: supplied Dice does not match the full-image masks.")
    return {**counts, "dice": expected}


def _error_map(reference, prediction):
    result = np.zeros(reference.shape, dtype=np.uint8)
    result[reference & prediction] = 1
    result[reference & ~prediction] = 2
    result[~reference & prediction] = 3
    return result


def _roi_bounds(reference, predictions):
    union = reference.copy()
    for prediction in predictions.values():
        union |= prediction
    height, width = reference.shape
    if not np.any(union):
        return 0, height, 0, width
    rows, columns = np.nonzero(union)
    return (
        max(0, int(rows.min()) - ROI_MARGIN_PIXELS),
        min(height, int(rows.max()) + 1 + ROI_MARGIN_PIXELS),
        max(0, int(columns.min()) - ROI_MARGIN_PIXELS),
        min(width, int(columns.max()) + 1 + ROI_MARGIN_PIXELS),
    )


def _draw_reference_contour(axis, reference, row_start, column_start):
    if not np.any(reference):
        axis.text(
            0.5, 0.04, "Reference ID5: empty", transform=axis.transAxes,
            ha="center", va="bottom", color="#ffe34d", fontsize=9,
            bbox={"facecolor": "black", "alpha": 0.65, "edgecolor": "none"},
        )
        return
    # A zero rim also draws a valid boundary for all-positive or one-pixel masks.
    padded = np.pad(reference.astype(np.uint8), 1, constant_values=0)
    rows = np.arange(row_start - 1, row_start + reference.shape[0] + 1)
    columns = np.arange(column_start - 1, column_start + reference.shape[1] + 1)
    axis.contour(columns, rows, padded, levels=[0.5], colors=["#ffe34d"], linewidths=1.1)


def save_failure_figure(output_path, image, reference_id5, predictions, metrics, metadata):
    """Save a full-slice and union-ROI ID5 diagnosis; return pixel ROI metadata.

    ``image`` is a normalized 2D NumPy MRI array. Reference and prediction masks
    are boolean original-size arrays. Scores/counts supplied for each model are
    verified against the full masks. The lower row is only a display crop; it
    never defines the metric support. PNG output is exclusively created.
    """
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite diagnostic image: {output_path}")
    if output_path.suffix.lower() != ".png":
        raise ValueError("Diagnostic figures use a .png output path.")
    image = np.asarray(image)
    if image.ndim != 2 or min(image.shape) <= 0 or not np.isfinite(image).all():
        raise ValueError("image must be a nonempty finite 2D array.")
    if image.min() < 0 or image.max() > 1:
        raise ValueError("image must already be normalized to [0,1].")
    reference = _boolean_mask(reference_id5, image.shape, "reference_id5")
    if set(predictions) != set(MODEL_LABELS) or set(metrics) != set(MODEL_LABELS):
        raise ValueError("Exactly baseline_best, improved_best and improved_last are required.")
    predictions = {
        key: _boolean_mask(predictions[key], image.shape, key) for key in MODEL_LABELS
    }
    scores = {
        key: _checked_statistics(reference, predictions[key], metrics[key], key)
        for key in MODEL_LABELS
    }
    for key in ("filename", "case_label", "selection_rule"):
        if key not in metadata:
            raise ValueError(f"metadata is missing {key}.")
    row0, row1, column0, column1 = _roi_bounds(reference, predictions)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_path.parent.resolve() / ".matplotlib"))
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle

    colors = ListedColormap(ERROR_COLORS)
    norm = BoundaryNorm((-0.5, 0.5, 1.5, 2.5, 3.5), colors.N)
    figure, axes = plt.subplots(2, 4, figsize=(15.8, 11.7), squeeze=False)
    # Reserve explicit space for a two-line title, two-row legend and footnotes.
    figure.subplots_adjust(left=0.05, right=0.985, top=0.86, bottom=0.205, hspace=0.34, wspace=0.28)
    figure.suptitle(
        f"{metadata['case_label']} | Validation: {metadata['filename']}\n"
        "ID5 error-mode diagnosis under the accepted label-preservation assumption",
        fontsize=13.5, y=0.975,
    )

    for row_index, bounds in enumerate(((0, image.shape[0], 0, image.shape[1]), (row0, row1, column0, column1))):
        start_row, stop_row, start_column, stop_column = bounds
        selection = np.s_[start_row:stop_row, start_column:stop_column]
        extent = (start_column - 0.5, stop_column - 0.5, stop_row - 0.5, start_row - 0.5)
        axis = axes[row_index, 0]
        axis.imshow(image[selection], cmap="gray", vmin=0, vmax=1,
                    interpolation="nearest", origin="upper", extent=extent)
        _draw_reference_contour(axis, reference[selection], start_row, start_column)
        if row_index == 0:
            axis.set_title(f"MRI + reference ID5 contour\nReference: {int(reference.sum())} pixels", fontsize=10.5)
            axis.add_patch(Rectangle(
                (column0 - 0.5, row0 - 0.5), column1 - column0, row1 - row0,
                fill=False, edgecolor="white", linestyle="--", linewidth=1.0,
            ))
        else:
            axis.set_title(
                f"Union zoom: rows {row0}:{row1}, cols {column0}:{column1}\n"
                "Stop coordinates are exclusive", fontsize=10.2,
            )
        for column_index, key in enumerate(MODEL_LABELS, start=1):
            axis = axes[row_index, column_index]
            axis.imshow(_error_map(reference, predictions[key])[selection],
                        cmap=colors, norm=norm, interpolation="nearest",
                        origin="upper", extent=extent)
            score = scores[key]
            dice_label = "NA" if score["dice"] is None else f"{score['dice']:.4f}"
            suffix = " (zoom)" if row_index else ""
            axis.set_title(
                f"{MODEL_LABELS[key]}{suffix}\n"
                f"Full-slice Dice {dice_label} | FN {score['fn']} | FP {score['fp']}",
                fontsize=10.2,
            )
        for axis in axes[row_index]:
            axis.set_xlim(start_column - 0.5, stop_column - 0.5)
            axis.set_ylim(stop_row - 0.5, start_row - 0.5)
            axis.set_xlabel("Column (pixels)", fontsize=9)
            axis.set_ylabel("Row (pixels)", fontsize=9)
            axis.tick_params(labelsize=8)
            axis.set_aspect("equal")

    handles = [
        Patch(facecolor=ERROR_COLORS[0], label="Neither reference nor predicted ID5"),
        Patch(facecolor=ERROR_COLORS[1], label="TP: correctly predicted ID5"),
        Patch(facecolor=ERROR_COLORS[2], label="FN: missed reference ID5"),
        Patch(facecolor=ERROR_COLORS[3], label="FP: ID5 outside reference"),
        Line2D([0], [0], color="#e2b900", linewidth=1.5, label="Yellow: reference ID5 contour"),
        Line2D([0], [0], color="#777777", linestyle="--", label="Zoom box (white on MRI)"),
    ]
    figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.133),
                  ncol=3, fontsize=9.2, frameon=False)
    notes = (
        "Error-map colors are TP/FN/FP categories, not raw class IDs. All scores/counts use the full slice; the bottom row is display-only.\n"
        "ID5 organ naming assumes derivative IDs preserve the provider mapping. Post-hoc validation error examples are not prevalence estimates; no test data.\n"
        "Axes and ROI bounds are pixels only; no millimetre or independent anatomical-alignment claim."
    )
    figure.text(0.05, 0.096, notes, fontsize=8.5, ha="left", va="top", linespacing=1.35)
    rule = textwrap.fill(f"Selection rule: {metadata['selection_rule']}", width=190)
    figure.text(0.05, 0.039, rule, fontsize=8.4, ha="left", va="top")
    try:
        # Exclusive creation also protects a file appearing after the first check.
        with output_path.open("xb") as handle:
            figure.savefig(handle, format="png", dpi=160, facecolor="white")
    finally:
        plt.close(figure)
    return {
        "roi_bounds_pixels": {
            "row_start": row0, "row_stop_exclusive": row1,
            "column_start": column0, "column_stop_exclusive": column1,
            "margin": ROI_MARGIN_PIXELS,
            "basis": "union of reference ID5 and all three predictions, clipped to image; empty union uses full image",
        },
        "image_filename": output_path.name,
    }
