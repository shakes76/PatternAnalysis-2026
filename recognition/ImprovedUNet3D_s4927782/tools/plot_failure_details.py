"""Render additional zoom panels for the pair analysis's exact selected cases.

This presentation tool does not alter predictions, scores, failure-case ranking,
or evaluation protocols. It reuses the original evidence-loading contracts and
copies full-volume Dice/FP/FN into captions and an auditable figure index. The
original full-field gallery remains necessary context for these cropped views.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

# Support direct execution from any working directory and module execution.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_pair


CLASS_NAMES = ("Background", "Body", "Bones", "Bladder", "Rectum", "Prostate")
CLASS_COLOURS = ("#000000", "#777777", "#FFD54F", "#2D77D2", "#ED8C28", "#39A64A")
TARGET_CLASS = 5
CONTEXT_VOXELS = 8


def choose_slice(target, predictions):
    actual = target == TARGET_CLASS
    error_counts = [np.logical_xor(prediction == TARGET_CLASS, actual).sum(axis=(1, 2))
                    for prediction in predictions]
    if int(error_counts[1].max()) > 0:
        depth = int(error_counts[1].argmax())
        method = "maximum_3d_target_fp_fn"
        description = "Maximum 3D prostate FP+FN over full depth; lowest-index tie"
    elif int(error_counts[0].max()) > 0:
        depth = int(error_counts[0].argmax())
        method = "perfect_3d_fallback_maximum_2d_target_fp_fn"
        description = "3D prostate is perfect: fallback to full-depth maximum 2D FP+FN"
    else:
        positive = np.flatnonzero(actual.any(axis=(1, 2)))
        if positive.size:
            depth = int((positive[0] + positive[-1]) // 2)
            method = "both_perfect_fallback_gt_extent_midpoint"
            description = "Both prostate predictions are perfect: fallback to GT-extent midpoint"
        else:
            depth = target.shape[0] // 2
            method = "both_empty_fallback_grid_midpoint"
            description = "GT and both prostate predictions are empty: grid-midpoint fallback"
    return {
        "depth_index": depth, "method": method, "description": description,
        "tie_break": "lowest depth index for error maxima; floor for GT-extent midpoint",
        "selected_slice_2d_target_error_voxels": int(error_counts[0][depth]),
        "selected_slice_3d_target_error_voxels": int(error_counts[1][depth]),
        "full_depth_maximum_2d_target_error_voxels": int(error_counts[0].max()),
        "full_depth_maximum_3d_target_error_voxels": int(error_counts[1].max()),
    }


def choose_crop(target, predictions, depth):
    union = target[depth] == TARGET_CLASS
    for prediction in predictions:
        union = union | (prediction[depth] == TARGET_CLASS)
    height, width = union.shape
    coordinates = np.argwhere(union)
    if coordinates.size:
        lower = np.maximum(coordinates.min(axis=0) - CONTEXT_VOXELS, 0)
        upper = np.minimum(coordinates.max(axis=0) + CONTEXT_VOXELS + 1, [height, width])
        h0, w0 = map(int, lower)
        h1, w1 = map(int, upper)
        fallback = False
    else:
        h0, h1, w0, w1 = 0, height, 0, width
        fallback = True
    return {
        "h_bounds": [h0, h1], "w_bounds": [w0, w1], "bounds_convention": "half-open",
        "context_voxels": CONTEXT_VOXELS, "full_plane_shape_hw": [height, width],
        "empty_union_full_plane_fallback": fallback,
        "description": "Full-plane fallback: no GT or predicted prostate on this plane" if fallback
        else "GT and both model prostate masks on this plane, plus eight voxels of context, clamped to grid",
    }


def render_case(image, target, predictions, case, selection, crop, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    depth = selection["depth_index"]
    h0, h1 = crop["h_bounds"]
    w0, w1 = crop["w_bounds"]
    window = np.percentile(image[depth], [1, 99]).astype(float)
    if window[1] <= window[0]:
        window[1] = window[0] + 1.0
    mri = image[depth, h0:h1, w0:w1]
    actual = target[depth, h0:h1, w0:w1]
    predicted = [prediction[depth, h0:h1, w0:w1] for prediction in predictions]
    categorical_cmap = ListedColormap(CLASS_COLOURS)
    categorical_norm = BoundaryNorm(np.arange(-0.5, 6.5), 6)
    error_cmap = ListedColormap(["#FF0000", "#00FFFF"])
    error_norm = BoundaryNorm([0.5, 1.5, 2.5], 2)
    figure, axes = plt.subplots(2, 3, figsize=(11.7, 8.7), squeeze=False)
    axes = axes.ravel()
    titles = ("A  MRI", "B  Ground truth IDs", "C  2D prediction IDs", "D  3D prediction IDs",
              "E  2D prostate FP/FN", "F  3D prostate FP/FN")
    for axis, title in zip(axes, titles):
        axis.set_title(title, fontsize=12.5, fontweight="bold", pad=7)
        axis.axis("off")
    axes[0].imshow(mri, cmap="gray", vmin=window[0], vmax=window[1], interpolation="nearest")
    for axis, mask in zip(axes[1:4], [actual, *predicted]):
        axis.imshow(mask, cmap=categorical_cmap, norm=categorical_norm, interpolation="nearest")
    for axis, mask in zip(axes[4:], predicted):
        axis.imshow(mri, cmap="gray", vmin=window[0], vmax=window[1], interpolation="nearest")
        errors = np.zeros(actual.shape, dtype=np.uint8)
        errors[(mask == TARGET_CLASS) & (actual != TARGET_CLASS)] = 1
        errors[(mask != TARGET_CLASS) & (actual == TARGET_CLASS)] = 2
        axis.imshow(np.ma.masked_where(errors == 0, errors), cmap=error_cmap,
                    norm=error_norm, alpha=0.75, interpolation="nearest")

    def score(value):
        return "undefined" if value is None else f"{value:.4f}"

    baseline, model = case["baseline"], case["model"]
    figure.suptitle(
        f"{case['patient']} / {case['visit']} | D={depth} | H[{h0}:{h1}), W[{w0}:{w1})",
        fontsize=14, fontweight="bold", y=0.984,
    )
    figure.text(0.5, 0.947,
                f"Full-volume prostate Dice: 2D {score(baseline['dice'])}   |   3D {score(model['dice'])}",
                ha="center", fontsize=11.5)
    figure.text(0.5, 0.922,
                f"Full-volume prostate errors: 2D FP {baseline['fp']:,}, FN {baseline['fn']:,}"
                f"   |   3D FP {model['fp']:,}, FN {model['fn']:,}", ha="center", fontsize=11.5)
    figure.text(0.5, 0.897, selection["description"], ha="center", fontsize=10.5)
    handles = [Patch(facecolor=colour, edgecolor="#555555", label=f"{index} {name}")
               for index, (name, colour) in enumerate(zip(CLASS_NAMES, CLASS_COLOURS))]
    handles += [Patch(facecolor="#FF0000", label="Prostate false positive"),
                Patch(facecolor="#00FFFF", label="Prostate false negative")]
    figure.legend(handles=handles, loc="lower center", ncol=4, fontsize=10.5,
                  bbox_to_anchor=(0.5, 0.035), frameon=False)
    footer = "Additional zoom only; all six panels share one resampled-grid crop. See full-field gallery for full context."
    if crop["empty_union_full_plane_fallback"]:
        footer += " Empty prostate union: full-plane fallback."
    figure.text(0.5, 0.012, footer, ha="center", fontsize=8.5)
    figure.subplots_adjust(left=0.025, right=0.975, top=0.858, bottom=0.13, wspace=0.07, hspace=0.23)
    figure.savefig(destination, dpi=180, facecolor="white")
    plt.close(figure)
    return {"mri_window_percentiles": [1, 99], "mri_window_values": window.tolist()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "baseline-eval", "model-eval", "pair-analysis", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    root, manifest, records, evaluations = analyze_pair.load_evidence(
        args.manifest, args.baseline_eval, args.model_eval)
    analysis_path = args.pair_analysis.expanduser().resolve()
    analysis_bytes = analysis_path.read_bytes()
    pair = json.loads(analysis_bytes)
    manifest_hash = hashlib.sha256(args.manifest.expanduser().resolve().read_bytes()).hexdigest()
    if pair["manifest_sha256"] != manifest_hash or pair["split"] != evaluations[0][1]["split"]:
        raise ValueError("Pair analysis does not match this manifest and evaluation split.")
    if pair["target_class"] != TARGET_CLASS or manifest["label_ids"] != list(range(6)):
        raise ValueError("This plotting palette requires the documented HipMRI class IDs 0..5, prostate=5.")
    selected = pair["selected_failure_keys"]
    if any(not isinstance(key, list) or len(key) != 2 or not all(isinstance(value, str) for value in key)
           for key in selected):
        raise ValueError("Selected failure keys must be patient/visit string pairs.")
    keys = [tuple(key) for key in selected]
    case_map = analyze_pair._case_map(pair["cases"])
    if len(keys) != len(set(keys)) or any(key not in records or key not in case_map for key in keys):
        raise ValueError("Selected failure keys are duplicated or missing from the paired evidence.")
    output = args.output_dir.expanduser().resolve()
    if (output / "figure_index.json").exists() or any(output.glob("failure_zoom_*.png")):
        raise ValueError("Choose a fresh zoom output directory; existing figures are not overwritten.")
    output.mkdir(parents=True, exist_ok=True)
    index = {
        "schema_version": 1, "split": pair["split"], "target_class": TARGET_CLASS,
        "target_name": "Prostate", "manifest_sha256": manifest_hash,
        "pair_analysis": str(analysis_path), "pair_analysis_sha256": hashlib.sha256(analysis_bytes).hexdigest(),
        "selected_failure_keys": selected,
        "case_selection": "Exactly the existing pair analysis selected_failure_keys, in their existing order; no new ranking.",
        "slice_selection": "Full-depth maximum 3D prostate FP+FN, ties at lowest D. If 3D is perfect: maximum 2D error, then GT-extent midpoint, then grid midpoint if all masks are empty.",
        "crop_protocol": "Same H/W crop for all six panels: union of GT and both predictions' class 5 on the selected plane, plus eight voxels of context, clamped to grid. Empty union uses the full plane.",
        "metric_source": "Full-volume Dice, FP and FN copied from pair_analysis.json; cropped views do not replace full-volume measurements or the full-field gallery.",
        "palette": [{"class_index": i, "name": name, "colour": colour}
                    for i, (name, colour) in enumerate(zip(CLASS_NAMES, CLASS_COLOURS))],
        "figures": [],
    }
    for rank, key in enumerate(keys, 1):
        image, target, _, predictions = analyze_pair._read_case(root, records[key], evaluations, key, manifest)
        selection = choose_slice(target, predictions)
        crop = choose_crop(target, predictions, selection["depth_index"])
        name = f"failure_zoom_{rank:02d}.png"
        display = render_case(image, target, predictions, case_map[key], selection, crop, output / name)
        index["figures"].append({
            "rank": rank, "patient": key[0], "visit": key[1], "file": name,
            "full_grid_shape_dhw": list(target.shape), "selection": selection, "crop": crop, **display,
            "full_case_metrics": {model: {metric: case_map[key][model][metric] for metric in ("dice", "fp", "fn")}
                                  for model in ("baseline", "model")},
        })
    (output / "figure_index.json").write_text(json.dumps(index, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Rendered {len(keys)} additional failure zooms: {output / 'figure_index.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
