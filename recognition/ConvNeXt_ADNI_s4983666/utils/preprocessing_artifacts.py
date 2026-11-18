"""Export factual scan preprocessing parameters and original/processed QA panels."""

import os
from pathlib import Path
from typing import Any

from adni_data.preprocessing import apply_preprocessing, canonical_image, crop_box
from utils.artifacts import write_csv, write_json


def export_preprocessing_audit(output: Path, dataset: Any, prefix: str,
                               max_images: int = 6) -> dict[str, Any]:
    """Record every scan and show deterministically selected independent patients."""
    config = dataset.preprocessing
    if config.name == "none":
        return {"status": "legacy_preprocessing_no_scan_statistics"}
    if type(max_images) is not int or max_images < 1:
        raise ValueError("Preprocessing QA image count must be positive.")
    output = Path(output)
    scans = dataset.scan_parameters
    write_json(output / f"{prefix}_preprocessing_scans.json", {
        "preprocessing_config": config.to_dict(), "role": dataset.role,
        "parameters": scans, "source_images_modified": False,
        "foreground_is_brain_mask": False})
    records = []
    for scan in sorted(scans.values(), key=lambda s: s["image_id"]):
        box = crop_box(scan, config) if config.crops else None
        records.append({"image_id": scan["image_id"], "patient_id": scan["patient_id"],
                        "slices": len(scan["slice_indices"]), "native_width": scan["native_size"][0],
                        "native_height": scan["native_size"][1], "foreground_bbox": str(scan["bbox"]),
                        "crop_box": str(box), "foreground_pixels": scan["foreground_pixels"],
                        "intensity_low": scan["intensity_low"], "intensity_high": scan["intensity_high"],
                        "intensity_status": scan["intensity_status"],
                        "source_binding_sha256": scan["source_binding_sha256"]})
    write_csv(output / f"{prefix}_preprocessing_scans.csv", records)
    selected, seen = [], set()
    for row in sorted(dataset.rows, key=lambda r: (r["patient_id"], r["image_id"], int(r["slice_index"]))):
        if row["patient_id"] not in seen:
            selected.append(row)
            seen.add(row["patient_id"])
        if len(selected) >= max_images:
            break
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    figure, axes = plt.subplots(len(selected), 3, figsize=(12, 3 * len(selected)),
                               squeeze=False, layout="constrained")
    for index, row in enumerate(selected):
        source = canonical_image(row, dataset.data_root)
        scan = scans[row["image_id"]]
        processed = apply_preprocessing(source, scan, config, dataset.image_size)
        axes[index, 0].imshow(source, cmap="gray", vmin=0, vmax=255)
        for box, color, label in ((scan["bbox"], "lime", "Foreground proxy"),
                                  (crop_box(scan, config) if config.crops else None, "orange", "Fixed crop")):
            if box:
                axes[index, 0].add_patch(Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                                  fill=False, edgecolor=color, label=label))
        axes[index, 0].set_title(f"Original | scan {row['image_id']} | slice {row['slice_index']}")
        if axes[index, 0].get_legend_handles_labels()[0]:
            axes[index, 0].legend(fontsize=6)
        axes[index, 1].imshow(processed, cmap="gray", vmin=0, vmax=255)
        axes[index, 1].set_title(f"Model input | {processed.height}x{processed.width}")
        axes[index, 0].axis("off")
        axes[index, 1].axis("off")
        histogram = scan["histogram"]
        count = sum(histogram)
        axes[index, 2].plot(range(256), [v / max(count, 1) for v in histogram])
        axes[index, 2].axvline(scan["intensity_low"], color="orange", label="Scan lower percentile")
        axes[index, 2].axvline(scan["intensity_high"], color="red", label="Scan upper percentile")
        axes[index, 2].set(title="Whole-scan foreground histogram", xlabel="Original uint8 intensity",
                           ylabel="Fraction of foreground pixels", xlim=(0, 255))
        axes[index, 2].legend(fontsize=6)
    figure.savefig(output / f"{prefix}_preprocessing_preview.png", dpi=120)
    plt.close(figure)
    return {"status": "exported", "scans": len(scans), "preview_slices": len(selected),
            "empty_foreground_scans": sum(s["foreground_pixels"] == 0 for s in scans.values()),
            "degenerate_intensity_scans": sum(s["intensity_high"] <= s["intensity_low"] for s in scans.values()),
            "crop_foreground_clipping": "refused_before_model_scoring" if config.crops else "not_cropped",
            "parameters_scope": "full_supplied_scan_reused_for_failure_subsets"}
