"""Exercise the real train/validation loaders without fitting or evaluating a model.

Only the three small manifests are read for the final test partition: its images
are not decoded. Report is aggregate-only; the training-image preview is PRIVATE.
Run from a file, not an interactive notebook, for Windows DataLoader safety.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

from PIL import Image, ImageDraw, __version__ as pillow_version
import numpy as np
import torch

from dataset import (ADNISliceDataset, DataError, counts_for_records,
                     load_finalized_splits, make_loader, preprocessing_spec, require)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def inspect_loader(dataset: ADNISliceDataset, *, batch_size: int,
                   num_workers: int) -> dict[str, Any]:
    """Read every train/validation sample once and verify shapes/ranges/coverage."""
    require(dataset.split in ("train", "validation"), "This check must not access test images.")
    loader = make_loader(dataset, batch_size=batch_size, num_workers=num_workers)
    seen: set[str] = set()
    batches = 0
    minimum, maximum = float("inf"), float("-inf")
    expected = {row.image_path for row in dataset.records}
    last_batch_size = 0
    for batch in loader:
        images, labels = batch["image"], batch["label"]
        n = images.shape[0]
        require(images.shape == (n, 1, 224, 224) and 0 < n <= batch_size,
                "Unexpected image batch shape.")
        require(images.dtype == torch.float32 and labels.dtype == torch.int64
                and labels.shape == (n,), "Unexpected tensor dtype or label shape.")
        require(bool(torch.isfinite(images).all()), "A tensor contains non-finite values.")
        low, high = images.amin().item(), images.amax().item()
        require(low >= -1.000001 and high <= 1.000001, "Tensor values are outside [-1,1].")
        require(bool(((labels == 0) | (labels == 1)).all()), "A batch label is invalid.")
        require(set(batch["split"]) == {dataset.split}, "A batch mixes final splits.")
        for path in batch["image_path"]:
            require(path in expected and path not in seen, "Loader repeated or introduced an image.")
            seen.add(path)
        batches += 1
        last_batch_size = n
        minimum, maximum = min(minimum, low), max(maximum, high)
        if batches % 250 == 0:
            print(f"{dataset.split}: {len(seen)}/{len(dataset)} transformed samples checked", flush=True)
    require(seen == expected, "Loader omitted images.")
    require(batches == (len(dataset) + batch_size - 1) // batch_size,
            "Loader produced an unexpected number of batches.")
    print(f"{dataset.split}: {len(seen)}/{len(dataset)} transformed samples checked", flush=True)
    return {"images_checked": len(seen), "batches": batches,
            "last_batch_size": last_batch_size, "every_sample_once": True,
            "shape_per_image": [1, 224, 224], "image_dtype": "float32",
            "label_dtype": "int64", "value_min": minimum, "value_max": maximum,
            "all_finite": True}


def make_private_preview(dataset: ADNISliceDataset, output: Path) -> None:
    """Before/after pairs for up to four distinct TRAIN patients per class.

    Patient/scan IDs are not displayed, but these are still medical images and
    must remain private unless the applicable sharing conditions authorize them.
    """
    require(dataset.split == "train" and dataset.rotation_degrees == 0,
            "Preview must use unaugmented training images.")
    selected: dict[int, list[int]] = {0: [], 1: []}
    seen: set[str] = set()
    for index, row in enumerate(dataset.records):
        if row.subject_id not in seen and len(selected[row.label]) < 4:
            selected[row.label].append(index)
            seen.add(row.subject_id)
    rows = max(len(value) for value in selected.values())
    cell_w, cell_h, top = 280, 292, 50
    canvas = Image.new("RGB", (4 * cell_w, top + rows * cell_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 12), "PRIVATE: TRAINING IMAGES ONLY. Original -> model input; no augmentation.", fill="black")
    for label, selected_indices in selected.items():
        for r, index in enumerate(selected_indices):
            col = label * 2
            with Image.open(dataset.source_path(index)) as opened:
                original = opened.convert("RGB")
            tensor = dataset[index]["image"][0]
            values = ((tensor + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8).numpy()
            processed = Image.fromarray(values).convert("RGB")
            for offset, image in enumerate((original, processed)):
                x, y = (col + offset) * cell_w, top + r * cell_h
                caption = f"{'NC' if label == 0 else 'AD'} example {r + 1}: {'original' if offset == 0 else 'input'}"
                draw.text((x + 8, y + 6), caption, fill="black")
                canvas.paste(image, (x + (cell_w - image.width) // 2,
                                     y + 28 + (256 - image.height) // 2))
    canvas.save(output)


def report_markdown(report: dict[str, Any]) -> str:
    lines = ["## Data pipeline", "",
             "The finalized patient-independent manifests, rather than the original directory names, "
             "determine the training, validation and test partitions. The CSV hashes and grouping "
             "consistency were verified before loading images. Model labels are NC = 0 and AD = 1; "
             "the original metadata codes 0 and 2 are retained as provenance.", "",
             "| Partition | Patients | Scans | Images |",
             "|---|---:|---:|---:|"]
    for split, counts in report["split_counts"].items():
        lines.append(f"| {split} | {counts['candidate_subjects']} | {counts['scans']} | {counts['images']} |")
    lines += ["", "### Preprocessing and checks", "",
              "Each supplied 256-by-240 greyscale JPEG is resized to 224-by-210 using bilinear "
              "interpolation and padded with seven black pixels above and below. No anatomy is "
              "cropped and the original aspect ratio is preserved. The result is one channel of "
              "224-by-224 float32 values, scaled using (pixel / 255 - 0.5) / 0.5. This fixed scaling "
              "does not fit statistics to validation or test images. It does not recover physical "
              "MRI intensities from the supplied JPEGs.", "",
              "The initial configuration uses no random augmentation. A small training-only rotation "
              "is implemented but disabled; enabling it later is a separate, documented experiment. "
              "Both the CNN baseline and ConvNeXt will use the same selected preprocessing.", "",
              "A CPU-only pass successfully checked every final training and validation sample for "
              "shape, dtype, finite/range-correct values and exactly-once coverage, including the last "
              "partial batch. It did not train a network or decode the held-out test images. The "
              "test manifest was read only for split-integrity verification.", "",
              "Subject and scan identifiers are retained privately in each batch for later "
              "hierarchical aggregation of probabilities; slice counts must not be confused with "
              "independent patient counts. Patient-level evaluation and confidence-based referral "
              "will be implemented with the training/evaluation stages.", "",
              "A private before/after preview was generated for separate visual inspection. A local-byte versus "
              "Rangpur-source hash comparison was not performed. Model training, GPU memory and "
              "runtime measurements remain outstanding.", "",
              "### Artificial Intelligence Usage Disclosure — update", "",
              "ChatGPT assisted with drafting the data loader, verification script and synthetic "
              "tests. The code was checked by automated synthetic tests and a local CPU loader pass. "
              "I will document my own code review, understanding, visual inspection and any changes; "
              "these are not implied by the automated pass.", ""]
    return "\n".join(lines)


def run_check(root: Path, splits_dir: Path, output_dir: Path,
              *, batch_size: int = 16, num_workers: int = 0) -> dict[str, Any]:
    root, splits_dir, output_dir = root.resolve(), splits_dir.resolve(), output_dir.resolve()
    require(not output_dir.exists(), "Choose a new output directory; existing results are not overwritten.")
    require(output_dir != root and root not in output_dir.parents
            and output_dir != splits_dir and splits_dir not in output_dir.parents,
            "Output must be outside the dataset and finalized split directories.")
    require(batch_size > 0 and num_workers >= 0, "Invalid batch size or worker count.")
    output_dir.mkdir(parents=True)
    report: dict[str, Any] = {
        "format_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_pipeline_check_passed": False, "baseline_training_can_begin_after_visual_review": False,
        "model_training_performed": False, "model_training_validated": False,
        "final_test_images_decoded": 0, "final_test_model_evaluation_performed": False,
        "source_rangpur_copy_hash_verified": False, "splits_or_dataset_modified": False,
        "device_for_check": "cpu", "batch_size": batch_size, "num_workers": num_workers,
        "torch_version": torch.__version__, "pillow_version": pillow_version,
        "numpy_version": np.__version__, "preprocessing": preprocessing_spec(),
        "visual_review_completed": False, "errors": [],
    }
    report_path = output_dir / "data_pipeline_summary.json"
    try:
        bundle = load_finalized_splits(root, splits_dir)
        report.update(manifest_sha256=bundle.manifest_hashes,
                      manifest_hashes_and_grouping_checked=True,
                      split_counts={key: counts_for_records(value) for key, value in bundle.records.items()})
        train = ADNISliceDataset(bundle, "train")
        validation = ADNISliceDataset(bundle, "validation")
        report["train_loader"] = inspect_loader(train, batch_size=batch_size, num_workers=num_workers)
        report["validation_loader"] = inspect_loader(validation, batch_size=batch_size, num_workers=num_workers)
        require(torch.equal(validation[0]["image"], validation[0]["image"]),
                "Validation preprocessing changed between identical reads.")
        first_a = next(iter(make_loader(train, batch_size=batch_size, seed=42)))
        first_b = next(iter(make_loader(train, batch_size=batch_size, seed=42)))
        require(first_a["image_path"] == first_b["image_path"]
                and torch.equal(first_a["image"], first_b["image"]),
                "Same-seed training batches were not reproduced.")
        report["same_seed_first_batch_reproduced"] = True
        report["validation_repeat_reproduced"] = True
        make_private_preview(train, output_dir / "PRIVATE_training_preview.png")
        report["preview_created"] = True
        report["preview_final_partition"] = "train"
        report["source_code_sha256"] = {
            name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
            for name in ("dataset.py", "check_data_pipeline.py")
        }
        report["data_pipeline_check_passed"] = True
        report["baseline_training_can_begin_after_visual_review"] = True
        report["next_step"] = "Implement baseline CNN and training loop; no further splitting is required."
        (output_dir / "README_data_pipeline_section.md").write_text(report_markdown(report), encoding="utf-8")
        print("DATA_PIPELINE_CHECK_PASSED")
        print("REVIEW_PRIVATE_TRAINING_PREVIEW_THEN_PROCEED_TO_BASELINE")
    except Exception as exc:
        report["errors"] = [str(exc) if isinstance(exc, DataError) else
                            f"{type(exc).__name__}: local loader check stopped; inspect inputs privately."]
        raise
    finally:
        write_json(report_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    # A small number of CPU threads avoids thread overhead for tiny tensors.
    torch.set_num_threads(1)
    torch.manual_seed(42)
    try:
        run_check(args.data_root, args.splits_dir, args.output_dir,
                  batch_size=args.batch_size, num_workers=args.num_workers)
        return 0
    except Exception as exc:
        text = str(exc) if isinstance(exc, DataError) else type(exc).__name__
        print(f"STOP: {text}. Do not upload patient-linked manifests or images.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
