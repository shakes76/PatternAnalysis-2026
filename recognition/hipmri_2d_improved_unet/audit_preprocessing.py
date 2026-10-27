"""Audit train/validation preprocessing contracts without training or saving tensors."""

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import time

import torch

from dataset import (
    DEFAULT_MULTIPLE,
    IGNORE_INDEX,
    collect_pairs,
    crop_to_original,
    load_pair,
    patient_ids,
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_sample(sample, multiple):
    image, target, valid = sample["image"], sample["target"], sample["valid_mask"]
    height, width = (int(value) for value in sample["original_size"])
    require(height > 0 and width > 0, "Original spatial dimensions must be positive.")
    require(image.ndim == 3 and image.shape[0] == 1, "Expected image shape [1,H,W].")
    require(target.ndim == 2 and valid.shape == target.shape, "Target/valid shapes differ.")
    require(image.shape[-2:] == target.shape, "Image/target shapes differ.")
    require(image.dtype == torch.float32, "Image must be float32.")
    require(target.dtype == torch.int64 and valid.dtype == torch.bool, "Invalid target/valid dtype.")
    require(image.device.type == target.device.type == valid.device.type == "cpu", "Expected CPU samples.")
    padded_height, padded_width = target.shape
    expected_size = (
        ((height + multiple - 1) // multiple) * multiple,
        ((width + multiple - 1) // multiple) * multiple,
    )
    require(tuple(target.shape) == expected_size, "Incorrect padding extent.")
    require(bool(torch.isfinite(image).all()), "Image contains non-finite values.")
    image_min, image_max = image.min().item(), image.max().item()
    require(0.0 <= image_min <= image_max <= 1.0, "Image range exceeds [0,1].")
    expected_valid = torch.zeros_like(valid)
    expected_valid[:height, :width] = True
    require(torch.equal(valid, expected_valid), "Valid region is not the original top-left rectangle.")
    require(int(valid.sum()) == height * width, "Valid pixel count differs from original area.")
    require(bool((target[valid] >= 0).all()), "Valid pixels contain negative class IDs.")
    require(bool((target[~valid] == IGNORE_INDEX).all()), "Padding target is not the ignore index.")
    require(bool((image[:, ~valid] == 0).all()), "Padding image is not zero.")
    cropped_image = crop_to_original(image, (height, width))
    cropped_target = crop_to_original(target, (height, width))
    require(tuple(cropped_image.shape) == (1, height, width), "Cropped image shape is wrong.")
    require(tuple(cropped_target.shape) == (height, width), "Cropped target shape is wrong.")
    require(torch.equal(cropped_image.reshape(-1), image[:, valid].reshape(-1)), "Image crop and valid selection disagree.")
    require(torch.equal(cropped_target.reshape(-1), target[valid]), "Crop and valid-region selection disagree.")
    return {
        "original_size": f"{height}x{width}",
        "class_ids": [int(value) for value in torch.unique(target[valid]).tolist()],
        "valid_pixels": height * width,
        "padding_pixels": padded_height * padded_width - height * width,
        "image_min": image_min,
        "image_max": image_max,
    }


def audit_split(name, pairs, multiple, limit, progress_every, report):
    selected = pairs if limit is None else pairs[:limit]
    summary = {
        "total_pairs": len(pairs),
        "case_group_count": len(patient_ids(pairs)),
        "selected_count": len(selected),
        "processed_count": 0,
        "passed_count": 0,
        "error_count": 0,
        "sampled": len(selected) < len(pairs),
        "all_pairs_attempted": False,
        "original_size_counts": {},
        "class_ids": [],
        "samples_with_padding": 0,
        "valid_pixels": 0,
        "padding_pixels": 0,
        "normalized_range": None,
        "errors": [],
    }
    report["splits"][name] = summary
    sizes, classes = Counter(), set()
    minimum, maximum = float("inf"), float("-inf")
    started = time.perf_counter()
    print(f"{name}: auditing {len(selected)} of {len(pairs)} filename pairs", flush=True)
    for image_path, mask_path in selected:
        try:
            sample = load_pair(image_path, mask_path, multiple=multiple)
            checked = check_sample(sample, multiple)
            sizes[checked["original_size"]] += 1
            classes.update(checked["class_ids"])
            summary["samples_with_padding"] += int(checked["padding_pixels"] > 0)
            summary["valid_pixels"] += checked["valid_pixels"]
            summary["padding_pixels"] += checked["padding_pixels"]
            minimum = min(minimum, checked["image_min"])
            maximum = max(maximum, checked["image_max"])
            summary["passed_count"] += 1
        except Exception as error:
            summary["error_count"] += 1
            summary["errors"].append({
                "image": str(image_path),
                "mask": str(mask_path),
                "error_type": type(error).__name__,
                "message": str(error),
            })
            print(f"ERROR {name}: {image_path.name}: {error}", flush=True)
        summary["processed_count"] += 1
        if summary["processed_count"] % progress_every == 0 or summary["processed_count"] == len(selected):
            print(
                f"{name}: {summary['processed_count']}/{len(selected)} selected; "
                f"passed={summary['passed_count']}, errors={summary['error_count']}",
                flush=True,
            )
    summary["all_pairs_attempted"] = summary["processed_count"] == len(pairs)
    summary["original_size_counts"] = dict(sorted(sizes.items()))
    summary["class_ids"] = sorted(classes)
    if summary["passed_count"]:
        summary["normalized_range"] = [minimum, maximum]
    summary["elapsed_seconds"] = round(time.perf_counter() - started, 3)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    workspace = Path(__file__).resolve().parents[4]
    parser.add_argument(
        "--data-root", type=Path,
        default=workspace / "Final Report/HipMRI_2D_Improved_UNet_学习/data/keras_slices_data",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/preprocessing_audit.json"))
    parser.add_argument("--multiple", type=int, default=DEFAULT_MULTIPLE)
    parser.add_argument("--limit", type=int, help="Maximum pairs per split; omit for the full audit.")
    parser.add_argument("--progress-every", type=int, default=1000)
    args = parser.parse_args()
    if args.multiple <= 0 or args.progress_every <= 0 or (args.limit is not None and args.limit <= 0):
        parser.error("multiple, progress-every and any limit must be positive")
    return args


def main():
    args = parse_args()
    torch.set_num_threads(1)
    report = {
        "audit_kind": "preprocessing_contract",
        "performed_by": "Codex automatic verification; not user-performed checks",
        "created_at": datetime.now(timezone(timedelta(hours=10))).isoformat(),
        "timezone": "Australia/Brisbane",
        "data_root": str(args.data_root.resolve()),
        "parameters": {"multiple": args.multiple, "ignore_index": IGNORE_INDEX, "limit_per_split": args.limit},
        "scope": ["training", "validation"],
        "test_split_read": False,
        "case_identifier_note": "Numeric case field is a grouping proxy; original identity mapping is not verified.",
        "claim_boundary": "This audit checks preprocessing contracts, not model performance, medical accuracy or complete leakage prevention.",
        "original_shape_note": "Uses load_pair original_size metadata; the auditor does not reread the original MRI or mask.",
        "shared_case_ids": None,
        "splits": {},
        "errors": [],
        "status": "running",
    }
    started = time.perf_counter()
    try:
        pairs = {
            "training": collect_pairs(args.data_root / "keras_slices_train", args.data_root / "keras_slices_seg_train"),
            "validation": collect_pairs(args.data_root / "keras_slices_validate", args.data_root / "keras_slices_seg_validate"),
        }
        report["shared_case_ids"] = sorted(patient_ids(pairs["training"]) & patient_ids(pairs["validation"]))
        if report["shared_case_ids"]:
            raise ValueError(f"Train/validation case IDs overlap: {report['shared_case_ids']}")
        with torch.inference_mode():
            for name, split_pairs in pairs.items():
                audit_split(name, split_pairs, args.multiple, args.limit, args.progress_every, report)
        failed = any(split["error_count"] for split in report["splits"].values())
        limited = any(split["sampled"] for split in report["splits"].values())
        report["status"] = "failed" if failed else ("passed_limited" if limited else "passed_full")
    except Exception as error:
        report["errors"].append({"error_type": type(error).__name__, "message": str(error)})
        report["status"] = "failed"
        print(f"ERROR audit setup: {error}", flush=True)
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        print("Audit interrupted; report contains only work completed so far.", flush=True)
    report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    report["error_count"] = len(report["errors"]) + sum(split["error_count"] for split in report["splits"].values())
    report["all_pairs_attempted"] = len(report["splits"]) == 2 and all(
        split["all_pairs_attempted"] for split in report["splits"].values()
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Audit status: {report['status']}; errors={report['error_count']}; report={args.output.resolve()}", flush=True)
    return 0 if report["status"] in ("passed_full", "passed_limited") else 1


if __name__ == "__main__":
    raise SystemExit(main())
