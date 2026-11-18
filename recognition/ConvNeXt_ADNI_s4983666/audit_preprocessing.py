"""Review scan-standardized inputs on frozen training/early-stop scans without a model."""

import argparse
from pathlib import Path
import sys

from dataset import load_fold
from adni_data.preprocessing import (add_preprocessing_arguments, fit_training_crop,
                                   prepare_scans, preprocessing_from_args)
from adni_data.slices import ADNISliceDataset
from utils.artifacts import code_fingerprints, validate_output, write_json
from utils.preprocessing_artifacts import export_preprocessing_audit


def run(args: argparse.Namespace) -> dict:
    """Fit only crop geometry on training scans and export private QA for two inner roles."""
    preprocessing = preprocessing_from_args(args)
    if preprocessing.name == "none":
        raise ValueError("Choose an opt-in scan preprocessing profile for this audit.")
    if min(args.image_height, args.image_width, args.max_images) < 1:
        raise ValueError("Image dimensions and max-images must be positive.")
    output = validate_output(args.output, args.data_root, args.splits_dir)
    data = load_fold(args.data_root, args.splits_dir, args.fold)
    scans = prepare_scans(data["train"], args.data_root, preprocessing)
    preprocessing, crop_fit = fit_training_crop(scans, preprocessing, role="train")
    image_size = ((preprocessing.crop_height, preprocessing.crop_width) if preprocessing.crops
                  else (args.image_height, args.image_width))
    # Check train and val before publishing any preview; val reuses the train-fitted crop.
    datasets = {role: ADNISliceDataset(data[role], args.data_root, image_size, role=role,
                                      preprocessing=preprocessing,
                                      scan_parameters=scans if role == "train" else None)
                for role in ("train", "val")}
    output.mkdir(parents=True, exist_ok=False)
    result = {"status": "complete", "kind": "input_preprocessing_audit_no_model_no_accuracy",
              "preprocessing_config": preprocessing.to_dict(), "crop_fit": crop_fit,
              "image_size": list(image_size), "fold": args.fold,
              "manifest_sha256": data["manifest_sha256"], "code_sha256": code_fingerprints(),
              "limitations": ["Thresholded foreground is a head proxy, not brain extraction or anatomical registration.",
                              "Percentile clipping can change relevant contrast; inspect original/processed previews.",
                              "Passing synthetic checks does not establish real-data quality or classification gains.",
                              "No model is trained; outer validation/calibration/final test are not previewed or scored.",
                              "Mandatory original source integrity checks still audit all partitions."],
              "roles": {role: export_preprocessing_audit(output, dataset, role, args.max_images)
                        for role, dataset in datasets.items()}}
    write_json(output / "summary.json", result)
    print(f"Preprocessing review: {output}; resolved input size: {image_size}")
    return result


def main(argv: list[str] | None = None) -> int:
    """Run a standalone input audit against immutable manifests, with fresh outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--image-height", type=int, default=240)
    parser.add_argument("--image-width", type=int, default=256)
    parser.add_argument("--max-images", type=int, default=6)
    add_preprocessing_arguments(parser)
    parser.set_defaults(preprocessing="scan_intensity_crop")
    args = parser.parse_args(argv)
    try:
        run(args)
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
