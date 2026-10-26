"""Whole-volume validation or explicitly authorized final-test evaluation.

Outputs are hard class IDs on the preprocessed manifest grid, not native-space
clinical masks. Validation is the default. Test requires --split test
--final-test and a checkpoint trained on a complete manifest.
"""

from __future__ import annotations

import argparse
import platform
from pathlib import Path

import numpy as np
import torch

from dataset import HipMRIVolumeDataset, load_manifest
from modules import ImprovedUNet3D, UNet2D
from utils import evaluate_volumes, file_sha256, plot_case, safe_case_name, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--final-test", action="store_true", help="Explicitly evaluate test after freezing all decisions using validation.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8, help="2D slices per batch; 3D evaluates one full volume at a time.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None,
                        help="Default uses the checkpoint's AMP setting on CUDA, full precision on CPU.")
    parser.add_argument("--target-class", type=int, default=5, help="Class index for FP/FN overlays; HipMRI class 5 is prostate.")
    parser.add_argument("--figure-count", type=int, default=5, help="Plot this many lowest-Dice cases; zero disables plots.")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.figure_count < 0:
        parser.error("Batch size must be positive and figure count nonnegative.")
    if args.split == "test" and not args.final_test:
        parser.error("Test evaluation requires --split test --final-test after validation decisions are frozen.")
    if args.final_test and args.split != "test":
        parser.error("--final-test is only valid with --split test.")
    manifest_path = args.manifest.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    manifest = load_manifest(manifest_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    if config.get("manifest_sha256") != file_sha256(manifest_path):
        raise ValueError("Manifest differs from the checkpoint; patient splits and preprocessing must remain fixed.")
    if config["label_ids"] != manifest["label_ids"] or config["num_classes"] != manifest["num_classes"]:
        raise ValueError("Checkpoint and manifest label mapping differ.")
    if config["target_shape_dhw"] != manifest["target_shape_dhw"]:
        raise ValueError("Checkpoint and manifest preprocessing grid differ.")
    development = bool(config.get("development_only") or manifest.get("partial") or manifest.get("development_only"))
    if args.split == "test" and development:
        raise ValueError("A development smoke/partial-data checkpoint cannot produce final test evidence.")
    if not 1 <= args.target_class < config["num_classes"]:
        parser.error("--target-class must identify a valid foreground class index.")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable.")
    amp = config["amp"] and device.type == "cuda" if args.amp is None else args.amp
    if amp and device.type != "cuda":
        raise ValueError("--amp requires CUDA; use --no-amp on CPU.")
    torch.manual_seed(config["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    model_class = UNet2D if config["model"] == "unet2d" else ImprovedUNet3D
    model = model_class(num_classes=config["num_classes"], base_channels=config["base_channels"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    dataset = HipMRIVolumeDataset(manifest_path, args.split, augment=False)
    if not len(dataset):
        raise ValueError(f"No volumes are available in the {args.split} split.")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "evaluation.json").exists():
        raise ValueError("This output already contains an evaluation; choose a new output directory.")
    prediction_dir = output_dir / "predictions"
    prediction_dir.mkdir(exist_ok=True)

    def save_prediction(item, prediction, record):
        name = safe_case_name(item["patient"], item["visit"]) + ".npz"
        storage_type = np.uint8 if config["num_classes"] <= 256 else np.uint16 if config["num_classes"] <= 65536 else np.int32
        np.savez_compressed(prediction_dir / name, prediction=prediction.numpy().astype(storage_type),
                            spacing_dhw=np.asarray(item["spacing"], dtype=np.float64),
                            label_ids=np.asarray(config["label_ids"], dtype=np.int64))
        record["prediction_file"] = f"predictions/{name}"

    evaluation = evaluate_volumes(model, dataset, config["model"], config["num_classes"], device,
                                  args.batch_size, amp, on_case=save_prediction)
    evaluation.update({
        "schema_version": 1, "split": args.split, "final_test_explicitly_requested": args.final_test,
        "development_only": development,
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": file_sha256(checkpoint_path),
        "checkpoint_epoch": checkpoint["epoch"], "checkpoint_best_epoch": checkpoint.get("best_epoch"),
        "manifest": str(manifest_path), "manifest_sha256": file_sha256(manifest_path),
        "model_config": config, "device": str(device), "amp": amp,
        "label_semantics": manifest.get("label_semantics", {"status": "unverified"}),
        "python_version": platform.python_version(), "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "metric_protocol": "Hard argmax labels on complete preprocessed volumes. Both-empty classes: null/excluded; absent-target false positives: zero. Background class 0 reported separately and excluded from foreground. Mean foreground per case, mean cases per patient, mean patients.",
        "geometry": "Predictions use the preprocessed manifest grid; native geometry and HD95 are not claimed.",
        "figure_selection": "Lowest foreground Dice cases, ties by patient/visit. Slices include target-extent endpoints and the largest target-class FP/FN area, including false positives outside the target extent. These are failure illustrations, not an independent score.",
    })
    # Save measurements first so an optional plotting failure cannot erase them.
    write_json(output_dir / "evaluation.json", evaluation)
    ranked = sorted(evaluation["cases"], key=lambda record: (
        record["foreground_mean_dice"] if record["foreground_mean_dice"] is not None else float("inf"),
        record["patient"], record["visit"]))[:args.figure_count]
    selected = {(record["patient"], record["visit"]): record for record in ranked}
    figures = []
    if selected:
        figure_dir = output_dir / "figures"
        figure_dir.mkdir(exist_ok=True)
        for index in range(len(dataset)):
            item = dataset[index]
            key = (str(item["patient"]), str(item["visit"]))
            if key not in selected:
                continue
            record = selected[key]
            with np.load(output_dir / record["prediction_file"], allow_pickle=False) as saved:
                prediction = saved["prediction"]
            name = safe_case_name(*key) + ".png"
            dice = record["foreground_mean_dice"]
            score = f"{dice:.3f}" if dice is not None else "undefined"
            plot_case(item["image"], item["mask"], prediction, figure_dir / name,
                      config["num_classes"], args.target_class,
                      title=f"{key[0]} / {key[1]} | {config['model']} | foreground Dice {score}",
                      class_names=[manifest.get("label_semantics", {}).get("class_index_to_name", {}).get(str(i), f"Class {i}") for i in range(config["num_classes"])])
            figures.append(f"figures/{name}")
    evaluation["figures"] = figures
    write_json(output_dir / "evaluation.json", evaluation)
    score = evaluation["aggregate"]["foreground_mean_dice"]
    score_text = f"{score:.4f}" if score is not None else "undefined"
    print(f"{args.split}: {len(dataset)} whole volumes, {evaluation['aggregate']['num_patients']} patients; patient-balanced foreground Dice={score_text}")
    print(f"Evidence: {output_dir / 'evaluation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
