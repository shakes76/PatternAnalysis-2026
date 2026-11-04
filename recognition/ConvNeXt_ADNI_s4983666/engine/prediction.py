"""Load a trained checkpoint and run inference on one patient role.

Final evaluation is two separate steps on patients never used for training or
model comparison:

1. ``--role calibration`` fits Platt scaling and the referral threshold on the
   calibration patients and writes ``calibration.json``.
2. ``--role test --calibration-file .../calibration.json`` applies both,
   unchanged, to the test patients. Run this once, for the final model only.

``--role val`` reproduces the development result of a training run.

Several ``--checkpoint`` files form an ensemble whose slice logit is the mean
of the members' logits (e.g. the five fold models). An ensemble may only score
calibration or test patients: each development patient was a training patient
of some member. Calibration binds to the whole set of checkpoint hashes.

When the metadata file names give acquisition dates, metrics are also split by
acquisition period (docs/ACQUISITION_CONFOUND.md).
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import torch

from dataset import load_fold, load_holdout, make_loader
from adni_data.acquisition import SHARED_PERIOD_START, acquisition_info
from adni_data.preprocessing import checkpoint_preprocessing
from evaluation.calibration import (apply_platt, choose_referral_threshold, fit_platt,
                                    referral_decision)
from evaluation.inference import predict_slices, score_slices
from evaluation.metrics import binary_metrics
from evaluation.reporting import acquisition_breakdown, aggregate_patients, prediction_report
from evaluation.resources import profile_inference
from modules import MODEL_NAMES, PRETRAINED_MODELS, create_model
from utils.artifacts import validate_output, write_csv, write_json
from utils.evaluation_artifacts import export_evaluation_artifacts, plot_patient_examples
from utils.runtime import seed_everything, select_device
from utils.training_controls import ExecutionControls, validate_device

# Format 5 (early-stop-selected best.pt) and 6 (final-epoch final.pt) store the same model state.
CHECKPOINT_FORMATS = (5, 6)


def load_checkpoint(path: Path, device: torch.device) -> tuple[torch.nn.Module, dict, ExecutionControls, int]:
    """Rebuild the saved architecture and weights; only the cited ImageNet ConvNeXt may be pretrained."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    if config.get("checkpoint_format_version") not in CHECKPOINT_FORMATS:
        raise ValueError("Checkpoint was written by an older version of train.py; retrain it.")
    pretrained = config.get("pretrained_weights")
    if config["model_name"] not in MODEL_NAMES or (pretrained is not None) != (config["model_name"] in PRETRAINED_MODELS):
        raise ValueError("Unsupported architecture or undeclared pretrained weights.")
    controls = ExecutionControls.from_dict(config["execution_controls"])
    validate_device(controls, device)
    model = create_model(config["model_name"], input_channels=controls.context_slices,
                         drop_path=controls.drop_path).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model.eval(), config, controls, checkpoint["epoch"]


# Settings every ensemble member must share so that one data pipeline serves all of them.
SHARED_SETTINGS = ("image_size", "expected_slices", "manifest_sha256", "preprocessing_config")


def load_members(paths: list[Path], device: torch.device) -> list[tuple]:
    """Load every checkpoint and check that they share input pipeline and manifests."""
    members = [load_checkpoint(path, device) for path in paths]
    first_config, first_controls = members[0][1], members[0][2]
    for _, config, controls, _ in members[1:]:
        if controls.context_slices != first_controls.context_slices or any(
                config.get(key) != first_config.get(key) for key in SHARED_SETTINGS):
            raise ValueError("Ensemble members must share context slices, image size, preprocessing and manifests.")
    return members


def ensemble_slices(per_member: list[list[dict]]) -> list[dict]:
    """Average the members' slice logits (rows are in the same loader order)."""
    rows = []
    for member_rows in zip(*per_member):
        logit = sum(row["logit"] for row in member_rows) / len(member_rows)
        if any(row["relative_path"] != member_rows[0]["relative_path"] for row in member_rows):
            raise ValueError("Ensemble members saw slices in a different order.")
        rows.append({**member_rows[0], "logit": logit,
                     "probability": 1 / (1 + math.exp(-logit)) if logit >= 0 else math.exp(logit) / (1 + math.exp(logit))})
    return rows


def load_role_rows(args: argparse.Namespace, config: dict) -> tuple[list[dict], str]:
    """Rows of the requested role from the same frozen manifests used in training."""
    if args.role in ("calibration", "test"):
        data = load_holdout(args.data_root, args.splits_dir, args.role)
    else:
        data = load_fold(args.data_root, args.splits_dir, int(config["fold"]))
    if data["manifest_sha256"] != config["manifest_sha256"]:
        raise ValueError("Checkpoint was trained with a different set of frozen manifests.")
    return data[args.role], data["manifest_sha256"]


def run(args: argparse.Namespace) -> dict:
    """Predict one role; fit calibration on ``calibration`` or apply it elsewhere."""
    if args.role == "test" and args.calibration_file is None:
        raise ValueError("Test evaluation requires --calibration-file from the calibration step.")
    if args.role == "calibration" and args.calibration_file is not None:
        raise ValueError("The calibration step fits a new calibration; do not pass --calibration-file.")
    if not 0.5 < args.target_accuracy <= 1:
        raise ValueError("--target-accuracy must be in (0.5, 1].")
    if len(args.checkpoint) > 1 and args.role == "val":
        raise ValueError("An ensemble of fold models cannot score development patients (they trained on them).")
    output = validate_output(args.output, args.data_root, args.splits_dir)
    device = select_device(args.device)
    hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in args.checkpoint]
    # One checkpoint keeps its own hash; an ensemble is identified by the sorted member hashes.
    checkpoint_sha256 = hashes[0] if len(hashes) == 1 else hashlib.sha256(
        "".join(sorted(hashes)).encode()).hexdigest()
    members = load_members(args.checkpoint, device)
    model, config, controls, epoch = members[0]
    seed_everything(config["seed"])
    torch.set_num_threads(args.threads)

    calibration = None
    if args.calibration_file is not None:
        calibration = json.loads(args.calibration_file.read_text())
        if calibration["checkpoint_sha256"] != checkpoint_sha256:
            raise ValueError("calibration.json was fitted for a different checkpoint.")

    preprocessing = checkpoint_preprocessing(config)  # Validate before reading any patient data.
    rows, manifest_sha256 = load_role_rows(args, config)
    image_size = tuple(config["image_size"])
    loader = make_loader(rows, args.data_root, image_size, args.batch_size, args.workers, config["seed"],
                         False, device, role=args.role, preprocessing=preprocessing,
                         context_slices=controls.context_slices)
    predictions = [predict_slices(member[0], loader, device, member[2]) for member in members]
    raw_slices = ensemble_slices([rows for rows, _ in predictions]) if len(members) > 1 else predictions[0][0]
    forward_seconds = sum(seconds for _, seconds in predictions)

    if args.role == "calibration":
        slope, intercept = fit_platt([r["logit"] for r in raw_slices], [r["label"] for r in raw_slices])
        slices = apply_platt(raw_slices, slope, intercept)
        fitted_patients = aggregate_patients(slices)
        referral = choose_referral_threshold([p["label"] for p in fitted_patients],
                                             [p["probability"] for p in fitted_patients], args.target_accuracy)
        calibration = {"checkpoint_sha256": checkpoint_sha256, "fitted_on": "calibration",
                       "platt_slope": slope, "platt_intercept": intercept, "referral_unit": "patient", **referral}
    else:
        slope, intercept = (calibration["platt_slope"], calibration["platt_intercept"]) if calibration else (1.0, 0.0)
        slices = apply_platt(raw_slices, slope, intercept)

    threshold = calibration["threshold"] if calibration else None
    scores, scans = score_slices(slices, config["expected_slices"])
    raw_scores, _ = score_slices(raw_slices, config["expected_slices"])
    raw_patients = aggregate_patients(raw_slices)
    raw_scores["patient"] = binary_metrics([p["label"] for p in raw_patients], [p["probability"] for p in raw_patients])
    report, patients = prediction_report(slices, scans, reject_threshold=threshold)
    scores["patient"] = report["patient_metrics"]
    for patient in patients:
        patient["decision"] = referral_decision(patient["probability"], threshold) if calibration else (
            "AD" if patient["prediction"] else "NC")
    years = {image_id: scan["year"] for image_id, scan in acquisition_info(args.data_root).items()}
    by_period = acquisition_breakdown(scans, patients, years, SHARED_PERIOD_START)

    output.mkdir(parents=True, exist_ok=False)
    if args.role == "calibration":
        write_json(output / "calibration.json", calibration)
    write_csv(output / "slice_predictions.csv", slices)
    write_csv(output / "scan_predictions.csv", scans)
    failures = export_evaluation_artifacts(output, report, slices, patients, rows, args.data_root, image_size,
                                           args.role, preprocessing=preprocessing,
                                           scan_parameters=loader.dataset.scan_parameters)
    plot_patient_examples(output / f"{args.role}_examples.png", patients, rows, args.data_root, image_size,
                          preprocessing=preprocessing, scan_parameters=loader.dataset.scan_parameters)
    profile = profile_inference(model, image_size, device, controls, enabled=not args.skip_inference_profile)
    result = {
        "status": "complete", "role": args.role, "checkpoint_sha256": checkpoint_sha256,
        "checkpoints": [{"path": str(path), "sha256": digest, "fold": member[1]["fold"], "epoch": member[3]}
                        for path, digest, member in zip(args.checkpoint, hashes, members)],
        "checkpoint_epoch": epoch,
        "manifest_sha256": manifest_sha256, "model_name": config["model_name"],
        "ensemble_size": len(members), "by_acquisition_period": by_period,
        "calibration": calibration, "metrics": scores, "uncalibrated_metrics": raw_scores,
        "coursework_report": report, "failure_examples": failures, "inference_profile": profile,
        "forward_ms_per_slice": forward_seconds * 1000 / len(slices),
    }
    write_json(output / "metrics.json", result)
    print_summary(args.role, patients, scores, calibration, by_period)
    print(f"Results: {output / 'metrics.json'}")
    return result


def print_summary(role: str, patients: list[dict], scores: dict, calibration: dict | None,
                  by_period: dict | None = None) -> None:
    """Print one line per patient, then accuracy at every evaluation unit."""
    print(f"{'patient':<14}{'true':<6}{'p(AD)':>8}  decision")
    for patient in patients:
        truth = "AD" if patient["label"] else "NC"
        print(f"{patient['patient_id']:<14}{truth:<6}{patient['probability']:>8.3f}  {patient['decision']}")
    if calibration:
        print(f"Platt slope {calibration['platt_slope']:.3f}, intercept {calibration['platt_intercept']:.3f}; "
              f"referral threshold {calibration['threshold']} "
              f"(fitted on calibration patients, target accuracy {calibration['target_accuracy']}).")
        referred = sum(p["decision"] == "REFER" for p in patients)
        print(f"Referred to a specialist: {referred}/{len(patients)} patients.")
    for unit in ("slice", "scan", "patient"):
        print(f"{role} {unit}: accuracy={scores[unit]['accuracy']:.4f} AUROC={scores[unit]['auroc']} "
              f"(n={scores[unit]['n_samples']})")
    for period, entry in (by_period or {}).items():
        print(f"{role} scans acquired {period}: accuracy={entry['scans']['accuracy']:.4f} "
              f"AUROC={entry['scans']['auroc']} counts={entry['scan_counts']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True,
                        help="One checkpoint, or several to average their logits (calibration/test only).")
    parser.add_argument("--data-root", type=Path, default=Path("/home/groups/comp3710/ADNI"))
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--role", choices=("val", "calibration", "test"), required=True)
    parser.add_argument("--calibration-file", type=Path, default=None,
                        help="calibration.json from the calibration step (required for --role test).")
    parser.add_argument("--target-accuracy", type=float, default=0.9,
                        help="Calibration step only: accuracy required on automated (non-referred) patients.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--skip-inference-profile", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run(args)
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
