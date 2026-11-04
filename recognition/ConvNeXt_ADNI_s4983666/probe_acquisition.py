"""Probe whether slices carry an acquisition signature that could stand in for the diagnosis.

In this copy of ADNI the acquisition period is entangled with the label: every
development scan acquired 2011-2016 is AD, and every NC scan was acquired
2017-2021. A diagnosis model could therefore partly learn "older acquisition
style = AD". This probe trains the same network on a *non-diagnostic* target
while holding the diagnosis fixed, using the same frozen folds:

* ``era_within_ad``: AD scans only; target = acquired 2017-2021 (1) vs 2011-2016 (0).
* ``vendor_within_nc``: NC scans only (all 2017-2021); target = GE IR-SPGR
  sequence (1) vs other (Siemens/Philips MPRAGE) sequences (0).

The acquisition year and sequence name are read from the scan's ``raw`` path
in ``meta_data_with_label.json``. For each fold the model trains on the fold's
training patients and is scored once on its validation patients; the
out-of-fold predictions of all folds are pooled. A target that is predicted
far better than the diagnosis itself (scan AUROC ~0.85) shows that the images
carry a strong acquisition signature. Calibration and test patients are never
loaded.
"""

import argparse
from collections import Counter
from pathlib import Path
import sys

import torch

from dataset import load_fold, make_loader
from adni_data.acquisition import SHARED_PERIOD_START, acquisition_info
from adni_data.augmentation import AUGMENTATION_NAMES, make_augmentation
from engine.ema import ModelEMA
from engine.scheduling import learning_rate
from engine.training import train_epoch
from evaluation.inference import evaluate, score_slices
from modules import MODEL_CHOICES, PRETRAINED_MODELS, create_model
from utils.artifacts import code_fingerprints, validate_output, write_csv, write_json
from utils.runtime import seed_everything, select_device
from utils.training_controls import ExecutionControls, make_criterion

PROBES = ("era_within_ad", "vendor_within_nc")


def probe_target(probe: str, row: dict, info: dict) -> int | None:
    """Return the probe label of one slice row, or None when the row is outside the probe."""
    scan = info.get(row["image_id"])
    if scan is None:
        return None
    if probe == "era_within_ad":
        return int(scan["year"] >= SHARED_PERIOD_START) if row["label"] == "1" else None
    return int("SPGR" in scan["sequence"].upper()) if row["label"] == "0" else None


def relabel(rows: list[dict], probe: str, info: dict) -> list[dict]:
    """Keep only rows inside the probe and replace the diagnosis label with the probe target."""
    kept = []
    for row in rows:
        target = probe_target(probe, row, info)
        if target is not None:
            kept.append({**row, "label": str(target)})
    return kept


def run(args: argparse.Namespace) -> dict:
    """Train and score the probe on every requested fold; pool out-of-fold predictions."""
    device = select_device(args.device)
    output = validate_output(args.output, args.data_root, args.splits_dir)
    info = acquisition_info(args.data_root)
    controls = ExecutionControls(context_slices=args.context_slices)
    augmentation = make_augmentation(args.augmentation)
    image_size = (args.image_height, args.image_width)
    torch.set_num_threads(args.threads)
    output.mkdir(parents=True, exist_ok=False)

    folds, pooled_slices, pooled_scans = [], [], []
    for fold in args.folds:
        data = load_fold(args.data_root, args.splits_dir, fold)
        expected_slices = int(data["report"]["config"]["expected_slices"])
        train_rows, val_rows = (relabel(data[role], args.probe, info) for role in ("train", "val"))
        counts = Counter(int(row["label"]) for row in train_rows)
        if min(counts[0], counts[1]) == 0:
            raise ValueError(f"Fold {fold}: the probe needs both targets in training.")
        seed = args.seed + fold
        seed_everything(seed)
        model = create_model(MODEL_CHOICES[args.model], input_channels=args.context_slices,
                             drop_path=None).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        ema = ModelEMA(model, args.ema_decay)
        criterion = make_criterion(controls, counts[0] / counts[1], device)
        loader_args = (args.data_root, image_size, args.batch_size, args.workers, seed)
        train_loader = make_loader(train_rows, *loader_args, shuffle=True, device=device, role="train",
                                   augmentation=augmentation, context_slices=args.context_slices)
        val_loader = make_loader(val_rows, *loader_args, shuffle=False, device=device, role="val",
                                 context_slices=args.context_slices)
        for epoch in range(1, args.epochs + 1):
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(epoch, args.epochs, args.lr, "warmup_cosine", args.warmup_epochs)
            train = train_epoch(model, train_loader, optimizer, criterion, device, controls, ema)
        # Final-epoch EMA weights, scored once on the fold's validation patients.
        scores, slices, scans = evaluate(ema.module, val_loader, device, expected_slices, controls)
        for row in slices + scans:
            row["fold"] = fold
        pooled_slices += slices
        pooled_scans += scans
        val_counts = Counter(row["label"] for row in scans)
        folds.append({"fold": fold, "train_scans_by_target": {str(k): v // expected_slices for k, v in counts.items()},
                      "val_scans_by_target": {str(k): v for k, v in sorted(val_counts.items())},
                      "final_train_accuracy": train["accuracy"],
                      "slice": scores["slice"], "scan": scores["scan"]})
        print(f"fold {fold}: train scans {dict(counts)} slices, val scans {dict(val_counts)}, "
              f"scan AUROC={scores['scan']['auroc']} slice AUROC={scores['slice']['auroc']}", flush=True)

    pooled, _ = score_slices(pooled_slices, expected_slices)
    result = {"status": "complete", "probe": args.probe, "folds": folds,
              "pooled_out_of_fold": {"slice": pooled["slice"], "scan": pooled["scan"]},
              "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
              "code_sha256": code_fingerprints()}
    write_csv(output / "scan_predictions.csv", pooled_scans)
    write_json(output / "metrics.json", result)
    print(f"pooled out-of-fold: scan AUROC={pooled['scan']['auroc']} "
          f"slice AUROC={pooled['slice']['auroc']}", flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=Path("/home/groups/comp3710/ADNI"))
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe", choices=PROBES, required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--model", choices=[k for k, v in MODEL_CHOICES.items() if v not in PRETRAINED_MODELS],
                        default="cnn")
    parser.add_argument("--context-slices", type=int, choices=(1, 3), default=3)
    parser.add_argument("--augmentation", choices=AUGMENTATION_NAMES, default="strong")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--image-height", type=int, default=240)
    parser.add_argument("--image-width", type=int, default=256)
    parser.add_argument("--seed", type=int, default=3710)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
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
