"""Example usage of a trained 3D U-Net (standard or residual): inference, Dice and overlays.

    python predict3d.py --checkpoint runs/unet3d_256_p96_seed3710_e20/best.pt \
        --metadata /path/to/metadata.csv --split val --out-dir predictions/unet3d_256_val

``--split`` is required and has no default, so the held-out test split is
only used when requested explicitly (after model selection is frozen).

Each volume is predicted whole by the same sliding-window inference as
validation in train3d.py (patch size from the checkpoint, 50% overlap by
default) and scored with the same metrics. Outputs match predict.py:
per-scan Dice, global and per-volume Dice, ``metrics.json`` and overlays.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dataset import (
    EXPECTED_PATIENTS_PER_SPLIT,
    EXPECTED_SPLIT_SHA256,
    NUM_CLASSES,
    SPLITS,
    load_manifest,
    load_volume_pair,
    to_model_layout,
)
from metrics import (
    CLASS_NAMES,
    confusion_matrix,
    dice_from_confusion,
    global_metrics,
    per_volume_metrics,
)
from predict import save_overlay
from train import resolve_device
from train3d import build_variant, predict_volume3d


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=SPLITS)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--max-volumes", type=int, default=None, help="Limit the number of volumes")
    parser.add_argument("--num-overlays", type=int, default=3)
    parser.add_argument(
        "--eval-batch-size", type=int, default=4, help="Sliding windows per inference batch"
    )
    parser.add_argument("--overlap", type=float, default=0.5, help="Sliding-window overlap")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def load_trained_model3d(path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = build_variant(checkpoint["variant"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model.eval(), checkpoint


def run_prediction3d(
    args: argparse.Namespace,
    expected_patients: dict[str, int] | None = EXPECTED_PATIENTS_PER_SPLIT,
    expected_sha256: str | None = EXPECTED_SPLIT_SHA256,
) -> dict:
    device = resolve_device(args.device)
    model, checkpoint = load_trained_model3d(args.checkpoint, device)
    size = (checkpoint["patch_size"],) * 3
    print(
        f"[model] {checkpoint['variant']} from {args.checkpoint} (epoch {checkpoint['epoch']}, "
        f"{size[0]}^3 windows, overlap {args.overlap})"
    )

    frame = load_manifest(args.metadata, expected_patients, expected_sha256)
    rows = frame[frame["split"] == args.split]
    if args.max_volumes:
        rows = rows.head(args.max_volumes)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    matrices, per_scan = [], {}
    for i, row in enumerate(rows.itertuples(index=False)):
        image, mask = load_volume_pair(row.image_path, row.mask_path)
        logits = predict_volume3d(
            model, image, device, size, args.eval_batch_size, args.amp, args.overlap
        )
        prediction = logits.argmax(dim=0).to(torch.uint8).cpu()  # [D, H, W]
        del logits
        matrix = confusion_matrix(torch.from_numpy(to_model_layout(mask)), prediction, NUM_CLASSES)
        matrices.append(matrix)
        dice = dice_from_confusion(matrix)
        per_scan[row.scan_id] = {
            n: (None if np.isnan(d) else float(d)) for n, d in zip(CLASS_NAMES, dice, strict=True)
        }
        print(
            f"{row.scan_id}: "
            + "  ".join(f"{n}={d:.3f}" for n, d in zip(CLASS_NAMES[1:], dice[1:], strict=True))
        )
        if i < args.num_overlays:
            # Back to [H, W, D] for the shared overlay helper.
            prediction_hwd = prediction.numpy().transpose(1, 2, 0)
            save_overlay(
                args.out_dir / f"{row.scan_id}_overlay.png",
                image,
                mask,
                prediction_hwd,
                row.scan_id,
            )

    results = {
        "checkpoint": str(args.checkpoint),
        "variant": checkpoint["variant"],
        "split": args.split,
        "patch_size": checkpoint["patch_size"],
        "overlap": args.overlap,
        "global": global_metrics(torch.stack(matrices).sum(dim=0)),
        "per_volume": per_volume_metrics(matrices),
        "per_scan_dice": per_scan,
    }
    g, pv = results["global"], results["per_volume"]["mean_foreground_dice"]
    print(f"[{args.split}] global mean foreground Dice {g['mean_foreground_dice']:.4f}")
    print(
        f"[{args.split}] per-volume mean foreground Dice {pv['mean']:.4f} +/- {pv['sd'] or 0:.4f} (n={pv['n']})"
    )
    (args.out_dir / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    return results


def main(argv: list[str] | None = None) -> None:
    run_prediction3d(parse_args(argv))


if __name__ == "__main__":
    main()
