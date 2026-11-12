"""Example usage of a trained 2D U-Net: inference, per-volume Dice and overlays.

    python predict.py --checkpoint runs/unet2d_256_seed3710_e20/best.pt \
        --metadata /path/to/metadata.csv --split val --out-dir predictions/unet2d_256_val

``--split`` is required and has no default, so the held-out test split is
only used when requested explicitly (after model selection is frozen).

Volumes are loaded one at a time (not preloaded). For each volume this prints
per-class Dice; it then prints global Dice and per-volume mean/SD, writes
``metrics.json`` and saves image / ground-truth / prediction overlays for the
first ``--num-overlays`` volumes on the axial slice with the most foreground.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from dataset import (  # noqa: E402
    EXPECTED_PATIENTS_PER_SPLIT,
    EXPECTED_SPLIT_SHA256,
    NUM_CLASSES,
    SPLITS,
    load_manifest,
    load_volume_pair,
)
from metrics import (  # noqa: E402
    CLASS_NAMES,
    confusion_matrix,
    dice_from_confusion,
    global_metrics,
    per_volume_metrics,
)
from modules import build_model  # noqa: E402
from train import predict_volume, resolve_device  # noqa: E402

# Background transparent, then one colour per foreground class.
LABEL_COLOURS = ListedColormap(
    [
        (0, 0, 0, 0),
        (0.12, 0.47, 0.71, 0.5),
        (1.0, 0.5, 0.05, 0.25),
        (0.6, 0.6, 0.6, 0.5),
        (0.17, 0.63, 0.17, 0.5),
        (0.84, 0.15, 0.16, 0.6),
    ]
)


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
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def load_trained_model(path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = build_model(checkpoint["variant"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model.eval(), checkpoint


def save_overlay(
    path: Path, image: np.ndarray, mask: np.ndarray, prediction: np.ndarray, title: str
) -> None:
    z = int(np.argmax((mask > 0).sum(axis=(0, 1))))
    # Rotate so anterior is up in the RAS-canonical array.
    panels = [np.rot90(a[:, :, z]) for a in (image, mask, prediction)]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    for ax, label, overlay in zip(
        axes, ("MRI", "Ground truth", "Prediction"), (None, panels[1], panels[2]), strict=True
    ):
        ax.imshow(panels[0], cmap="gray")
        if overlay is not None:
            ax.imshow(
                overlay, cmap=LABEL_COLOURS, vmin=0, vmax=NUM_CLASSES - 1, interpolation="nearest"
            )
        ax.set_title(label)
        ax.axis("off")
    fig.suptitle(f"{title} (axial slice {z})")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run_prediction(
    args: argparse.Namespace,
    expected_patients: dict[str, int] | None = EXPECTED_PATIENTS_PER_SPLIT,
    expected_sha256: str | None = EXPECTED_SPLIT_SHA256,
) -> dict:
    device = resolve_device(args.device)
    model, checkpoint = load_trained_model(args.checkpoint, device)
    print(f"[model] {checkpoint['variant']} from {args.checkpoint} (epoch {checkpoint['epoch']})")

    frame = load_manifest(args.metadata, expected_patients, expected_sha256)
    rows = frame[frame["split"] == args.split]
    if args.max_volumes:
        rows = rows.head(args.max_volumes)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    matrices, per_scan = [], {}
    for i, row in enumerate(rows.itertuples(index=False)):
        image, mask = load_volume_pair(row.image_path, row.mask_path)
        prediction, _ = predict_volume(model, image, device, args.batch_size, args.amp)
        matrix = confusion_matrix(torch.from_numpy(mask), torch.from_numpy(prediction), NUM_CLASSES)
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
            save_overlay(
                args.out_dir / f"{row.scan_id}_overlay.png", image, mask, prediction, row.scan_id
            )

    results = {
        "checkpoint": str(args.checkpoint),
        "variant": checkpoint["variant"],
        "split": args.split,
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
    run_prediction(parse_args(argv))


if __name__ == "__main__":
    main()
