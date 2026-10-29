"""Load saved weights and visualize validation or test failure cases."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import VolumeDataset, load_split, read_manifest, verify_manifest_provenance
from metrics import case_metrics
from modules import ImprovedUNet3D, UNet2D
from train import choose_device, predict_volume


def load_model(path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    """Recreate the architecture recorded in a training checkpoint."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint["kind"] == "2d":
        model = UNet2D(width=checkpoint["width"])
    else:
        model = ImprovedUNet3D(width=checkpoint["width"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint


def require_test_results(run_dir: Path) -> None:
    """Keep validation-only tuning runs from opening the held-out test split."""
    report = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    if report.get("provenance", {}).get("validation_only"):
        raise ValueError("This is a validation-only run; test prediction is disabled")
    if any("test" not in report.get(kind, {}) for kind in ("2d", "3d")):
        raise ValueError("Both models need completed test results before prediction")


def _error_overlay(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Show false positives in red and false negatives in blue."""
    overlay = np.zeros((*truth.shape, 4), dtype=float)
    overlay[np.logical_and(prediction, ~truth)] = (1, 0, 0, 0.65)
    overlay[np.logical_and(~prediction, truth)] = (0, 0.4, 1, 0.65)
    return overlay


def select_failure_cases(evaluated: list[dict], count: int) -> list[dict]:
    """Choose low-Dice cases from different patients before repeating a patient."""
    ranked = sorted(
        evaluated, key=lambda case: case["improved_metrics"]["prostate_dice"]
    )
    selected = []
    used_patients = set()
    for case in ranked:
        patient_id = case["sample"]["patient_id"]
        if patient_id not in used_patients:
            selected.append(case)
            used_patients.add(patient_id)
        if len(selected) == count:
            return selected
    selected_ids = {case["sample"]["case_id"] for case in selected}
    selected.extend(
        case for case in ranked if case["sample"]["case_id"] not in selected_ids
    )
    return selected[:count]


def draw_case(
    sample: dict, pred_2d: np.ndarray, pred_3d: np.ndarray,
    path: Path, split_name: str,
) -> None:
    """Save apex, middle, and base slices for one held-out case."""
    image = sample["image"][0].numpy()
    truth = sample["mask"].numpy().astype(bool)
    prostate_slices = np.flatnonzero(truth.any(axis=(1, 2)))
    slices = [
        int(prostate_slices[0]),
        int(prostate_slices[len(prostate_slices) // 2]),
        int(prostate_slices[-1]),
    ]
    fig, axes = plt.subplots(3, 6, figsize=(19, 10))
    titles = [
        "MRI input", "Expert contour", "2D U-Net",
        "2D errors: FP red / FN blue", "3D Improved U-Net",
        "3D errors: FP red / FN blue",
    ]
    for row, slice_index in enumerate(slices):
        gray = image[slice_index]
        vmin, vmax = np.percentile(gray, (1, 99))
        for col, ax in enumerate(axes[row]):
            ax.imshow(gray, cmap="gray", vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(titles[col], fontsize=10)

        for col, mask in (
            (1, truth[slice_index]),
            (2, pred_2d[slice_index]),
            (4, pred_3d[slice_index]),
        ):
            if mask.any():
                axes[row, col].contour(mask, levels=[0.5], colors=["#00e5ff"], linewidths=1.5)

        axes[row, 3].imshow(_error_overlay(pred_2d[slice_index], truth[slice_index]))
        axes[row, 5].imshow(_error_overlay(pred_3d[slice_index], truth[slice_index]))
        axes[row, 0].set_ylabel(("apex", "mid", "base")[row] + f"  z={slice_index}")
    fig.suptitle(
        f"{split_name.title()} case {sample['case_id']} | patient {sample['patient_id']}"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.split == "test":
        require_test_results(args.run_dir)
    report = json.loads((args.run_dir / "results.json").read_text(encoding="utf-8"))
    verify_manifest_provenance(report["provenance"], args.manifest)
    default_dir = "failure_cases" if args.split == "test" else "validation_failure_cases"
    out = args.out or args.run_dir / default_dir
    out.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    baseline, base_config = load_model(args.run_dir / "2d_best.pt", device)
    improved, new_config = load_model(args.run_dir / "3d_best.pt", device)
    if (
        base_config["size"] != new_config["size"]
        or base_config["prostate_label"] != new_config["prostate_label"]
        or base_config.get("crop_xy") != new_config.get("crop_xy")
    ):
        raise ValueError("Models were trained with different preprocessing")
    split = load_split(read_manifest(args.manifest), args.run_dir / "split.json")
    dataset = VolumeDataset(
        split[args.split], tuple(new_config["size"]), new_config["prostate_label"],
        crop_xy=new_config.get("crop_xy"),
    )
    evaluated = []
    for sample in dataset:
        pred_2d = predict_volume(baseline, sample["image"], "2d", device)
        pred_3d = predict_volume(improved, sample["image"], "3d", device)
        truth = sample["mask"].numpy()
        evaluated.append({
            "sample": sample,
            "pred_2d": pred_2d,
            "pred_3d": pred_3d,
            "baseline_metrics": case_metrics(pred_2d, truth),
            "improved_metrics": case_metrics(pred_3d, truth),
        })
    rows = []
    for rank, case in enumerate(select_failure_cases(evaluated, args.count), 1):
        sample = case["sample"]
        baseline_metrics = case["baseline_metrics"]
        improved_metrics = case["improved_metrics"]
        name = f"{rank:02d}_{sample['case_id']}.png"
        draw_case(
            sample, case["pred_2d"], case["pred_3d"], out / name, args.split
        )
        rows.append({
            "rank": rank,
            "split": args.split,
            "case_id": sample["case_id"],
            "patient_id": sample["patient_id"],
            "figure": name,
            "baseline_dice": baseline_metrics["prostate_dice"],
            "improved_dice": improved_metrics["prostate_dice"],
            "improved_apex_dice": improved_metrics["apex_dice"],
            "improved_base_dice": improved_metrics["base_dice"],
            "false_positive_voxels": improved_metrics["false_positive_voxels"],
            "false_negative_voxels": improved_metrics["false_negative_voxels"],
        })
        print(
            f"{name}: 2D Dice={baseline_metrics['prostate_dice']:.3f}; "
            f"3D Dice={improved_metrics['prostate_dice']:.3f}"
        )
    with (out / "autopsy_index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
