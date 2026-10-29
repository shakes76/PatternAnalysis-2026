"""Train 2D baseline and residual 3D model on identical patient splits."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset import (
    Case,
    SliceDataset,
    TrainingVolumeDataset,
    VolumeDataset,
    central_xy_slices,
    load_native_anatomy,
    load_split,
    manifest_sha256,
    read_manifest,
    save_split,
    split_by_patient,
)
from metrics import case_metrics
from modules import ImprovedUNet3D, UNet2D


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def memory_mb(device: torch.device) -> float | None:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 2**20
    if device.type == "mps":
        return torch.mps.current_allocated_memory() / 2**20
    return None


def reserved_memory_mb(device: torch.device) -> float | None:
    """Peak CUDA allocator reservation, including cached blocks."""
    if device.type == "cuda":
        return torch.cuda.max_memory_reserved() / 2**20
    return None


def segmentation_loss(
    logits: torch.Tensor, target: torch.Tensor, foreground_weight: float = 1.0
) -> torch.Tensor:
    class_weights = logits.new_tensor([1.0, foreground_weight])
    cross_entropy = F.cross_entropy(logits, target, weight=class_weights)
    probabilities = logits.softmax(dim=1)[:, 1]
    foreground = F.one_hot(target, num_classes=2).movedim(-1, 1)[:, 1].float()
    spatial_dims = tuple(range(1, foreground.ndim))
    intersection = (probabilities * foreground).sum(dim=spatial_dims)
    predicted_area = probabilities.sum(dim=spatial_dims)
    target_area = foreground.sum(dim=spatial_dims)
    soft_dice = (2 * intersection + 1) / (predicted_area + target_area + 1)
    return cross_entropy + 1 - soft_dice.mean()


def model_loss(
    output: torch.Tensor | dict, target: torch.Tensor,
    foreground_weight: float = 1.0,
) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return segmentation_loss(output, target, foreground_weight)
    loss = segmentation_loss(output["logits"], target, foreground_weight)
    for weight, auxiliary_logits in zip((0.25, 0.125), output["aux"]):
        scaled_target = F.interpolate(
            target[:, None].float(), size=auxiliary_logits.shape[2:], mode="nearest"
        )[:, 0].long()
        loss += weight * segmentation_loss(
            auxiliary_logits, scaled_target, foreground_weight
        )
    return loss


@torch.no_grad()
def predict_volume(model: torch.nn.Module, image: torch.Tensor, kind: str,
                   device: torch.device, slice_batch: int = 16) -> np.ndarray:
    """Run either model on one [1,D,H,W] image and return a [D,H,W] mask."""
    model.eval()
    if kind == "2d":
        image_slices = image[0].unsqueeze(1)  # [D,1,H,W]
        predicted_slices = []
        for batch in image_slices.split(slice_batch):
            predicted_slices.append(model(batch.to(device)).argmax(dim=1).cpu())
        return torch.cat(predicted_slices).numpy().astype(bool)

    logits = model(image[None].to(device))["logits"]
    return logits.argmax(dim=1)[0].cpu().numpy().astype(bool)


def _mean_case_metrics(rows: list[dict]) -> dict:
    """Report both scan-weighted and patient-balanced metric averages."""
    summary = {}
    metric_names = set(rows[0]) - {"case_id", "patient_id"}
    patient_ids = sorted({row["patient_id"] for row in rows})
    for name in sorted(metric_names):
        values = [row[name] for row in rows if row[name] is not None]
        summary[name] = float(np.mean(values)) if values else None
        patient_values = []
        for patient_id in patient_ids:
            visits = [
                row[name] for row in rows
                if row["patient_id"] == patient_id and row[name] is not None
            ]
            if visits:
                patient_values.append(float(np.mean(visits)))
        summary[f"patient_mean_{name}"] = (
            float(np.mean(patient_values)) if patient_values else None
        )
    summary["cases"] = len(rows)
    summary["patients"] = len(patient_ids)
    return summary


def native_case_metrics(
    prediction: np.ndarray, case: Case, prostate_label: int,
    bladder_label: int | None, rectum_label: int | None,
    crop_xy: int | None = None,
) -> dict:
    """Restore the full native grid, pasting any cropped prediction in place."""
    anatomy, spacing_dhw = load_native_anatomy(case)
    x_range, y_range = central_xy_slices(
        (anatomy.shape[2], anatomy.shape[1]), crop_xy
    )
    crop_shape = (
        anatomy.shape[0], y_range.stop - y_range.start,
        x_range.stop - x_range.start,
    )
    prediction_tensor = torch.from_numpy(prediction.astype(np.float32))[None, None]
    restored_crop = F.interpolate(
        prediction_tensor, size=crop_shape, mode="nearest"
    )[0, 0].numpy().astype(bool)
    native_prediction = np.zeros_like(anatomy, dtype=bool)
    native_prediction[:, y_range, x_range] = restored_crop
    native_scores = case_metrics(
        native_prediction, anatomy == prostate_label, anatomy,
        bladder_label, rectum_label, spacing_dhw,
    )
    return {f"native_{name}": value for name, value in native_scores.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module, dataset: VolumeDataset, kind: str,
             device: torch.device, bladder_label: int | None = None,
             rectum_label: int | None = None,
             native: bool = False) -> tuple[dict, list[dict]]:
    rows = []
    for index, sample in enumerate(dataset):
        sync(device)
        started = time.perf_counter()
        pred = predict_volume(model, sample["image"], kind, device)
        sync(device)
        elapsed_ms = (time.perf_counter() - started) * 1000
        metrics = case_metrics(
            pred,
            sample["mask"].numpy(),
            sample["anatomy"].numpy(),
            bladder_label,
            rectum_label,
        )
        row = {
            "case_id": sample["case_id"],
            "patient_id": sample["patient_id"],
            "latency_ms": elapsed_ms,
            **metrics,
        }
        if native:
            row.update(native_case_metrics(
                pred, dataset.cases[index], dataset.prostate_label,
                bladder_label, rectum_label, dataset.crop_xy,
            ))
        rows.append(row)
    return _mean_case_metrics(rows), rows


def _training_loader(
    kind: str, volumes: VolumeDataset, args: argparse.Namespace
) -> DataLoader:
    """Use balanced slice sampling for 2D and whole volumes for 3D."""
    if kind == "2d":
        slices = SliceDataset(volumes, augment=args.augment)
        sampler = WeightedRandomSampler(
            slices.weights, num_samples=len(slices), replacement=True
        )
        return DataLoader(slices, batch_size=args.batch_2d, sampler=sampler)
    training_volumes = TrainingVolumeDataset(volumes) if args.augment else volumes
    return DataLoader(training_volumes, batch_size=args.batch_3d, shuffle=True)


def _train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    foreground_weight: float,
) -> float:
    model.train()
    losses = []
    for batch in loader:
        image = batch["image"].to(device)
        mask = batch["mask"].to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = model_loss(model(image), mask, foreground_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def _plot_history(history: list[dict], path: Path) -> None:
    epochs = [record["epoch"] for record in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(epochs, [record["train_loss"] for record in history])
    axes[0].set(xlabel="Epoch", ylabel="CE + Dice loss", title="Training loss")

    for label, key in (
        ("whole volume", "val_prostate_dice"),
        ("patient-balanced", "val_patient_mean_prostate_dice"),
        ("apex", "val_apex_dice"),
        ("base", "val_base_dice"),
    ):
        axes[1].plot(epochs, [record[key] for record in history], label=label)
    axes[1].set(xlabel="Epoch", ylabel="Dice", ylim=(0, 1), title="Validation")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def train_one(
    kind: str,
    train_volumes: VolumeDataset,
    val_volumes: VolumeDataset,
    device: torch.device,
    out: Path,
    args: argparse.Namespace,
) -> tuple[torch.nn.Module, dict]:
    """Fit one model and restore its best validation checkpoint."""
    width = args.width_2d if kind == "2d" else args.width_3d
    model = UNet2D(width=width) if kind == "2d" else ImprovedUNet3D(width=width)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loader = _training_loader(kind, train_volumes, args)

    history = []
    best_score = -1.0
    stale_epochs = 0
    checkpoint = out / f"{kind}_best.pt"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    training_started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_loss = _train_epoch(
            model, loader, optimizer, device, args.foreground_weight
        )
        validation, _ = evaluate(
            model, val_volumes, kind, device, args.bladder_label, args.rectum_label
        )
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_prostate_dice": validation["prostate_dice"],
            "val_patient_mean_prostate_dice": validation["patient_mean_prostate_dice"],
            "val_apex_dice": validation["apex_dice"],
            "val_base_dice": validation["base_dice"],
        }
        history.append(record)
        print(json.dumps({"model": kind, **record}), flush=True)

        selection_score = validation["patient_mean_prostate_dice"]
        if selection_score > best_score + 1e-5:
            best_score = selection_score
            stale_epochs = 0
            torch.save({
                "state_dict": model.state_dict(),
                "kind": kind,
                "width": width,
                "size": args.size,
                "prostate_label": args.prostate_label,
                "crop_xy": args.crop_xy,
                "best_epoch": epoch,
            }, checkpoint)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    best_checkpoint = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(best_checkpoint["state_dict"])
    (out / f"{kind}_history.json").write_text(json.dumps(history, indent=2) + "\n")
    _plot_history(history, out / f"{kind}_training_curves.png")
    profile = {
        "best_epoch": best_checkpoint["best_epoch"],
        "completed_epochs": len(history),
        "optimizer_steps": len(loader) * len(history),
        "fit_seconds_including_validation": time.perf_counter() - training_started,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "peak_device_memory_mb": memory_mb(device),
    }
    return model, profile


def parse_arguments() -> argparse.Namespace:
    """Define the experiment configuration in one place."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--prostate-label", type=int, required=True,
        help="Inspect original mask IDs first; never assume the prostate label",
    )
    parser.add_argument("--bladder-label", type=int, default=None)
    parser.add_argument("--rectum-label", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path("runs/hipmri"))
    parser.add_argument(
        "--size", nargs=3, type=int, default=[64, 128, 128],
        metavar=("D", "H", "W"),
    )
    parser.add_argument(
        "--crop-xy", type=int, default=None,
        help="Optional even-width central crop of canonical X/Y before resizing",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-2d", type=int, default=16)
    parser.add_argument("--batch-3d", type=int, default=1)
    parser.add_argument("--width-2d", type=int, default=16)
    parser.add_argument("--width-3d", type=int, default=12)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument(
        "--foreground-weight", type=float, default=1.0,
        help="Cross-entropy weight for prostate voxels; tune using validation only",
    )
    parser.add_argument("--seed", type=int, default=3710)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--augment", action="store_true",
        help="Random training-only left-right flips for both models",
    )
    evaluation_mode = parser.add_mutually_exclusive_group()
    evaluation_mode.add_argument(
        "--validation-only", action="store_true",
        help="Train and select checkpoints without opening the test set (default)",
    )
    evaluation_mode.add_argument(
        "--evaluate-test", action="store_true",
        help="Also evaluate the test split after training; use only for a prespecified final run",
    )
    args = parser.parse_args()
    # Keep older --validation-only commands working while making that mode safe by default.
    args.validation_only = not args.evaluate_test
    chosen_labels = [
        label for label in (
            args.prostate_label, args.bladder_label, args.rectum_label
        ) if label is not None
    ]
    if len(set(chosen_labels)) != len(chosen_labels):
        parser.error("Prostate, bladder, and rectum IDs must be distinct")
    if args.foreground_weight <= 0:
        parser.error("--foreground-weight must be positive")
    if args.crop_xy is not None and (args.crop_xy <= 0 or args.crop_xy % 2):
        parser.error("--crop-xy must be a positive even integer")
    return args


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_arguments()
    args.out.mkdir(parents=True, exist_ok=True)
    existing_outputs = [
        name for name in (
            "results.json", "per_case.csv", "validation_per_case.csv",
            "2d_best.pt", "3d_best.pt",
            "2d_history.json", "3d_history.json",
        ) if (args.out / name).exists()
    ]
    if existing_outputs:
        raise FileExistsError(
            f"Run directory {args.out} already contains experiment outputs "
            f"({', '.join(existing_outputs)}). Choose a new --out directory so "
            "checkpoints and held-out results cannot be overwritten."
        )
    set_random_seed(args.seed)

    manifest_digest = manifest_sha256(args.manifest)
    cases = read_manifest(args.manifest)
    split_path = args.out / "split.json"
    if split_path.exists():
        split = load_split(cases, split_path)
    else:
        split = split_by_patient(cases, args.seed)
        save_split(split, split_path)

    size = tuple(args.size)
    train_data = VolumeDataset(split["train"], size, args.prostate_label, args.crop_xy)
    val_data = VolumeDataset(split["val"], size, args.prostate_label, args.crop_xy)
    device = choose_device(args.device)
    split_counts = {name: len(group) for name, group in split.items()}
    print(f"device={device}; cases={split_counts}", flush=True)

    results = {}
    validation_rows = []
    trained_models = {}
    for kind in ("2d", "3d"):
        model, profile = train_one(kind, train_data, val_data, device, args.out, args)
        validation, rows = evaluate(
            model, val_data, kind, device, args.bladder_label,
            args.rectum_label, native=not args.validation_only,
        )
        validation_rows.extend({"model": kind, **row} for row in rows)
        profile["peak_device_memory_mb"] = memory_mb(device)
        profile["peak_reserved_device_memory_mb"] = reserved_memory_mb(device)
        results[kind] = {"profile": profile, "validation": validation}
        trained_models[kind] = model.cpu()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    with (args.out / "validation_per_case.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(validation_rows[0]))
        writer.writeheader()
        writer.writerows(validation_rows)

    if manifest_sha256(args.manifest) != manifest_digest:
        raise ValueError("Manifest contents changed during training")

    if not args.validation_only:
        # The held-out test split is opened only after both models are selected.
        test_data = VolumeDataset(split["test"], size, args.prostate_label, args.crop_xy)
        all_rows = []
        for kind in ("2d", "3d"):
            model = trained_models[kind].to(device)
            test, rows = evaluate(
                model, test_data, kind, device, args.bladder_label,
                args.rectum_label, native=True,
            )
            results[kind]["test"] = test
            all_rows.extend({"model": kind, **row} for row in rows)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        with (args.out / "per_case.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
            writer.writeheader()
            writer.writerows(all_rows)

    if manifest_sha256(args.manifest) != manifest_digest:
        raise ValueError("Manifest contents changed during training")
    results["provenance"] = {
        "manifest": str(Path(args.manifest).resolve()),
        "manifest_sha256": manifest_digest,
        "seed": args.seed,
        "size_dhw": args.size,
        "crop_xy": args.crop_xy,
        "prostate_label": args.prostate_label,
        "bladder_label": args.bladder_label,
        "rectum_label": args.rectum_label,
        "foreground_weight": args.foreground_weight,
        "augment": args.augment,
        "learning_rate": args.lr,
        "max_epochs": args.epochs,
        "early_stopping_patience": args.patience,
        "batch_2d": args.batch_2d,
        "batch_3d": args.batch_3d,
        "width_2d": args.width_2d,
        "width_3d": args.width_3d,
        "torch": torch.__version__,
        "device": str(device),
        "gpu_device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "validation_only": args.validation_only,
    }
    (args.out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
