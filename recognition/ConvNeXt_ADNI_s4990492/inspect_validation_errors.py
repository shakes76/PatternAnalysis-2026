"""Inspect errors on the fixed validation set; never load test images."""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from dataset import ADNISliceDataset
from modules import ConvNeXtTiny


def select_errors(labels, predictions, confidence, rows, true_label, limit=3):
    """Choose highest-confidence errors from distinct patients per error type."""
    indices = torch.where((labels == true_label) & (predictions != labels))[0].tolist()
    indices.sort(key=lambda i: (-confidence[i].item(), i))
    selected, seen = [], set()
    for i in indices:
        patient = rows[i]["patient_id"]
        if patient not in seen:
            seen.add(patient)
            selected.append(i)
            if len(selected) == limit:
                break
    return selected


def main():
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")
    base = Path(__file__).resolve().parent
    manifests = base.parents[1] / "local_data" / "splits_local"
    manifest = manifests / "val.csv"
    run = base / "runs" / "local_convnext_aug_lr3e5_20261025_123556_489058"
    output = run / "validation_error_inspection"
    if output.exists():
        raise FileExistsError("Output already exists; preserving it.")
    dataset = ADNISliceDataset(manifest)
    train = ADNISliceDataset(manifests / "train.csv")
    if {r["patient_id"] for r in dataset.rows} & {r["patient_id"] for r in train.rows}:
        raise ValueError("Train/validation patient overlap.")
    checkpoint = torch.load(run / "best_convnext.pth", map_location="cpu", weights_only=True)
    if checkpoint["config"]["model"] != "ConvNeXtTiny":
        raise ValueError("Expected the local control ConvNeXt checkpoint.")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if checkpoint["config"]["manifest_sha256"]["val"] != digest:
        raise ValueError("Validation manifest differs from training configuration.")
    model = ConvNeXtTiny()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.cuda().eval()
    probabilities, labels = [], []
    loss_sum = 0.0
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    with torch.inference_mode():
        for images, targets in loader:
            logits = model(images.cuda())
            if not torch.isfinite(logits).all():
                raise RuntimeError("Non-finite logits.")
            loss_sum += F.cross_entropy(logits, targets.cuda(), reduction="sum").item()
            probabilities.append(logits.softmax(dim=1).cpu())
            labels.append(targets)
    probabilities, labels = torch.cat(probabilities), torch.cat(labels)
    if not torch.equal(labels, torch.tensor([int(r["target"]) for r in dataset.rows])):
        raise RuntimeError("Manifest label-order mismatch.")
    predictions = probabilities.argmax(dim=1)
    confidence = probabilities.max(dim=1).values
    correct = predictions == labels
    groups = [select_errors(labels, predictions, confidence, dataset.rows, t) for t in (0, 1)]
    summary = {
        "split": "validation", "evaluation_unit": "slice", "epoch": checkpoint["epoch"],
        "images": len(dataset), "accuracy": correct.double().mean().item(),
        "loss": loss_sum / len(dataset),
        "recorded_val_loss": checkpoint["val_loss"],
        "loss_difference": loss_sum / len(dataset) - checkpoint["val_loss"],
        "manifest_sha256": digest,
        "errors": int((~correct).sum()),
        "nc_predicted_ad": int(((labels == 0) & (predictions == 1)).sum()),
        "ad_predicted_nc": int(((labels == 1) & (predictions == 0)).sum()),
        "mean_confidence_correct": confidence[correct].mean().item() if correct.any() else None,
        "mean_confidence_wrong": confidence[~correct].mean().item() if (~correct).any() else None,
        "errors_with_confidence_at_least_90pct": int(((~correct) & (confidence >= 0.9)).sum()),
        "probabilities_calibrated": False,
        "figure_selection": "Up to three distinct patients per error type, descending error confidence",
        "selected_manifest_rows": {"nc_predicted_ad": groups[0], "ad_predicted_nc": groups[1]},
    }
    serialized = json.dumps(summary, indent=2, allow_nan=False)
    output.mkdir()
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["row_index", "target", "prediction", "prob_ad", "confidence"])
        for i in range(len(dataset)):
            writer.writerow([i, labels[i].item(), predictions[i].item(),
                             probabilities[i, 1].item(), confidence[i].item()])
    if any(groups):
        fig, axes = plt.subplots(2, 3, figsize=(12, 10), layout="constrained")
        for axis in axes.flat:
            axis.set_axis_off()
        names = ["NC", "AD"]
        for row, indices in enumerate(groups):
            for column, i in enumerate(indices):
                axis = axes[row, column]
                image, _ = dataset[i]
                axis.imshow(image.squeeze(0), cmap="gray", vmin=0, vmax=1)
                axis.set_title(
                    f"True {names[labels[i].item()]} -> Predicted {names[predictions[i].item()]}"
                    f"\nConfidence {confidence[i].item():.1%}"
                )
        fig.suptitle("High-confidence validation errors: distinct patients per error type")
        fig.savefig(output / "high_confidence_errors.png", dpi=160)
        plt.close(fig)
    (output / "summary.json").write_text(serialized, encoding="utf-8")
    print(serialized, flush=True)
    print("Saved:", output, flush=True)


if __name__ == "__main__":
    main()
