"""Export private validation predictions and aggregate by patient."""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import ADNISliceDataset
from modules import SimpleCNN, ConvNeXtTiny
from predict import classification_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Run in a Slurm GPU job.")
    torch.set_num_threads(2)

    base = Path(__file__).resolve().parent
    manifest = base / "splits" / "val.csv"
    dataset = ADNISliceDataset(manifest)
    train = ADNISliceDataset(base / "splits" / "train.csv")

    val_patients = {r["patient_id"] for r in dataset.rows}
    if val_patients & {r["patient_id"] for r in train.rows}:
        raise ValueError("Train/validation patient overlap.")

    output = args.checkpoint.parent / "validation_export"
    if output.exists():
        raise FileExistsError(f"Preserving existing output: {output}")

    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=True
    )
    models = {"SimpleCNN": SimpleCNN, "ConvNeXtTiny": ConvNeXtTiny}
    model = models[checkpoint["config"]["model"]]()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.cuda().eval()

    logits_list, labels_list = [], []
    loader = DataLoader(
        dataset, batch_size=8, shuffle=False, num_workers=0
    )
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.cuda()).cpu()
            if not torch.isfinite(logits).all():
                raise RuntimeError("Non-finite logits.")
            logits_list.append(logits)
            labels_list.append(labels)

    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)
    probabilities = logits.softmax(dim=1)
    predictions = logits.argmax(dim=1)

    # Confirm predictions remain aligned with the ordered manifest.
    expected_labels = torch.tensor(
        [int(r["target"]) for r in dataset.rows]
    )
    if not torch.equal(labels, expected_labels):
        raise RuntimeError("Prediction/manifest alignment mismatch.")

    scan_indices = defaultdict(list)
    patient_labels = {}
    for index, row in enumerate(dataset.rows):
        patient = row["patient_id"]
        label = int(row["target"])
        if patient in patient_labels and patient_labels[patient] != label:
            raise ValueError("Conflicting patient labels.")
        patient_labels[patient] = label
        scan_indices[(patient, row["scan_id"])].append(index)

    patient_scans = defaultdict(list)
    for (patient, scan), indices in scan_indices.items():
        patient_scans[patient].append(
            probabilities[indices].double().mean(dim=0)
        )

    patient_ids = sorted(patient_scans)
    patient_probabilities = torch.stack([
        torch.stack(patient_scans[p]).mean(dim=0)
        for p in patient_ids
    ])
    patient_targets = torch.tensor([
        patient_labels[p] for p in patient_ids
    ])
    patient_predictions = patient_probabilities.argmax(dim=1)

    slice_metrics = classification_metrics(
        labels, predictions, probabilities
    )
    patient_metrics = classification_metrics(
        patient_targets, patient_predictions, patient_probabilities
    )
    summary = {
        "split": "validation",
        "model": checkpoint["config"]["model"],
        "checkpoint_epoch": checkpoint["epoch"],
        "images": len(dataset),
        "scans": len(scan_indices),
        "patients": len(patient_ids),
        "aggregation": "mean slices per scan, then mean scans per patient",
        "decision_rule": "argmax; exact ties assigned to NC",
        "probabilities_calibrated": False,
        "manifest_sha256": hashlib.sha256(
            manifest.read_bytes()
        ).hexdigest(),
        "slice_metrics": slice_metrics,
        "patient_metrics": patient_metrics,
    }

    output.mkdir()
    with (output / "slice_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow([
            "row_index", "patient_id", "scan_id", "target",
            "prediction", "logit_nc", "logit_ad", "prob_ad", "confidence",
        ])
        for index, row in enumerate(dataset.rows):
            writer.writerow([
                index, row["patient_id"], row["scan_id"],
                labels[index].item(), predictions[index].item(),
                logits[index, 0].item(), logits[index, 1].item(),
                probabilities[index, 1].item(),
                probabilities[index].max().item(),
            ])

    with (output / "patient_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow([
            "patient_id", "scans", "target",
            "prediction", "prob_ad", "confidence",
        ])
        for index, patient in enumerate(patient_ids):
            writer.writerow([
                patient, len(patient_scans[patient]),
                patient_targets[index].item(),
                patient_predictions[index].item(),
                patient_probabilities[index, 1].item(),
                patient_probabilities[index].max().item(),
            ])

    with (output / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, allow_nan=False)

    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)
    print("Private predictions saved under:", output, flush=True)


if __name__ == "__main__":
    main()