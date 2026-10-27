"""Evaluate saved ADNI models at slice, scan and patient levels."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, roc_auc_score,
)
from torch.utils.data import DataLoader

from dataset import ADNIDataset
from modules import build_model
from train import load_manifest


def calculate_metrics(rows):
    labels = [r["label"] for r in rows]
    scores = [r["prob_ad"] for r in rows]
    predictions = [int(p >= 0.5) for p in scores]

    tn, fp, fn, tp = confusion_matrix(
        labels, predictions, labels=[0, 1]
    ).ravel()

    return {
        "n": len(rows),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(
            labels, predictions, labels=[0, 1],
            average="macro", zero_division=0,
        )),
        "auroc": (
            float(roc_auc_score(labels, scores))
            if len(set(labels)) == 2 else None
        ),
        "sensitivity_ad": float(tp / (tp + fn)) if tp + fn else None,
        "specificity_nc": float(tn / (tn + fp)) if tn + fp else None,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def aggregate(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)

    result = []
    for key, members in sorted(groups.items()):
        labels = {r["label"] for r in members}
        if len(labels) != 1:
            raise ValueError(f"Conflicting labels: {key}")

        result.append({
            **dict(zip(keys, key)),
            "label": labels.pop(),
            "prob_ad": float(np.mean([r["prob_ad"] for r in members])),
            "n_members": len(members),
        })
    return result


def save_csv(path, rows):
    output = []
    for row in rows:
        prediction = int(row["prob_ad"] >= 0.5)
        output.append({
            **row,
            "prediction": prediction,
            "correct": int(prediction == row["label"]),
        })

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path,
        default=Path.home() / "data/comp3710_a3_adni_split.csv",
    )
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Run evaluation through a GPU Slurm job.")

    device = torch.device("cuda")
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=True,
    )
    records = load_manifest(args.manifest)
    dataset = ADNIDataset(records, split=args.split)
    if not len(dataset):
        raise ValueError("Evaluation dataset is empty.")

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    # All trained weights come from the checkpoint; no download needed.
    model = build_model(
        checkpoint["model_name"], num_classes=2, pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()

    criterion = nn.CrossEntropyLoss(reduction="sum")
    scores, labels = [], []
    total_loss = 0.0

    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            total_loss += criterion(logits, targets).item()
            scores.extend(torch.softmax(logits, dim=1)[:, 1].cpu().tolist())
            labels.extend(targets.cpu().tolist())

    if labels != [r["label"] for r in dataset.records]:
        raise ValueError("Predictions and metadata are misaligned.")

    slices = [
        {
            "path": r["path"],
            "patient_id": r["patient_id"],
            "scan_id": r["scan_id"],
            "label": label,
            "prob_ad": score,
        }
        for r, label, score in zip(dataset.records, labels, scores)
    ]
    scans = aggregate(slices, ["patient_id", "scan_id"])
    patients = aggregate(scans, ["patient_id"])

    metrics = {
        "checkpoint": str(args.checkpoint.resolve()),
        "model": checkpoint["model_name"],
        "saved_epoch": checkpoint["epoch"],
        "pretrained": checkpoint["pretrained"],
        "split": args.split,
        "threshold": 0.5,
        "patient_aggregation": "mean slices per scan, then mean scans per patient",
        "slice": calculate_metrics(slices),
        "scan": calculate_metrics(scans),
        "patient": calculate_metrics(patients),
    }
    metrics["slice"]["loss"] = total_loss / len(dataset)

    if args.split == "val":
        saved = checkpoint["validation_metrics"]
        for key in ("accuracy", "macro_f1", "auroc", "loss"):
            if not np.isclose(
                metrics["slice"][key], saved[key], atol=1e-5, rtol=1e-5):
                raise ValueError(
                    f"Validation reproduction failed for {key}: "
                    f"{metrics['slice'][key]} vs {saved[key]}"
                )
        print("Validation reproduction: PASSED", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("slice", slices), ("scan", scans), ("patient", patients),
    ):
        save_csv(args.output_dir / f"{name}_predictions.csv", rows)

    with open(args.output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, allow_nan=False)

    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
