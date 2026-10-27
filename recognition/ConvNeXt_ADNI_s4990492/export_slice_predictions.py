"""Export validation slice predictions for confidence analysis."""

import argparse
import csv
import hashlib
import json
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
    if ({r["patient_id"] for r in dataset.rows}
            & {r["patient_id"] for r in train.rows}):
        raise ValueError("Train/validation patient overlap.")
    output = args.checkpoint.parent / "validation_slice_export"
    if output.exists():
        raise FileExistsError(f"Preserving existing output: {output}")
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=True
    )
    model_classes = {"SimpleCNN": SimpleCNN, "ConvNeXtTiny": ConvNeXtTiny}
    model_name = checkpoint["config"]["model"]
    if model_name not in model_classes:
        raise ValueError("Only the self-implemented models are supported.")
    model = model_classes[model_name]()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.cuda().eval()
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    logits_list, labels_list = [], []
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.cuda()).cpu()
            if not torch.isfinite(logits).all():
                raise RuntimeError("Non-finite logits.")
            logits_list.append(logits)
            labels_list.append(labels)
    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)
    expected = torch.tensor([int(r["target"]) for r in dataset.rows])
    if not torch.equal(labels, expected):
        raise RuntimeError("Manifest/prediction label alignment mismatch.")
    probabilities = logits.softmax(dim=1)
    predictions = logits.argmax(dim=1)
    confidence = probabilities.max(dim=1).values
    summary = classification_metrics(labels, predictions, probabilities)
    summary.update({
        "model": model_name,
        "checkpoint_epoch": checkpoint["epoch"],
        "split": "validation",
        "evaluation_unit": "slice",
        "images": len(dataset),
        "probabilities_calibrated": False,
        "row_index_definition": "zero-based data-row index in validation manifest",
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    })
    serialized = json.dumps(summary, indent=2, allow_nan=False)
    output.mkdir()
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "row_index", "target", "prediction", "logit_nc", "logit_ad",
            "prob_ad", "confidence", "correct",
        ])
        for i in range(len(dataset)):
            writer.writerow([
                i, labels[i].item(), predictions[i].item(),
                logits[i, 0].item(), logits[i, 1].item(),
                probabilities[i, 1].item(), confidence[i].item(),
                int(predictions[i].item() == labels[i].item()),
            ])
    with (output / "summary.json").open("w", encoding="utf-8") as file:
        file.write(serialized + "\n")
    print(serialized, flush=True)
    print("Saved:", output, flush=True)


if __name__ == "__main__":
    main()
