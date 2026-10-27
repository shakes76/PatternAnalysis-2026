"""Evaluate saved models on the fixed validation split only."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from dataset import ADNISliceDataset
from modules import SimpleCNN, ConvNeXtTiny


def binary_auc(labels, scores):
    """AD-positive ROC AUC with tied scores grouped before integration."""
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    if positives == 0 or negatives == 0:
        raise ValueError("AUROC requires both classes.")
    order = torch.argsort(scores, descending=True)
    scores, labels = scores[order], labels[order]
    ends = torch.ones(len(scores), dtype=torch.bool)
    ends[:-1] = scores[:-1] != scores[1:]
    tp = (labels == 1).cumsum(0)[ends].double()
    fp = (labels == 0).cumsum(0)[ends].double()
    zero = torch.zeros(1, dtype=torch.float64)
    tpr = torch.cat([zero, tp / positives])
    fpr = torch.cat([zero, fp / negatives])
    return torch.trapezoid(tpr, fpr).item()


def classification_metrics(labels, predictions, probabilities):
    """Rows are true classes, columns predicted classes; NC=0, AD=1."""
    confusion = torch.bincount(
        labels * 2 + predictions, minlength=4
    ).reshape(2, 2)
    per_class = {}
    for index, name in enumerate(("NC", "AD")):
        tp = confusion[index, index].item()
        support = confusion[index].sum().item()
        predicted = confusion[:, index].sum().item()
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if precision + recall else 0.0)
        per_class[name] = dict(
            precision=precision, recall=recall, f1=f1, support=support
        )
    return {
        "accuracy": (labels == predictions).double().mean().item(),
        "balanced_accuracy": sum(x["recall"] for x in per_class.values()) / 2,
        "macro_f1": sum(x["f1"] for x in per_class.values()) / 2,
        "auroc_ad_positive": binary_auc(labels, probabilities[:, 1]),
        "confusion_matrix": confusion.tolist(),
        "confusion_order": ["NC", "AD"],
        "confusion_axes": "rows=true, columns=predicted",
        "per_class": per_class,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("Batch size must be positive.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; run in a GPU job.")
    output = args.checkpoint.parent / "validation_metrics.json"
    if output.exists():
        raise FileExistsError(f"Preserving existing evaluation: {output}")
    torch.set_num_threads(2)
    base = Path(__file__).resolve().parent
    manifest = base / "splits" / "val.csv"
    dataset = ADNISliceDataset(manifest)
    train_dataset = ADNISliceDataset(base / "splits" / "train.csv")
    train_patients = {r["patient_id"] for r in train_dataset.rows}
    val_patients = {r["patient_id"] for r in dataset.rows}
    if train_patients & val_patients:
        raise ValueError("Train/validation patient overlap.")
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=True
    )
    config = checkpoint["config"]
    models = {"SimpleCNN": SimpleCNN, "ConvNeXtTiny": ConvNeXtTiny}
    if config["model"] not in models:
        raise ValueError("Unsupported checkpoint model.")
    model = models[config["model"]]()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    device = torch.device(args.device)
    model.to(device).eval()
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=False, num_workers=0)
    loss_sum = 0.0
    all_labels, all_probabilities, all_predictions = [], [], []
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            if not torch.isfinite(logits).all():
                raise RuntimeError("Non-finite logits.")
            loss_sum += F.cross_entropy(logits, labels, reduction="sum").item()
            all_labels.append(labels.cpu())
            all_probabilities.append(logits.softmax(dim=1).cpu())
            all_predictions.append(logits.argmax(dim=1).cpu())
    labels = torch.cat(all_labels)
    probabilities = torch.cat(all_probabilities)
    predictions = torch.cat(all_predictions)
    metrics = classification_metrics(labels, predictions, probabilities)
    metrics.update({
        "loss": loss_sum / len(dataset),
        "images": len(dataset), "patients": len(val_patients),
        "evaluation_unit": "slice", "split": "validation",
        "model": config["model"], "checkpoint_epoch": checkpoint["epoch"],
        "recorded_val_loss": checkpoint["val_loss"],
        "loss_difference": loss_sum / len(dataset) - checkpoint["val_loss"],
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "probabilities_calibrated": False,
        "torch_version": str(torch.__version__),
        "device": str(device), "batch_size": args.batch_size,
    })
    serialized = json.dumps(metrics, indent=2, allow_nan=False)
    with output.open("x", encoding="utf-8") as file:
        file.write(serialized + "\n")
    print(serialized, flush=True)
    print("Saved:", output, flush=True)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'example':
        sys.argv.pop(1)
        from predict_rgb224_example import main as example_main
        example_main()
    else:
        main()
