
"""Training pipeline for ADNI MRI classification.

Models:
    - SimpleCNN baseline
    - ConvNeXt-Tiny

Patient-independent splits are loaded from a fixed CSV manifest.
"""

import argparse
import csv
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)

from torch.utils.data import DataLoader, Subset

from dataset import ADNIDataset
from modules import build_model, count_parameters


DEFAULT_MANIFEST = (
    Path.home() / "data" / "comp3710_a3_adni_split.csv"
)


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_manifest(path):
    """Load saved splits and check patient independence."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Manifest not found: {path}. Run dataset.py first."
        )

    with open(path, newline="") as f:
        records = list(csv.DictReader(f))

    if not records:
        raise ValueError("Manifest is empty.")

    patient_splits = defaultdict(set)
    patient_labels = defaultdict(set)

    for record in records:
        record["label"] = int(record["label"])

        if record["split"] not in ("train", "val", "test"):
            raise ValueError("Invalid split in manifest.")

        patient_id = record["patient_id"]
        patient_splits[record["split"]].add(patient_id)
        patient_labels[patient_id].add(record["label"])

    for a, b in (
        ("train", "val"),
        ("train", "test"),
        ("val", "test"),
    ):
        overlap = patient_splits[a] & patient_splits[b]

        if overlap:
            raise ValueError(
                f"Patient leakage detected between {a} and {b}"
            )

    if any(len(labels) != 1 for labels in patient_labels.values()):
        raise ValueError("Conflicting patient labels detected.")

    print("Patient-level split verification: PASSED")

    for split in ("train", "val", "test"):
        print(
            f"{split} patients:",
            len(patient_splits[split])
        )

    return records


def make_loader(
    records,
    split,
    batch_size,
    workers,
    smoke_samples=0,
    seed=42,
):
    """Create a DataLoader with optional balanced smoke sampling."""
    dataset = ADNIDataset(records, split=split)

    if smoke_samples > 0:
        if smoke_samples < 2:
            raise ValueError(
                "Smoke test requires at least two samples."
            )

        class_indices = {0: [], 1: []}

        for index, record in enumerate(dataset.records):
            class_indices[record["label"]].append(index)

        rng = random.Random(seed)

        for label in class_indices:
            rng.shuffle(class_indices[label])

        n_nc = smoke_samples // 2
        n_ad = smoke_samples - n_nc

        if (
            len(class_indices[0]) < n_nc
            or len(class_indices[1]) < n_ad
        ):
            raise ValueError(
                "Not enough samples for balanced smoke test."
            )

        selected = (
            class_indices[0][:n_nc]
            + class_indices[1][:n_ad]
        )

        rng.shuffle(selected)

        dataset = Subset(dataset, selected)

        print(
            f"{split} smoke labels:",
            {"NC": n_nc, "AD": n_ad}
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )


def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
):
    """Run one training or validation epoch."""
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_samples = 0

    all_labels = []
    all_scores = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        probs = torch.softmax(
            logits.detach(), dim=1
        )[:, 1]

        all_labels.extend(
            labels.cpu().tolist()
        )

        all_scores.extend(
            probs.cpu().tolist()
        )

    if total_samples == 0:
        raise ValueError("DataLoader is empty.")

    predictions = [
        int(score >= 0.5)
        for score in all_scores
    ]

    accuracy = accuracy_score(
        all_labels,
        predictions,
    )

    macro_f1 = f1_score(
        all_labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    auroc = None

    if len(set(all_labels)) == 2:
        auroc = roc_auc_score(
            all_labels,
            all_scores,
        )

    return {
        "loss": total_loss / total_samples,
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "auroc": (
            float(auroc)
            if auroc is not None
            else None
        ),
    }



def plot_training_history(history, output_path):
    """Save updated training and validation curves after each epoch."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    fig = Figure(figsize=(12, 3.5))
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 3)
    epochs = [row["epoch"] for row in history]

    for ax, metric, title in zip(
        axes,
        ("loss", "accuracy", "macro_f1"),
        ("Cross-entropy loss", "Accuracy", "Macro-F1"),
    ):
        for split in ("train", "val"):
            ax.plot(
                epochs,
                [row[split][metric] for row in history],
                label=split,
            )
        ax.set_xlabel("Epoch")
        ax.set_title(title)
        ax.grid(alpha=0.2)
        ax.legend()
        if metric != "loss":
            ax.set_ylim(0, 1.03)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    fig.clear()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=["cnn", "convnext"],
        default="cnn",
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--smoke-samples",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--pretrained",
        action="store_true",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.home() / "data" / "comp3710_a3_runs",
    )

    parser.add_argument(
        "--test-checkpoint", type=Path,
        help="Evaluate a saved checkpoint on the test split without training.",
    )
    parser.add_argument(
        "--test-output-dir", type=Path,
        help="Directory for test metrics and prediction CSVs.",
    )

    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        parser.error(
            "Epochs and batch size must be positive."
        )

    if args.test_checkpoint is not None:
        if args.test_output_dir is None:
            parser.error("--test-checkpoint requires --test-output-dir.")
        import subprocess
        import sys

        subprocess.run([
            sys.executable, "-u",
            str(Path(__file__).resolve().with_name("evaluate.py")),
            "--checkpoint", str(args.test_checkpoint),
            "--output-dir", str(args.test_output_dir),
            "--manifest", str(args.manifest),
            "--split", "test",
            "--batch-size", str(args.batch_size),
            "--workers", str(args.workers),
        ], check=True)
        return

    if args.test_output_dir is not None:
        parser.error("--test-output-dir requires --test-checkpoint.")

    set_seed(args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    if device.type == "cpu" and args.smoke_samples == 0:
        raise RuntimeError(
            "Full training requires a GPU allocation. "
            "Use --smoke-samples for a CPU smoke test."
        )

    records = load_manifest(args.manifest)

    train_loader = make_loader(
        records,
        "train",
        args.batch_size,
        args.workers,
        args.smoke_samples,
        args.seed,
    )

    val_loader = make_loader(
        records,
        "val",
        args.batch_size,
        args.workers,
        args.smoke_samples,
        args.seed,
    )

    model = build_model(
        args.model,
        num_classes=2,
        pretrained=args.pretrained,
    ).to(device)

    total_params, trainable_params = count_parameters(model)

    print("Model:", args.model)
    print("Total parameters:", total_params)
    print("Trainable parameters:", trainable_params)
    print("Training samples:", len(train_loader.dataset))
    print("Validation samples:", len(val_loader.dataset))

    criterion = nn.CrossEntropyLoss()

    learning_rate = args.lr

    if learning_rate is None:
        learning_rate = (
            1e-4
            if args.model == "convnext"
            else 1e-3
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
    )

    # Smoke tests and full runs use separate directories.
    run_type = (
        "smoke"
        if args.smoke_samples > 0
        else "full"
    )

    run_dir = (
        args.output_dir
        / args.model
        / run_type
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_f1 = -1.0
    history = []

    start_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
        )

        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            device,
        )

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train loss {train_metrics['loss']:.4f} | "
            f"Train acc {train_metrics['accuracy']:.4f} | "
            f"Val loss {val_metrics['loss']:.4f} | "
            f"Val acc {val_metrics['accuracy']:.4f} | "
            f"Val macro-F1 {val_metrics['macro_f1']:.4f} | "
            f"Val AUROC {val_metrics['auroc']}"
        )

        history.append({
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        })

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]

            torch.save({
                "model_state_dict": model.state_dict(),
                "model_name": args.model,
                "epoch": epoch,
                "validation_metrics": val_metrics,
                "seed": args.seed,
                "pretrained": args.pretrained,
            }, run_dir / "best_model.pth")

        plot_training_history(
            history, run_dir / "training_curves.png"
        )

    elapsed = time.perf_counter() - start_time

    config = vars(args).copy()
    config["manifest"] = str(args.manifest)
    config["output_dir"] = str(args.output_dir)
    config["lr"] = learning_rate

    results = {
        "config": config,
        "history": history,
        "training_seconds": elapsed,
        "parameters": total_params,
        "trainable_parameters": trainable_params,
        "device": str(device),
        "best_validation_macro_f1": best_f1,
    }

    with open(run_dir / "history.json", "w") as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    print("\nTraining completed.")
    print("Best validation macro-F1:", round(best_f1, 4))
    print("Training time:", round(elapsed, 2), "seconds")
    print("Results saved to:", run_dir)


if __name__ == "__main__":
    main()

