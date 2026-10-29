"""Train the MLP baseline or two-layer GCN on Facebook page data.

The training mask updates model weights, the validation mask selects the best
checkpoint, and the test mask is used once after model selection is complete.
Separate training and split seeds let repeated runs use identical held-out pages.

References:
    GCN architecture: T. N. Kipf and M. Welling, ICLR 2017.
    https://arxiv.org/abs/1609.02907
    Course-linked dataset: https://github.com/gayanku/facebook_page-page_dataset
    See modules.py and dataset.py for model/operator and data attribution.

AI assistance:
    OpenAI Codex assisted in generating, revising and debugging this code.
    The README section "Artificial Intelligence Usage Disclosure" documents
    the scope of assistance and the data audits, smoke runs, complete training
    runs and saved-output checks used to verify the implementation.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch_geometric.data import Data

from dataset import load_facebook_page_page
from modules import MLPBaseline, TwoLayerGCN


def set_seed(seed: int) -> None:
    """Set random seeds so an experiment can be repeated."""

    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    """Select the CPU or an available CUDA device."""

    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(device_name)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but no CUDA device is available"
        )

    return device


def create_model(
    model_name: str,
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    dropout_p: float,
) -> nn.Module:
    """Create either the MLP baseline or the two-layer GCN."""

    if model_name == "mlp":
        model = MLPBaseline(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            dropout_p=dropout_p,
        )
    elif model_name == "gcn":
        model = TwoLayerGCN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            dropout_p=dropout_p,
        )
    else:
        raise ValueError(f"Unknown model name: {model_name!r}")

    return model


def forward_model(
    model: nn.Module,
    data: Data,
    model_name: str,
) -> torch.Tensor:
    """Run the correct forward pass for the selected model."""

    if model_name == "mlp":
        outputs = model(data.x)
    elif model_name == "gcn":
        outputs = model(data.x, data.edge_index)
    else:
        raise ValueError(f"Unknown model name: {model_name!r}")

    return outputs


def train_one_epoch(
    model: nn.Module,
    data: Data,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    model_name: str,
) -> float:
    """Train the selected model for one epoch."""

    model.train()

    # Forward pass and loss on training nodes only.
    outputs = forward_model(model, data, model_name)
    loss = criterion(
        outputs[data.train_mask],
        data.y[data.train_mask],
    )

    if not bool(torch.isfinite(loss)):
        raise RuntimeError("Training produced a non-finite loss")

    # Backward pass and parameter update.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return float(loss.item())


def macro_f1_from_predictions(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> float:
    """Calculate the unweighted mean F1 score across all classes."""

    class_f1_scores: list[torch.Tensor] = []

    for class_index in range(num_classes):
        predicted_class = predictions == class_index
        actual_class = labels == class_index

        true_positive = (predicted_class & actual_class).sum().float()
        false_positive = (predicted_class & ~actual_class).sum().float()
        false_negative = (~predicted_class & actual_class).sum().float()

        denominator = (
            2.0 * true_positive
            + false_positive
            + false_negative
        )

        if float(denominator.item()) == 0.0:
            class_f1 = torch.tensor(
                0.0,
                device=labels.device,
            )
        else:
            class_f1 = 2.0 * true_positive / denominator

        class_f1_scores.append(class_f1)

    return float(torch.stack(class_f1_scores).mean().item())


def evaluate(
    model: nn.Module,
    data: Data,
    mask: torch.Tensor,
    criterion: nn.Module,
    model_name: str,
    num_classes: int,
) -> dict[str, float]:
    """Calculate loss, accuracy, and macro-F1 without updating weights."""

    model.eval()

    with torch.no_grad():
        outputs = forward_model(model, data, model_name)
        selected_outputs = outputs[mask]
        selected_labels = data.y[mask]

        loss = criterion(
            selected_outputs,
            selected_labels,
        )

        predictions = selected_outputs.argmax(dim=1)
        correct = (predictions == selected_labels).sum().item()
        accuracy = correct / int(mask.sum().item())
        macro_f1 = macro_f1_from_predictions(
            predictions,
            selected_labels,
            num_classes,
        )

    return {
        "loss": float(loss.item()),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
    }


def save_history(
    path: Path,
    history: list[dict[str, float | int]],
) -> None:
    """Save epoch-level results as a CSV file."""

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "epoch",
                "train_loss",
                "train_accuracy",
                "train_macro_f1",
                "validation_loss",
                "validation_accuracy",
                "validation_macro_f1",
            ),
        )
        writer.writeheader()
        writer.writerows(history)


def save_training_curves(
    path: Path,
    history: list[dict[str, float | int]],
    model_name: str,
    seed: int,
    best_epoch: int,
) -> None:
    """Plot training and validation metrics recorded after each epoch."""

    if not history:
        raise ValueError("Training history cannot be empty")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for axis, metric, title in zip(
        axes,
        ("loss", "accuracy", "macro_f1"),
        ("Loss", "Accuracy", "Macro-F1"),
    ):
        axis.plot(
            epochs,
            [row[f"train_{metric}"] for row in history],
            marker="o" if len(epochs) <= 20 else None,
            label="Training",
        )
        axis.plot(
            epochs,
            [row[f"validation_{metric}"] for row in history],
            marker="o" if len(epochs) <= 20 else None,
            linestyle="--",
            label="Validation",
        )
        axis.axvline(
            best_epoch,
            color="black",
            linestyle="--",
            alpha=0.6,
            label=f"Validation-selected epoch: {best_epoch}",
        )
        axis.set(title=title, xlabel="Epoch", ylabel=title)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))
        axis.ticklabel_format(axis="y", style="plain", useOffset=False)

        if len(epochs) == 1:
            axis.set_xlim(epochs[0] - 0.5, epochs[0] + 0.5)
        else:
            axis.set_xlim(epochs[0], epochs[-1])
        if metric != "loss":
            axis.set_ylim(0.0, 1.0)

        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)

    figure.suptitle(f"{model_name.upper()} training history (seed {seed})")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def load_checkpoint(
    path: Path,
    device: torch.device,
) -> dict[str, Any]:
    """Load a saved checkpoint on the selected device."""

    try:
        checkpoint = torch.load(
            path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            path,
            map_location=device,
        )

    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(f"Invalid checkpoint: {path}")

    return checkpoint


def run_training(args: argparse.Namespace) -> None:
    """Train one model, select it on validation F1, and test it once."""

    if args.epochs < 1:
        raise ValueError("epochs must be at least 1")
    if args.hidden_channels < 1:
        raise ValueError("hidden-channels must be at least 1")
    if args.learning_rate <= 0.0:
        raise ValueError("learning-rate must be positive")
    if args.weight_decay < 0.0:
        raise ValueError("weight-decay cannot be negative")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("dropout must be in the range [0, 1)")
    if args.log_every < 1:
        raise ValueError("log-every must be at least 1")

    set_seed(args.seed)
    device = resolve_device(args.device)

    print(f"> Device: {device}")
    print(f"> Training seed: {args.seed} | Data split seed: {args.split_seed}")
    print("> Loading dataset")

    data = load_facebook_page_page(
        args.data_path,
        seed=args.split_seed,
    )
    data = data.to(device)

    num_classes = int(data.num_classes)
    model_config = {
        "model_name": args.model,
        "in_channels": int(data.x.shape[1]),
        "hidden_channels": args.hidden_channels,
        "out_channels": num_classes,
        "dropout_p": args.dropout,
    }

    model = create_model(**model_config).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    run_name = f"{args.model}_seed{args.seed}"
    checkpoint_path = args.output_dir / f"{run_name}_best.pt"
    history_path = args.output_dir / f"{run_name}_history.csv"
    curves_path = args.output_dir / f"{run_name}_training_curves.png"
    results_path = args.output_dir / f"{run_name}_results.json"

    history: list[dict[str, float | int]] = []
    best_validation_f1 = float("-inf")
    best_validation_loss = float("inf")
    best_epoch = 0

    print(f"> Training {args.model.upper()}")
    start_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        train_one_epoch(
            model,
            data,
            criterion,
            optimizer,
            args.model,
        )

        train_metrics = evaluate(
            model,
            data,
            data.train_mask,
            criterion,
            args.model,
            num_classes,
        )
        validation_metrics = evaluate(
            model,
            data,
            data.val_mask,
            criterion,
            args.model,
            num_classes,
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "train_macro_f1": train_metrics["macro_f1"],
                "validation_loss": validation_metrics["loss"],
                "validation_accuracy": validation_metrics["accuracy"],
                "validation_macro_f1": validation_metrics["macro_f1"],
            }
        )

        validation_f1 = validation_metrics["macro_f1"]
        validation_loss = validation_metrics["loss"]
        improved = (
            validation_f1 > best_validation_f1
            or (
                validation_f1 == best_validation_f1
                and validation_loss < best_validation_loss
            )
        )

        if improved:
            best_validation_f1 = validation_f1
            best_validation_loss = validation_loss
            best_epoch = epoch

            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_config": model_config,
                    "best_epoch": best_epoch,
                    "best_validation_macro_f1": best_validation_f1,
                    "validation_loss": best_validation_loss,
                    "seed": args.seed,
                    "split_seed": int(data.split_seed),
                    "split_ratios": list(data.split_ratios),
                    "class_values": list(data.class_values),
                },
                checkpoint_path,
            )

        if (
            epoch == 1
            or epoch % args.log_every == 0
            or epoch == args.epochs
        ):
            print(
                f"Epoch [{epoch}/{args.epochs}] "
                f"Train Loss: {train_metrics['loss']:.4f} "
                f"Train Accuracy: {train_metrics['accuracy']:.4f} "
                f"Train Macro-F1: {train_metrics['macro_f1']:.4f} "
                f"Validation Loss: {validation_metrics['loss']:.4f} "
                f"Validation Accuracy: "
                f"{validation_metrics['accuracy']:.4f} "
                f"Validation Macro-F1: "
                f"{validation_metrics['macro_f1']:.4f} | "
                f"Elapsed: {time.perf_counter() - start_time:.3f} s"
            )

    training_seconds = time.perf_counter() - start_time

    # Restore the best validation checkpoint before using the test mask.
    checkpoint = load_checkpoint(
        checkpoint_path,
        device,
    )
    model.load_state_dict(checkpoint["state_dict"])

    final_train_metrics = evaluate(
        model,
        data,
        data.train_mask,
        criterion,
        args.model,
        num_classes,
    )
    final_validation_metrics = evaluate(
        model,
        data,
        data.val_mask,
        criterion,
        args.model,
        num_classes,
    )
    final_test_metrics = evaluate(
        model,
        data,
        data.test_mask,
        criterion,
        args.model,
        num_classes,
    )

    results = {
        "model": args.model,
        "seed": args.seed,
        "split_seed": int(data.split_seed),
        "split_ratios": list(data.split_ratios),
        "epochs_requested": args.epochs,
        "best_epoch": best_epoch,
        "device": str(device),
        "parameter_count": sum(
            parameter.numel()
            for parameter in model.parameters()
        ),
        "training_seconds": training_seconds,
        "split_counts": {
            "train": int(data.train_mask.sum().item()),
            "validation": int(data.val_mask.sum().item()),
            "test": int(data.test_mask.sum().item()),
        },
        "train_metrics": final_train_metrics,
        "validation_metrics": final_validation_metrics,
        "test_metrics": final_test_metrics,
        "checkpoint": str(checkpoint_path),
    }

    save_history(
        history_path,
        history,
    )

    with results_path.open("w", encoding="utf-8") as handle:
        json.dump(
            results,
            handle,
            indent=2,
        )

    save_training_curves(
        curves_path,
        history,
        args.model,
        args.seed,
        best_epoch,
    )

    print("> Final results")
    print(f"Training loop time (including validation): {training_seconds:.3f} seconds")
    print(
        f"Best epoch: {best_epoch} | "
        f"Validation Macro-F1: "
        f"{final_validation_metrics['macro_f1']:.4f} | "
        f"Test Accuracy: {final_test_metrics['accuracy']:.4f} | "
        f"Test Macro-F1: {final_test_metrics['macro_f1']:.4f}"
    )
    print(f"Checkpoint: {checkpoint_path}")
    print(f"History: {history_path}")
    print(f"Training curves: {curves_path}")
    print(f"Results: {results_path}")


def parse_args() -> argparse.Namespace:
    """Read experiment settings from the command line."""

    parser = argparse.ArgumentParser(
        description=(
            "Train an MLP or GCN on the Facebook Page-Page dataset"
        )
    )

    parser.add_argument(
        "--data-path",
        "--data",
        dest="data_path",
        type=Path,
        required=True,
        help="Path to facebook.npz",
    )
    parser.add_argument(
        "--model",
        choices=("mlp", "gcn"),
        required=True,
        help="Model architecture to train",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--hidden-channels",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=5e-4,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for model initialization and training randomness",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Seed for the data split; keep fixed across repeated training runs",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device selection: auto, cpu, cuda, or cuda:N",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
    )

    return parser.parse_args()


def main() -> None:
    """Run the training program."""

    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
