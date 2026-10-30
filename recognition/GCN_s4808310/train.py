"""
Training file
"""
import argparse
import copy
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn

from dataset import load_facebook
from modules import GCN, MLP


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def forward_model(model, graph, model_name):
    if model_name == "gcn":
        return model(graph.x, graph.edge_index)
    return model(graph.x)


def evaluate(model, graph, mask, model_name):
    model.eval()

    with torch.no_grad():
        logits = forward_model(model, graph, model_name)
        loss = nn.functional.cross_entropy(
            logits[mask], graph.y[mask]
        )

        predictions = logits[mask].argmax(dim=1).cpu().numpy()
        labels = graph.y[mask].cpu().numpy()

    return {
        "loss": float(loss.item()),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(
            labels, predictions, average="macro", zero_division=0
        )),
    }


def plot_history(history, model_name, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Validation")
    axes[0].set_title(f"{model_name.upper()} Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy")
    axes[0].legend()

    axes[1].plot(epochs, history["val_accuracy"], label="Accuracy")
    axes[1].plot(epochs, history["val_macro_f1"], label="Macro-F1")
    axes[1].set_title(f"{model_name.upper()} Validation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_dir / f"{model_name}_training.png", dpi=150)
    plt.close(fig)


def train_model(model, graph, model_name, args, output_dir):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    criterion = nn.CrossEntropyLoss()

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_macro_f1": [],
    }

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    stale_epochs = 0

    start_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()

        logits = forward_model(model, graph, model_name)

        # Only training-node labels contribute to optimisation.
        loss = criterion(
            logits[graph.train_mask],
            graph.y[graph.train_mask]
        )

        loss.backward()
        optimizer.step()

        val_metrics = evaluate(
            model, graph, graph.val_mask, model_name
        )

        history["train_loss"].append(float(loss.item()))
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])

        if val_metrics["loss"] < best_val_loss - 1e-4:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"{model_name.upper()} | Epoch {epoch:03d} | "
                f"Train loss: {loss.item():.4f} | "
                f"Val loss: {val_metrics['loss']:.4f} | "
                f"Val F1: {val_metrics['macro_f1']:.4f}",
                flush=True
            )

        if stale_epochs >= args.patience:
            print(f"{model_name.upper()} early stopping at epoch {epoch}")
            break

    training_seconds = time.perf_counter() - start_time

    # Restore best validation checkpoint before final test evaluation.
    model.load_state_dict(best_state)

    test_metrics = evaluate(
        model, graph, graph.test_mask, model_name
    )

    torch.save(
        model.state_dict(),
        output_dir / f"{model_name}_best.pt"
    )

    plot_history(history, model_name, output_dir)

    results = {
        "model": model_name,
        "best_epoch": best_epoch,
        "parameters": sum(p.numel() for p in model.parameters()),
        "training_seconds": training_seconds,
        "best_validation_loss": best_val_loss,
        "test": test_metrics,
        "history": history,
    }

    print(
        f"\n{model_name.upper()} Test Results: "
        f"Accuracy={test_metrics['accuracy']:.4f}, "
        f"Macro-F1={test_metrics['macro_f1']:.4f}\n",
        flush=True
    )

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"],
                        default="auto")
    parser.add_argument("--data-dir",
                        default="data/facebook_uci/facebook_large")
    args = parser.parse_args()

    set_seed(args.seed)

    device = get_device() if args.device == "auto" else torch.device(args.device)
    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Device:", device, flush=True)

    graph = load_facebook(args.data_dir, seed=args.seed).to(device)

    print(
        f"Dataset: {graph.num_nodes} nodes, "
        f"{graph.num_edges} edges, "
        f"{graph.num_node_features} features",
        flush=True
    )

    results = {}

    for model_name in ("mlp", "gcn"):
        # Same seed and splits for both experiments.
        set_seed(args.seed)

        model_class = MLP if model_name == "mlp" else GCN

        model = model_class(
            input_dim=graph.num_node_features,
            hidden_dim=64,
            num_classes=int(graph.y.max().item()) + 1
        ).to(device)

        print(f"\nTraining {model_name.upper()}...", flush=True)

        results[model_name] = train_model(
            model, graph, model_name, args, output_dir
        )

        del model

    with open(output_dir / "training_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nFinal Comparison")
    print("Model | Test Accuracy | Test Macro-F1")

    for name, result in results.items():
        print(
            f"{name.upper():5} | "
            f"{result['test']['accuracy']:.4f}        | "
            f"{result['test']['macro_f1']:.4f}"
        )

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
