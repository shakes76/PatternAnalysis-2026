"""Train the MLP baseline and two-layer GCN, recording a full epoch history.

Each new training run writes one CSV row per epoch for each model and
saves loss/validation plots automatically when training completes.

Existing best checkpoints are protected by default. To collect fresh plots
without replacing models used in the final report, run:
    python train.py --no-save-checkpoints
"""

import argparse
import copy
import csv
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save figures in non-interactive/headless environments
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from sklearn.metrics import accuracy_score, f1_score

from dataset import load_facebook, split_nodes
from modules import GCN, MLP


SEED = 42
MAX_EPOCHS = 100
PATIENCE = 15
BASE_DIR = Path(__file__).resolve().parent
FIGURES_DIR = BASE_DIR / "figures"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def train_one_epoch(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()

    logits = model(data.x, data.edge_index)
    loss = criterion(logits[data.train_mask], data.y[data.train_mask])

    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def evaluate(model, data, mask):
    model.eval()

    logits = model(data.x, data.edge_index)
    predictions = logits[mask].argmax(dim=1)

    y_true = data.y[mask].cpu().numpy()
    y_pred = predictions.cpu().numpy()

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(
        y_true, y_pred, average="macro", zero_division=0
    )

    return accuracy, macro_f1



def run_experiment(name, model, data, device, save_checkpoint=True):
    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.01, weight_decay=5e-4
    )
    criterion = nn.CrossEntropyLoss()

    best_f1 = -1.0
    best_epoch = 0
    best_state = None
    waiting = 0
    history = []

    num_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    print(f"\n===== {name} =====")
    print(f"Parameters: {num_params}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = FIGURES_DIR / f"{name.lower()}_epoch_history.csv"

    # Write and flush metrics EVERY epoch: a partial trace survives interruption.
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = ["epoch", "train_loss", "val_accuracy", "val_macro_f1"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for epoch in range(1, MAX_EPOCHS + 1):
            loss = train_one_epoch(model, data, optimizer, criterion)
            val_acc, val_f1 = evaluate(model, data, data.val_mask)

            record = {
                "epoch": epoch,
                "train_loss": loss,
                "val_accuracy": val_acc,
                "val_macro_f1": val_f1,
            }
            history.append(record)
            writer.writerow(record)
            csv_file.flush()

            if val_f1 > best_f1 + 1e-4:
                best_f1 = val_f1
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                waiting = 0
            else:
                waiting += 1

            if epoch == 1 or epoch % 10 == 0:
                print(
                    f"Epoch {epoch:03d} | "
                    f"Loss: {loss:.4f} | "
                    f"Val Acc: {val_acc:.4f} | "
                    f"Val Macro-F1: {val_f1:.4f}"
                )

            if waiting >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"Saved full epoch history: {csv_path}")

    # Restore the best validation checkpoint, as in the original training code.
    model.load_state_dict(best_state)
    final_acc, final_f1 = evaluate(model, data, data.val_mask)

    print(f"Best epoch: {best_epoch}")
    print(f"Validation Accuracy: {final_acc:.4f}")
    print(f"Validation Macro-F1: {final_f1:.4f}")

    if save_checkpoint:
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        checkpoint = CHECKPOINT_DIR / f"{name.lower()}_best.pt"
        torch.save(model.state_dict(), checkpoint)
        print(f"Saved checkpoint: {checkpoint}")
    else:
        print("Checkpoint NOT saved (--no-save-checkpoints).")

    return {
        "model": model,
        "accuracy": final_acc,
        "macro_f1": final_f1,
        "best_epoch": best_epoch,
        "history": history,
    }



def plot_training_curves(results):
    """Save plots directly from full per-epoch metrics recorded in this run."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    charts = [
        ("train_loss", "Training cross-entropy loss", "training_loss_full.png"),
        ("val_accuracy", "Validation accuracy", "validation_accuracy_full.png"),
        ("val_macro_f1", "Validation macro-F1", "validation_macro_f1_full.png"),
    ]

    for metric, ylabel, filename in charts:
        fig, ax = plt.subplots(figsize=(8, 5))
        for name, result in results.items():
            rows = result["history"]
            ax.plot(
                [r["epoch"] for r in rows],
                [r[metric] for r in rows],
                label=name,
                linewidth=2,
            )
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} per epoch (seed {SEED})")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()

        output = FIGURES_DIR / filename
        fig.savefig(output, dpi=180)
        plt.close(fig)
        print(f"Saved training figure: {output}")



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-save-checkpoints",
        action="store_true",
        help="Re-train and generate full histories/plots without overwriting saved models",
    )
    parser.add_argument(
        "--overwrite-checkpoints",
        action="store_true",
        help="Explicitly allow replacement of existing model checkpoints",
    )
    args = parser.parse_args()

    if args.no_save_checkpoints and args.overwrite_checkpoints:
        parser.error("Choose only one checkpoint option")

    existing = [
        CHECKPOINT_DIR / f"{name}_best.pt"
        for name in ("mlp", "gcn")
        if (CHECKPOINT_DIR / f"{name}_best.pt").exists()
    ]
    if existing and not args.no_save_checkpoints and not args.overwrite_checkpoints:
        parser.error(
            "Existing checkpoints detected. Use --no-save-checkpoints "
            "to preserve the frozen models, or --overwrite-checkpoints "
            "if you explicitly intend to replace them."
        )

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    data = split_nodes(load_facebook(), seed=SEED).to(device)
    num_features = data.num_node_features
    num_classes = int(data.y.max().item()) + 1

    print("Training:", data.train_mask.sum().item())
    print("Validation:", data.val_mask.sum().item())
    print("Testing:", data.test_mask.sum().item())

    # Same model definitions, optimiser and random seeds as the original run.
    set_seed(SEED)
    mlp_result = run_experiment(
        "MLP", MLP(num_features, num_classes), data, device,
        save_checkpoint=not args.no_save_checkpoints,
    )

    set_seed(SEED)
    gcn_result = run_experiment(
        "GCN", GCN(num_features, num_classes), data, device,
        save_checkpoint=not args.no_save_checkpoints,
    )

    results = {"MLP": mlp_result, "GCN": gcn_result}
    plot_training_curves(results)

    print("\n===== Validation Comparison =====")
    print(f"{'Model':<8} {'Accuracy':<12} {'Macro-F1':<12}")
    for name, result in results.items():
        print(
            f"{name:<8} "
            f"{result['accuracy']:<12.4f} "
            f"{result['macro_f1']:<12.4f}"
        )


if __name__ == "__main__":
    main()
