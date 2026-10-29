"""COMP3710 TimeGAN V5: causal next-step mid-price movement classifier.

A diagnostic baseline, NOT a generative TimeGAN head.
Requires dataset.py and dataset_v4.py alongside this script.

Labels: 0 = unchanged, 1 = up, 2 = down.
Input: past 24 normalized 41-dimensional observations.
Target: movement from final input observation to the following one.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dataset import chronological_split, load_lobster
from dataset_v4 import prepare_dataset_v4


class TickHead(nn.Module):
    def __init__(self, feature_dim=41, hidden_dim=32, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_dim, 3)

    def forward(self, x):
        _, hidden = self.gru(x)
        return self.head(hidden[-1])


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_labels(mid2, indices, seq_len, stride):
    """Targets use only transitions inside their chronological partition."""
    values = mid2[indices]
    starts = np.arange(0, len(values) - seq_len, stride, dtype=np.int64)
    last_input = starts + seq_len - 1
    target = starts + seq_len
    movement = values[target] - values[last_input]
    if not np.all(movement % 100 == 0):
        raise ValueError("Non-integer tick movement detected")
    y = np.zeros(len(starts), dtype=np.int64)
    y[movement > 0] = 1
    y[movement < 0] = 2
    return y


def metrics(true, predicted):
    cm = np.zeros((3, 3), dtype=np.int64)
    for actual, guess in zip(true, predicted):
        cm[int(actual), int(guess)] += 1
    per_class = []
    for i, name in enumerate(("unchanged", "up", "down")):
        tp = int(cm[i, i])
        support = int(cm[i].sum())
        predicted_count = int(cm[:, i].sum())
        recall = tp / support if support else 0.0
        precision = tp / predicted_count if predicted_count else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append({"name": name, "support": support,
                          "precision": precision, "recall": recall, "f1": f1})
    return {
        "accuracy": float(np.mean(true == predicted)),
        "macro_f1": float(np.mean([entry["f1"] for entry in per_class])),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    targets, predictions = [], []
    for features, labels in loader:
        logits = model(features.to(device))
        targets.append(labels.numpy())
        predictions.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(targets), np.concatenate(predictions)


def show_metrics(title, result):
    print(f"\n=== {title} ===")
    print(f"Accuracy: {result['accuracy']:.5f}")
    print(f"Macro-F1: {result['macro_f1']:.5f}")
    for c in result["per_class"]:
        print(f"{c['name']:>9}: precision={c['precision']:.5f} "
              f"recall={c['recall']:.5f} f1={c['f1']:.5f} "
              f"support={c['support']}")
    print("Confusion matrix (rows=true; columns=pred; classes=0/1/2):")
    for row in result["confusion_matrix"]:
        print(" ", row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs_v5_tick_head")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error("epochs, batch-size, lr must be positive")
    set_seed(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seq_len, stride = 24, 24
    data = prepare_dataset_v4(args.data_dir, seq_len=seq_len, stride=stride)
    orderbook, _ = load_lobster(args.data_dir)
    raw = orderbook.to_numpy()
    mid2 = raw[:, 0].astype(np.int64) + raw[:, 2].astype(np.int64)
    indices = chronological_split(np.arange(len(mid2)))

    arrays = {}
    for name, ids in zip(("train", "val", "test"), indices):
        x = np.asarray(data[name], dtype=np.float32)
        y = make_labels(mid2, ids, seq_len, stride)
        if x.shape != (len(y), seq_len, 41):
            raise RuntimeError(f"{name} input-label mismatch: {x.shape}, labels={len(y)}")
        if not np.isfinite(x).all():
            raise RuntimeError(f"Non-finite values in {name} inputs")
        arrays[name] = (x, y)
        print(f"{name}: X={x.shape}, y={y.shape}, counts={np.bincount(y, minlength=3).tolist()}")

    def loader(name, shuffle=False):
        x, y = arrays[name]
        return DataLoader(
            TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
            batch_size=args.batch_size,
            shuffle=shuffle,
        )

    train_loader = loader("train", shuffle=True)
    val_loader = loader("val")
    test_loader = loader("test")

    # Moderate weighting avoids completely overwhelming the majority class.
    counts = np.bincount(arrays["train"][1], minlength=3).astype(np.float64)
    weights = np.sqrt(counts.sum() / np.maximum(counts, 1.0))
    weights /= weights.mean()
    print("Class weights (train only):", weights.tolist())

    model = TickHead(feature_dim=41, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))

    best_f1 = -1.0
    best_epoch = -1
    history = []
    checkpoint_path = output_dir / "best_tick_head.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        n_examples = 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            logits = model(features)
            loss = criterion(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(labels)
            n_examples += len(labels)

        y_val, pred_val = predict(model, val_loader, device)
        result = metrics(y_val, pred_val)
        train_loss = loss_sum / n_examples
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_macro_f1": result["macro_f1"],
                        "val_accuracy": result["accuracy"]})
        print(f"Epoch {epoch}/{args.epochs} | train CE={train_loss:.5f} | "
              f"val Macro-F1={result['macro_f1']:.5f} | val Acc={result['accuracy']:.5f}", flush=True)
        if result["macro_f1"] > best_f1:
            best_f1 = result["macro_f1"]
            best_epoch = epoch
            torch.save({
                "model_state_dict": model.state_dict(),
                "feature_dim": 41, "hidden_dim": args.hidden_dim,
                "seed": args.seed, "epoch": epoch,
                "val_macro_f1": best_f1,
            }, checkpoint_path)

    # Evaluate once on test after selecting by validation Macro-F1.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    y_val, pred_val = predict(model, val_loader, device)
    y_test, pred_test = predict(model, test_loader, device)
    baseline_val = np.zeros_like(y_val)
    baseline_test = np.zeros_like(y_test)

    results = {
        "best_epoch": best_epoch, "best_val_macro_f1": best_f1,
        "config": vars(args), "history": history,
        "validation_model": metrics(y_val, pred_val),
        "validation_always_unchanged": metrics(y_val, baseline_val),
        "test_model": metrics(y_test, pred_test),
        "test_always_unchanged": metrics(y_test, baseline_test),
    }
    for name in ("validation_model", "validation_always_unchanged",
                 "test_model", "test_always_unchanged"):
        show_metrics(name, results[name])
    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nBest epoch:", best_epoch)
    print("Saved checkpoint:", checkpoint_path)
    print("Saved metrics:", output_dir / "results.json")
    print("This is a next-step prediction probe, NOT a trained generative TimeGAN tick head.")


if __name__ == "__main__":
    main()
