"""
Train, validate, test and save the MLP baseline and the GCN.

Training is full-batch: the whole graph fits in memory, so every epoch does one
forward pass over all nodes, and the loss is computed on training nodes only.
The checkpoint with the best validation accuracy is kept (early stopping), then
evaluated once on the held-out test nodes.

Usage (from this folder):
    python train.py --model both
    python train.py --model gcn --epochs 300 --lr 0.01
"""

import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")  # write figures to file; no display needed (e.g. on Rangpur)
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, f1_score, roc_auc_score

from dataset import CLASS_NAMES, load_graph
from modules import build_model, count_parameters

HERE = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(HERE, "checkpoints")
FIGURE_DIR = os.path.join(HERE, "figures")
RESULTS_DIR = os.path.join(HERE, "results")


def train_epoch(model, data, optimiser):
    """One full-batch gradient step. Only training-node labels enter the loss."""
    model.train()
    optimiser.zero_grad()
    logits = model(data.x, data.edge_index)
    loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimiser.step()
    return loss.item()


@torch.no_grad()
def predict_proba(model, data):
    """Class probabilities for every node (dropout off)."""
    model.eval()
    return F.softmax(model(data.x, data.edge_index), dim=1)


@torch.no_grad()
def split_loss_acc(model, data, mask):
    """Cross-entropy loss and accuracy on the nodes selected by ``mask``."""
    model.eval()
    logits = model(data.x, data.edge_index)
    loss = F.cross_entropy(logits[mask], data.y[mask]).item()
    acc = (logits[mask].argmax(dim=1) == data.y[mask]).float().mean().item()
    return loss, acc


def compute_metrics(probs, y):
    """Accuracy, per-class precision/recall/F1, macro-F1 and one-vs-rest AUROC."""
    y_true, p = y.cpu().numpy(), probs.cpu().numpy()
    y_pred = p.argmax(axis=1)
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES,
                                   output_dict=True, zero_division=0)
    return {
        "accuracy": report["accuracy"],
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "macro_auroc": roc_auc_score(y_true, p, multi_class="ovr", average="macro"),
        "per_class": {c: {k: report[c][k] for k in ("precision", "recall", "f1-score", "support")}
                      for c in CLASS_NAMES},
    }


@torch.no_grad()
def inference_latency_ms(model, data, repeats=20):
    """Median wall-clock time of one full-graph forward pass, in milliseconds."""
    model.eval()
    model(data.x, data.edge_index)  # warm-up
    times = []
    for _ in range(repeats):
        if data.x.is_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        model(data.x, data.edge_index)
        if data.x.is_cuda:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    return sorted(times)[len(times) // 2]


def plot_curves(history, name):
    """Save loss and accuracy curves (train vs validation) for one model."""
    os.makedirs(FIGURE_DIR, exist_ok=True)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax_loss.plot(epochs, history["train_loss"], label="train")
    ax_loss.plot(epochs, history["val_loss"], label="validation")
    ax_loss.set(xlabel="epoch", ylabel="cross-entropy loss", title=f"{name.upper()} loss")
    ax_acc.plot(epochs, history["train_acc"], label="train")
    ax_acc.plot(epochs, history["val_acc"], label="validation")
    ax_acc.set(xlabel="epoch", ylabel="accuracy", title=f"{name.upper()} accuracy")
    for ax in (ax_loss, ax_acc):
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    path = os.path.join(FIGURE_DIR, f"{name}_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def train_model(name, data, args, device):
    """Train one model with early stopping, save the best weights, return test results."""
    torch.manual_seed(args.seed)
    model = build_model(name, data.num_node_features, len(CLASS_NAMES),
                        args.hidden, args.layers, args.dropout).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc, best_epoch, best_state = -1.0, 0, None
    start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        train_epoch(model, data, optimiser)
        # Re-evaluate train loss/acc with dropout off so both curves are comparable.
        train_loss, train_acc = split_loss_acc(model, data, data.train_mask)
        val_loss, val_acc = split_loss_acc(model, data, data.val_mask)
        for key, value in zip(history, (train_loss, val_loss, train_acc, val_acc)):
            history[key].append(value)

        if val_acc > best_val_acc:
            best_val_acc, best_epoch = val_acc, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= args.patience:
            print(f"[{name}] early stop at epoch {epoch} (best val acc {best_val_acc:.4f} @ {best_epoch})")
            break

        if epoch % 25 == 0 or epoch == 1:
            print(f"[{name}] epoch {epoch:4d} | train loss {train_loss:.4f} acc {train_acc:.4f} "
                  f"| val loss {val_loss:.4f} acc {val_acc:.4f}")

    train_seconds = time.perf_counter() - start
    model.load_state_dict(best_state)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{name}.pt")
    torch.save({"state_dict": best_state, "hparams": {
        "hidden_dim": args.hidden, "num_layers": args.layers, "dropout": args.dropout},
        "seed": args.seed}, ckpt_path)

    probs = predict_proba(model, data)
    metrics = compute_metrics(probs[data.test_mask], data.y[data.test_mask])
    metrics.update({
        "model": name,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "parameters": count_parameters(model),
        "train_seconds": train_seconds,
        "inference_ms_full_graph": inference_latency_ms(model, data),
        "peak_gpu_mb": (torch.cuda.max_memory_allocated(device) / 2**20
                        if device.type == "cuda" else None),
        "device": str(device),
    })

    curves = plot_curves(history, name)
    print(f"[{name}] test acc {metrics['accuracy']:.4f} | macro-F1 {metrics['macro_f1']:.4f} "
          f"| macro-AUROC {metrics['macro_auroc']:.4f} | saved {ckpt_path}, {curves}")
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=["mlp", "gcn", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=50, help="epochs without val improvement before stopping")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42, help="seed for weights and the data split")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_graph(seed=args.seed).to(device)
    print(f"Device: {device} | {data}")

    names = ["mlp", "gcn"] if args.model == "both" else [args.model]
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for name in names:
        metrics = train_model(name, data, args, device)
        with open(os.path.join(RESULTS_DIR, f"{name}_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
