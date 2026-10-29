"""
Train, validate, test and save a node classifier (MLP baseline or GCN).

Usage:
    python train.py --model mlp                 # 3 layers, seeds 0, 1, 2
    python train.py --model gcn                 # 4 layers, seeds 0, 1, 2
    python train.py --model gcn --layers 2 --seeds 0   # override depth

Training is transductive and semi-supervised: the whole graph goes through
the model every epoch, but the loss only uses the labels of training nodes.
Validation labels are used only to pick the best epoch (early stopping), and
test labels are used once, at the very end, to report the final result.

Each seed changes the random weight initialisation and dropout, while the
train/val/test split stays fixed (--split-seed), so differences between
seeds show how much results depend on luck.
"""
import argparse
import json
import os
import statistics
import time

import matplotlib
matplotlib.use("Agg")  # save plots to files without needing a screen
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from dataset import load_data
from modules import GCN, MLP

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = {"mlp": MLP, "gcn": GCN}
# Best depth per model, chosen on the VALIDATION set by grid search over
# layers x weight decay x dropout x hidden size (confirmed over 3 seeds)
DEFAULT_LAYERS = {"mlp": 3, "gcn": 4}


def train_step(model, data, optimizer):
    """One epoch: predict every node, compute the loss on training nodes
    only, and update the weights. Returns the training loss."""
    model.train()                 # dropout ON
    optimizer.zero_grad()         # clear gradients left over from last step
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()               # work out how each weight should change
    optimizer.step()              # change the weights a little
    return loss.item()


@torch.no_grad()  # no learning here, so skip the gradient bookkeeping
def evaluate(model, data, mask):
    """Loss, accuracy and macro-F1 on the nodes selected by `mask`."""
    model.eval()                  # dropout OFF
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[mask], data.y[mask]).item()
    pred = out[mask].argmax(dim=1)   # highest score = predicted class
    true = data.y[mask]
    acc = (pred == true).float().mean().item()
    macro_f1 = f1_score(true, pred, average="macro")
    return loss, acc, macro_f1


def plot_curves(history, path, title):
    """Save training/validation loss and accuracy curves side by side."""
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 4))
    ax_loss.plot(history["train_loss"], label="train")
    ax_loss.plot(history["val_loss"], label="validation")
    ax_loss.set(xlabel="epoch", ylabel="cross-entropy loss", title="Loss")
    ax_acc.plot(history["train_acc"], label="train")
    ax_acc.plot(history["val_acc"], label="validation")
    ax_acc.set(xlabel="epoch", ylabel="accuracy", title="Accuracy")
    for ax in (ax_loss, ax_acc):
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def train_one(args, data, seed):
    """Train one model from one random seed, save it, and return its
    final validation/test metrics as a dict."""
    torch.manual_seed(seed)
    model = MODELS[args.model](in_dim=data.num_features, hidden=args.hidden,
                               n_classes=int(data.y.max()) + 1,
                               num_layers=args.layers, dropout=args.dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss, best_epoch, best_state = float("inf"), 0, None
    start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        train_step(model, data, optimizer)
        # Re-measure training loss/accuracy with dropout off, so the train
        # and validation curves are directly comparable
        train_loss, train_acc, _ = evaluate(model, data, data.train_mask)
        val_loss, val_acc, _ = evaluate(model, data, data.val_mask)
        for key, value in zip(history, (train_loss, train_acc, val_loss, val_acc)):
            history[key].append(value)

        # Early stopping on VALIDATION loss (never test): remember the best
        # weights so far, and stop if they have not improved for a while
        if val_loss < best_val_loss:
            best_val_loss, best_epoch = val_loss, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= args.patience:
            break

        if epoch % 25 == 0:
            print(f"epoch {epoch:3d}  train loss {train_loss:.3f} acc {train_acc:.3f}"
                  f"  |  val loss {val_loss:.3f} acc {val_acc:.3f}")

    train_time = time.perf_counter() - start

    # Restore the best epoch's weights, then touch the test set exactly once
    model.load_state_dict(best_state)
    _, val_acc, val_f1 = evaluate(model, data, data.val_mask)
    _, test_acc, test_f1 = evaluate(model, data, data.test_mask)

    print(f"\n{args.model.upper()} (seed {seed}): best epoch {best_epoch}, "
          f"trained {epoch} epochs in {train_time:.1f}s")
    print(f"  validation  accuracy {val_acc:.4f}  macro-F1 {val_f1:.4f}")
    print(f"  test        accuracy {test_acc:.4f}  macro-F1 {test_f1:.4f}")

    # Save weights (ignored by git) and training curves (for the README)
    name = f"{args.model}_seed{seed}"
    torch.save(model.state_dict(), os.path.join(HERE, "checkpoints", f"{name}.pt"))
    plot_curves(history, os.path.join(HERE, "figures", f"{name}_curves.png"),
                f"{args.model.upper()} training curves (seed {seed})")
    print(f"  saved checkpoints/{name}.pt and figures/{name}_curves.png\n")

    return {"seed": seed, "best_epoch": best_epoch, "train_time_s": train_time,
            "val_acc": val_acc, "val_macro_f1": val_f1,
            "test_acc": test_acc, "test_macro_f1": test_f1}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--model", choices=MODELS, default="gcn")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                        help="one run per seed (weight initialisation and dropout)")
    parser.add_argument("--split-seed", type=int, default=42,
                        help="seed for the train/val/test split (keep fixed)")
    parser.add_argument("--layers", type=int, default=None,
                        help="number of layers (default: 3 for mlp, 4 for gcn)")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=1000,
                        help="maximum number of epochs")
    parser.add_argument("--patience", type=int, default=50,
                        help="stop after this many epochs without "
                             "validation-loss improvement")
    args = parser.parse_args()
    if args.layers is None:
        args.layers = DEFAULT_LAYERS[args.model]

    for folder in ("checkpoints", "figures", "results"):
        os.makedirs(os.path.join(HERE, folder), exist_ok=True)

    # Same split for every seed, so all runs are scored on the same test pages
    data = load_data(seed=args.split_seed)
    runs = [train_one(args, data, seed) for seed in args.seeds]

    # Summary over seeds: mean ± standard deviation of each metric
    print(f"=== {args.model.upper()} summary over seeds {args.seeds} ===")
    summary = {}
    for metric in ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1",
                   "train_time_s"):
        values = [run[metric] for run in runs]
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        summary[metric] = {"mean": mean, "std": std}
        print(f"  {metric:14s} {mean:.4f} ± {std:.4f}")

    # Save every run plus the summary, so the README numbers can be traced
    path = os.path.join(HERE, "results", f"{args.model}_results.json")
    with open(path, "w") as f:
        json.dump({"args": vars(args), "runs": runs, "summary": summary}, f, indent=2)
    print(f"  saved results/{args.model}_results.json")


if __name__ == "__main__":
    main()
