"""
Choose model settings by grid search, using the VALIDATION set only.

Usage:
    python tune.py --model mlp               # phase 1: every combination, seed 0
    python tune.py --model gcn
    python tune.py --model gcn --confirm 3   # phase 2: top 3 + starting point, seeds 0, 1, 2
    python tune.py --model lp                # label propagation reference (seconds)

Phase 1 trains one model (seed 0) for every combination of layers x weight
decay x dropout x hidden size. A single seed can be lucky, so phase 2 reruns
the best few combinations, plus the starting point (2 layers, wd 5e-4,
dropout 0.5, hidden 64), with three seeds and reports mean ± std.

Each run uses the same training loop and early stopping as train.py, but only
validation metrics are computed: test labels are never touched, so the test
set stays a fair final check. Results are appended to
results/tune_{model}.jsonl, one line per run; runs already in the file are
skipped, so an interrupted search can simply be restarted.

Label propagation has no weights to train: it spreads the training labels
along the links, keeping a share alpha of what arrives from the neighbours at
each of num_layers steps. Only alpha and the number of steps are searched.
"""
import argparse
import itertools
import json
import os
import statistics
import time

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch_geometric.nn import LabelPropagation

from dataset import load_data
from train import HERE, MODELS, train_step

START = {"layers": 2, "weight_decay": 5e-4, "dropout": 0.5, "hidden": 64}
METRICS = ("val_acc", "val_macro_f1", "val_loss", "val_ece")


def ece(probs, y, n_bins=15):
    """Expected calibration error: the average gap between confidence and
    accuracy, over equal-width confidence bins weighted by their size."""
    conf, pred = probs.max(dim=1)
    correct = (pred == y).float()
    edges = torch.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        if in_bin.any():
            gap = (correct[in_bin].mean() - conf[in_bin].mean()).abs()
            total += in_bin.float().mean().item() * gap.item()
    return total


def label_propagation(data, num_layers, alpha):
    """Spread the TRAINING labels along the links and return log-probabilities
    for every page, so the result can be scored like a model's output."""
    lp = LabelPropagation(num_layers=num_layers, alpha=alpha)
    scores = lp(data.y, data.edge_index, mask=data.train_mask)
    probs = scores / scores.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return probs.clamp_min(1e-12).log()


def val_metrics(logits, data):
    """Validation accuracy, macro-F1, loss and calibration error."""
    val = data.val_mask
    probs = logits[val].softmax(dim=1)
    pred = probs.argmax(dim=1)
    return {"val_acc": (pred == data.y[val]).float().mean().item(),
            "val_macro_f1": f1_score(data.y[val], pred, average="macro"),
            "val_loss": F.cross_entropy(logits[val], data.y[val]).item(),
            "val_ece": ece(probs, data.y[val])}


def run(config, data, args):
    """Train one model with early stopping on validation loss and return its
    validation metrics at the best epoch."""
    torch.manual_seed(config["seed"])
    model = MODELS[config["model"]](in_dim=data.num_features, hidden=config["hidden"],
                                    n_classes=int(data.y.max()) + 1,
                                    num_layers=config["layers"], dropout=config["dropout"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"],
                                 weight_decay=config["weight_decay"])
    val = data.val_mask
    best_val_loss, best_epoch, best_out = float("inf"), 0, None
    start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        train_step(model, data, optimizer)
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
        val_loss = F.cross_entropy(out[val], data.y[val]).item()
        if val_loss < best_val_loss:
            best_val_loss, best_epoch, best_out = val_loss, epoch, out
        elif epoch - best_epoch >= args.patience:
            break

    return {**val_metrics(best_out, data),
            "best_epoch": best_epoch,
            "params": sum(p.numel() for p in model.parameters()),
            "train_time_s": time.perf_counter() - start}


def run_lp(config, data):
    start = time.perf_counter()
    logits = label_propagation(data, config["layers"], config["alpha"])
    return {**val_metrics(logits, data), "params": 0,
            "train_time_s": time.perf_counter() - start}


def describe(config):
    if config["model"] == "lp":
        return f"L={config['layers']} alpha={config['alpha']:g}"
    return (f"L={config['layers']} wd={config['weight_decay']:g} "
            f"drop={config['dropout']:g} h={config['hidden']}")


def same_settings(a, b):
    return all(a[k] == b[k] for k in ("layers", "weight_decay", "dropout", "hidden"))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--model", choices=[*MODELS, "lp"], default="gcn")
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="default: 2 3 4 (lp: 10 50 propagation steps)")
    parser.add_argument("--alpha", type=float, nargs="+", default=[0.5, 0.7, 0.9, 0.99],
                        help="lp only: share kept from the neighbours at each step")
    parser.add_argument("--weight-decay", type=float, nargs="+", default=[0.0, 5e-5, 5e-4])
    parser.add_argument("--dropout", type=float, nargs="+", default=[0.0, 0.5])
    parser.add_argument("--hidden", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--confirm", type=int, default=0, metavar="K",
                        help="phase 2: rerun the top K phase-1 settings with seeds 0, 1, 2")
    args = parser.parse_args()
    if args.layers is None:
        args.layers = [10, 50] if args.model == "lp" else [2, 3, 4]

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    path = os.path.join(HERE, "results", f"tune_{args.model}.jsonl")
    done = []
    if os.path.exists(path):
        with open(path) as f:
            done = [json.loads(line) for line in f if line.strip()]

    if args.confirm:
        # Rank phase-1 runs (seed 0) by validation accuracy, keep the top K
        phase1 = sorted((r for r in done if r["config"]["seed"] == 0),
                        key=lambda r: r["val_acc"], reverse=True)
        chosen = [r["config"] for r in phase1[:args.confirm]]
        if not any(same_settings(c, START) for c in chosen):
            chosen.append({**chosen[0], **START})
        configs = [{**c, "seed": s} for c in chosen for s in (0, 1, 2)]
    elif args.model == "lp":
        # No random start, so one run per setting is enough
        configs = [{"model": "lp", "layers": L, "alpha": a, "seed": 0}
                   for L, a in itertools.product(args.layers, args.alpha)]
    else:
        configs = [{"model": args.model, "layers": L, "weight_decay": wd,
                    "dropout": p, "hidden": h, "lr": args.lr, "seed": 0}
                   for L, wd, p, h in itertools.product(
                       args.layers, args.weight_decay, args.dropout, args.hidden)]

    data = load_data(seed=42)
    for config in configs:
        if any(r["config"] == config for r in done):
            continue
        result = {"config": config, **(run_lp(config, data) if args.model == "lp"
                                       else run(config, data, args))}
        done.append(result)
        with open(path, "a") as f:
            f.write(json.dumps(result) + "\n")
        print(f"{describe(config)} seed {config['seed']}: val acc {result['val_acc']:.4f}"
              f"  ECE {result['val_ece']:.4f}  ({result['train_time_s']:.0f}s)")

    # Ranking: mean over whatever seeds have been run for each setting
    groups = {}
    for r in done:
        groups.setdefault(describe(r["config"]), []).append(r)
    rows = sorted(groups.items(), key=lambda kv: -statistics.mean(r["val_acc"] for r in kv[1]))
    print(f"\n=== {args.model.upper()} settings ranked by validation accuracy ===")
    for name, runs in rows:
        cells = []
        for metric in METRICS:
            values = [r[metric] for r in runs]
            std = statistics.stdev(values) if len(values) > 1 else 0.0
            cells.append(f"{metric} {statistics.mean(values):.4f} ± {std:.4f}")
        print(f"  {name:28s} seeds {len(runs)}  " + "  ".join(cells))


if __name__ == "__main__":
    main()
