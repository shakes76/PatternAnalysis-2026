import argparse
import copy
import json
import os
import time

import matplotlib
matplotlib.use("Agg")                    # draw to files, no display needed on Rangpur
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score

from dataset import get_data
from modules import GCN, MLP


def to_device(d, device):
    """Move every tensor in the data dict (including the sparse adjacency) to the device."""
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in d.items()}


def evaluate(model, d, idx):
    """Loss, accuracy and softmax probabilities for the nodes in idx."""
    model.eval()
    with torch.no_grad():
        logits = model(d["x"], d["adj"])[idx]
        loss = F.cross_entropy(logits, d["y"][idx]).item()
        probs = F.softmax(logits, dim=1)
        acc = (probs.argmax(1) == d["y"][idx]).float().mean().item()
    return loss, acc, probs


def train_model(model, d, epochs=200, lr=0.01, weight_decay=5e-4):
    """Train one model with Adam and cross-entropy on the training nodes only.

    The weights from the epoch with the best VALIDATION accuracy are restored
    at the end. Returns (history, train_seconds, peak_gpu_mb).
    """
    device = next(model.parameters()).device
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val, best_state = -1.0, None

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.time()

    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(d["x"], d["adj"])
        loss = F.cross_entropy(out[d["train_idx"]], d["y"][d["train_idx"]])
        loss.backward()
        opt.step()

        val_loss, val_acc, _ = evaluate(model, d, d["val_idx"])
        history["train_loss"].append(loss.item())
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        if val_acc > best_val:
            best_val = val_acc
            best_state = copy.deepcopy(model.state_dict())

    if device.type == "cuda":
        torch.cuda.synchronize()          # wait for the GPU before stopping the clock
    train_seconds = time.time() - start
    peak_mb = torch.cuda.max_memory_allocated() / 1e6 if device.type == "cuda" else None

    model.load_state_dict(best_state)
    return history, train_seconds, peak_mb

def expected_calibration_error(conf, correct, bins=10):
    """ECE: average gap between stated confidence and actual accuracy, in confidence bins."""
    edges = torch.linspace(0, 1, bins + 1, device=conf.device)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        if in_bin.any():
            gap = (correct[in_bin].float().mean() - conf[in_bin].mean()).abs()
            ece += in_bin.float().mean() * gap
    return float(ece)


def full_metrics(model, d, idx):
    """Classification and confidence metrics for the nodes in idx (use the TEST set)."""
    _, acc, probs = evaluate(model, d, idx)
    conf, pred = probs.max(1)
    y = d["y"][idx]
    correct = pred == y

    y_np, pred_np = y.cpu().numpy(), pred.cpu().numpy()
    p_np = probs.double().cpu().numpy()
    p_np = p_np / p_np.sum(axis=1, keepdims=True)       # rows sum to exactly 1
    precision, recall, _, _ = precision_recall_fscore_support(y_np, pred_np, zero_division=0)

    metrics = {
        "accuracy": acc,
        "macro_f1": float(f1_score(y_np, pred_np, average="macro")),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "auroc_ovr": float(roc_auc_score(y_np, p_np, multi_class="ovr", average="macro")),
        "conf_correct": conf[correct].mean().item(),
        "conf_wrong": conf[~correct].mean().item(),
        # errors are the positive class; low confidence should flag them
        "error_auroc": float(roc_auc_score((~correct).cpu().numpy(), (1 - conf).cpu().numpy())),
        "ece": expected_calibration_error(conf, correct),
    }
    metrics["conf_gap"] = metrics["conf_correct"] - metrics["conf_wrong"]
    return metrics, conf, correct


def reject_rule(val_conf, val_correct, test_conf, test_correct, target_acc=0.95):
    """Pick the lowest confidence threshold whose VALIDATION accuracy meets the target
    (lowest = most pages automated), then report that threshold's result on TEST."""
    for t in torch.linspace(0.30, 0.99, 70).tolist():
        keep_val = val_conf >= t
        if keep_val.sum() < 50:                      # too few pages left to trust
            break
        if val_correct[keep_val].float().mean() >= target_acc:
            keep_test = test_conf >= t
            return {
                "threshold": round(t, 2),
                "val_accuracy": val_correct[keep_val].float().mean().item(),
                "val_coverage": keep_val.float().mean().item(),
                "test_accuracy": test_correct[keep_test].float().mean().item(),
                "test_coverage": keep_test.float().mean().item(),
            }
    return None                                      # target never reached


def threshold_table(val_conf, val_correct, test_conf, test_correct):
    """Accuracy and coverage at a fixed set of thresholds, for the README."""
    rows = []
    for t in (0.5, 0.6, 0.7, 0.8, 0.9):
        kv, kt = val_conf >= t, test_conf >= t
        rows.append({
            "threshold": t,
            "val_accuracy": val_correct[kv].float().mean().item(),
            "val_coverage": kv.float().mean().item(),
            "test_accuracy": test_correct[kt].float().mean().item(),
            "test_coverage": kt.float().mean().item(),
        })
    return rows


def inference_ms(model, d, runs=20):
    """Average time (ms) for one forward pass over the whole graph."""
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        model(d["x"], d["adj"])                      # warm-up
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        for _ in range(runs):
            model(d["x"], d["adj"])
        if device.type == "cuda":
            torch.cuda.synchronize()
    return (time.time() - start) / runs * 1000


def plot_history(histories, path):
    """Training/validation loss and validation accuracy curves for all models."""
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))
    colours = {"MLP": "tab:blue", "GCN": "tab:orange"}
    for name, h in histories.items():
        c = colours[name]
        ax_loss.plot(h["train_loss"], color=c, label=f"{name} train")
        ax_loss.plot(h["val_loss"], "--", color=c, label=f"{name} val")
        ax_acc.plot(h["val_acc"], color=c, label=name)
    ax_loss.set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Loss")
    ax_acc.set(xlabel="Epoch", ylabel="Validation accuracy", title="Validation accuracy")
    ax_loss.legend()
    ax_acc.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def print_summary(results):
    names = list(results)
    print("\n" + "=" * 62)
    print(f"{'Metric':<28}" + "".join(f"{n:>14}" for n in names))
    print("-" * 62)
    for key, label in [("accuracy", "Test accuracy"), ("macro_f1", "Test macro-F1"),
                       ("auroc_ovr", "Class AUROC (OvR)"), ("conf_correct", "Conf. (correct)"),
                       ("conf_wrong", "Conf. (wrong)"), ("conf_gap", "Confidence gap"),
                       ("error_auroc", "Error-detection AUROC"), ("ece", "ECE")]:
        print(f"{label:<28}" + "".join(f"{results[n][key]:>14.3f}" for n in names))
    print(f"{'Parameters':<28}" + "".join(f"{results[n]['params']:>14,}" for n in names))
    print(f"{'Train time (s)':<28}" + "".join(f"{results[n]['train_seconds']:>14.1f}" for n in names))
    print(f"{'Inference (ms/pass)':<28}" + "".join(f"{results[n]['inference_ms']:>14.1f}" for n in names))
    peaks = [results[n]["peak_gpu_mb"] for n in names]
    print(f"{'Peak GPU memory (MB)':<28}" + "".join(f"{'n/a' if p is None else f'{p:.1f}':>14}" for p in peaks))
    print("-" * 62)
    for n in names:
        r = results[n]["reject_rule"]
        if r is None:
            print(f"{n}: no threshold reached the target accuracy on validation")
        else:
            print(f"{n} reject rule: threshold {r['threshold']} -> test accuracy "
                  f"{r['test_accuracy']:.3f}, coverage {r['test_coverage']:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Train and compare the MLP baseline and the GCN.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=0, help="model initialisation seed")
    parser.add_argument("--split-seed", type=int, default=0, help="train/val/test split seed")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = to_device(get_data(root=args.data_root, seed=args.split_seed), device)
    in_dim, n_classes = d["x"].size(1), d["num_classes"]

    results, histories = {}, {}
    for name, Model in (("MLP", MLP), ("GCN", GCN)):
        torch.manual_seed(args.seed)
        model = Model(in_dim, args.hidden, n_classes,
                      num_layers=args.layers, dropout=args.dropout).to(device)
        print(f"\nTraining {name} ({sum(p.numel() for p in model.parameters()):,} parameters)...")
        history, secs, peak = train_model(model, d, args.epochs, args.lr, args.weight_decay)
        torch.save(model.state_dict(), os.path.join(args.out, f"{name}.pt"))

        metrics, test_conf, test_correct = full_metrics(model, d, d["test_idx"])
        _, _, val_probs = evaluate(model, d, d["val_idx"])
        val_conf, val_pred = val_probs.max(1)
        val_correct = val_pred == d["y"][d["val_idx"]]

        metrics["reject_rule"] = reject_rule(val_conf, val_correct, test_conf, test_correct)
        metrics["threshold_table"] = threshold_table(val_conf, val_correct, test_conf, test_correct)
        metrics.update(params=sum(p.numel() for p in model.parameters()),
                       train_seconds=secs, peak_gpu_mb=peak,
                       inference_ms=inference_ms(model, d),
                       best_val_acc=max(history["val_acc"]))
        results[name], histories[name] = metrics, history

    plot_history(histories, os.path.join(args.out, "training_curves.png"))
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    print_summary(results)
    print(f"\nSaved weights, curves and results.json in '{args.out}/'")


if __name__ == "__main__":
    main()