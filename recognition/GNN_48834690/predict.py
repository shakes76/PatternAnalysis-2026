"""
Inference and analysis with the trained MLP baseline and GCN.

Loads the saved checkpoints from ``train.py`` and, on the held-out test nodes:
  1. prints test metrics for both models;
  2. plots a 2D t-SNE of each model's hidden embeddings, coloured by true class;
  3. measures calibration (reliability diagram + expected calibration error)
     and whether confidence separates correct from wrong predictions;
  4. picks a confidence threshold on the *validation* nodes (so the test set
     stays untouched) above which predictions are automated and below which
     they are routed to a human reviewer, then reports how that rule does on test;
  5. lists the GCN's most confident test errors for the failure-case autopsy.

Usage (from this folder, after training):
    python predict.py
    python predict.py --target-accuracy 0.99 --num-failures 5
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score

from dataset import CLASS_NAMES, load_graph, load_page_info
from modules import build_model
from train import CHECKPOINT_DIR, FIGURE_DIR, RESULTS_DIR, compute_metrics, predict_proba

MODEL_NAMES = ["mlp", "gcn"]
CLASS_COLOURS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


def load_trained(name, data, device):
    """Rebuild a model from its checkpoint's hyperparameters and load the weights."""
    ckpt = torch.load(os.path.join(CHECKPOINT_DIR, f"{name}.pt"), map_location=device, weights_only=True)
    model = build_model(name, data.num_node_features, len(CLASS_NAMES), **ckpt["hparams"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()


def expected_calibration_error(conf, correct, n_bins=15):
    """ECE: average |accuracy - confidence| over equal-width confidence bins, weighted by bin size."""
    edges = np.linspace(0, 1, n_bins + 1)
    ece, bins = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        if in_bin.any():
            acc, avg_conf = correct[in_bin].mean(), conf[in_bin].mean()
            ece += in_bin.mean() * abs(acc - avg_conf)
            bins.append((avg_conf, acc, int(in_bin.sum())))
    return ece, bins


def coverage_curve(conf, correct):
    """Risk-coverage curve: automate the k most confident nodes, for every k.

    Returns the fraction of nodes automated and the accuracy on them. Sorting by
    confidence (rather than sweeping fixed thresholds) keeps full resolution
    where softmax confidences pile up just below 1.0.
    """
    order = np.argsort(-conf, kind="stable")
    k = np.arange(1, len(conf) + 1)
    return k / len(conf), np.cumsum(correct[order]) / k


def choose_threshold(conf, correct, target_accuracy):
    """Lowest threshold whose automated predictions reach ``target_accuracy`` (max coverage)."""
    for t in np.unique(conf):  # ascending, so the first hit automates the most nodes
        keep = conf >= t
        if correct[keep].mean() >= target_accuracy:
            return float(t)
    return 1.0


def plot_tsne(embeddings, labels, path, seed):
    """Side-by-side t-SNE of each model's test-node embeddings, coloured by true class."""
    fig, axes = plt.subplots(1, len(embeddings), figsize=(6 * len(embeddings), 5.5))
    for ax, (name, emb) in zip(axes, embeddings.items()):
        xy = TSNE(n_components=2, perplexity=30, init="pca", random_state=seed).fit_transform(emb)
        for c, (cls, colour) in enumerate(zip(CLASS_NAMES, CLASS_COLOURS)):
            pts = xy[labels == c]
            ax.scatter(pts[:, 0], pts[:, 1], s=3, alpha=0.6, color=colour, label=cls)
        ax.set(title=f"{name.upper()} hidden embeddings (test nodes)", xticks=[], yticks=[])
        ax.legend(markerscale=4, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_calibration(stats, path):
    """Reliability diagram and confidence histograms (correct vs wrong) for each model."""
    fig, axes = plt.subplots(1, 1 + len(stats), figsize=(5 * (1 + len(stats)), 4.5))
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    for name, s in stats.items():
        conf, acc, _ = zip(*s["bins"])
        ax.plot(conf, acc, "o-", label=f"{name.upper()} (ECE {s['ece']:.3f})")
    ax.set(xlabel="mean predicted confidence", ylabel="observed accuracy",
           title="Reliability diagram (test)", xlim=(0.25, 1), ylim=(0.25, 1))
    ax.legend()
    ax.grid(alpha=0.3)

    bins = np.linspace(0.25, 1, 31)
    for ax, (name, s) in zip(axes[1:], stats.items()):
        ax.hist(s["conf"][s["correct"]], bins=bins, alpha=0.6, density=True, label="correct")
        ax.hist(s["conf"][~s["correct"]], bins=bins, alpha=0.6, density=True, label="wrong")
        ax.set(xlabel="max softmax confidence", ylabel="density",
               title=f"{name.upper()} confidence (error-detection AUROC {s['error_auroc']:.3f})")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_coverage(stats, path):
    """Accuracy of automated predictions vs fraction of pages automated, per model."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for name, s in stats.items():
        cov, acc = coverage_curve(s["conf"], s["correct"])
        line, = ax.plot(cov, acc, label=name.upper())
        r = s["rule"]
        ax.plot(r["test_coverage"], r["test_accuracy"], "o", color=line.get_color(), markersize=8,
                label=f"{name.upper()} rule: conf >= {r['threshold']:.4f}")
    ax.set(xlabel="coverage (fraction of test pages automated)",
           ylabel="accuracy on automated pages", title="Reject-option trade-off (test)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def neighbour_label_mix(node, data):
    """Fraction of a node's neighbours in each true class (analysis only, not model input)."""
    nbrs = data.edge_index[1][data.edge_index[0] == node]
    if nbrs.numel() == 0:
        return {}
    counts = torch.bincount(data.y[nbrs], minlength=len(CLASS_NAMES)).float() / nbrs.numel()
    return {c: round(float(f), 2) for c, f in zip(CLASS_NAMES, counts) if f > 0}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target-accuracy", type=float, default=0.99,
                        help="accuracy required on automated predictions when choosing the threshold")
    parser.add_argument("--num-failures", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42, help="must match the seed used in train.py")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_graph(seed=args.seed).to(device)
    test, val = data.test_mask.cpu().numpy(), data.val_mask.cpu().numpy()
    y = data.y.cpu().numpy()
    os.makedirs(FIGURE_DIR, exist_ok=True)

    stats, embeddings, probs_all = {}, {}, {}
    for name in MODEL_NAMES:
        model = load_trained(name, data, device)
        probs = predict_proba(model, data).cpu().numpy()
        probs_all[name] = probs
        with torch.no_grad():
            embeddings[name] = model.embed(data.x, data.edge_index).cpu().numpy()[test]

        metrics = compute_metrics(torch.from_numpy(probs[test]), data.y[data.test_mask].cpu())
        conf, correct = probs.max(axis=1), probs.argmax(axis=1) == y
        ece, bins = expected_calibration_error(conf[test], correct[test])

        # Threshold is chosen on validation nodes, then evaluated once on test nodes.
        t = choose_threshold(conf[val], correct[val], args.target_accuracy)
        keep = conf[test] >= t
        rule = {"threshold": t, "test_coverage": float(keep.mean()),
                "test_accuracy": float(correct[test][keep].mean()),
                "routed_to_human": int((~keep).sum()),
                "errors_caught": int((~keep & ~correct[test]).sum()),
                "errors_total": int((~correct[test]).sum())}

        stats[name] = {
            "conf": conf[test], "correct": correct[test], "bins": bins, "ece": float(ece),
            # How well confidence ranks correct above wrong predictions (0.5 = useless).
            "error_auroc": float(roc_auc_score(correct[test], conf[test])),
            "mean_conf_correct": float(conf[test][correct[test]].mean()),
            "mean_conf_wrong": float(conf[test][~correct[test]].mean()),
            "rule": rule,
        }
        print(f"\n=== {name.upper()} (test) ===")
        print(f"accuracy {metrics['accuracy']:.4f} | macro-F1 {metrics['macro_f1']:.4f} "
              f"| macro-AUROC {metrics['macro_auroc']:.4f}")
        for cls, m in metrics["per_class"].items():
            print(f"  {cls:<11} P {m['precision']:.3f}  R {m['recall']:.3f}  F1 {m['f1-score']:.3f}")
        s = stats[name]
        print(f"ECE {s['ece']:.4f} | mean confidence: correct {s['mean_conf_correct']:.3f}, "
              f"wrong {s['mean_conf_wrong']:.3f} | error-detection AUROC {s['error_auroc']:.3f}")
        print(f"Reject rule (target {args.target_accuracy:.0%} on val): automate if confidence >= {t:.3f} -> "
              f"automates {rule['test_coverage']:.1%} of test pages at {rule['test_accuracy']:.2%} accuracy, "
              f"routes {rule['routed_to_human']} to review, catching {rule['errors_caught']}/{rule['errors_total']} errors")

    # Combined guardrail: automate only if the GCN is confident AND the feature-only
    # MLP agrees. Targets confident errors caused by misleading neighbourhoods.
    gcn_pred, mlp_pred = probs_all["gcn"].argmax(axis=1), probs_all["mlp"].argmax(axis=1)
    gcn_conf, gcn_ok = probs_all["gcn"].max(axis=1), gcn_pred == y
    auto = test & (gcn_conf >= stats["gcn"]["rule"]["threshold"]) & (gcn_pred == mlp_pred)
    combined = {"test_coverage": float(auto.sum() / test.sum()),
                "test_accuracy": float(gcn_ok[auto].mean()),
                "errors_automated": int((auto & ~gcn_ok).sum())}
    stats["gcn"]["combined_rule"] = combined
    print(f"\nCombined rule (GCN conf >= {stats['gcn']['rule']['threshold']:.3f} AND MLP agrees): "
          f"automates {combined['test_coverage']:.1%} of test pages at {combined['test_accuracy']:.2%} "
          f"accuracy, {combined['errors_automated']} errors slip through")

    plot_tsne(embeddings, y[test], os.path.join(FIGURE_DIR, "tsne.png"), args.seed)
    plot_calibration(stats, os.path.join(FIGURE_DIR, "calibration.png"))
    plot_coverage(stats, os.path.join(FIGURE_DIR, "reject_option.png"))

    # Failure autopsy: the GCN's most confident wrong answers on the test set.
    names, _ = load_page_info()
    gcn, mlp = probs_all["gcn"], probs_all["mlp"]
    test_idx = np.where(test)[0]
    wrong = test_idx[gcn[test_idx].argmax(axis=1) != y[test_idx]]
    wrong = wrong[np.argsort(-gcn[wrong].max(axis=1))]
    degree = torch.bincount(data.edge_index[0], minlength=data.num_nodes).cpu().numpy()
    failures = []
    print(f"\n=== GCN: {args.num_failures} most confident test errors (of {len(wrong)}) ===")
    for node in wrong[:args.num_failures]:
        case = {"node": int(node), "page_name": names[node],
                "true": CLASS_NAMES[y[node]], "gcn_pred": CLASS_NAMES[gcn[node].argmax()],
                "gcn_conf": round(float(gcn[node].max()), 3),
                "mlp_pred": CLASS_NAMES[mlp[node].argmax()],
                "mlp_conf": round(float(mlp[node].max()), 3),
                "degree": int(degree[node]),
                "neighbour_classes": neighbour_label_mix(node, data)}
        failures.append(case)
        print(json.dumps(case, ensure_ascii=False))

    summary = {name: {k: v for k, v in s.items() if k not in ("conf", "correct", "bins")}
               for name, s in stats.items()}
    summary["gcn_failures"] = failures
    with open(os.path.join(RESULTS_DIR, "analysis.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved figures to {FIGURE_DIR} and analysis to {RESULTS_DIR}/analysis.json")


if __name__ == "__main__":
    main()
