import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")                      # draw to files, no display needed
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, f1_score

from dataset import get_data
from modules import GCN, MLP
from train import to_device

# Class names identified by inspecting the page_name field of facebook.npz.
CLASS_NAMES = ["Companies", "Governmental organisations", "Politicians", "Television shows"]


def load_run(run_dir, device):
    """Load the data, split and trained models exactly as used in training."""
    with open(os.path.join(run_dir, "results.json")) as f:
        saved = json.load(f)
    cfg = saved["args"]
    d = to_device(get_data(root=cfg["data_root"], seed=cfg["split_seed"]), device)
    models = {}
    for name, Model in (("MLP", MLP), ("GCN", GCN)):
        model = Model(d["x"].size(1), cfg["hidden"], d["num_classes"],
                      num_layers=cfg["layers"], dropout=cfg["dropout"]).to(device)
        model.load_state_dict(torch.load(os.path.join(run_dir, f"{name}.pt"),
                                         map_location=device))
        model.eval()                       # dropout off for inference
        models[name] = model
    return d, models, saved


def predict(model, d):
    """Softmax probabilities and hidden-layer embeddings for every node."""
    with torch.no_grad():
        logits, emb = model(d["x"], d["adj"], return_embedding=True)
    return F.softmax(logits, dim=1), emb


def plot_confusion(y_true, preds, path):
    """Row-normalised confusion matrices (each row sums to 1), with raw counts."""
    k = len(CLASS_NAMES)
    fig, axes = plt.subplots(1, len(preds), figsize=(6.5 * len(preds), 5.5))
    for ax, (name, y_pred) in zip(np.atleast_1d(axes), preds.items()):
        cm = confusion_matrix(y_true, y_pred, labels=range(k))
        cm_norm = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues")
        for i in range(k):
            for j in range(k):
                ax.text(j, i, f"{cm_norm[i, j]:.2f}\n({cm[i, j]})", ha="center",
                        va="center", fontsize=9,
                        color="white" if cm_norm[i, j] > 0.5 else "black")
        ax.set_xticks(range(k))
        ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
        ax.set_yticks(range(k))
        ax.set_yticklabels(CLASS_NAMES)
        ax.set(xlabel="Predicted category", ylabel="True category",
               title=f"{name} (test set)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_tsne(embeddings, y_true, path, max_points=3000, seed=0):
    """2D t-SNE of the last hidden layer for a random sample of test nodes."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y_true), size=min(max_points, len(y_true)), replace=False)
    fig, axes = plt.subplots(1, len(embeddings), figsize=(6.5 * len(embeddings), 5.5))
    axes = np.atleast_1d(axes)
    for ax, (name, emb) in zip(axes, embeddings.items()):
        xy = TSNE(n_components=2, init="pca", perplexity=30,
                  random_state=seed).fit_transform(emb[idx])
        for c, cname in enumerate(CLASS_NAMES):
            mask = y_true[idx] == c
            ax.scatter(xy[mask, 0], xy[mask, 1], s=6, label=cname)
        ax.set_title(f"{name}: hidden-layer embeddings (t-SNE)")
        ax.set_xticks([])
        ax.set_yticks([])
    axes[-1].legend(markerscale=3, loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate and visualise trained models.")
    parser.add_argument("--run-dir", default="outputs/seed0",
                        help="folder written by train.py (holds MLP.pt, GCN.pt, results.json)")
    parser.add_argument("--fig-dir", default="figures", help="where to save the figures")
    parser.add_argument("--md-out", default="outputs/failure_cases.md",
                        help="markdown file for the failure-case text")
    args = parser.parse_args()

    os.makedirs(args.fig_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.md_out) or ".", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d, models, saved = load_run(args.run_dir, device)

    test_idx = d["test_idx"]
    y_test = d["y"][test_idx].cpu().numpy()
    preds, embeddings, probs_all = {}, {}, {}
    for name, model in models.items():
        probs, emb = predict(model, d)
        probs_all[name] = probs
        y_pred = probs.argmax(1)[test_idx].cpu().numpy()
        preds[name] = y_pred
        embeddings[name] = emb[test_idx].cpu().numpy()
        acc = float((y_pred == y_test).mean())
        f1 = float(f1_score(y_test, y_pred, average="macro"))
        stored = saved["results"][name]
        print(f"{name}: test accuracy {acc:.3f}, macro-F1 {f1:.3f} "
              f"(training run: {stored['accuracy']:.3f}, {stored['macro_f1']:.3f})")

    plot_confusion(y_test, preds, os.path.join(args.fig_dir, "confusion_matrices.png"))
    plot_tsne(embeddings, y_test, os.path.join(args.fig_dir, "tsne_embeddings.png"))

    # ---- failure analysis: GCN errors on the test set ----
    names = load_page_names(saved["args"]["data_root"])
    degree, counts, hom = neighbour_stats(d["edge_index"], d["y"], d["num_classes"])
    gcn_conf, gcn_pred = probs_all["GCN"].max(1)
    mlp_conf, mlp_pred = probs_all["MLP"].max(1)
    t = {k: v[test_idx].cpu().numpy() for k, v in dict(
        y=d["y"], gcn_pred=gcn_pred, gcn_conf=gcn_conf, mlp_pred=mlp_pred,
        mlp_conf=mlp_conf, deg=degree, hom=hom).items()}
    counts_t = counts[test_idx].cpu().numpy()
    node_ids = test_idx.cpu().numpy()

    rule = saved["results"]["GCN"]["reject_rule"]
    threshold = rule["threshold"] if rule else 0.5      # threshold chosen on validation data

    md = [f"## Failure cases (GCN, test set, run `{args.run_dir}`)\n",
          f"Reject threshold (chosen on validation data): {threshold:.2f}\n"]
    for title, i in select_cases(t, counts_t, threshold):
        md.append(case_markdown(title, i, t, counts_t, names, node_ids, threshold))
    md += ["### Error rate by neighbourhood agreement\n"]
    md += error_table(t, "hom", HOMOPHILY_BINS, "Share of neighbours with the same category") + [""]
    md += ["### Error rate by number of links\n"]
    md += error_table(t, "deg", DEGREE_BINS, "Links") + [""]
    text = "\n".join(md)
    print("\n" + text)
    with open(args.md_out, "w", encoding="utf-8") as f:
        f.write(text)
    plot_error_by_homophily(t, os.path.join(args.fig_dir, "error_by_homophily.png"))
    print(f"Saved figures in '{args.fig_dir}/' and failure cases in '{args.md_out}'")


def load_page_names(data_root):
    """Page names stored in facebook.npz (a pickled object array, hence allow_pickle).
    The file comes from the course coordinators."""
    path = os.path.join(data_root, "FacebookPagePage", "raw", "facebook.npz")
    return np.load(path, allow_pickle=True)["page_name"]


def neighbour_stats(edge_index, y, num_classes):
    """Per node: degree, neighbour counts per TRUE class, and node homophily
    (share of neighbours with the same true class). Self-loops are ignored.
    These use true labels, so they are for post-hoc analysis only, never training."""
    src, dst = edge_index
    keep = src != dst
    src, dst = src[keep], dst[keep]
    onehot = F.one_hot(y, num_classes).float()
    counts = torch.zeros(y.size(0), num_classes, device=y.device)
    counts = counts.index_add(0, src, onehot[dst])
    degree = counts.sum(1)
    same = counts.gather(1, y[:, None]).squeeze(1)
    return degree, counts, same / degree.clamp(min=1)


def select_cases(t, counts_t, threshold):
    """Pick up to five representative GCN errors on the test set, one per failure type."""
    wrong = t["gcn_pred"] != t["y"]
    rows = np.arange(len(wrong))
    deg = t["deg"]
    share_pred = counts_t[rows, t["gcn_pred"]] / np.maximum(deg, 1)
    specs = [
        ("Confident error (would be automated)", wrong & (t["gcn_conf"] >= threshold), t["gcn_conf"]),
        ("Misled by its neighbours", wrong & (counts_t.argmax(1) == t["gcn_pred"]) & (deg >= 5), share_pred),
        ("Low-degree page", wrong & (deg <= 3), t["gcn_conf"]),
        ("Uncertain error (routed to a human)", wrong & (t["gcn_conf"] < threshold), t["gcn_conf"]),
        ("Graph hurt: the MLP was right", wrong & (t["mlp_pred"] == t["y"]), t["gcn_conf"]),
    ]
    used, cases = set(), []
    for title, mask, key in specs:
        mask = mask.copy()
        for u in used:
            mask[u] = False
        if mask.any():
            i = int(np.argmax(np.where(mask, key, -np.inf)))
            used.add(i)
            cases.append((title, i))
    return cases


def case_markdown(title, i, t, counts_t, names, node_ids, threshold):
    """One failure case as a short markdown block."""
    neigh = ", ".join(f"{CLASS_NAMES[c]} {int(counts_t[i, c])}"
                      for c in range(len(CLASS_NAMES)) if counts_t[i, c] > 0)
    routed = "yes" if t["gcn_conf"][i] < threshold else "no"
    return (f"**{title}**\n"
            f"- Page: {names[node_ids[i]]}\n"
            f"- True category: {CLASS_NAMES[t['y'][i]]}\n"
            f"- GCN predicted: {CLASS_NAMES[t['gcn_pred'][i]]} (confidence {t['gcn_conf'][i]:.2f}); "
            f"MLP predicted: {CLASS_NAMES[t['mlp_pred'][i]]} (confidence {t['mlp_conf'][i]:.2f})\n"
            f"- Links: {int(t['deg'][i])}; neighbours' true categories: {neigh}\n"
            f"- Sent to a human at threshold {threshold:.2f}: {routed}\n")


HOMOPHILY_BINS = [("below 0.5", 0.0, 0.5), ("0.5 to 0.8", 0.5, 0.8), ("0.8 to 1.0", 0.8, 1.01)]
DEGREE_BINS = [("1 to 3", 1, 4), ("4 to 10", 4, 11), ("11 or more", 11, 1e9)]


def error_table(t, key, bins, label):
    """Markdown table of MLP and GCN error rates for groups of test pages."""
    lines = [f"| {label} | Test pages | MLP error rate | GCN error rate |", "|---|---|---|---|"]
    for name, lo, hi in bins:
        m = (t[key] >= lo) & (t[key] < hi)
        if m.any():
            lines.append(f"| {name} | {int(m.sum())} ({m.mean():.0%}) | "
                         f"{(t['mlp_pred'][m] != t['y'][m]).mean():.3f} | "
                         f"{(t['gcn_pred'][m] != t['y'][m]).mean():.3f} |")
    return lines


def plot_error_by_homophily(t, path):
    """Bar chart: test error rate vs how many of a page's neighbours share its category."""
    labels, mlp_err, gcn_err = [], [], []
    for name, lo, hi in HOMOPHILY_BINS:
        m = (t["hom"] >= lo) & (t["hom"] < hi)
        if m.any():
            labels.append(f"{name}\n(n={int(m.sum())})")
            mlp_err.append((t["mlp_pred"][m] != t["y"][m]).mean())
            gcn_err.append((t["gcn_pred"][m] != t["y"][m]).mean())
    x, w = np.arange(len(labels)), 0.38
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar(x - w / 2, mlp_err, w, label="MLP", color="tab:blue")
    ax.bar(x + w / 2, gcn_err, w, label="GCN", color="tab:orange")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set(xlabel="Share of a page's neighbours with the same true category",
           ylabel="Test error rate", title="Error rate by neighbourhood agreement")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()