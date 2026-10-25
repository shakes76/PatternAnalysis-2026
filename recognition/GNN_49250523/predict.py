"""predict.py: Evaluate saved checkpoints: test metrics, calibration, reject option, failures.

Author: 49250523 (COMP3710 Pattern Recognition project, 2026)

Usage:
    python predict.py                    # metrics, figures and failure candidates
    python predict.py --nodes 12 345     # also draw these pages' 2-hop neighbourhoods

For each model and each final seed, the temperature and the reject threshold are
fitted on validation logits only, frozen, and then applied to the test logits
(see ``calibrate.py``). Results are written to ``results/``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import textwrap
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch_geometric.data import Data
from torch_geometric.utils import degree

from calibrate import (
    ECE_BINS,
    TARGET_ACCURACY,
    choose_threshold,
    ece,
    fit_temperature,
    reject_report,
)
from dataset import CLASS_NAMES, SPLIT_SEED, get_data, load_page_names
from modules import NodeClassifier, build_model
from train import CHECKPOINTS, FINAL_SEEDS, RESULTS

ECE_LIMIT = 0.03  # AC2
COVERAGE_TARGET = 0.80  # AC3
LATENCY_REPEATS = 20
CONFIDENT = 0.9  # a wrong page at or above this confidence is "confidently wrong"
LOW_DEGREE = 2  # a page with this many links or fewer has little graph signal
SAFETY_TARGET = 0.98  # Section 7.9: AC3 rule with a 1-point safety margin
OPERATING_TARGETS = (0.97, 0.98, 0.99)  # Section 7.9: accuracy targets offered to the client


def load_model(path: str, data: Data, seed: int) -> NodeClassifier:
    """Rebuild a model from a checkpoint written by ``train.py final``.

    Args:
        path: Checkpoint path.
        data: Graph, used for the input and output sizes.
        seed: Model seed the checkpoint must have been trained with.

    Returns:
        The model in evaluation mode.

    Raises:
        ValueError: If the checkpoint was trained with a different seed or split.
    """
    ckpt = torch.load(path, weights_only=True)
    if ckpt["seed"] != seed:
        raise ValueError(f"{path} was trained with seed {ckpt['seed']}, expected {seed}")
    if "split_seed" in ckpt and ckpt["split_seed"] != SPLIT_SEED:
        raise ValueError(f"{path} was trained on split {ckpt['split_seed']}, not {SPLIT_SEED}")
    cfg = ckpt["config"]
    model = build_model(
        cfg["model"],
        data.num_features,
        len(CLASS_NAMES),
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
    )
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()


@torch.no_grad()
def forward(model: NodeClassifier, data: Data) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the logits and the hidden embeddings of every node."""
    h = model.embed(data.x, data.edge_index)
    return model.classify(h, data.edge_index), h


def score_seed(logits: torch.Tensor, data: Data) -> tuple[dict, torch.Tensor]:
    """Fit calibration on validation, then score the test split once.

    Args:
        logits: Logits of every node from one trained model.
        data: Graph with real labels and split masks.

    Returns:
        The test metrics, and the calibrated probabilities of every node.
    """
    val, test = data.val_mask, data.test_mask
    temperature = fit_temperature(logits[val], data.y[val])
    probs_raw = logits.softmax(1)
    probs = (logits / temperature).softmax(1)
    threshold = choose_threshold(probs[val], data.y[val])

    y = data.y[test].numpy()
    pred = probs[test].argmax(1).numpy()
    precision, recall, _, _ = precision_recall_fscore_support(y, pred, zero_division=0)
    metrics = {
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
        "auroc_macro_ovr": roc_auc_score(y, probs[test].numpy(), multi_class="ovr"),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "temperature": temperature,
        "ece_before": ece(probs_raw[test], data.y[test]),
        "ece_after": ece(probs[test], data.y[test]),
        "confusion": confusion_matrix(y, pred, labels=range(len(CLASS_NAMES))).tolist(),
    }
    if threshold is None:
        metrics["reject"] = None
    else:
        metrics["reject"] = reject_report(probs[test], data.y[test], threshold, len(CLASS_NAMES))
    return metrics, probs


@torch.no_grad()
def latency(model: NodeClassifier, data: Data, repeats: int = LATENCY_REPEATS) -> float:
    """Return the median wall-clock time of one full-graph forward pass, in seconds."""
    for _ in range(3):  # warm-up
        model(data.x, data.edge_index)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        model(data.x, data.edge_index)
        times.append(time.perf_counter() - start)
    return float(np.median(times))


def summarise(runs: list[dict]) -> dict:
    """Return the mean and standard deviation over seeds of each scalar metric."""
    out = {}
    scalars = ("accuracy", "macro_f1", "auroc_macro_ovr", "temperature", "ece_before", "ece_after")
    for key in scalars:
        values = np.array([r[key] for r in runs])
        out[key] = {"mean": values.mean(), "std": values.std(ddof=1)}
    for key in ("precision", "recall"):
        values = np.array([r[key] for r in runs])
        out[key] = {"mean": values.mean(0).tolist(), "std": values.std(0, ddof=1).tolist()}
    rejects = [r["reject"] for r in runs if r["reject"] is not None]
    if rejects:
        for key in ("threshold", "coverage", "accuracy"):
            values = np.array([r[key] for r in rejects])
            out[f"reject_{key}"] = {"mean": values.mean(), "std": values.std(ddof=1)}
        out["reject_coverage_per_class"] = np.mean(
            [r["coverage_per_class"] for r in rejects], axis=0
        ).tolist()
    out["seeds_without_threshold"] = len(runs) - len(rejects)
    out["confusion_sum"] = np.sum([r["confusion"] for r in runs], axis=0).tolist()
    return out


def routing_on_validation(
    data: Data, gcn_probs: list[torch.Tensor], mlp_probs: list[torch.Tensor]
) -> dict:
    """Score the extra routing rules of Section 7.9 on the validation split only.

    The rules were proposed after reading test failures, so they are checked on
    validation, never on test. For each final seed, a policy decides which
    validation pages the GCN auto-labels; the rest go to a human.

    * ``threshold_97``: the AC3 rule (confidence at or above the 97 % threshold).
    * ``+disagree``: also refer pages where the GCN and the MLP predict differently.
    * ``+low_degree``: also refer pages with ``LOW_DEGREE`` links or fewer.
    * ``threshold_98``: the AC3 rule with a ``SAFETY_TARGET`` accuracy target.

    Args:
        data: Graph with real labels and split masks.
        gcn_probs: Calibrated GCN probabilities of every node, one tensor per seed.
        mlp_probs: Calibrated MLP probabilities, same seed order as ``gcn_probs``.

    Returns:
        For each policy, the mean over seeds of validation coverage, accuracy of
        the auto-labelled pages, and number of auto-labelled errors.
    """
    val = data.val_mask
    y = data.y[val]
    low = degree(data.edge_index[0], data.num_nodes)[val] <= LOW_DEGREE
    results: dict[str, list[tuple[float, float, int]]] = {}
    for g_all, m_all in zip(gcn_probs, mlp_probs, strict=True):
        g, m = g_all[val], m_all[val]
        conf, pred = g.max(1)
        disagree = pred != m.argmax(1)
        t97, t98 = choose_threshold(g, y), choose_threshold(g, y, SAFETY_TARGET)
        if t97 is None or t98 is None:
            continue
        base = conf >= t97
        policies = {
            "threshold_97": base,
            "threshold_97+disagree": base & ~disagree,
            "threshold_97+low_degree": base & ~low,
            "threshold_97+disagree+low_degree": base & ~disagree & ~low,
            "threshold_98": conf >= t98,
        }
        for name, keep in policies.items():
            wrong = int((pred[keep] != y[keep]).sum())
            accuracy = 1 - wrong / max(int(keep.sum()), 1)
            results.setdefault(name, []).append((keep.float().mean().item(), accuracy, wrong))
    return {
        name: {
            "coverage": float(np.mean([r[0] for r in rows])),
            "accuracy": float(np.mean([r[1] for r in rows])),
            "errors_auto_labelled": float(np.mean([r[2] for r in rows])),
            "seeds": len(rows),
        }
        for name, rows in results.items()
    }


def operating_points(data: Data, gcn_probs: list[torch.Tensor]) -> dict:
    """Report the coverage-accuracy trade-off for several accuracy targets.

    For each target and each final seed, the threshold (minimum confidence score)
    is chosen on validation with the AC3 rule, frozen, and then applied to test.
    Nothing here is chosen on test; the test numbers are reported only.

    Args:
        data: Graph with real labels and split masks.
        gcn_probs: Calibrated GCN probabilities of every node, one tensor per seed.

    Returns:
        For each target, the mean over seeds of the threshold and, for the
        validation and test splits, coverage, accuracy of the auto-labelled pages
        and the number of auto-labelled errors.
    """
    out = {}
    for target in OPERATING_TARGETS:
        rows = []
        for probs in gcn_probs:
            threshold = choose_threshold(probs[data.val_mask], data.y[data.val_mask], target)
            if threshold is None:
                continue
            row = {"threshold": threshold}
            for split, mask in (("val", data.val_mask), ("test", data.test_mask)):
                conf, pred = probs[mask].max(1)
                keep = conf >= threshold
                wrong = int((pred[keep] != data.y[mask][keep]).sum())
                row[f"{split}_coverage"] = keep.float().mean().item()
                row[f"{split}_accuracy"] = 1 - wrong / max(int(keep.sum()), 1)
                row[f"{split}_errors"] = wrong
            rows.append(row)
        out[str(target)] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]} | {
            "seeds": len(rows)
        }
    return out


def plot_reliability(before: torch.Tensor, after: torch.Tensor, y: torch.Tensor, path: str) -> None:
    """Draw reliability diagrams (accuracy against confidence) before and after scaling."""
    edges = np.linspace(0, 1, ECE_BINS + 1)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    titles = ("Before scaling", "After scaling")
    for ax, probs, title in zip(axes, (before, after), titles, strict=True):
        conf, pred = probs.max(1)
        conf, correct = conf.numpy(), (pred == y).numpy()
        accs = [
            correct[(conf > lo) & (conf <= hi)].mean() if ((conf > lo) & (conf <= hi)).any() else 0
            for lo, hi in zip(edges[:-1], edges[1:], strict=True)
        ]
        ax.bar(edges[:-1], accs, width=1 / ECE_BINS, align="edge", edgecolor="k", alpha=0.7)
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
        ax.set(xlabel="confidence", title=f"{title} (ECE {ece(probs, y):.3f})", xlim=(0, 1))
        ax.legend(loc="upper left")
    axes[0].set_ylabel("accuracy")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_tsne(embeddings: dict[str, torch.Tensor], y: torch.Tensor, path: str) -> None:
    """Draw a 2-D t-SNE of each model's hidden embeddings, coloured by true class."""
    fig, axes = plt.subplots(1, len(embeddings), figsize=(5 * len(embeddings), 4.5))
    for ax, (name, h) in zip(axes, embeddings.items(), strict=True):
        xy = TSNE(init="pca", random_state=0).fit_transform(h.numpy())
        for c, cls in enumerate(CLASS_NAMES):
            mask = (y == c).numpy()
            ax.scatter(xy[mask, 0], xy[mask, 1], s=3, alpha=0.6, label=cls)
        ax.set(title=f"{name.upper()} hidden layer (test pages)", xticks=[], yticks=[])
    axes[-1].legend(markerscale=4, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def failure_candidates(data: Data, probs: list[torch.Tensor], path: str) -> int:
    """List the test pages the GCN gets wrong for every seed, with a failure type.

    A page wrong for all five seeds is a stable failure, not bad luck. Each page
    is tagged with the rules below; the README discusses 3-5 of them.

    * ``wrong-neighbours``: fewer than half of its links go to its own class.
    * ``few-links``: degree ``LOW_DEGREE`` or less.
    * ``confident``: mean confidence ``CONFIDENT`` or more.

    Args:
        data: Graph with real labels and split masks.
        probs: Calibrated GCN probabilities of every node, one tensor per seed.
        path: Output CSV path.

    Returns:
        The number of pages listed.
    """
    names = load_page_names() or [""] * data.num_nodes
    stacked = torch.stack(probs)  # (seeds, nodes, classes)
    preds = stacked.argmax(2)
    always_wrong = (preds != data.y).all(0) & data.test_mask
    src, dst = data.edge_index
    deg = degree(src, data.num_nodes).long()
    rows = []
    for node in always_wrong.nonzero().flatten().tolist():
        neighbours = dst[src == node]
        counts = torch.bincount(data.y[neighbours], minlength=len(CLASS_NAMES))
        true, pred = int(data.y[node]), int(torch.mode(preds[:, node]).values)
        own_share = counts[true].item() / max(int(deg[node]), 1)
        conf = stacked[:, node].max(1).values.mean().item()
        tags = [
            tag
            for tag, hit in (
                ("wrong-neighbours", own_share < 0.5),
                ("few-links", deg[node] <= LOW_DEGREE),
                ("confident", conf >= CONFIDENT),
            )
            if hit
        ]
        rows.append(
            {
                "node": node,
                "page_name": names[node],
                "true": CLASS_NAMES[true],
                "predicted": CLASS_NAMES[pred],
                "mean_confidence": round(conf, 3),
                "degree": int(deg[node]),
                "own_class_share": round(own_share, 3),
                **{f"n_{c}": int(counts[i]) for i, c in enumerate(CLASS_NAMES)},
                "type": "+".join(tags) or "other",
            }
        )
    rows.sort(key=lambda r: -r["mean_confidence"])
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["node"])
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def draw_neighbourhood(
    ax: plt.Axes, data: Data, node: int, pred: int, names: list[str], cap: int = 40
) -> None:
    """Draw one page's 2-hop neighbourhood on ``ax``, coloured by true class.

    Large neighbourhoods are cut to ``cap`` first-hop pages and 3 second-hop
    pages each (first in index order), so the picture stays readable.

    Args:
        ax: Axes to draw on.
        data: Graph with real labels.
        node: Page to draw (shown large, outlined and labelled with its name).
        pred: The class the model predicted for ``node``.
        names: Page name for every node index.
        cap: Maximum number of first-hop pages drawn.
    """
    src, dst = data.edge_index
    hop1 = dst[src == node].tolist()[:cap]
    graph = nx.Graph()
    graph.add_node(node)
    for n1 in hop1:
        graph.add_edge(node, n1)
        for n2 in [n for n in dst[src == n1].tolist() if n != node][:3]:
            graph.add_edge(n1, n2)
    colours = [f"C{int(data.y[n])}" for n in graph.nodes]
    sizes = [400 if n == node else 120 if n in hop1 else 40 for n in graph.nodes]
    pos = nx.spring_layout(graph, seed=0)
    nx.draw_networkx_edges(graph, pos, ax=ax, alpha=0.3)
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=colours, node_size=sizes)
    nx.draw_networkx_nodes(
        graph, pos, nodelist=[node], ax=ax, node_color=colours[0], node_size=400, edgecolors="k"
    )
    ax.annotate(
        textwrap.shorten(names[node], 30, placeholder="…"),
        pos[node],
        xytext=(0, 16),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontweight="bold",
        bbox={"boxstyle": "round", "fc": "white", "ec": "k", "alpha": 0.85},
        zorder=5,
    )
    ax.set_title(
        f"true: {CLASS_NAMES[int(data.y[node])]}, predicted: {CLASS_NAMES[pred]}\n"
        f"degree {len(dst[src == node])} (showing {len(hop1)})"
    )
    ax.axis("off")


def plot_neighbourhoods(data: Data, nodes: list[int], preds: list[int], path: str) -> None:
    """Draw the 2-hop neighbourhoods of several pages side by side in one figure.

    Args:
        data: Graph with real labels.
        nodes: Pages to draw.
        preds: The class the model predicted for each page in ``nodes``.
        path: Output PNG path.
    """
    names = load_page_names() or [str(i) for i in range(data.num_nodes)]
    cols = min(3, len(nodes) + 1)  # one spare panel holds the legend
    rows = -(-(len(nodes) + 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), squeeze=False)
    axes = axes.ravel()
    for i, (ax, node, pred) in enumerate(zip(axes, nodes, preds, strict=False)):
        draw_neighbourhood(ax, data, node, pred, names)
        ax.text(0, 1, str(i + 1), transform=ax.transAxes, fontsize=16, fontweight="bold")
    for ax in axes[len(nodes) :]:
        ax.axis("off")
    legend = axes[len(nodes)]
    for c, cls in enumerate(CLASS_NAMES):
        legend.scatter([], [], color=f"C{c}", s=80, label=cls)
    legend.scatter([], [], color="w", edgecolors="k", s=200, label="analysed page")
    legend.scatter([], [], color="grey", s=60, label="direct link")
    legend.scatter([], [], color="grey", s=20, label="two links away")
    legend.legend(loc="center", fontsize=12, title="True class", title_fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Score every checkpoint and write metrics, figures and failure candidates."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--nodes", type=int, nargs="*", default=[], help="pages to draw")
    args = parser.parse_args()
    plt.switch_backend("Agg")
    data = get_data()
    test = data.test_mask
    report, embeddings = {}, {}
    probs_by_model: dict[str, list[torch.Tensor]] = {"gcn": [], "mlp": []}
    for model_name in ("gcn", "mlp"):
        # Same seed order for both models, so GCN and MLP runs pair by seed.
        runs = []
        for i, seed in enumerate(FINAL_SEEDS):
            path = os.path.join(CHECKPOINTS, f"{model_name}_seed{seed}.pt")
            model = load_model(path, data, seed)
            logits, h = forward(model, data)
            metrics, probs = score_seed(logits, data)
            runs.append(metrics)
            probs_by_model[model_name].append(probs)
            if i == 0:  # figures and latency use the first final seed
                embeddings[model_name] = h[test]
                report[f"{model_name}_latency_seconds"] = latency(model, data)
                if model_name == "gcn":
                    plot_reliability(
                        logits[test].softmax(1),
                        probs[test],
                        data.y[test],
                        os.path.join(RESULTS, "reliability.png"),
                    )
            print(f"{model_name} {os.path.basename(path)}: F1 {metrics['macro_f1']:.4f}")
        report[model_name] = {"per_seed": runs, "summary": summarise(runs)}

    s = report["gcn"]["summary"]
    report["acceptance"] = {
        "AC2_ece": bool(s["ece_after"]["mean"] <= ECE_LIMIT),
        "AC3_reject": bool(
            s["seeds_without_threshold"] == 0
            and s["reject_coverage"]["mean"] >= COVERAGE_TARGET
            and s["reject_accuracy"]["mean"] >= TARGET_ACCURACY
        ),
        "AC5_latency": bool(report["gcn_latency_seconds"] <= 1.0),
    }
    report["routing_validation"] = routing_on_validation(
        data, probs_by_model["gcn"], probs_by_model["mlp"]
    )
    report["operating_points"] = operating_points(data, probs_by_model["gcn"])
    with open(os.path.join(RESULTS, "predict.json"), "w") as f:
        json.dump(report, f, indent=2)

    plot_tsne(embeddings, data.y[test], os.path.join(RESULTS, "tsne.png"))
    gcn_probs = probs_by_model["gcn"]
    n = failure_candidates(data, gcn_probs, os.path.join(RESULTS, "failures.csv"))
    mean_pred = torch.stack(gcn_probs).mean(0).argmax(1)
    if args.nodes:
        out = os.path.join(RESULTS, "neighbourhoods.png")
        plot_neighbourhoods(data, args.nodes, [int(mean_pred[n]) for n in args.nodes], out)

    for model_name in ("gcn", "mlp"):
        m = report[model_name]["summary"]
        line = f"{model_name}: F1 {m['macro_f1']['mean']:.4f}, ECE {m['ece_before']['mean']:.4f}"
        line += f" -> {m['ece_after']['mean']:.4f}"
        if "reject_coverage" in m:
            line += f", coverage {m['reject_coverage']['mean']:.3f}"
            line += f" at accuracy {m['reject_accuracy']['mean']:.3f}"
        print(line)
    print(f"latency: gcn {report['gcn_latency_seconds'] * 1000:.1f} ms")
    print(f"acceptance: {report['acceptance']}")
    print(f"failure candidates (GCN wrong for every seed): {n}")
    for target, r in report["operating_points"].items():
        print(
            f"target {target}: threshold {r['threshold']:.3f}, "
            f"test coverage {r['test_coverage']:.3f}, test accuracy {r['test_accuracy']:.4f}"
        )
    for name, r in report["routing_validation"].items():
        print(f"validation {name}: coverage {r['coverage']:.3f}, accuracy {r['accuracy']:.4f}")


if __name__ == "__main__":
    main()
