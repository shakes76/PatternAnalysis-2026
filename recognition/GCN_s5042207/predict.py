"""
Evaluate the trained MLP and GCN on the test set and analyse their confidence.

Usage:
    python predict.py      # needs the checkpoints saved by train.py for both
                           # models, and results/tune_lp.jsonl from tune.py

Loads the saved model of every seed, predicts every page once, and reports:
  1. per-type precision, recall, F1 and AUROC, macro-F1, and confusion matrices
  2. calibration: does confidence match accuracy? Before and after
     temperature scaling, with the temperature fitted on validation
  3. triage: a confidence threshold chosen on validation to reach 97%
     accuracy, applied to the test set (how many pages can be automated?)
  4. neighbourhood analysis: accuracy and confidence grouped by the share of a
     page's labelled neighbours that have the same type (local homophily)
  5. t-SNE pictures of what each model's last hidden layer has learned
  6. the most confident GCN mistakes, page by page
  7. resources: parameters, inference time and peak memory
  8. a politician-safety rule for the political-advertising use case: clear
     a page as "not a politician" automatically only when its politician
     probability is tiny (GCN alone, or GCN and MLP both), else ask a human

Label propagation (graph only, no features, no training) is included as a
reference in the tables. Numbers are mean ± std over the seeds. All settings
were fixed beforehand on the validation set, so this is the first and only
use of the test labels.
"""
import ctypes
import json
import os
import statistics
import sys
import time

import matplotlib
matplotlib.use("Agg")  # save plots to files without needing a screen
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.metrics import (confusion_matrix, f1_score,
                             precision_recall_fscore_support, roc_auc_score)

from dataset import CLASS_NAMES, load_data, load_page_names
from train import HERE, MODELS, evaluate, train_step
from tune import ece, label_propagation

TARGET_ACC = 0.97        # triage target from the feasibility review (criterion 3)
MAX_MISSED = 0.01        # politician-safety rule: miss at most 1% of real politicians
HOMOPHILY_BINS = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]
# Okabe-Ito colour-blind-safe colours, same as Figure 1 in the README
TYPE_COLOURS = ["#009E73", "#0072B2", "#CC79A7", "#E69F00"]   # in CLASS_NAMES order
MODEL_COLOURS = {"mlp": "#D55E00", "gcn": "#0072B2", "lp": "#009E73"}
NAMES = {"mlp": "MLP", "gcn": "GCN", "lp": "Label propagation"}


# ---------------------------------------------------------------- loading

def load_models(name, data):
    """Rebuild every seed's model with the settings train.py saved, and load
    its trained weights. Returns {seed: model}."""
    with open(os.path.join(HERE, "results", f"{name}_results.json")) as f:
        args = json.load(f)["args"]
    models = {}
    for seed in args["seeds"]:
        model = MODELS[name](in_dim=data.num_features, hidden=args["hidden"],
                             n_classes=len(CLASS_NAMES), num_layers=args["layers"],
                             dropout=args["dropout"])
        path = os.path.join(HERE, "checkpoints", f"{name}_seed{seed}.pt")
        model.load_state_dict(torch.load(path, weights_only=True))
        model.eval()                  # dropout OFF
        models[seed] = model
    return models


def best_lp_settings():
    """The label propagation setting with the best validation accuracy in
    results/tune_lp.jsonl (written by `python tune.py --model lp`)."""
    with open(os.path.join(HERE, "results", "tune_lp.jsonl")) as f:
        runs = [json.loads(line) for line in f if line.strip()]
    return max(runs, key=lambda r: r["val_acc"])["config"]


@torch.no_grad()
def predict_logits(model, data):
    """The 4 raw scores of every page (softmax turns them into probabilities)."""
    return model(data.x, data.edge_index)


# ---------------------------------------------------------------- 1. classification

def classification_metrics(y, probs):
    """Accuracy, macro-F1, macro-AUROC and per-type precision/recall/F1/AUROC.

    AUROC (one type vs the rest) asks: if we pick a random page of this type
    and a random page of another type, how often does the model give the first
    one a higher probability for this type? 1.0 = always, 0.5 = coin flip."""
    pred = probs.argmax(dim=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, pred, labels=range(len(CLASS_NAMES)), zero_division=0)
    auroc = [roc_auc_score(y == c, probs[:, c]) for c in range(len(CLASS_NAMES))]
    metrics = {"accuracy": (pred == y).float().mean().item(),
               "macro_f1": f1_score(y, pred, average="macro"),
               "macro_auroc": statistics.mean(auroc)}
    for c, name in enumerate(CLASS_NAMES):
        metrics[f"{name}_precision"] = float(precision[c])
        metrics[f"{name}_recall"] = float(recall[c])
        metrics[f"{name}_f1"] = float(f1[c])
        metrics[f"{name}_auroc"] = auroc[c]
    return metrics


def mean_std(runs):
    """{metric: {"mean", "std"}} over a list of per-seed metric dicts."""
    return {k: {"mean": statistics.mean(r[k] for r in runs),
                "std": statistics.stdev(r[k] for r in runs) if len(runs) > 1 else 0.0}
            for k in runs[0]}


def print_table(title, report, models, labels=NAMES):
    print(f"\n=== {title} (mean ± std over seeds) ===")
    print(f"{'':24s}" + "".join(f"{labels[m]:>20s}" for m in models))
    for metric in report[models[0]]:
        cells = "".join(f"{report[m][metric]['mean']:>12.3f} ± {report[m][metric]['std']:.3f}"
                        for m in models)
        print(f"{metric:24s}{cells}")


def plot_confusion(y, probs, path):
    """Row-normalised confusion matrices (rows = true type, sum to 100%),
    averaged over seeds, one panel per model."""
    fig, axes = plt.subplots(1, len(probs), figsize=(5.2 * len(probs), 4.6))
    for ax, (name, seed_probs) in zip(axes, probs.items()):
        cm = np.mean([confusion_matrix(y, p.argmax(dim=1), normalize="true")
                      for p in seed_probs.values()], axis=0) * 100
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=100)
        for i in range(len(CLASS_NAMES)):
            for j in range(len(CLASS_NAMES)):
                ax.text(j, i, f"{cm[i, j]:.1f}%", ha="center", va="center", fontsize=10,
                        color="white" if cm[i, j] > 50 else "#222222")
        ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=30)
        ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
        ax.set(xlabel="predicted type", ylabel="true type", title=name.upper())
    fig.suptitle("Test confusion matrices (% of each true type, mean over seeds)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------- 2. calibration

def fit_temperature(logits, y):
    """Find the temperature T that minimises the cross-entropy of
    softmax(logits / T) on the given (VALIDATION) pages.

    Dividing every score by T > 1 makes the probabilities less extreme
    (fixes over-confidence); T < 1 makes them sharper (fixes
    under-confidence). The predicted type never changes, only the confidence.
    T is optimised as log T so that it stays positive."""
    log_t = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), y)
        loss.backward()
        return loss

    optimizer.step(closure)
    return log_t.exp().item()


def reliability_curve(probs, y, n_bins=15, min_count=30):
    """Mean confidence and accuracy in each equal-width confidence bin that
    holds at least `min_count` pages (tiny bins are too noisy to plot)."""
    conf, pred = probs.max(dim=1)
    correct = (pred == y).float()
    edges = torch.linspace(0, 1, n_bins + 1)
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        if in_bin.sum() >= min_count:
            xs.append(conf[in_bin].mean().item())
            ys.append(correct[in_bin].mean().item())
    return xs, ys


def plot_reliability(y, raw, calibrated, path):
    """Reliability diagrams before and after temperature scaling. Test pages
    of all seeds are pooled. A perfectly calibrated model lies on the diagonal;
    below it = over-confident, above it = under-confident."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    for ax, (title, probs) in zip(axes, (("Before temperature scaling", raw),
                                         ("After temperature scaling", calibrated))):
        ax.plot([0, 1], [0, 1], color="#999999", lw=1, ls="--", label="perfect calibration")
        for name, seed_probs in probs.items():
            xs, ys = reliability_curve(torch.cat(list(seed_probs.values())),
                                       y.repeat(len(seed_probs)))
            ax.plot(xs, ys, marker="o", ms=5, lw=2, color=MODEL_COLOURS[name],
                    label=NAMES[name])
        ax.set(xlim=(0.25, 1.0), ylim=(0.25, 1.0), xlabel="confidence (probability of predicted type)",
               title=title)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("accuracy")
    axes[0].legend(loc="upper left")
    fig.suptitle("Test reliability diagrams (all seeds pooled; bins with ≥ 30 pages)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------- 3. triage

def choose_threshold(conf, correct, target=TARGET_ACC):
    """The LOWEST confidence threshold at which the accepted pages
    (confidence >= threshold) are still at least `target` accurate. A lower
    threshold automates more pages, so the lowest one that meets the target
    is the best. Chosen on VALIDATION pages only."""
    conf_sorted, order = conf.sort(descending=True)
    running_acc = correct[order].cumsum(0) / torch.arange(1, len(conf) + 1)
    ok = (running_acc >= target).nonzero()
    return conf_sorted[ok.max()].item() if len(ok) else 1.0


def triage(conf, correct, y, threshold):
    """What happens on a set of pages if every page with confidence >=
    threshold is decided automatically and the rest go to a human."""
    auto = conf >= threshold
    result = {"threshold": threshold,
              "coverage": auto.float().mean().item(),
              "automated_accuracy": correct[auto].float().mean().item(),
              "automated_errors": int((auto & ~correct).sum()),
              "human_review_accuracy": correct[~auto].float().mean().item()}
    for c, name in enumerate(CLASS_NAMES):
        result[f"{name}_coverage"] = auto[y == c].float().mean().item()
    return result


def plot_triage(curves, chosen, path):
    """Accuracy of the automated pages against the share of pages automated,
    as the threshold moves from strict (left) to lenient (right)."""
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for name, (cov, acc) in curves.items():
        ax.plot(cov * 100, acc * 100, lw=2, color=MODEL_COLOURS[name], label=NAMES[name])
        c, a = chosen[name]
        ax.plot(c * 100, a * 100, "o", ms=9, color=MODEL_COLOURS[name],
                markeredgecolor="white", markeredgewidth=2)
        ax.annotate(f"{c:.0%} automated\nat {a:.1%}", (c * 100, a * 100),
                    textcoords="offset points", xytext=(-10, -34), ha="right", fontsize=9)
    ax.axhline(TARGET_ACC * 100, color="#999999", lw=1, ls="--")
    ax.text(1, TARGET_ACC * 100 + 0.3, f"{TARGET_ACC:.0%} target", fontsize=9, color="#555555")
    ax.set(xlabel="pages decided automatically (%)", ylabel="accuracy of automated pages (%)",
           xlim=(0, 100), ylim=(80, 100.5),
           title="Test triage: automate the most confident pages")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def coverage_curve(conf, correct, grid):
    """Accuracy of the most confident pages, for each coverage in `grid`."""
    order = conf.argsort(descending=True)
    running_acc = correct[order].float().cumsum(0) / torch.arange(1, len(conf) + 1)
    idx = (grid * len(conf)).long().clamp(1, len(conf)) - 1
    return running_acc[idx]


# ---------------------------------------------------------------- 4. neighbourhoods

def local_homophily(data):
    """For every page: the share of its neighbours WITH KNOWN LABELS
    (training or validation pages) that have the same type as the page itself.
    NaN when none of its neighbours is labelled.

    The page's own type is its true label, which is fine for analysing results
    after the fact; the neighbours' labels never include test labels, so this
    is the same information the model could have had."""
    src, dst = data.edge_index
    known = data.train_mask | data.val_mask
    src, dst = src[known[dst]], dst[known[dst]]
    n = data.num_nodes
    n_known = torch.zeros(n).index_add_(0, src, torch.ones(len(src)))
    n_same = torch.zeros(n).index_add_(0, src, (data.y[src] == data.y[dst]).float())
    return torch.where(n_known > 0, n_same / n_known.clamp_min(1), torch.nan)


def homophily_bins(h):
    """(label, mask) for every homophily bin, plus pages with no labelled neighbour."""
    bins = []
    for lo, hi in HOMOPHILY_BINS:
        mask = (h > lo) & (h <= hi) if lo > 0 else (h >= lo) & (h <= hi)
        bins.append((f"{lo:.0%}–{hi:.0%}", mask))
    bins.append(("no labelled\nneighbour", torch.isnan(h)))
    return bins


def plot_homophily(rows, path):
    """Accuracy of each model, and the GCN's confidence, per homophily bin."""
    labels = [r["bin"] for r in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for name in ("mlp", "gcn"):
        ax.plot(x, [r[f"{name}_accuracy"] * 100 for r in rows], marker="o", ms=7, lw=2,
                color=MODEL_COLOURS[name], label=f"{NAMES[name]} accuracy")
    ax.plot(x, [r["gcn_confidence"] * 100 for r in rows], marker="s", ms=6, lw=2, ls="--",
            color=MODEL_COLOURS["gcn"], label="GCN confidence (calibrated)")
    ax.set_xticks(x, [f"{lab}\n(n = {r['pages']:.0f})" for lab, r in zip(labels, rows)])
    ax.set(xlabel="share of labelled neighbours with the same type as the page",
           ylabel="%", ylim=(0, 102), title="Test accuracy and confidence by neighbourhood")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------- 5. t-SNE

@torch.no_grad()
def hidden_features(model, data):
    """What the model passes into its last layer: the 64 numbers per page
    after the final hidden layer, captured with a forward hook."""
    captured = {}
    hook = model.layers[-1].register_forward_pre_hook(
        lambda module, inputs: captured.update(h=inputs[0]))
    model(data.x, data.edge_index)
    hook.remove()
    return captured["h"]


def plot_tsne(embeddings, y, wrong, path):
    """2-D t-SNE maps of the test pages' hidden features, coloured by true type.
    Pages the model gets wrong are marked with a black cross."""
    fig, axes = plt.subplots(1, len(embeddings), figsize=(6 * len(embeddings), 5.6))
    for ax, (name, xy) in zip(axes, embeddings.items()):
        for c, type_name in enumerate(CLASS_NAMES):
            pts = xy[y == c]
            ax.scatter(pts[:, 0], pts[:, 1], s=4, alpha=0.6, color=TYPE_COLOURS[c],
                       label=type_name, rasterized=True)
        bad = xy[wrong[name]]
        ax.scatter(bad[:, 0], bad[:, 1], s=10, marker="x", lw=0.8, color="#111111",
                   label="wrong prediction")
        ax.set(title=f"{NAMES[name]} (seed 0): {wrong[name].mean():.1%} wrong",
               xticks=[], yticks=[])
    axes[0].legend(loc="best", markerscale=3, fontsize=9)
    fig.suptitle("t-SNE of the last hidden layer, test pages, coloured by true type")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- 6. failure cases

def neighbour_summary(data, page):
    """Counts of each type among the page's labelled (train/val) neighbours,
    and the number of unlabelled (test) neighbours."""
    src, dst = data.edge_index
    nbrs = dst[src == page]
    known = (data.train_mask | data.val_mask)[nbrs]
    counts = {name: int((data.y[nbrs[known]] == c).sum()) for c, name in enumerate(CLASS_NAMES)}
    return {"links": len(nbrs), "labelled_neighbour_types": counts,
            "test_neighbours": int((~known).sum())}


# ---------------------------------------------------------------- 8. politician safety

def choose_clear_threshold(score, is_politician, max_missed=MAX_MISSED):
    """The HIGHEST cut-off t such that clearing every page with score <= t
    as "not a politician" misses at most `max_missed` of the real
    politicians. A higher cut-off clears more pages automatically (fewer
    human reviews), so the highest one that meets the target is the best.
    Chosen on VALIDATION pages only; capped at 0.5, where a page is more
    likely a politician than not."""
    pol_scores = score[is_politician].sort().values
    allowed = int(max_missed * len(pol_scores))      # politicians we may miss
    if allowed >= len(pol_scores):
        return 0.5
    # just below the score of the first politician we may NOT miss
    t = torch.nextafter(pol_scores[allowed], torch.tensor(float("-inf"))).item()
    return min(t, 0.5)


def politician_outcomes(cleared, auto_political, is_politician):
    """How a political / not political / human-review split performs."""
    human = ~cleared & ~auto_political
    return {"missed_politicians": (cleared & is_politician).float().sum().item()
                                  / is_politician.sum().item(),
            "missed_count": int((cleared & is_politician).sum()),
            "human_review": human.float().mean().item(),
            "auto_political": auto_political.float().mean().item(),
            "false_political": (auto_political & ~is_politician).float().sum().item()
                               / (~is_politician).sum().item()}


def safety_rule(score, gcn_p, threshold):
    """Clear as not political if score <= threshold; mark as political if the
    GCN thinks a politician is more likely than not; otherwise ask a human."""
    cleared = score <= threshold
    return cleared, ~cleared & (gcn_p >= 0.5)


def plot_politician(curves, points, path):
    """Politicians missed against the share of pages sent to a human, as the
    cut-off for clearing pages moves (test set, mean over seeds)."""
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    styles = {"gcn_only": ("GCN only", MODEL_COLOURS["gcn"], "-"),
              "gcn_mlp": ("GCN + MLP second opinion", MODEL_COLOURS["mlp"], "-")}
    for key, (human, missed) in curves.items():
        label, colour, ls = styles[key]
        ax.plot(human * 100, missed * 100, lw=2, ls=ls, color=colour, label=label)
    for key, (human, missed, text) in points.items():
        colour = styles[key][1] if key in styles else "#555555"
        ax.plot(human * 100, missed * 100, "o", ms=9, color=colour,
                markeredgecolor="white", markeredgewidth=2)
        ax.annotate(text, (human * 100, missed * 100), textcoords="offset points",
                    xytext=(8, 6), fontsize=9)
    ax.axhline(MAX_MISSED * 100, color="#999999", lw=1, ls="--")
    ax.text(29, MAX_MISSED * 100 + 0.08, f"{MAX_MISSED:.0%} target", fontsize=9, color="#555555")
    ax.set(xlabel="pages sent to a human (%)", ylabel="real politicians cleared as not political (%)",
           xlim=(0, 35), ylim=(0, 4.5),
           title="Test: politicians missed vs human workload")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------- 7. resources

def peak_memory_mb():
    """Peak memory (resident set) used by this process so far, in MB."""
    if os.name == "nt":
        class Counters(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + [
                (f, ctypes.c_size_t) for f in (
                    "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
        psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters),
                                   counters.cb)
        return counters.PeakWorkingSetSize / 2**20
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**20 if sys.platform == "darwin" else peak / 2**10


def time_inference(model, data, repeats=20):
    """Median wall-clock time of one full-graph prediction (all 22,470 pages)."""
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        predict_logits(model, data)
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def time_training_epochs(name, data, epochs=10):
    """Average time of one train.py epoch (one training step + the two
    evaluations used for the curves and early stopping), measured on a fresh
    model. The models from this timing run are thrown away."""
    with open(os.path.join(HERE, "results", f"{name}_results.json")) as f:
        args = json.load(f)["args"]
    torch.manual_seed(0)
    model = MODELS[name](in_dim=data.num_features, hidden=args["hidden"],
                         n_classes=len(CLASS_NAMES), num_layers=args["layers"],
                         dropout=args["dropout"])
    optimizer = torch.optim.Adam(model.parameters(), lr=args["lr"],
                                 weight_decay=args["weight_decay"])
    start = time.perf_counter()
    for _ in range(epochs):
        train_step(model, data, optimizer)
        evaluate(model, data, data.train_mask)
        evaluate(model, data, data.val_mask)
    return (time.perf_counter() - start) / epochs


# ---------------------------------------------------------------- main

def main():
    # Page names include many scripts; never crash on a terminal that can't show them
    sys.stdout.reconfigure(errors="backslashreplace")
    data = load_data(seed=42)
    val, test = data.val_mask, data.test_mask
    y_val, y_test = data.y[val], data.y[test]

    # Predict once per model and seed; every analysis below reuses these scores
    models = {name: load_models(name, data) for name in ("mlp", "gcn")}
    logits = {name: {seed: predict_logits(m, data) for seed, m in seed_models.items()}
              for name, seed_models in models.items()}
    lp = best_lp_settings()
    logits["lp"] = {0: label_propagation(data, lp["layers"], lp["alpha"])}
    print(f"label propagation: {lp['layers']} steps, alpha {lp['alpha']} (best on validation)")
    order = ("mlp", "gcn", "lp")
    report = {}
    # Memory needed to load the graph and all models and predict every page,
    # measured now, before the heavier analyses (t-SNE) below
    resources = {"peak_memory_inference_mb": peak_memory_mb()}

    # 1. Classification quality ------------------------------------------------
    report["classification"] = {
        name: mean_std([classification_metrics(y_test, lg[test].softmax(dim=1))
                        for lg in logits[name].values()])
        for name in order}
    print_table("1. Test classification", report["classification"], order)
    plot_confusion(y_test, {name: {s: lg[test].softmax(dim=1) for s, lg in logits[name].items()}
                            for name in ("mlp", "gcn")},
                   os.path.join(HERE, "figures", "confusion_matrices.png"))

    # 2. Calibration: temperature fitted on VALIDATION, judged on test --------
    temperature = {name: {s: fit_temperature(lg[val], y_val) for s, lg in logits[name].items()}
                   for name in order}
    probs = {name: {s: (lg / temperature[name][s]).softmax(dim=1)
                    for s, lg in logits[name].items()} for name in order}
    report["calibration"] = {}
    for name in order:
        runs = []
        for s, lg in logits[name].items():
            runs.append({"temperature": temperature[name][s],
                         "ece_before": ece(lg[test].softmax(dim=1), y_test),
                         "ece_after": ece(probs[name][s][test], y_test),
                         "nll_before": F.cross_entropy(lg[test], y_test).item(),
                         "nll_after": F.cross_entropy(lg[test] / temperature[name][s],
                                                      y_test).item(),
                         "mean_confidence_after": probs[name][s][test].max(dim=1)
                                                  .values.mean().item()})
        report["calibration"][name] = mean_std(runs)
    print_table("2. Test calibration (temperature fitted on validation)",
                report["calibration"], order)
    plot_reliability(y_test,
                     {n: {s: lg[test].softmax(dim=1) for s, lg in logits[n].items()}
                      for n in ("mlp", "gcn")},
                     {n: {s: p[test] for s, p in probs[n].items()} for n in ("mlp", "gcn")},
                     os.path.join(HERE, "figures", "reliability.png"))

    # 3. Triage: threshold chosen on VALIDATION, applied to test --------------
    report["triage"], thresholds = {}, {}
    curves, chosen = {}, {}
    grid = torch.linspace(0.01, 1.0, 100)
    for name in order:
        runs, seed_curves = [], []
        for s, p in probs[name].items():
            conf, pred = p.max(dim=1)
            correct = pred == data.y
            thresholds[(name, s)] = choose_threshold(conf[val], correct[val])
            runs.append(triage(conf[test], correct[test], y_test, thresholds[(name, s)]))
            seed_curves.append(coverage_curve(conf[test], correct[test], grid))
        report["triage"][name] = mean_std(runs)
        curves[name] = (grid, torch.stack(seed_curves).mean(dim=0))
        chosen[name] = (report["triage"][name]["coverage"]["mean"],
                        report["triage"][name]["automated_accuracy"]["mean"])
    print_table(f"3. Test triage (threshold for {TARGET_ACC:.0%} chosen on validation)",
                report["triage"], order)
    plot_triage({n: curves[n] for n in ("mlp", "gcn")}, chosen,
                os.path.join(HERE, "figures", "triage.png"))

    # 4. Neighbourhoods: where do the models fail? ----------------------------
    h = local_homophily(data)
    rows = []
    for label, in_bin in homophily_bins(h[test]):
        row = {"bin": label, "pages": int(in_bin.sum())}
        for name in order:
            accs, confs, auto_errs = [], [], []
            for s, p in probs[name].items():
                conf, pred = p[test].max(dim=1)
                correct = pred == y_test
                auto = conf >= thresholds[(name, s)]
                accs.append(correct[in_bin].float().mean().item())
                confs.append(conf[in_bin].mean().item())
                auto_errs.append(int((auto & ~correct & in_bin).sum()))
            row[f"{name}_accuracy"] = statistics.mean(accs)
            row[f"{name}_confidence"] = statistics.mean(confs)
            row[f"{name}_automated_errors"] = statistics.mean(auto_errs)
        rows.append(row)
    report["neighbourhood"] = rows
    print("\n=== 4. Test pages by share of same-type labelled neighbours (mean over seeds) ===")
    print(f"{'bin':>22s} {'pages':>6s}" + "".join(
        f"{NAMES[n][:5] + ' acc':>11s}" for n in order) + f"{'GCN conf':>10s}{'GCN auto-err':>14s}")
    for r in rows:
        print(f"{r['bin'].replace(chr(10), ' '):>22s} {r['pages']:6d}" + "".join(
            f"{r[n + '_accuracy']:>11.3f}" for n in order)
            + f"{r['gcn_confidence']:>10.3f}{r['gcn_automated_errors']:>14.1f}")
    plot_homophily(rows, os.path.join(HERE, "figures", "homophily.png"))

    # 5. t-SNE of the last hidden layer (seed 0) -------------------------------
    embeddings, wrong = {}, {}
    for name in ("mlp", "gcn"):
        hidden = hidden_features(models[name][0], data)[test].numpy()
        embeddings[name] = TSNE(n_components=2, init="pca", perplexity=30,
                                random_state=0).fit_transform(hidden)
        wrong[name] = (logits[name][0][test].argmax(dim=1) != y_test).numpy()
    plot_tsne(embeddings, y_test.numpy(), wrong, os.path.join(HERE, "figures", "tsne.png"))

    # 6. The most confident GCN mistakes (seed 0, calibrated) -----------------
    test_idx = test.nonzero().squeeze(1)
    conf, pred = probs["gcn"][0].max(dim=1)
    mistakes = [i for i in test_idx.tolist() if pred[i] != data.y[i]
                and conf[i] >= thresholds[("gcn", 0)]]
    mistakes.sort(key=lambda i: -conf[i].item())
    page_names = load_page_names()
    cases = []
    for i in mistakes[:5]:
        mlp_conf, mlp_pred = probs["mlp"][0][i].max(dim=0)
        cases.append({
            "page": i, "name": page_names[i], "true_type": CLASS_NAMES[data.y[i]],
            "gcn_prediction": CLASS_NAMES[pred[i]], "gcn_confidence": conf[i].item(),
            "gcn_predictions_all_seeds": [CLASS_NAMES[probs["gcn"][s][i].argmax()]
                                          for s in probs["gcn"]],
            "mlp_prediction": CLASS_NAMES[mlp_pred], "mlp_confidence": mlp_conf.item(),
            "local_homophily": None if torch.isnan(h[i]) else h[i].item(),
            **neighbour_summary(data, i)})
    report["failure_cases"] = {"confident_mistakes_seed0": len(mistakes), "cases": cases}
    print(f"\n=== 6. GCN seed 0: {len(mistakes)} test mistakes above the triage "
          f"threshold ({thresholds[('gcn', 0)]:.3f}); the 5 most confident ===")
    for c in cases:
        print(f"  page {c['page']:5d} {c['name']!r}:\n"
              f"      true {c['true_type']:10s} GCN {c['gcn_prediction']:10s}"
              f" ({c['gcn_confidence']:.3f}, seeds {c['gcn_predictions_all_seeds']})"
              f"  MLP {c['mlp_prediction']} ({c['mlp_confidence']:.2f})"
              f"  links {c['links']}, labelled neighbours {c['labelled_neighbour_types']}")

    # 7. Resources ------------------------------------------------------------
    for name in ("mlp", "gcn"):
        with open(os.path.join(HERE, "results", f"{name}_results.json")) as f:
            runs = json.load(f)["runs"]
        model = models[name][0]
        epoch_s = time_training_epochs(name, data)
        epochs_run = [r["best_epoch"] + 50 for r in runs]   # + patience
        resources[name] = {
            "parameters": sum(p.numel() for p in model.parameters()),
            "inference_s": time_inference(model, data),
            "train_epoch_s": epoch_s,
            "epochs_run_mean": statistics.mean(epochs_run),
            "estimated_train_time_s": epoch_s * statistics.mean(epochs_run)}
    resources["peak_memory_whole_run_mb"] = peak_memory_mb()   # includes t-SNE and training
    report["resources"] = resources
    print("\n=== 7. Resources (this machine) ===")
    for name in ("mlp", "gcn"):
        r = resources[name]
        print(f"  {NAMES[name]}: {r['parameters']:,} parameters, inference "
              f"{r['inference_s'] * 1000:.0f} ms for all pages, {r['train_epoch_s']:.3f} s per "
              f"epoch x {r['epochs_run_mean']:.0f} epochs = ~{r['estimated_train_time_s']:.0f} s")
    print(f"  peak memory: {resources['peak_memory_inference_mb']:.0f} MB (inference), "
          f"{resources['peak_memory_whole_run_mb']:.0f} MB (whole run, incl. t-SNE and training)")

    # 8. Politician safety: cut-offs chosen on VALIDATION, judged on test ----
    pol = CLASS_NAMES.index("politician")
    is_pol = data.y == pol
    rules = {"current_triage": "Current triage (3.)", "gcn_only": "GCN only",
             "gcn_mlp": "GCN + MLP"}
    runs = {key: [] for key in rules}
    seed_curves = {"gcn_only": [], "gcn_mlp": []}
    rule_thresholds, rescued = {}, []
    grid = torch.linspace(0, 0.5, 201).tolist()
    for s in probs["gcn"]:
        gcn_p = probs["gcn"][s][:, pol]
        scores = {"gcn_only": gcn_p,
                  # second opinion: the more worried of the two models
                  "gcn_mlp": torch.maximum(gcn_p, probs["mlp"][s][:, pol])}
        # Section 3's triage rule, seen from the politician angle
        conf, pred = probs["gcn"][s].max(dim=1)
        auto = conf >= thresholds[("gcn", s)]
        runs["current_triage"].append({"threshold": thresholds[("gcn", s)], **politician_outcomes(
            (auto & (pred != pol))[test], (auto & (pred == pol))[test], is_pol[test])})
        cleared_by = {}
        for key, score in scores.items():
            t = choose_clear_threshold(score[val], is_pol[val])
            rule_thresholds[(key, s)] = t
            cleared, auto_pol = safety_rule(score, gcn_p, t)
            cleared_by[key] = cleared
            runs[key].append({"threshold": t, **politician_outcomes(
                cleared[test], auto_pol[test], is_pol[test])})
            curve = []
            for g in grid:
                c, a = safety_rule(score[test], gcn_p[test], g)
                o = politician_outcomes(c, a, is_pol[test])
                curve.append((o["human_review"], o["missed_politicians"]))
            seed_curves[key].append(torch.tensor(curve))
        # Real politicians that GCN alone would clear, but the second opinion catches
        rescued.append(int((cleared_by["gcn_only"] & ~cleared_by["gcn_mlp"] & is_pol & test).sum()))

    report["politician_safety"] = {key: mean_std(r) for key, r in runs.items()}
    report["politician_safety"]["rescued_by_second_opinion"] = statistics.mean(rescued)
    print_table(f"8. Politician safety (cut-offs chosen on validation to miss <= {MAX_MISSED:.0%})",
                report["politician_safety"], tuple(rules), labels=rules)
    print(f"  politicians caught by the second opinion that GCN alone would clear: "
          f"{statistics.mean(rescued):.1f} per seed")

    # What the rules do with the politicians among the failure cases (seed 0)
    for c in cases:
        if c["true_type"] == "politician":
            i = c["page"]
            decisions = {}
            for key in ("gcn_only", "gcn_mlp"):
                g = probs["gcn"][0][i, pol]
                score = g if key == "gcn_only" else torch.maximum(g, probs["mlp"][0][i, pol])
                decisions[key] = ("cleared as not political" if score <= rule_thresholds[(key, 0)]
                                  else "political" if g >= 0.5 else "sent to a human")
            c["politician_rule"] = decisions
            print(f"  failure case {c['name']!r}: GCN only -> {decisions['gcn_only']}, "
                  f"GCN + MLP -> {decisions['gcn_mlp']}")

    points = {key: (report["politician_safety"][key]["human_review"]["mean"],
                    report["politician_safety"][key]["missed_politicians"]["mean"],
                    rules[key].replace(" (3.)", "")) for key in rules}
    plot_politician({key: (torch.stack(c).mean(dim=0)[:, 0], torch.stack(c).mean(dim=0)[:, 1])
                     for key, c in seed_curves.items()}, points,
                    os.path.join(HERE, "figures", "politician_safety.png"))

    with open(os.path.join(HERE, "results", "test_metrics.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("\nsaved results/test_metrics.json and figures/{confusion_matrices, reliability, "
          "triage, homophily, tsne, politician_safety}.png")


if __name__ == "__main__":
    main()
