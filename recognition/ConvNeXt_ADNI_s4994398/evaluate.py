"""
Open-dilemma analysis: ConvNeXt vs baseline on calibration, selective prediction,
computational cost and failure cases. Run after training both models:

    python train.py --arch convnext
    python train.py --arch baseline --lr 1e-3 --weight-decay 1e-4
    python evaluate.py

Writes figures and ``summary.json`` to results/ and stores the fitted temperature and
validation-chosen referral threshold inside each checkpoint for predict.py.
"""

import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from dataset import CLASSES, load_patient_map, load_patient_splits, preprocess, scan_id
from modules import build_model
from utils import (apply_threshold, classification_report, fit_temperature, reliability_bins,
                   risk_coverage, roc_points, patient_level, scan_level, softmax_np, threshold_for_accuracy)

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "checkpoints")
RESULTS_IN = os.path.join(HERE, "results")  # where train.py wrote <tag>_test_metrics.json
RES = RESULTS_IN                             # where figures / summary go (overridable with --out-dir)
NAMES = {"convnext": "ConvNeXt-T", "baseline": "Baseline CNN"}
COLORS = {"convnext": "#1f77b4", "baseline": "#d62728"}


def load_run(tag):
    z = np.load(os.path.join(CKPT, f"{tag}_logits.npz"))
    with open(os.path.join(RESULTS_IN, f"{tag}_test_metrics.json")) as f:
        meta = json.load(f)
    return dict(z), meta


def slice_rank(paths):
    """Position (0..19) of each slice within its scan's sorted slice numbers."""
    names = [os.path.basename(str(p)) for p in paths]
    groups = {}
    for i, n in enumerate(names):
        s, k = n.split("_")[0], int(n.split("_")[1].split(".")[0])
        groups.setdefault(s, []).append((k, i))
    rank = np.zeros(len(names), dtype=int)
    for items in groups.values():
        for r, (_, i) in enumerate(sorted(items)):
            rank[i] = r
    return rank


# --------------------------------------------------------------------------- profiling
@torch.no_grad()
def profile(tag, device, reps=50):
    ckpt = torch.load(os.path.join(CKPT, f"{tag}_best.pt"), map_location=device)
    model = build_model(ckpt["arch"], ckpt.get("variant", "tiny"), pretrained=False).to(device).eval()
    model.load_state_dict(ckpt["model"])
    out = {"params_M": sum(p.numel() for p in model.parameters()) / 1e6}
    # bs=1: single-slice latency; bs=20: one whole scan (20 slices), i.e. per-scan latency;
    # bs=64: throughput at the training batch size
    for bs in (1, 20, 64):
        x = torch.randn(bs, 1, 224, 224, device=device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        for _ in range(10):  # warm-up
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model(x)
        torch.cuda.synchronize()
        times = []
        for _ in range(reps):
            t = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model(x)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t)
        out[f"latency_ms_bs{bs}"] = 1000 * float(np.median(times))
        out[f"throughput_img_s_bs{bs}"] = bs / float(np.median(times))
        out[f"infer_peak_vram_mb_bs{bs}"] = (torch.cuda.max_memory_allocated() - base) / 2**20 \
            + sum(p.numel() * p.element_size() for p in model.parameters()) / 2**20
    return out


# --------------------------------------------------------------------------- Grad-CAM
def grad_cam(model, layer, x, target):
    """Grad-CAM (Selvaraju et al., 2017) heat-map for class ``target`` at ``layer``."""
    acts, grads = {}, {}
    h1 = layer.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("g", go[0]))
    model.zero_grad()
    logits = model(x)
    logits[:, target].sum().backward()
    h1.remove()
    h2.remove()
    w = grads["g"].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * acts["a"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    return (cam / (cam.max() + 1e-8)).detach().cpu().numpy()


def failure_autopsy(tag, probs, labels, paths, images, device, n_fn=3, n_fp=2):
    """Most confident false negatives / false positives with Grad-CAM overlays."""
    ckpt = torch.load(os.path.join(CKPT, f"{tag}_best.pt"), map_location=device)
    model = build_model(ckpt["arch"], ckpt.get("variant", "tiny"), pretrained=False).to(device).eval()
    model.load_state_dict(ckpt["model"])
    layer = model.stages[-1] if ckpt["arch"] == "convnext" else model.features[-2]

    pred = probs.argmax(1)
    conf = probs.max(1)
    scans = np.array([scan_id(p) for p in paths])
    rank = slice_rank(paths)
    # scan-level P(AD): does averaging the scan's 20 slices rescue this error?
    scan_p = {s: probs[scans == s, 1].mean() for s in np.unique(scans)}

    def most_confident(mask, n):
        """Indices of the n most confident errors in ``mask``, at most one slice per scan,
        so the autopsy shows n different scans rather than neighbouring slices of one."""
        out, seen = [], set()
        for i in np.where(mask)[0][np.argsort(-conf[mask])]:
            if scans[i] not in seen:
                out.append(i)
                seen.add(scans[i])
            if len(out) == n:
                break
        return out

    picks = [("FN", i) for i in most_confident((pred == 0) & (labels == 1), n_fn)] + \
            [("FP", i) for i in most_confident((pred == 1) & (labels == 0), n_fp)]

    fig, axes = plt.subplots(2, len(picks), figsize=(3.2 * len(picks), 7.2))
    cases = []
    for col, (kind, i) in enumerate(picks):
        x = preprocess(images[i:i + 1].to(device), ckpt["mean"], ckpt["std"]).requires_grad_(True)
        cam = grad_cam(model, layer, x, int(pred[i]))
        img = F.interpolate(images[i:i + 1, None].float(), size=(224, 224), mode="bilinear")[0, 0].numpy()
        name = os.path.basename(str(paths[i]))
        n_wrong = int((pred[scans == scans[i]] != labels[i]).sum())
        axes[0, col].imshow(img, cmap="gray")
        axes[0, col].set_title(f"{kind}: scan {scans[i]}, slice {rank[i] + 1}/20\n"
                               f"true {CLASSES[labels[i]]}, slice P(AD)={probs[i, 1]:.3f}\n"
                               f"scan P(AD)={scan_p[scans[i]]:.3f}, {n_wrong}/20 slices wrong", fontsize=8.5)
        axes[1, col].imshow(img, cmap="gray")
        axes[1, col].imshow(cam, cmap="jet", alpha=0.45)
        axes[1, col].set_title(f"Grad-CAM for predicted '{CLASSES[pred[i]]}'", fontsize=8.5)
        cases.append({"type": kind, "file": name, "scan": str(scans[i]), "slice_rank": int(rank[i]),
                      "true": CLASSES[labels[i]], "p_ad": float(probs[i, 1]), "confidence": float(conf[i]),
                      "scan_p_ad": float(scan_p[scans[i]]), "scan_slices_wrong": n_wrong})
    for a in axes.flat:
        a.axis("off")
    fig.suptitle(f"{NAMES[tag]}: most confident test errors, one per scan (after temperature scaling)")
    fig.tight_layout(h_pad=2.0)
    fig.savefig(os.path.join(RES, "failure_cases.png"), dpi=120)
    plt.close(fig)
    return cases


def scan_disagreement(runs, a, b):
    """How confident is each model on the *scans the other one gets wrong*?

    If model A is confidently wrong where model B is unsure (P(AD) near 0.5), B's
    errors are catchable by a referral rule and A's are not, even if A is more accurate.
    """
    out = {}
    for wrong_tag, other in ((a, b), (b, a)):
        rw, ro = runs[wrong_tag], runs[other]
        wrong = rw["subj_probs"].argmax(1) != rw["subj_labels"]
        auto = wrong & (rw["subj_probs"].max(1) >= rw["scan_thr"])  # errors that would NOT be referred
        out[f"{wrong_tag}_wrong_scans"] = int(wrong.sum())
        out[f"{wrong_tag}_wrong_scans_automated"] = int(auto.sum())
        out[f"{wrong_tag}_mean_conf_on_own_errors"] = float(rw["subj_probs"].max(1)[wrong].mean())
        out[f"{other}_mean_conf_on_{wrong_tag}_errors"] = float(ro["subj_probs"].max(1)[wrong].mean())
        out[f"{other}_correct_on_{wrong_tag}_errors"] = int((ro["subj_probs"].argmax(1) == ro["subj_labels"])[wrong].sum())

    # Two cheap combinations, both at scan level:
    # (1) ensemble = average the two models' probabilities;
    # (2) "agreement" rule = automate only scans where both models predict the same class
    #     and each clears its own validation-chosen threshold; everything else is referred.
    ra, rb = runs[a], runs[b]
    y = ra["subj_labels"]
    ens = (ra["subj_probs"] + rb["subj_probs"]) / 2
    out["ensemble_scan_accuracy"] = float((ens.argmax(1) == y).mean())
    pa, pb = ra["subj_probs"].argmax(1), rb["subj_probs"].argmax(1)
    keep = (pa == pb) & (ra["subj_probs"].max(1) >= ra["scan_thr"]) & (rb["subj_probs"].max(1) >= rb["scan_thr"])
    out["agreement_rule"] = {
        "coverage": float(keep.mean()), "referred": int((~keep).sum()),
        "accepted_accuracy": float((pa[keep] == y[keep]).mean()),
        "auto_false_negatives": int(((pa == 0) & (y == 1) & keep).sum()),
        "auto_false_positives": int(((pa == 1) & (y == 0) & keep).sum()),
    }
    return out


# --------------------------------------------------------------------------- plots
def plot_reliability(runs):
    fig, axes = plt.subplots(1, len(runs), figsize=(5.2 * len(runs), 4.6), squeeze=False)
    for ax, (tag, r) in zip(axes[0], runs.items()):
        for key, style, lab in (("raw", "o-", "uncalibrated"), ("ts", "s--", f"temp. scaled (T={r['T']:.2f})")):
            bc, ba, bn, _ = reliability_bins(r[f"probs_{key}"], r["labels"], n_bins=10)
            m = bn > 0
            ece = classification_report(r[f"probs_{key}"], r["labels"])["ece"]
            ax.plot(bc[m], ba[m], style, label=f"{lab}, ECE={ece:.3f}")
        ax.plot([0.5, 1], [0.5, 1], "k:", label="perfect calibration")
        ax.set(xlabel="predicted confidence", ylabel="observed accuracy", title=NAMES[tag],
               xlim=(0.5, 1), ylim=(0.3, 1))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "reliability.png"), dpi=130)
    plt.close(fig)


def plot_confidence_hist(runs):
    fig, axes = plt.subplots(1, len(runs), figsize=(5.2 * len(runs), 3.8), squeeze=False)
    bins = np.linspace(0.5, 1, 26)
    for ax, (tag, r) in zip(axes[0], runs.items()):
        p = r["probs_ts"]
        conf, ok = p.max(1), p.argmax(1) == r["labels"]
        ax.hist(conf[ok], bins, alpha=0.6, label=f"correct (n={ok.sum()})", color="tab:green", density=True)
        ax.hist(conf[~ok], bins, alpha=0.6, label=f"wrong (n={(~ok).sum()})", color="tab:red", density=True)
        ax.set(xlabel="confidence (temp. scaled)", ylabel="density", title=NAMES[tag])
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "confidence_hist.png"), dpi=130)
    plt.close(fig)


def plot_risk_coverage(runs):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for tag, r in runs.items():
        cov, acc, _ = risk_coverage(r["probs_ts"], r["labels"])
        axes[0].plot(cov, acc, color=COLORS[tag], label=f"{NAMES[tag]} (slice)")
        cov, acc, _ = risk_coverage(r["subj_probs"], r["subj_labels"])
        axes[0].plot(cov, acc, color=COLORS[tag], ls="--", label=f"{NAMES[tag]} (scan)")
        fpr, tpr = roc_points(r["probs_ts"], r["labels"])
        auc = classification_report(r["probs_ts"], r["labels"])["auroc"]
        axes[1].plot(fpr, tpr, color=COLORS[tag], label=f"{NAMES[tag]} AUROC={auc:.3f}")
    axes[0].set(xlabel="coverage (fraction automated)", ylabel="accuracy on automated cases",
                title="Selective prediction (risk-coverage)", ylim=(0.6, 1.0))
    axes[1].plot([0, 1], [0, 1], "k:")
    axes[1].set(xlabel="false positive rate", ylabel="true positive rate (AD sensitivity)", title="ROC, test slices")
    for a in axes:
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "risk_coverage_roc.png"), dpi=130)
    plt.close(fig)


def plot_slice_position(runs):
    """Accuracy vs. the slice's rank within its scan (0 = first of the 20 slices).

    Raw slice numbers are not comparable across scans (each scan's 20 slices
    start at a different index), so the rank is used as the position.
    """
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for tag, r in runs.items():
        rank = slice_rank(r["paths"])
        ok = r["probs_ts"].argmax(1) == r["labels"]
        xs = np.unique(rank)
        ax.plot(xs, [ok[rank == s].mean() for s in xs], "o-", color=COLORS[tag], label=NAMES[tag])
    ax.set(xlabel="slice rank within scan (0-19)", ylabel="test accuracy", title="Accuracy by slice position")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "slice_position.png"), dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tags", nargs="+", default=["convnext", "baseline"])
    ap.add_argument("--target-acc", type=float, default=0.95,
                    help="accepted-case accuracy the referral threshold must reach on validation")
    ap.add_argument("--out-dir", default=None,
                    help="write figures and summary.json here instead of results/ (e.g. for ablations)")
    args = ap.parse_args()
    global RES
    if args.out_dir:
        RES = args.out_dir
        os.makedirs(RES, exist_ok=True)
    # tags such as "convnext_noflip" reuse the display name / colour of their architecture
    for tag in args.tags:
        if tag not in NAMES:
            base = "convnext" if tag.startswith("convnext") else "baseline"
            NAMES[tag] = f"{NAMES[base]} ({tag[len(base):].strip('_') or tag})"
            COLORS[tag] = COLORS[base]
    device = torch.device("cuda")

    runs, summary = {}, {}
    patient_of = load_patient_map()
    val_paths = load_patient_splits()[0]["val"][2]
    for tag in args.tags:
        z, meta = load_run(tag)
        T = fit_temperature(z["val_logits"], z["val_labels"])
        r = {"labels": z["test_labels"], "paths": z["test_paths"], "T": T,
             "probs_raw": softmax_np(z["test_logits"]), "probs_ts": softmax_np(z["test_logits"], T)}
        r["subj_probs"], r["subj_labels"], _ = scan_level(r["probs_ts"], r["labels"], r["paths"])
        r["pat_probs"], r["pat_labels"], _ = patient_level(r["probs_ts"], r["labels"], r["paths"], patient_of)
        val_ts = softmax_np(z["val_logits"], T)
        thr, val_cov = threshold_for_accuracy(val_ts, z["val_labels"], args.target_acc)
        # Same rule at scan level (one decision per MRI = what a radiologist reads).
        # Validation logits are stored in split order, so val_paths lines them up.
        vs_p, vs_l, _ = scan_level(val_ts, z["val_labels"], val_paths)
        scan_thr, scan_val_cov = threshold_for_accuracy(vs_p, vs_l, args.target_acc)
        r["scan_thr"] = scan_thr
        runs[tag] = r

        s = {
            "temperature": T,
            "slice_raw": classification_report(r["probs_raw"], r["labels"]),
            "slice_ts": classification_report(r["probs_ts"], r["labels"]),
            "scan": classification_report(r["subj_probs"], r["subj_labels"]),
            "patient": classification_report(r["pat_probs"], r["pat_labels"]),
            "val_threshold": {"target_acc": args.target_acc, "threshold": thr, "val_coverage": val_cov,
                              "test": apply_threshold(r["probs_ts"], r["labels"], thr)},
            "fixed_thresholds": {str(t): apply_threshold(r["probs_ts"], r["labels"], t)
                                 for t in (0.6, 0.7, 0.8, 0.9, 0.95)},
            "scan_thresholds": {str(t): apply_threshold(r["subj_probs"], r["subj_labels"], t)
                                   for t in (0.6, 0.7, 0.8, 0.9)},
            "scan_val_threshold": {"target_acc": args.target_acc, "threshold": scan_thr,
                                   "val_coverage": scan_val_cov,
                                   "test": apply_threshold(r["subj_probs"], r["subj_labels"], scan_thr)},
            "test_threshold_for_95": threshold_for_accuracy(r["probs_ts"], r["labels"], 0.95),
            "train_time_min": meta["train_time_s"] / 60, "train_peak_vram_mb": meta["train_peak_vram_mb"],
            "best_epoch": meta["best_epoch"], "epochs": meta["args"]["epochs"],
            "profile": profile(tag, device),
        }
        ok = r["probs_ts"].argmax(1) == r["labels"]
        conf = r["probs_ts"].max(1)
        s["mean_conf_correct"], s["mean_conf_wrong"] = float(conf[ok].mean()), float(conf[~ok].mean())
        s["wrong_with_conf_over_0.9"] = int(((~ok) & (conf > 0.9)).sum())
        # "Does confidence track correctness?" as one number: AUROC of confidence for
        # separating correct from wrong predictions (0.5 = confidence is useless for
        # spotting errors, 1.0 = every error is less confident than every correct case).
        # Temperature scaling is monotonic in the margin, so this is the same raw or scaled.
        s["confidence_auroc_slice"] = float(roc_auc_score(ok, conf))
        s_ok = r["subj_probs"].argmax(1) == r["subj_labels"]
        s["confidence_auroc_scan"] = float(roc_auc_score(s_ok, r["subj_probs"].max(1)))
        summary[tag] = s

        # store calibration + referral rule in the checkpoint for predict.py
        path = os.path.join(CKPT, f"{tag}_best.pt")
        ck = torch.load(path, map_location="cpu")
        ck["temperature"], ck["threshold"] = T, thr
        torch.save(ck, path)

    plot_reliability(runs)
    plot_confidence_hist(runs)
    plot_risk_coverage(runs)
    plot_slice_position(runs)
    images, _, _ = load_patient_splits()[0]["test"]  # same split as training (default seed/fractions)
    main_tag = args.tags[0]
    r = runs[main_tag]
    if len(args.tags) == 2:
        summary["scan_disagreement"] = scan_disagreement(runs, *args.tags)
    summary["failure_cases"] = failure_autopsy(main_tag, r["probs_ts"], r["labels"], r["paths"], images, device)

    with open(os.path.join(RES, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)

    # markdown tables for the README
    print("| Model | Acc | Macro-F1 | AUROC | Sens (AD) | Spec (NC) | ECE raw | ECE TS | Scan acc | Patient acc |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for tag in args.tags:
        s = summary[tag]
        a, b = s["slice_raw"], s["slice_ts"]
        print(f"| {NAMES[tag]} | {a['accuracy']:.4f} | {a['macro_f1']:.4f} | {a['auroc']:.4f} | "
              f"{a['sensitivity']:.4f} | {a['specificity']:.4f} | {a['ece']:.4f} | {b['ece']:.4f} | "
              f"{s['scan']['accuracy']:.4f} | {s['patient']['accuracy']:.4f} |")
    print(json.dumps({t: {k: summary[t][k] for k in ("temperature", "val_threshold", "profile",
                                                       "mean_conf_correct", "mean_conf_wrong",
                                                       "wrong_with_conf_over_0.9")} for t in args.tags}, indent=1))


if __name__ == "__main__":
    main()
