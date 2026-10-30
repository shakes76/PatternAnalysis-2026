"""Calibration and referral analysis at scan level.

Takes per-scan probabilities for validation and test (from predict.py), then:
  1. measures calibration on test (ECE, Brier, NLL) and draws a reliability diagram,
  2. fits a temperature on validation only, and reports test calibration after it,
  3. picks a referral threshold on validation only (reject scans whose confidence is below it),
     then reports what that threshold does on test.

Example:
  python calibrate.py --val ~/adni_work/gfnet_clean_r5_stem_s42_val_preds.json \
                      --test ~/adni_work/gfnet_clean_r5_stem_s42_test_preds.json --name final
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPS = 1e-6


def load(path):
    d = json.load(open(os.path.expanduser(path)))
    ids = sorted(d["scan_prob"])
    p = np.array([d["scan_prob"][i] for i in ids], dtype=float)
    y = np.array([d["scan_label"][i] for i in ids], dtype=int)
    return p, y


def confidence(p):
    """Confidence in the predicted class (0.5 = coin flip, 1.0 = certain)."""
    return np.maximum(p, 1 - p)


def reliability(p, y, bins=5):
    conf, correct = confidence(p), ((p > 0.5).astype(int) == y)
    idx = np.minimum(((conf - 0.5) / 0.5 * bins).astype(int), bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        rows.append({"lo": 0.5 + 0.5 * b / bins, "hi": 0.5 + 0.5 * (b + 1) / bins, "n": int(m.sum()),
                     "conf": float(conf[m].mean()) if m.any() else float("nan"),
                     "acc": float(correct[m].mean()) if m.any() else float("nan")})
    return rows


def ece(p, y, bins=5):
    rows = reliability(p, y, bins)
    return float(sum(r["n"] / len(y) * abs(r["acc"] - r["conf"]) for r in rows if r["n"]))


def nll(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def scale(p, T):
    z = np.log(np.clip(p, EPS, 1 - EPS) / (1 - np.clip(p, EPS, 1 - EPS)))
    return 1 / (1 + np.exp(-z / T))


def fit_temperature(p, y):
    grid = np.linspace(0.3, 6.0, 571)
    return float(grid[int(np.argmin([nll(scale(p, T), y) for T in grid]))])


def referral(p, y, t):
    conf, pred = confidence(p), (p > 0.5).astype(int)
    keep = conf >= t
    n, na = len(y), int(keep.sum())
    acc = lambda m: float((pred[m] == y[m]).mean()) if m.any() else float("nan")
    return {"threshold": float(t), "accepted": na, "referred": n - na, "coverage": na / n,
            "acc_accepted": acc(keep), "acc_referred": acc(~keep), "acc_all": acc(np.ones(n, bool)),
            "ad_total": int((y == 1).sum()),
            "ad_missed_no_referral": int(((pred == 0) & (y == 1)).sum()),
            "ad_missed_after_referral": int(((pred == 0) & (y == 1) & keep).sum()),
            "ad_referred": int(((y == 1) & ~keep).sum())}


def choose_threshold(p, y, target, min_cov):
    """Smallest threshold (most scans kept) whose accepted accuracy on validation reaches target."""
    cands = sorted(set(confidence(p).tolist()))
    ok = [t for t in cands if referral(p, y, t)["coverage"] >= min_cov]
    for t in ok:
        if referral(p, y, t)["acc_accepted"] >= target:
            return t, True
    best = max(ok, key=lambda t: (round(referral(p, y, t)["acc_accepted"], 6), -t))
    return best, False


def reliability_plot(ax, p, y, bins, title):
    rows = [r for r in reliability(p, y, bins) if r["n"]]
    ax.plot([0.5, 1], [0.5, 1], "k--", lw=1, label="perfect calibration")
    ax.bar([r["conf"] for r in rows], [r["acc"] for r in rows], width=0.4 / bins * 1.0, color="#4c78a8", label="observed accuracy")
    for r in rows:
        ax.text(r["conf"], r["acc"] + 0.015, f"n={r['n']}", ha="center", fontsize=8)
    ax.set_xlim(0.5, 1.0); ax.set_ylim(0, 1.08)
    ax.set_xlabel("confidence in predicted class"); ax.set_ylabel("accuracy")
    ax.set_title(f"{title}\nECE {ece(p, y, bins):.3f}"); ax.legend(loc="upper left", fontsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--name", default="final")
    ap.add_argument("--bins", type=int, default=5)
    ap.add_argument("--target", type=float, default=0.95, help="target accuracy on accepted validation scans")
    ap.add_argument("--min-coverage", type=float, default=0.5, help="never refer more than 1 minus this fraction")
    args = ap.parse_args()

    pv, yv = load(args.val)
    pt, yt = load(args.test)
    T = fit_temperature(pv, yv)
    pt_T, pv_T = scale(pt, T), scale(pv, T)

    cal = {"n_val": len(yv), "n_test": len(yt), "temperature": T}
    for tag, pp in [("test_raw", pt), ("test_scaled", pt_T)]:
        cal[tag] = {"ece": ece(pp, yt, args.bins), "brier": brier(pp, yt), "nll": nll(pp, yt)}

    t, reached = choose_threshold(pv_T, yv, args.target, args.min_coverage)
    ref = {"val": referral(pv_T, yv, t), "test": referral(pt_T, yt, t), "target_reached_on_val": reached}

    print(json.dumps({"calibration": cal, "referral": ref}, indent=2))
    print("\nMarkdown for README:")
    print("| | ECE | Brier | NLL |\n|---|---|---|---|")
    for k, lab in [("test_raw", "raw"), ("test_scaled", f"after temperature scaling (T={T:.2f})")]:
        c = cal[k]
        print(f"| test, {lab} | {c['ece']:.3f} | {c['brier']:.3f} | {c['nll']:.3f} |")
    print("\n| split | threshold | scans referred | accuracy, all | accuracy, accepted | AD missed (no referral -> with referral) |")
    print("|---|---|---|---|---|---|")
    for s in ["val", "test"]:
        r = ref[s]
        print(f"| {s} | {r['threshold']:.3f} | {r['referred']}/{r['accepted'] + r['referred']} "
              f"({100 * (1 - r['coverage']):.0f}%) | {r['acc_all']:.3f} | {r['acc_accepted']:.3f} | "
              f"{r['ad_missed_no_referral']} -> {r['ad_missed_after_referral']} of {r['ad_total']} |")
    if not reached:
        print(f"\nNOTE: target accuracy {args.target} was not reachable on validation with coverage >= "
              f"{args.min_coverage}; used the best available threshold.")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    reliability_plot(ax[0], pt, yt, args.bins, "Test, raw probabilities")
    reliability_plot(ax[1], pt_T, yt, args.bins, f"Test, temperature scaled (T={T:.2f})")
    fig.tight_layout(); fig.savefig(f"calibration_{args.name}.png", dpi=140)

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for lab, pp, yy, col in [("validation", pv_T, yv, "#999999"), ("test", pt_T, yt, "#4c78a8")]:
        ts = sorted(set(confidence(pp).tolist()))
        pts = [(referral(pp, yy, x)["coverage"], referral(pp, yy, x)["acc_accepted"]) for x in ts]
        pts = [q for q in pts if q[0] >= 0.2]
        ax.plot([q[0] for q in pts], [q[1] for q in pts], color=col, label=lab)
    ax.axvline(ref["test"]["coverage"], color="#e45756", ls="--", lw=1, label=f"chosen threshold {t:.2f}")
    ax.set_xlabel("coverage (fraction of scans kept)"); ax.set_ylabel("accuracy on kept scans")
    ax.set_title("Accuracy vs referral"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"referral_{args.name}.png", dpi=140)
    json.dump({"calibration": cal, "referral": ref}, open(f"calibration_{args.name}.json", "w"), indent=2)
    print(f"\nsaved calibration_{args.name}.png, referral_{args.name}.png, calibration_{args.name}.json")


if __name__ == "__main__":
    main()
