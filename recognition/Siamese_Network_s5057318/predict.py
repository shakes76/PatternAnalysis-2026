"""Evaluate the trained baseline and Siamese models on the SAME test split (image level).

Thresholds are chosen on the validation split only; the test split is only
ever scored. Metrics are implemented by hand (no sklearn).
"""
import argparse
import math
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import dataset as d
from modules import BaselineCNN, SiameseEncoder
from train import subsample

HERE = os.path.dirname(os.path.abspath(__file__))
N_REF, N_REF_DEBUG = 200, 20
N_BOOT = 1000
RECALL_TARGET, PRECISION_TARGET = 0.95, 0.5


# ---------------------------------------------------------------- metrics
def auroc(scores, labels):
    """Area under the ROC curve via the rank-sum formula (ties get average ranks)."""
    scores, labels = np.asarray(scores, dtype=float), np.asarray(labels).astype(int)
    n_pos, n_neg = int(labels.sum()), int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(counts) - (counts - 1) / 2.0)[inverse]  # 1-based average ranks
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _precision_recall_points(scores, labels):
    """Precision/recall at every distinct threshold (score >= t), plus those thresholds."""
    scores, labels = np.asarray(scores, dtype=float), np.asarray(labels).astype(int)
    order = np.argsort(-scores, kind="mergesort")
    s, y = scores[order], labels[order]
    tp = np.cumsum(y)
    last_of_group = np.r_[s[1:] != s[:-1], True]  # one point per distinct score
    n_pred = np.arange(1, len(s) + 1)[last_of_group]
    tp = tp[last_of_group]
    return tp / n_pred, tp / max(int(y.sum()), 1), s[last_of_group]


def average_precision(scores, labels):
    """PR-AUC as average precision: sum over thresholds of (R_k - R_{k-1}) * P_k."""
    if int(np.asarray(labels).sum()) == 0:
        return float("nan")
    precision, recall, _ = _precision_recall_points(scores, labels)
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def point_metrics(scores, labels, threshold):
    """Recall, precision, F1 and balanced accuracy for the rule score >= threshold."""
    pred = np.asarray(scores) >= threshold
    y = np.asarray(labels).astype(bool)
    tp, fp = int((pred & y).sum()), int((pred & ~y).sum())
    fn, tn = int((~pred & y).sum()), int((~pred & ~y).sum())
    recall = tp / (tp + fn) if tp + fn else float("nan")
    precision = tp / (tp + fp) if tp + fp else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if tp else 0.0
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    return {"recall": recall, "precision": precision, "f1": f1,
            "balanced_acc": (recall + specificity) / 2, "accuracy": (tp + tn) / len(y)}


def select_threshold(scores, labels, rule, min_accuracy=0.80):
    """Pick a threshold on `scores` (rule: predict malignant when score >= threshold).

    Candidates are the distinct scores. rule "balanced": maximise balanced accuracy,
    ties to the lowest threshold. rule "recall_acc": highest recall among thresholds
    with accuracy >= min_accuracy, ties to the higher accuracy, then the lowest
    threshold. Returns (threshold, metrics dict), or (None, None) if none qualifies.
    """
    scores, y = np.asarray(scores, dtype=float), np.asarray(labels).astype(bool)
    candidates = np.unique(scores)  # ascending
    # Row c of `pred` holds the predictions at threshold candidates[c]; tp and tn are
    # then counted per candidate threshold.
    pred = scores[None, :] >= candidates[:, None]
    tp = (pred & y).sum(axis=1)
    tn = (~pred & ~y).sum(axis=1)
    eps = 1e-12  # float tolerance when comparing ties
    if rule == "balanced":
        balanced = (tp / y.sum() + tn / (~y).sum()) / 2
        index = int(np.flatnonzero(balanced >= balanced.max() - eps)[0])
    elif rule == "recall_acc":
        eligible = np.flatnonzero((tp + tn) / len(y) >= min_accuracy - eps)
        if len(eligible) == 0:
            return None, None
        keys = list(zip(tp[eligible], (tp + tn)[eligible]))  # (recall, accuracy) as counts
        index = int(eligible[keys.index(max(keys))])  # first max = lowest threshold on full ties
    else:
        raise ValueError(f"unknown rule {rule!r}")
    return float(candidates[index]), point_metrics(scores, y, candidates[index])


def choose_thresholds(val_scores, val_labels):
    """Operating points from the VALIDATION split only."""
    val_scores, val_labels = np.asarray(val_scores, dtype=float), np.asarray(val_labels)
    # t_low: sort the val malignant scores; `keep` = ceil(95% of them). t_low is the
    # keep-th highest malignant score, so exactly `keep` (or more, with ties) are >= t_low.
    malignant = np.sort(val_scores[val_labels == 1])
    keep = math.ceil(RECALL_TARGET * len(malignant) - 1e-9)  # malignant that must stay >= t_low
    t_low = float(malignant[len(malignant) - keep])  # highest t keeping >= 95% of val malignant
    # t_high: the lowest distinct val score whose precision (for score >= t) is >= 0.5;
    # None if no threshold reaches that precision.
    precision, _, thresholds = _precision_recall_points(val_scores, val_labels)
    ok = thresholds[precision >= PRECISION_TARGET]
    t_high = float(ok.min()) if len(ok) else None  # lowest t with val precision >= 0.5
    return t_low, t_high


def assign_tiers(scores, t_low, t_high):
    """0 = auto-clear, 1 = refer to specialist, 2 = flag."""
    scores = np.asarray(scores)
    tier = np.ones(len(scores), dtype=int)
    tier[scores < t_low] = 0
    if t_high is not None:
        tier[(scores >= t_high) & (scores >= t_low)] = 2
    return tier


def bootstrap_by_id(scores_a, scores_b, labels, ids, n_boot, seed):
    """Resample real IDs with replacement; return arrays of metric values per resample."""
    groups = {}
    for row, real_id in enumerate(ids):
        groups.setdefault(real_id, []).append(row)
    members = [np.array(rows) for rows in groups.values()]
    rng = np.random.default_rng(seed)
    out = {k: [] for k in ("auroc_a", "auroc_b", "ap_a", "ap_b")}
    skipped = 0
    for _ in range(n_boot):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])
        y = labels[idx]
        if y.sum() == 0 or y.sum() == len(y):
            skipped += 1
            continue
        out["auroc_a"].append(auroc(scores_a[idx], y))
        out["auroc_b"].append(auroc(scores_b[idx], y))
        out["ap_a"].append(average_precision(scores_a[idx], y))
        out["ap_b"].append(average_precision(scores_b[idx], y))
    return {k: np.array(v) for k, v in out.items()}, skipped


def wilson_interval(k, n, z=1.96):
    """Wilson score interval for a proportion k / n (95% by default)."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def print_tier_table(scores, labels, t_low, t_high, indent=""):
    """Tier counts for `scores` (same format as the main tier table)."""
    tier = assign_tiers(scores, t_low, t_high)
    for k, tn in enumerate(["auto-clear", "refer", "flag"]):
        sel = tier == k
        n_k, mal_k = int(sel.sum()), int(labels[sel].sum())
        print(f"{indent}{tn:10s} {n_k:5d} images ({pct(n_k, len(scores)):>6})  malignant in tier: "
              f"{mal_k} (malignant share of tier: {pct(mal_k, n_k)})")
    missed, total = int(labels[tier == 0].sum()), int(labels.sum())
    print(f"{indent}malignant images in auto-clear: {missed} of {total} ({pct(missed, total)})")


# ---------------------------------------------------------------- inference
def sync(device):
    """Wait for queued GPU work so timings are accurate (does nothing on the CPU)."""
    if device.type == "cuda":
        torch.cuda.synchronize()


def run_scores(score_fn, loader, device):
    """Score every image in `loader`. Returns (scores, labels, seconds spent in score_fn)."""
    scores, labels, seconds = [], [], 0.0
    with torch.no_grad():
        for images, y in loader:
            images = images.to(device)
            sync(device)
            t0 = time.perf_counter()
            s = score_fn(images)
            sync(device)
            seconds += time.perf_counter() - t0
            scores.append(s.float().cpu())
            labels.append(y)
    return torch.cat(scores).numpy(), torch.cat(labels).numpy().astype(int), seconds


def load_checkpoint(model, path, device):
    """Load the weights saved by train.py into `model`; return (model in eval mode, checkpoint dict)."""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt


def n_params(model):
    """Number of trainable parameters in `model`."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def pct(a, b):
    """a / b as a percentage string with one decimal, or "n/a" when b is 0."""
    return f"{100 * a / b:.1f}%" if b else "n/a"


def main():
    """Score val and test with both models, choose thresholds on val only and print every section."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", help="folder with cleaned_metadata.csv and combined_images/")
    parser.add_argument("--out_dir", default="outputs",
                        help="relative paths are taken from this script's folder")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_seed", type=int, default=None,
                        help="seed of the train/val/test split (default: same as --seed)")
    parser.add_argument("--k", type=int, default=5,
                        help="nearest references averaged per class in the siamese score")
    parser.add_argument("--n_refs", type=int, default=None,
                        help=f"reference photos per class (default {N_REF}, or {N_REF_DEBUG} with --debug; "
                             "0 = every training image)")
    parser.add_argument("--score", choices=["knn", "prototype"], default="knn",
                        help="knn: k nearest refs per class; prototype: distance to each class's mean ref")
    parser.add_argument("--preprocess", choices=["none", "grayworld"], default="none",
                        help="colour-constancy step for all images (must match training)")
    parser.add_argument("--no_cudnn", action="store_true", help="disable cuDNN")
    parser.add_argument("--debug", action="store_true",
                        help="128 val / 128 test images, 20 benign + 20 malignant references")
    args = parser.parse_args()
    if args.split_seed is None:
        args.split_seed = args.seed
    if args.no_cudnn:
        torch.backends.cudnn.enabled = False
    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(HERE, args.out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    print(f"device: {device} | seed: {args.seed} | cudnn: {torch.backends.cudnn.enabled}"
          f"{' | DEBUG' if args.debug else ''}")
    n_ref = args.n_refs if args.n_refs is not None else (N_REF_DEBUG if args.debug else N_REF)
    print(f"siamese scoring: --score {args.score} | --n_refs {n_ref}"
          f"{' (every training image)' if n_ref == 0 else ' per class'}"
          f"{f' | --k {args.k}' if args.score == 'knn' else ''}")

    image_dir = os.path.join(args.data_dir, "combined_images")
    splits = d.split_by_id(d.load_clean_metadata(args.data_dir), seed=args.split_seed)
    d.check_no_id_overlap(splits)
    if args.debug:  # same subsets train.py --debug used, so the normalisation matches
        splits = subsample(splits, args.seed)
    # Eval transform (no augmentation) for every split; mean/std from the train split only.
    _, val_loader, test_loader = d.get_dataloaders(splits, image_dir, batch_size=64, seed=args.seed,
                                                 preprocess=args.preprocess)
    test_ds = test_loader.dataset
    mean, std = test_ds.mean, test_ds.std
    test_df = splits["test"].reset_index(drop=True)
    print(f"val: {len(splits['val'])} images | test: {len(test_df)} images "
          f"({int(test_df['target'].sum())} malignant)")

    # ------------------------------------------------ 1. scores
    baseline, base_ckpt = load_checkpoint(BaselineCNN(), os.path.join(out_dir, "baseline_best.pt"), device)
    encoder, siam_ckpt = load_checkpoint(SiameseEncoder(), os.path.join(out_dir, "siamese_best.pt"), device)
    for name, ck in (("baseline", base_ckpt), ("siamese", siam_ckpt)):
        ck_args = ck.get("args", {})
        ck_seed = ck_args.get("seed")
        ck_split = ck_args.get("split_seed", ck_seed)  # older checkpoints: split used --seed
        print(f"{name} checkpoint: seed {ck_seed}, split_seed {ck_split} | "
              f"using: seed {args.seed}, split_seed {args.split_seed}")
        if (ck_seed, ck_split) != (args.seed, args.split_seed):
            print(f"WARNING: {name} checkpoint seeds differ from the ones being used")
        ck_pre = ck_args.get("preprocess", "none")  # older checkpoints: no preprocessing
        if ck_pre != args.preprocess:
            print(f"WARNING: {name} checkpoint was trained with preprocess={ck_pre}, "
                  f"but --preprocess {args.preprocess} was passed")

    # Fixed reference set drawn from the TRAIN split only (seeded).
    train_df = splits["train"].reset_index(drop=True)
    gen = torch.Generator().manual_seed(args.seed)
    ref_rows = {}
    for cls in (0, 1):
        pool = torch.from_numpy(np.flatnonzero(train_df["target"].to_numpy() == cls))
        ref_rows[cls] = pool[torch.randperm(len(pool), generator=gen)[:n_ref or None]].tolist()  # reference set drawn here
        if len(ref_rows[cls]) < n_ref:
            print(f"note: only {len(ref_rows[cls])} class-{cls} images in the train split")
    ref_loaders = {cls: torch.utils.data.DataLoader(
        torch.utils.data.Subset(d.ISICDataset(train_df, image_dir, test_ds.transform), rows),
        batch_size=64) for cls, rows in ref_rows.items()}

    ref_emb, t_ref = {}, 0.0
    with torch.no_grad():
        for cls, loader in ref_loaders.items():
            embs = []
            for images, _ in loader:
                images = images.to(device)
                sync(device)
                t0 = time.perf_counter()
                embs.append(encoder(images))
                sync(device)
                t_ref += time.perf_counter() - t0
            ref_emb[cls] = torch.cat(embs)
    n_refs_total = len(ref_emb[0]) + len(ref_emb[1])

    # Prototype of each class = mean of its reference embeddings (not re-normalised).
    prototypes = {cls: ref_emb[cls].mean(dim=0, keepdim=True) for cls in (0, 1)}

    # Siamese score, higher = more malignant-like. prototype: distance to the benign
    # prototype minus distance to the malignant prototype. knn: mean distance to the k
    # nearest benign references minus mean distance to the k nearest malignant references.
    def siamese_score(images):
        """Siamese score for a batch: distance to the benign references minus distance to the malignant references."""
        emb = encoder(images)
        if args.score == "prototype":
            return (torch.cdist(emb, prototypes[0]) - torch.cdist(emb, prototypes[1])).squeeze(1)

        def mean_nearest(refs):
            """Mean distance from each embedding to its k nearest references in refs."""
            return torch.cdist(emb, refs).topk(args.k, dim=1, largest=False).values.mean(dim=1)
        return mean_nearest(ref_emb[0]) - mean_nearest(ref_emb[1])  # higher = more malignant-like

    models = {"baseline": lambda x: baseline(x).squeeze(1), "siamese": siamese_score}
    val_scores, test_scores, resources = {}, {}, {}
    for name, fn in models.items():
        val_scores[name], val_labels, _ = run_scores(fn, val_loader, device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        test_scores[name], test_labels, secs = run_scores(fn, test_loader, device)
        peak = (f"{torch.cuda.max_memory_allocated() / 2**20:.1f} MB" if device.type == "cuda"
                else "n/a (cpu)")
        resources[name] = (peak, 1000 * secs / len(test_labels))
    assert (test_labels == test_df["target"].to_numpy().astype(int)).all()
    ids = test_df["real_id"].tolist()
    n_test, n_mal = len(test_labels), int(test_labels.sum())

    # ------------------------------------------------ 3. operating points (VAL only)
    thresholds = {}
    print("\n== Operating points (chosen on VAL only) ==")
    for name in models:
        t_low, t_high = choose_thresholds(val_scores[name], val_labels)
        thresholds[name] = (t_low, t_high)
        high_txt = "none" if t_high is None else f"{t_high:.4f}"
        note = "  (t_high < t_low: refer tier is empty)" if t_high is not None and t_high < t_low else ""
        print(f"{name:9s} t_low={t_low:.4f}  t_high={high_txt}{note}")

    # ------------------------------------------------ 2. metrics on test
    print(f"\n== Test metrics ({n_test} images, {n_mal} malignant) ==")
    print(f"always-benign accuracy: {(n_test - n_mal) / n_test:.4f}")
    for name in models:
        print(f"{name:9s} AUROC={auroc(test_scores[name], test_labels):.4f}  "
              f"PR-AUC={average_precision(test_scores[name], test_labels):.4f}")
        t_low, t_high = thresholds[name]
        for label, t in (("t_low", t_low), ("t_high", t_high)):
            if t is None:
                print(f"   at {label}: none")
                continue
            m = point_metrics(test_scores[name], test_labels, t)
            print(f"   at {label}={t:.4f}: recall={m['recall']:.4f}  precision={m['precision']:.4f}  "
                  f"F1={m['f1']:.4f}  balanced acc={m['balanced_acc']:.4f}")

    # ------------------------------------------------ 3. tiers on test
    print("\n== Triage tiers on test (thresholds from val) ==")
    tier_names = ["auto-clear", "refer", "flag"]
    for name in models:
        t_low, t_high = thresholds[name]
        tier = assign_tiers(test_scores[name], t_low, t_high)
        print(name)
        for k, tn in enumerate(tier_names):
            sel = tier == k
            n_k, mal_k = int(sel.sum()), int(test_labels[sel].sum())
            print(f"   {tn:10s} {n_k:5d} images ({pct(n_k, n_test):>6})  malignant in tier: {mal_k} "
                  f"(malignant share of tier: {pct(mal_k, n_k)})")
        missed = int(test_labels[tier == 0].sum())
        print(f"   malignant images in auto-clear: {missed} of {n_mal} ({pct(missed, n_mal)})")

    # ------------------------------------------------ 4. bootstrap
    boot, skipped = bootstrap_by_id(test_scores["baseline"], test_scores["siamese"],
                                    test_labels, ids, N_BOOT, args.seed)

    def ci(x):
        """Format the 95% bootstrap interval (2.5th to 97.5th percentile) of x as '[low, high]'."""
        lo, hi = np.percentile(x, [2.5, 97.5])
        return f"[{lo:.4f}, {hi:.4f}]"
    print(f"\n== Bootstrap by real_id ({N_BOOT} resamples, {len(set(ids))} test IDs, "
          f"{skipped} skipped for one class only) ==")
    print(f"AUROC   baseline {ci(boot['auroc_a'])}  siamese {ci(boot['auroc_b'])}  "
          f"siamese-baseline {ci(boot['auroc_b'] - boot['auroc_a'])}")
    print(f"PR-AUC  baseline {ci(boot['ap_a'])}  siamese {ci(boot['ap_b'])}  "
          f"siamese-baseline {ci(boot['ap_b'] - boot['ap_a'])}")

    # ------------------------------------------------ extra analyses (thresholds unchanged)
    names = list(models)
    ids_arr = np.array(ids)
    sources = np.array(["IP" if i.startswith("IP_") else "HAM" if i.startswith("HAM_") else "other"
                        for i in ids])
    tiers = {name: assign_tiers(test_scores[name], *thresholds[name]) for name in names}

    # By source: same t_low / t_high as above (chosen on all val images).
    print("\n== By source (same t_low / t_high as above, chosen on all val images) ==")
    for src in ("IP", "HAM"):
        m = sources == src
        n_s, mal_s = int(m.sum()), int(test_labels[m].sum())
        if n_s == 0:
            print(f"source {src}: no test images")
            continue
        print(f"source {src}: {n_s} test images, {mal_s} malignant, "
              f"always-benign accuracy {(n_s - mal_s) / n_s:.4f}")
        if mal_s < 20:
            print(f"   WARNING: only {mal_s} malignant test images in {src}; estimates are unreliable")
        for name in names:
            s_src = test_scores[name][m]
            print(f"   {name:9s} AUROC={auroc(s_src, test_labels[m]):.4f}  "
                  f"PR-AUC={average_precision(s_src, test_labels[m]):.4f}")
            print_tier_table(s_src, test_labels[m], *thresholds[name], indent="      ")
    n_other = int((sources == "other").sum())
    if n_other:
        print(f"note: {n_other} test images have an ID that is neither IP_ nor HAM_")

    # Bootstrap by real_id within each source.
    print(f"\n== AUROC bootstrap by real_id within each source ({N_BOOT} resamples) ==")
    for src in ("IP", "HAM"):
        m = sources == src
        if not m.any():
            continue
        b, skipped = bootstrap_by_id(test_scores["baseline"][m], test_scores["siamese"][m],
                                     test_labels[m], ids_arr[m].tolist(), N_BOOT, args.seed)
        if len(b["auroc_a"]) == 0:
            print(f"{src}: no usable resamples")
            continue
        print(f"{src} ({len(set(ids_arr[m]))} IDs, {skipped} skipped): baseline {ci(b['auroc_a'])}  "
              f"siamese {ci(b['auroc_b'])}  siamese-baseline {ci(b['auroc_b'] - b['auroc_a'])}")

    # Confidence of mistakes.
    cap = 25
    print("\n== Confidence of mistakes ==")
    for name in names:
        t_low, t_high = thresholds[name]
        s_all, tier = test_scores[name], tiers[name]
        print(name)
        bad = np.flatnonzero((tier == 0) & (test_labels == 1))
        bad = bad[np.argsort(-(t_low - s_all[bad]))]  # most confidently wrong first
        print(f"   malignant images in auto-clear: {len(bad)}  "
              "(isic_id, real_id, source, score, t_low - score)")
        for row in bad[:cap]:
            print(f"      {test_df.at[row, 'isic_id']}  {ids[row]}  {sources[row]}  "
                  f"score={s_all[row]:.4f}  t_low-score={t_low - s_all[row]:.4f}")
        if len(bad) > cap:
            print(f"      ... and {len(bad) - cap} more")
        ok = (tier == 0) & (test_labels == 0)
        print(f"   median (t_low - score) over {int(ok.sum())} benign images in auto-clear: "
              f"{np.median(t_low - s_all[ok]) if ok.any() else float('nan'):.4f}"
              + (f"   [malignant mistakes: {np.median(t_low - s_all[bad]):.4f}]" if len(bad) else ""))
        if t_high is None:
            print("   no flag tier (t_high = none)")
            continue
        bad = np.flatnonzero((tier == 2) & (test_labels == 0))
        bad = bad[np.argsort(-(s_all[bad] - t_high))]
        print(f"   benign images in flag: {len(bad)}  "
              "(isic_id, real_id, source, score, score - t_high)")
        for row in bad[:cap]:
            print(f"      {test_df.at[row, 'isic_id']}  {ids[row]}  {sources[row]}  "
                  f"score={s_all[row]:.4f}  score-t_high={s_all[row] - t_high:.4f}")
        if len(bad) > cap:
            print(f"      ... and {len(bad) - cap} more")
        ok = (tier == 2) & (test_labels == 1)
        print(f"   median (score - t_high) over {int(ok.sum())} malignant images in flag: "
              f"{np.median(s_all[ok] - t_high) if ok.any() else float('nan'):.4f}"
              + (f"   [benign mistakes: {np.median(s_all[bad] - t_high):.4f}]" if len(bad) else ""))

    # Score deciles.
    print("\n== Malignant rate by test-score decile (10 equal-count bins, lowest scores first) ==")
    for name in names:
        order = np.argsort(test_scores[name], kind="stable")
        print(name)
        print("   bin  images  score range                malignant  rate")
        for k, rows in enumerate(np.array_split(order, 10), 1):
            sc, mal = test_scores[name][rows], int(test_labels[rows].sum())
            print(f"   {k:3d}  {len(rows):6d}  [{sc.min():9.4f}, {sc.max():9.4f}]  {mal:9d}  "
                  f"{pct(mal, len(rows)):>6}")

    # Wilson interval for the auto-clear miss rate.
    print("\n== Auto-clear miss rate (malignant in auto-clear / all malignant), Wilson 95% ==")
    for name in names:
        k = int(((tiers[name] == 0) & (test_labels == 1)).sum())
        lo, hi = wilson_interval(k, n_mal)
        print(f"{name:9s} {k} of {n_mal} = {pct(k, n_mal)}  95% interval [{100 * lo:.2f}%, {100 * hi:.2f}%]")

    # ------------------------------------------------ 5. plots
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, name in zip(axes, models):
        s = test_scores[name]
        bins = np.histogram_bin_edges(s, bins=30)
        ax.hist(s[test_labels == 0], bins=bins, alpha=0.6, density=True, label="benign")
        ax.hist(s[test_labels == 1], bins=bins, alpha=0.6, density=True, label="malignant")
        t_low, t_high = thresholds[name]
        ax.axvline(t_low, color="k", ls="--", lw=1, label="t_low (val)")
        if t_high is not None:
            ax.axvline(t_high, color="r", ls="--", lw=1, label="t_high (val)")
        ax.set_title(f"{name}: test scores")
        ax.set_xlabel("score (higher = more malignant-like)")
        ax.set_ylabel("density")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "score_hist.png"), dpi=120)
    plt.close(fig)

    m_t, s_t = torch.tensor(mean).view(3, 1, 1), torch.tensor(std).view(3, 1, 1)
    fig, axes = plt.subplots(len(models), 5, figsize=(15, 3.8 * len(models)))
    for r, name in enumerate(models):
        s = test_scores[name]
        mal_rows = np.flatnonzero(test_labels == 1)
        ben_rows = np.flatnonzero(test_labels == 0)
        picks = (mal_rows[np.argsort(s[mal_rows])[:3]].tolist()      # 3 lowest-scoring malignant
                 + ben_rows[np.argsort(-s[ben_rows])[:2]].tolist())  # 2 highest-scoring benign
        for c in range(5):
            ax = axes[r, c]
            ax.axis("off")
            if c >= len(picks):
                continue
            row = picks[c]
            image = test_ds[row][0]
            ax.imshow((image * s_t + m_t).clamp(0, 1).permute(1, 2, 0).numpy())
            ax.set_title(f"{name}\n{test_df.at[row, 'isic_id']}\n{ids[row]}\n"
                         f"label={'malignant' if test_labels[row] else 'benign'}  score={s[row]:.3f}",
                         fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "failures.png"), dpi=100)
    plt.close(fig)
    print(f"\nsaved {os.path.join(out_dir, 'score_hist.png')} and {os.path.join(out_dir, 'failures.png')}")

    # ------------------------------------------------ 6. resources
    print("\n== Resources ==")
    print(f"{'model':9s} {'parameters':>11s} {'peak GPU mem (test inference)':>31s} {'ms/image':>10s}")
    for name, model in (("baseline", baseline), ("siamese", encoder)):
        peak, ms = resources[name]
        print(f"{name:9s} {n_params(model):>11d} {peak:>31s} {ms:>10.2f}")
    print(f"siamese reference-set embedding (separate): {n_refs_total} images in {t_ref:.2f}s "
          f"= {1000 * t_ref / n_refs_total:.2f} ms/image (one-off cost)")

    # ------------------------------------------------ extra: alternative thresholds (VAL only)
    def fmt(x):
        """Format x to 4 decimal places, or return 'n/a' if x is NaN."""
        return "n/a" if x != x else f"{x:.4f}"  # NaN -> n/a

    print("\n== Alternative thresholds chosen on VAL only (predict malignant when score >= threshold) ==")
    rules = (("(a) maximise balanced accuracy", "balanced"),
             ("(b) highest recall with val accuracy >= 0.80", "recall_acc"))
    for name in names:
        for title, rule in rules:
            t, val_m = select_threshold(val_scores[name], val_labels, rule)
            if t is None:
                print(f"{name} {title}: none")
                continue
            print(f"{name} {title}: threshold={t:.4f}")
            rows = [("val (overall)", val_scores[name], val_labels)]
            rows += [("test overall", test_scores[name], test_labels)]
            rows += [(f"test {src}", test_scores[name][sources == src], test_labels[sources == src])
                     for src in ("IP", "HAM")]
            for label, s_rows, y_rows in rows:
                m = point_metrics(s_rows, y_rows, t)
                print(f"   {label:13s} n={len(y_rows):5d} malignant={int(y_rows.sum()):4d}  "
                      f"accuracy={fmt(m['accuracy'])}  recall={fmt(m['recall'])}  "
                      f"precision={fmt(m['precision'])}  F1={fmt(m['f1'])}  "
                      f"balanced acc={fmt(m['balanced_acc'])}")

    # ------------------------------------------------ extra: IP_-only validation thresholds
    print("\n== IP-only validation thresholds ==")
    val_df = splits["val"].reset_index(drop=True)
    assert (val_labels == val_df["target"].to_numpy().astype(int)).all()
    ip_val = val_df["real_id"].str.startswith("IP_").to_numpy()
    ip_test = test_df["real_id"].str.startswith("IP_").to_numpy()
    for what, got, want in (
            ("IP val images", int(ip_val.sum()), 1595),
            ("IP val malignant", int(val_labels[ip_val].sum()), 86),
            ("IP test images", int(ip_test.sum()), 1721),
            ("IP test malignant", int(test_labels[ip_test].sum()), 89)):
        print(f"{'PASS' if got == want else 'FAIL'}  {what}: {got} (expected {want})")
    y_ip_test = test_labels[ip_test]
    for name in names:
        s_val, y_val = val_scores[name][ip_val], val_labels[ip_val]
        s_test = test_scores[name][ip_test]
        t_low, t_high = choose_thresholds(s_val, y_val)
        t_bal, _ = select_threshold(s_val, y_val, "balanced")
        print(f"\n{name}: thresholds from IP validation images only")
        for title, t in (("t_low", t_low), ("t_high", t_high), ("balanced-accuracy", t_bal)):
            if t is None:
                print(f"  {title}: none")
                continue
            m = point_metrics(s_test, y_ip_test, t)
            print(f"  {title}={t:.4f}  IP test images={len(y_ip_test)} malignant={int(y_ip_test.sum())}  "
                  f"accuracy={fmt(m['accuracy'])}  recall={fmt(m['recall'])}  "
                  f"precision={fmt(m['precision'])}  F1={fmt(m['f1'])}  "
                  f"balanced acc={fmt(m['balanced_acc'])}")
        print("  three-tier table on IP test images (t_low / t_high above):")
        print_tier_table(s_test, y_ip_test, t_low, t_high, indent="     ")
        k = int(((assign_tiers(s_test, t_low, t_high) == 0) & (y_ip_test == 1)).sum())
        lo, hi = wilson_interval(k, int(y_ip_test.sum()))
        print(f"     auto-clear miss rate: {k} of {int(y_ip_test.sum())} = {pct(k, int(y_ip_test.sum()))}"
              f"  (Wilson 95% interval {100 * lo:.1f}% to {100 * hi:.1f}%)")


    # ------------------------------------------------ extra: validation metrics
    print("\n== Validation metrics ==")
    for name in names:
        print(f"{name:9s} AUROC={auroc(val_scores[name], val_labels):.4f}  "
              f"PR-AUC={average_precision(val_scores[name], val_labels):.4f}")

if __name__ == "__main__":
    main()
