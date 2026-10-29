"""COMP3710 TimeGAN: training-only discrete mid-price movement baseline.

This is an independent empirical/i.i.d. baseline, NOT a TimeGAN improvement.
It generates returns only; it does not produce a consistent full order book.

Run in the Rangpur project folder:
    python evaluate_v4_4_discrete.py --data-dir data \
        --v43-file outputs_v4_3_generated_norm.npy \
        --output-dir outputs_v4_4_discrete
"""
import argparse
import json
from pathlib import Path

import numpy as np

from dataset import load_lobster, chronological_split
from dataset_v4 import prepare_dataset_v4


def histogram_kl(p_samples, q_samples, bins=50, smoothing=1e-8):
    """KL(P_real || Q_candidate) on shared real-data histogram edges.

    Fixed bins and smoothing across candidates; mass outside real range is
    included in extreme bins (rather than silently discarded).
    """
    p = np.asarray(p_samples, dtype=np.float64).ravel()
    q = np.asarray(q_samples, dtype=np.float64).ravel()
    if not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError("Non-finite values found in histogram inputs")
    lo, hi = float(p.min()), float(p.max())
    if lo == hi:
        lo -= 1e-12
        hi += 1e-12
    edges = np.linspace(lo, hi, bins + 1)
    # Histogram includes all candidate values by clipping outliers to endpoints.
    p_count, _ = np.histogram(np.clip(p, lo, hi), bins=edges)
    q_count, _ = np.histogram(np.clip(q, lo, hi), bins=edges)
    ph = p_count.astype(float) + smoothing
    qh = q_count.astype(float) + smoothing
    ph /= ph.sum()
    qh /= qh.sum()
    return float(np.sum(ph * np.log(ph / qh)))


def lag1_abs_autocorrelation(returns):
    x = np.abs(np.asarray(returns, dtype=float))
    left, right = x[:, :-1].ravel(), x[:, 1:].ravel()
    if left.std() <= 1e-16 or right.std() <= 1e-16:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def describe(name, values, threshold):
    x = np.asarray(values, dtype=float)
    moves = np.abs(x) > threshold
    metrics = {
        "mean": float(x.mean()),
        "std": float(x.std()),
        "near_zero_fraction": float(np.mean(~moves)),
        "positive_fraction": float(np.mean(x > threshold)),
        "negative_fraction": float(np.mean(x < -threshold)),
        "mean_absolute_nonzero": float(np.mean(np.abs(x[moves]))) if moves.any() else 0.0,
        "cross_sample_std": float(x.std(axis=0).mean()),
        "lag1_absolute_return_correlation": lag1_abs_autocorrelation(x),
    }
    print(f"\n{name}:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--v43-file", default="outputs_v4_3_generated_norm.npy")
    parser.add_argument("--output-dir", default="outputs_v4_4_discrete")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bins", type=int, default=50)
    parser.add_argument("--near-zero-threshold", type=float, default=1e-10)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    orderbook, _ = load_lobster(args.data_dir)
    raw = orderbook.to_numpy()
    mid2 = raw[:, 0].astype(np.int64) + raw[:, 2].astype(np.int64)
    train_ids, val_ids, test_ids = chronological_split(np.arange(len(mid2)))
    train_mid2 = mid2[train_ids]
    train_deltas = np.diff(train_mid2)
    if not np.all(train_deltas % 100 == 0):
        raise ValueError("Training changes do not all follow the observed 100-unit mid2 grid")
    train_ticks = train_deltas // 100
    tick_values, tick_counts = np.unique(train_ticks, return_counts=True)
    tick_probabilities = tick_counts.astype(float) / tick_counts.sum()
    print("=== V4.4 Training-Only Empirical Tick Baseline ===")
    print("Training transitions:", len(train_deltas))
    print("Training zero fraction:", float(np.mean(train_ticks == 0)))
    print("Empirical tick categories:", len(tick_values))

    dataset = prepare_dataset_v4(args.data_dir)
    normalizer = dataset["normalizer"]
    real_seq = normalizer.inverse_transform(dataset["test"])[..., -1]
    fake_v43_norm = np.load(args.v43_file)
    if fake_v43_norm.shape != dataset["test"].shape:
        raise ValueError(
            f"V4.3 shape {fake_v43_norm.shape} does not match test {dataset['test'].shape}"
        )
    fake_v43 = normalizer.inverse_transform(fake_v43_norm)[..., -1]
    n, t = real_seq.shape

    # I.i.d. transitions sampled exclusively from the training distribution.
    sampled_ticks = rng.choice(tick_values, size=(n, t), p=tick_probabilities)
    # Fixed per-sequence *initial* mid2, drawn only from the training period.
    # Independent initial prices and moves are a simplifying baseline assumption.
    initial_mid2 = rng.choice(train_mid2, size=(n, 1)).astype(np.int64)
    sampled_delta_mid2 = sampled_ticks * 100
    current_mid2 = initial_mid2 + np.cumsum(sampled_delta_mid2, axis=1)
    previous_mid2 = current_mid2 - sampled_delta_mid2
    if (current_mid2 <= 0).any() or (previous_mid2 <= 0).any():
        raise ValueError("Nonpositive synthetic mid-price; cannot compute log returns")
    fake_discrete = np.log(current_mid2.astype(float) / previous_mid2.astype(float))

    threshold = args.near_zero_threshold
    metrics = {
        "real_test": describe("Real test returns", real_seq, threshold),
        "timegan_v43": describe("TimeGAN V4.3 returns", fake_v43, threshold),
        "iid_tick_baseline": describe("Independent empirical tick baseline", fake_discrete, threshold),
    }
    kl_v43 = histogram_kl(real_seq, fake_v43, args.bins)
    kl_iid = histogram_kl(real_seq, fake_discrete, args.bins)
    print("\n=== Comparable Histogram KL (this script's estimator) ===")
    print("KL(real || V4.3):", kl_v43)
    print("KL(real || independent tick baseline):", kl_iid)
    print("IMPORTANT: These KL values may differ from evaluate.histogram_kl")
    print("due to bin edges, out-of-range handling and smoothing.")
    print("Baseline is NOT a learned generator and does NOT guarantee")
    print("temporal dependence or consistency with LOB price ladders.")

    np.save(out_dir / "discrete_returns.npy", fake_discrete)
    np.save(out_dir / "discrete_ticks.npy", sampled_ticks)
    results = {
        "seed": args.seed,
        "histogram_bins": args.bins,
        "near_zero_threshold": threshold,
        "training_only_zero_fraction": float(np.mean(train_ticks == 0)),
        "kl_real_vs_v43": kl_v43,
        "kl_real_vs_discrete": kl_iid,
        "metrics": metrics,
        "note": "Independent training-derived empirical baseline, not TimeGAN output.",
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\nSaved:", out_dir / "discrete_returns.npy")
    print("Saved:", out_dir / "discrete_ticks.npy")
    print("Saved:", out_dir / "results.json")


if __name__ == "__main__":
    main()
