"""COMP3710 Project 7 - TimeGAN baseline evaluation (V2 normalized data).

Compare generated sequences with (1) randomly selected *training* sequences
and (2) a repeated mean training sequence. Both baselines operate on full
40-feature, 24-step order-book sequences.

IMPORTANT: real training resampling is an oracle-like reference, not a genuine
synthetic data generator. Random-pair SSIM does not measure aligned reconstruction.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from dataset_v2 import prepare_dataset_v2
from evaluate import evaluate  # Same histogram KL / MAE calculation as prior runs
from market_constraints import check_market_constraints
from ssim_evaluation import depth_heatmaps, get_ssim, pairwise_ssim, describe


def require_shape(name, x, reference):
    x = np.asarray(x)
    if x.ndim != 3 or x.shape[1:] != reference.shape[1:]:
        raise ValueError(
            f"{name}: expected (N, {reference.shape[1]}, {reference.shape[2]}), "
            f"got {x.shape}"
        )
    if len(x) == 0 or not np.isfinite(x).all():
        raise ValueError(f"{name} is empty or contains nonfinite values")
    return x.astype(np.float32, copy=False)


def independent_ssim(real_heatmaps, candidate_heatmaps, count, rng, ssim_fn):
    real_idx = rng.integers(len(real_heatmaps), size=count)
    candidate_idx = rng.integers(len(candidate_heatmaps), size=count)
    cross = pairwise_ssim(
        real_heatmaps[real_idx], candidate_heatmaps[candidate_idx], ssim_fn
    )
    # Draw distinct candidate indices when possible, to avoid identical-index pairs.
    if len(candidate_heatmaps) >= 2:
        other_idx = rng.integers(len(candidate_heatmaps) - 1, size=count)
        other_idx += other_idx >= candidate_idx
    else:
        other_idx = candidate_idx
    within = pairwise_ssim(
        candidate_heatmaps[candidate_idx], candidate_heatmaps[other_idx], ssim_fn
    )
    return describe(cross), describe(within)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--generated", required=True, help="V2 normalized .npy file")
    parser.add_argument("--output-dir", default="outputs_baselines")
    parser.add_argument("--pairs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.pairs <= 0:
        parser.error("--pairs must be positive")

    dataset = prepare_dataset_v2(args.data_dir)
    train = np.asarray(dataset["train"], dtype=np.float32)
    test = np.asarray(dataset["test"], dtype=np.float32)
    test = require_shape("test", test, train)
    fake = require_shape(
        "TimeGAN", np.load(args.generated, allow_pickle=False), test
    )
    if fake.shape != test.shape:
        raise ValueError(
            f"TimeGAN must match test shape for the existing evaluation: "
            f"{fake.shape} != {test.shape}"
        )

    rng = np.random.default_rng(args.seed)
    n = len(test)
    # No test leakage: both baselines are made from training sequences only.
    sampled_ids = rng.choice(len(train), size=n, replace=(n > len(train)))
    resampled = train[sampled_ids]
    mean_sequence = train.mean(axis=0, keepdims=True)
    repeated_mean = np.repeat(mean_sequence, n, axis=0).astype(np.float32)

    candidates = {
        "TimeGAN_V3.3_10ep": fake,
        "Training_resample": resampled,
        "Repeated_train_mean": repeated_mean,
    }
    ssim_fn = get_ssim()
    real_heatmaps = depth_heatmaps(test)
    rows = []
    for name, candidate in candidates.items():
        distribution = evaluate(test, candidate)
        decoded = dataset["transform"].inverse_transform(candidate)
        market = check_market_constraints(decoded)
        cross_ssim, within_ssim = independent_ssim(
            real_heatmaps, depth_heatmaps(candidate), args.pairs, rng, ssim_fn
        )
        rows.append({
            "name": name,
            **distribution,
            **market,
            "real_vs_candidate_ssim_mean": cross_ssim["mean"],
            "candidate_vs_candidate_ssim_mean": within_ssim["mean"],
            "ssim_random_cross": cross_ssim,
            "ssim_random_within": within_ssim,
        })

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "preprocessing": "v2", "split_evaluated": "test",
            "baseline_source": "train only", "seed": args.seed,
            "ssim_random_pairs": args.pairs,
            "ssim_note": "Independent random pairs; NOT aligned reconstruction SSIM",
            "kl_note": "Original feature-wise 50-bin histogram KL; estimator is sensitive to sparse bins",
        },
        "results": rows,
    }
    json_path = output_dir / "baseline_results.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    csv_path = output_dir / "baseline_results.csv"
    fields = ["name", "mean_mae", "std_mae", "kl_mean", "kl_median",
              "kl_max", "all_constraints_pct", "valid_spread_pct",
              "valid_ask_order_pct", "valid_bid_order_pct",
              "nonnegative_sizes_pct", "mean_best_spread", "std_best_spread",
              "real_vs_candidate_ssim_mean", "candidate_vs_candidate_ssim_mean"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print("Train/test sequences:", len(train), "/", n)
    print("TimeGAN shape:", fake.shape)
    print("\n=== Unified Baseline Comparison ===")
    for row in rows:
        print(f"\n{row['name']}")
        for key in fields[1:]:
            print(f"  {key}: {row[key]:.6f}")
    print("\nJSON saved:", json_path)
    print("CSV saved:", csv_path)
    print("Note: training resampling is a reference, not a novel synthetic generator.")
    print("Note: random-pair SSIM is not sample-aligned SSIM.")


if __name__ == "__main__":
    main()
