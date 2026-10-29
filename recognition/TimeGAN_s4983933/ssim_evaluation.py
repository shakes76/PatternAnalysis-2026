"""COMP3710 Project 7 - TimeGAN order-book depth SSIM evaluation.

Uses 24 x 20 heatmaps of normalized LOBSTER depth (size) features.
Real and synthetic samples are NOT time-aligned, so random-pair SSIM is a
reference distribution, not a paired reconstruction score.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from dataset_v2 import prepare_dataset_v2


def depth_heatmaps(sequences):
    """Extract interleaved ask/bid size columns from normalized 40-dim data."""
    x = np.asarray(sequences, dtype=np.float32)
    if x.ndim != 3 or x.shape[2] != 40:
        raise ValueError(f"Expected (N, T, 40); got {x.shape}")
    if x.shape[1] < 7:
        raise ValueError("At least 7 time steps required for SSIM")
    if not np.all(np.isfinite(x)):
        raise ValueError("Input contains NaN or infinity")
    return np.clip(x[:, :, 1::2], 0.0, 1.0)


def get_ssim():
    try:
        from skimage.metrics import structural_similarity
    except ImportError as exc:
        raise SystemExit("Install dependency first: pip install scikit-image") from exc
    return structural_similarity


def pairwise_ssim(a, b, ssim_fn):
    return np.array([
        ssim_fn(left, right, data_range=1.0, win_size=7)
        for left, right in zip(a, b)
    ], dtype=np.float64)


def describe(values):
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "median": float(np.median(values)),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
    }


def plot_examples(real, fake, output_dir, rng):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("Install dependency first: pip install matplotlib") from exc

    ridx = int(rng.integers(len(real)))
    fidx = int(rng.integers(len(fake)))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, img, heading in zip(
        axes, [real[ridx], fake[fidx]], ["Real LOBSTER", "TimeGAN generated"]
    ):
        im = ax.imshow(img.T, origin="lower", aspect="auto", vmin=0, vmax=1,
                       interpolation="nearest", cmap="viridis")
        ax.set_title(heading)
        ax.set_xlabel("Time step")
        ax.set_ylabel("Size feature (ask/bid, levels 1-10)")
    fig.colorbar(im, ax=axes, label="Train-scaled log(1+size)", shrink=0.8)
    path = output_dir / "depth_heatmaps.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--generated", required=True,
                        help="Path to generated normalized V2 .npy")
    parser.add_argument("--output-dir", default="outputs_ssim")
    parser.add_argument("--pairs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.pairs < 1:
        parser.error("--pairs must be positive")

    # V2 size features are log1p-transformed and train-split min-max normalized.
    # Use a shared fixed data_range=1; never normalize each image separately.
    real = depth_heatmaps(prepare_dataset_v2(args.data_dir)["test"])
    fake = depth_heatmaps(np.load(args.generated, allow_pickle=False))
    if len(real) < 2 or len(fake) < 2:
        raise ValueError("Need at least 2 real and 2 generated sequences")
    rng = np.random.default_rng(args.seed)
    ssim_fn = get_ssim()

    n = args.pairs
    # Sampling with replacement allows a chosen pair count for any dataset size.
    idx_real = rng.integers(0, len(real), size=n)
    idx_fake = rng.integers(0, len(fake), size=n)
    idx_real_other = rng.integers(0, len(real) - 1, size=n)
    idx_real_other += (idx_real_other >= idx_real)
    idx_fake_other = rng.integers(0, len(fake) - 1, size=n)
    idx_fake_other += (idx_fake_other >= idx_fake)

    scores = {
        "real_vs_real_random": pairwise_ssim(real[idx_real], real[idx_real_other], ssim_fn),
        "real_vs_generated_random": pairwise_ssim(real[idx_real], fake[idx_fake], ssim_fn),
        "generated_vs_generated_random": pairwise_ssim(fake[idx_fake], fake[idx_fake_other], ssim_fn),
    }
    summary = {name: describe(values) for name, values in scores.items()}
    summary["metadata"] = {
        "heatmap_shape": list(real.shape[1:]),
        "real_sequences": len(real),
        "generated_sequences": len(fake),
        "pairs": n,
        "seed": args.seed,
        "scaling": "V2 train-split log1p-size min-max, clipped to [0,1]",
        "data_range": 1.0,
        "pairing": "Independent random pairs, not temporally aligned",
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_file = output_dir / "ssim_results.json"
    report_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    example_file = plot_examples(real, fake, output_dir, rng)

    print("Real heatmap shape:", real.shape)
    print("Generated heatmap shape:", fake.shape)
    print("\n=== SSIM: independent random pairs ===")
    for name, vals in scores.items():
        stats = describe(vals)
        print(f"{name}: mean={stats['mean']:.6f}, std={stats['std']:.6f}, "
              f"median={stats['median']:.6f}, "
              f"p05={stats['p05']:.6f}, p95={stats['p95']:.6f}")
    print("\nResults saved:", report_file)
    print("Heatmaps saved:", example_file)
    print("Note: random-pair SSIM is NOT a sample-aligned reconstruction metric.")


if __name__ == "__main__":
    main()
