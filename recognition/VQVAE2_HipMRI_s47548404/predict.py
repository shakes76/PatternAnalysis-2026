"""
predict.py - Generate new HipMRI slices and evaluate them (COMP3710, task 2.4).

Steps (all outputs go to --out-dir):
  1. Reconstruction quality on the TEST split: SSIM and MSE for VQ-VAE2 and the baseline
     -> recon_comparison.png
  2. Generation: sample top codes (top prior) -> bottom codes (bottom prior, given top)
     -> decode with VQ-VAE2 -> samples.npz, samples_grid.png
  3. Memorisation audit: nearest TRAINING slice (pixel MSE) for every generated slice,
     compared with the same distance for real held-out slices (validate + test pooled)
     -> nn_pairs.png, nn_hist.png
  4. Diversity and intensity checks -> intensity_hist.png
  All numbers -> metrics.json

Usage:
    python predict.py --data-dir "<path>/keras_slices_data" --n-samples 16
    # re-run the evaluation without re-sampling (sampling is slow):
    python predict.py --data-dir "<path>/keras_slices_data" --load-samples
    # Plan B (merged codes): priors trained on merged code ids
    python predict.py --data-dir "<path>/keras_slices_data" --prior-dir runs/prior_merged \
        --merge-file runs/prior_merged/merge.pt --out-dir runs/predict_merged
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
from skimage.metrics import structural_similarity
from torch.utils.data import ConcatDataset, DataLoader

from dataset import HipMRIDataset
from modules import BaselineAE, PixelCNNPrior
from train import get_device, set_seed, sync
from train_prior import load_vqvae

TOP_SHAPE, BOTTOM_SHAPE = (32, 16), (64, 32)


def parse_args():
    p = argparse.ArgumentParser(description="Generate and evaluate HipMRI slices")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--vqvae-ckpt", default="runs/vqvae2/best.pt")
    p.add_argument("--baseline-ckpt", default="runs/baseline/best.pt")
    p.add_argument("--prior-dir", default="runs/prior")
    p.add_argument("--out-dir", default="runs/predict")
    p.add_argument("--n-samples", type=int, default=16)
    p.add_argument("--sample-batch", type=int, default=16)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--load-samples", action="store_true", help="reuse samples.npz")
    p.add_argument("--merge-file", default=None,
                   help="merge.pt from merge_codes.py, if the priors were trained on merged codes")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# loading
def load_baseline(path, device):
    ckpt = torch.load(path, map_location=device)
    model = BaselineAE().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded baseline from {path} (epoch {ckpt['epoch']}, val recon {ckpt['val_recon']:.5f})")
    return model


def load_prior(path, level, device):
    ckpt = torch.load(path, map_location=device)
    k = ckpt.get("num_codes", 512)
    prior = PixelCNNPrior(num_codes=k, channels=ckpt["channels"], n_layers=ckpt["n_layers"],
                          cond_num_codes=k if level == "bottom" else None).to(device)
    prior.load_state_dict(ckpt["model"])
    prior.eval()
    print(f"Loaded {level} prior from {path} (epoch {ckpt['epoch']}, val CE {ckpt['val_ce']:.4f})")
    return prior


def split_tensor(data_dir, splits):
    ds = ConcatDataset([HipMRIDataset(data_dir, s) for s in splits])
    return torch.cat(list(DataLoader(ds, batch_size=128)))  # (N, 1, 256, 128)


# 1. reconstruction
@torch.no_grad()
def reconstruct(fn, x, device, batch=64):
    out = [fn(x[i:i + batch].to(device)).clamp(0, 1).cpu() for i in range(0, len(x), batch)]
    return torch.cat(out)


def ssim_scores(x, x_hat):
    a, b = x[:, 0].numpy(), x_hat[:, 0].numpy()
    return np.array([structural_similarity(a[i], b[i], data_range=1.0) for i in range(len(a))])


def recon_figure(x, recons, path, n=8):
    idx = np.linspace(0, len(x) - 1, n).astype(int)
    rows = [("original", x)] + list(recons.items())
    fig, axes = plt.subplots(len(rows), n, figsize=(1.6 * n, 3.3 * len(rows)))
    for r, (name, imgs) in enumerate(rows):
        for c, i in enumerate(idx):
            axes[r, c].imshow(imgs[i, 0], cmap="gray", vmin=0, vmax=1)
            axes[r, c].axis("off")
        axes[r, 0].set_title(name, loc="left", fontsize=9)
    fig.suptitle("Test-set reconstructions")
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close(fig)


# 2. generation
@torch.no_grad()
def generate(vqvae, top_prior, bottom_prior, n, batch, device, temperature, merge=None):
    """Sample code grids and decode them. With `merge`, the priors output merged group ids,
    which are mapped to each group's representative code before decoding."""
    images, tops, bottoms, seconds = [], [], [], []
    for start in range(0, n, batch):
        b = min(batch, n - start)
        sync(device)
        t0 = time.time()
        top = top_prior.sample(b, TOP_SHAPE, device, temperature=temperature)
        bottom = bottom_prior.sample(b, BOTTOM_SHAPE, device, cond_ids=top, temperature=temperature)
        if merge is not None:
            top, bottom = merge["top_rep"][top], merge["bottom_rep"][bottom]
        x = vqvae.decode_code(top, bottom).clamp(0, 1)
        sync(device)
        seconds.append(time.time() - t0)
        images.append(x.cpu())
        tops.append(top.cpu())
        bottoms.append(bottom.cpu())
        print(f"  sampled {start + b}/{n} ({seconds[-1] / 60:.1f} min for this batch of {b})")
    return torch.cat(images), torch.cat(tops), torch.cat(bottoms), sum(seconds) / n


def grid_figure(imgs, path, title, cols=8):
    n = len(imgs)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(1.6 * cols, 3.2 * rows), squeeze=False)
    for k, ax in enumerate(axes.flat):
        ax.axis("off")
        if k < n:
            ax.imshow(imgs[k, 0], cmap="gray", vmin=0, vmax=1)
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close(fig)


# 3. memorisation
@torch.no_grad()
def nearest_train(queries, train_x, device, batch=256):
    """For each query, MSE to its nearest training slice and that slice's index."""
    q = queries.reshape(len(queries), -1).to(device)
    P = q.shape[1]
    q2 = (q ** 2).sum(1, keepdim=True)
    best = torch.full((len(q),), float("inf"), device=device)
    best_idx = torch.zeros(len(q), dtype=torch.long, device=device)
    for start in range(0, len(train_x), batch):
        t = train_x[start:start + batch].reshape(-1, P).to(device)
        d = ((q2 + (t ** 2).sum(1) - 2 * q @ t.T) / P).clamp_min(0)  # (Q, batch) pixel MSE
        m, i = d.min(1)
        better = m < best
        best[better] = m[better]
        best_idx[better] = i[better] + start
    return best.cpu().numpy(), best_idx.cpu().numpy()


def nn_figures(samples, train_x, s_dist, s_idx, real_dist, out_dir, n_pairs=6):
    order = np.argsort(s_dist)[:n_pairs]  # the most suspicious: samples closest to training
    fig, axes = plt.subplots(2, n_pairs, figsize=(1.8 * n_pairs, 6.8))
    for c, k in enumerate(order):
        axes[0, c].imshow(samples[k, 0], cmap="gray", vmin=0, vmax=1)
        axes[0, c].set_title(f"sample {k}\nMSE {s_dist[k]:.4f}", fontsize=8)
        axes[1, c].imshow(train_x[s_idx[k], 0], cmap="gray", vmin=0, vmax=1)
        axes[1, c].set_title(f"nearest train #{s_idx[k]}", fontsize=8)
        axes[0, c].axis("off")
        axes[1, c].axis("off")
    fig.suptitle("Generated slices closest to the training set (top) and their nearest training slice")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "nn_pairs.png"), dpi=110)
    plt.close(fig)

    plt.figure(figsize=(7, 3.8))
    bins = np.linspace(0, max(s_dist.max(), real_dist.max()), 40)
    plt.hist(real_dist, bins=bins, alpha=0.6, density=True, label="real held-out slices")
    plt.hist(s_dist, bins=bins, alpha=0.6, density=True, label="generated slices")
    plt.axvline(np.percentile(real_dist, 5), color="k", ls="--", lw=1,
                label="5th percentile of real")
    plt.xlabel("MSE to nearest training slice")
    plt.ylabel("density")
    plt.title("Memorisation audit: distance to nearest training slice")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "nn_hist.png"), dpi=110)
    plt.close()


# 4. diversity
def mean_pairwise_mse(x):
    f = x.reshape(len(x), -1)
    d = ((f[:, None, :] - f[None, :, :]) ** 2).mean(-1)
    return d[~torch.eye(len(x), dtype=torch.bool)].mean().item()


def intensity_figure(real, samples, path):
    plt.figure(figsize=(7, 3.8))
    bins = np.linspace(0, 1, 60)
    plt.hist(real.flatten().numpy(), bins=bins, alpha=0.6, density=True, label="real test slices")
    plt.hist(samples.flatten().numpy(), bins=bins, alpha=0.6, density=True, label="generated slices")
    plt.yscale("log")
    plt.xlabel("normalised intensity")
    plt.ylabel("density (log)")
    plt.title("Pixel intensity distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    os.makedirs(args.out_dir, exist_ok=True)
    metrics = {"device": str(device)}

    vqvae = load_vqvae(args.vqvae_ckpt, device)
    baseline = load_baseline(args.baseline_ckpt, device)

    # 1. Reconstruction on the test split
    test_x = split_tensor(args.data_dir, ["test"])
    recons = {"baseline": reconstruct(baseline, test_x, device),
              "VQ-VAE2": reconstruct(lambda x: vqvae(x)[0], test_x, device)}
    for name, x_hat in recons.items():
        s = ssim_scores(test_x, x_hat)
        metrics[f"test_{name}"] = {"ssim_mean": float(s.mean()), "ssim_std": float(s.std()),
                                   "ssim_min": float(s.min()),
                                   "frac_ssim_above_0.6": float((s > 0.6).mean()),
                                   "mse": float(((x_hat - test_x) ** 2).mean())}
        print(f"[test] {name}: SSIM {s.mean():.4f} +/- {s.std():.4f} (min {s.min():.4f}), "
              f"{(s > 0.6).mean():.1%} of slices above 0.6")
    recon_figure(test_x, recons, os.path.join(args.out_dir, "recon_comparison.png"))

    # 2. Generation
    sample_path = os.path.join(args.out_dir, "samples.npz")
    if args.load_samples:
        data = np.load(sample_path)
        samples = torch.from_numpy(data["images"])
        tops, bottoms = torch.from_numpy(data["top"]), torch.from_numpy(data["bottom"])
        print(f"Loaded {len(samples)} samples from {sample_path}")
    else:
        top_prior = load_prior(os.path.join(args.prior_dir, "top_best.pt"), "top", device)
        bottom_prior = load_prior(os.path.join(args.prior_dir, "bottom_best.pt"), "bottom", device)
        print(f"Sampling {args.n_samples} slices (temperature {args.temperature})...")
        merge = None
        if args.merge_file:
            m = torch.load(args.merge_file)
            merge = {"top_rep": m["top_rep"].to(device), "bottom_rep": m["bottom_rep"].to(device)}
            print(f"Using merged codes from {args.merge_file} (k={m['k']})")
        samples, tops, bottoms, sec = generate(vqvae, top_prior, bottom_prior, args.n_samples,
                                               args.sample_batch, device, args.temperature, merge)
        metrics["sampling_seconds_per_slice"] = sec
        np.savez(sample_path, images=samples.numpy(), top=tops.numpy(), bottom=bottoms.numpy())
        print(f"Sampling: {sec:.1f} s per slice on {device}")
    grid_figure(samples, os.path.join(args.out_dir, "samples_grid.png"),
                f"Generated slices (VQ-VAE2 + PixelCNN priors, T={args.temperature})")

    # 3. Memorisation audit
    print("Memorisation audit: nearest training slice for each sample and each held-out slice...")
    train_x = split_tensor(args.data_dir, ["train"])
    heldout_x = split_tensor(args.data_dir, ["validate", "test"])
    s_dist, s_idx = nearest_train(samples, train_x, device)
    r_dist, _ = nearest_train(heldout_x, train_x, device)
    p5 = float(np.percentile(r_dist, 5))
    metrics["memorisation"] = {
        "heldout_reference": "validate + test pooled (7 patients, 1200 slices)",
        "generated_nn_mse_median": float(np.median(s_dist)),
        "generated_nn_mse_min": float(s_dist.min()),
        "heldout_nn_mse_median": float(np.median(r_dist)),
        "heldout_nn_mse_5th_percentile": p5,
        "generated_below_heldout_5th_percentile": int((s_dist < p5).sum()),
        "n_generated": int(len(s_dist))}
    print(f"  generated: median NN-MSE {np.median(s_dist):.5f} | held-out real: median {np.median(r_dist):.5f}"
          f" | {(s_dist < p5).sum()}/{len(s_dist)} samples closer than the real 5th percentile ({p5:.5f})")
    nn_figures(samples, train_x, s_dist, s_idx, r_dist, args.out_dir)

    # 4. Diversity and intensity
    real_subset = test_x[torch.randperm(len(test_x))[:len(samples)]]
    metrics["diversity"] = {"generated_mean_pairwise_mse": mean_pairwise_mse(samples),
                            "real_test_mean_pairwise_mse": mean_pairwise_mse(real_subset),
                            "top_codes_used_in_samples": int(tops.unique().numel()),
                            "bottom_codes_used_in_samples": int(bottoms.unique().numel())}
    print(f"  diversity: mean pairwise MSE generated {metrics['diversity']['generated_mean_pairwise_mse']:.5f}"
          f" vs real {metrics['diversity']['real_test_mean_pairwise_mse']:.5f}")
    intensity_figure(test_x, samples, os.path.join(args.out_dir, "intensity_hist.png"))

    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Done. Figures and metrics.json in {args.out_dir}/")


if __name__ == "__main__":
    main()