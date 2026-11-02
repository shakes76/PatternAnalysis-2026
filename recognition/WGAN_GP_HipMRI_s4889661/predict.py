"""
predict.py — Example usage and evaluation of the trained HipMRI GANs

Loads the trained DCGAN (baseline) and WGAN-GP generators and produces the
evidence for the fidelity / memorisation / privacy investigation:
    1. Uncurated sample grids across several random seeds (diversity)
    2. Fidelity: SSIM between generated slices and their closest real slice
    3. Memorisation audit: nearest-neighbour distances vs unseen test patients, 
        with contrast and SSIM confound checks
    4. Latent interpolation between random noise vectors
    5. Failure candidates: lowest-SSIM samples per model
    6. Generation runtime

Usage
-----
    python predict.py --root ~/keras_slices_data \
        --dcgan ~/Desktop/dcgan_baseline/generator.pt \
        --wgan  ~/Desktop/wgan_gp/generator.pt

Outputs are written to figures/ (gitignored); selected figures are copied
into images/ for the README.
"""

import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")                 # save figures to file; no display needed
import matplotlib.pyplot as plt
import numpy as np                    # allowed in predict.py for visualisation
import torch
import torch.nn.functional as F

from modules import Generator, Z_DIM, count_params
from dataset import load_split

# =============================================================================
# COMMAND-LINE ARGUMENTS
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate trained HipMRI GANs")
    p.add_argument("--root", default=os.path.expanduser("~/keras_slices_data"),
                   help="path to keras_slices_data (for real reference slices)")
    p.add_argument("--dcgan", required=True, help="path to DCGAN generator.pt")
    p.add_argument("--wgan", required=True, help="path to WGAN-GP generator.pt")
    p.add_argument("--out-dir", default="figures")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3],
                   help="random seeds for uncurated sample grids")
    p.add_argument("--n-per-seed", type=int, default=8)
    p.add_argument("--n-eval", type=int, default=512,
                   help="generated samples per model for quantitative metrics")
    return p.parse_args()

# =============================================================================
# LOADING TRAINED GENERATORS AND SAMPLING
# =============================================================================

def load_generator(path, device):
    """Rebuild the Generator and load trained weights (eval mode: BatchNorm
    uses the running statistics learned during training)."""
    G = Generator()
    G.load_state_dict(torch.load(os.path.expanduser(path), map_location=device))
    return G.to(device).eval()

@torch.no_grad()
def generate(G, n, seed, device):
    """Generate n slices from a fixed seed, so every figure is reproducible."""
    gen = torch.Generator().manual_seed(seed)
    z = torch.randn(n, Z_DIM, generator=gen).to(device)
    return G(z).cpu()                                  # [n, 1, 128, 64] in [-1, 1]

# =============================================================================
# DISPLAY HELPERS
# =============================================================================

def to_display(img):
    """[1, H, W] tensor in [-1, 1] -> H x W array in [0, 1], rotated 90°.

    The NIfTI slices load rotated relative to the usual radiological view;
    rotation is applied for display only and never affects training or metrics.
    """
    return np.rot90(((img[0].numpy() + 1) / 2).clip(0, 1))

# =============================================================================
# 1. UNCURATED SAMPLE GRIDS (DIVERSITY)
# =============================================================================

def plot_uncurated_grid(G, name, seeds, n_per_seed, out_dir, device):
    """One row per seed, n_per_seed samples per row, nothing selected or
    removed, so the grid honestly reflects the model's output diversity."""
    fig, axes = plt.subplots(len(seeds), n_per_seed,
                             figsize=(n_per_seed * 1.6, len(seeds) * 0.9 + 0.6))
    for r, seed in enumerate(seeds):
        imgs = generate(G, n_per_seed, seed, device)
        for c in range(n_per_seed):
            ax = axes[r, c]
            ax.imshow(to_display(imgs[c]), cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
        axes[r, 0].set_title(f"seed {seed}", fontsize=8, loc="left")

    fig.suptitle(f"{name}: uncurated samples "
                 f"({len(seeds)} seeds × {n_per_seed}, no selection)")
    plt.tight_layout()
    path = os.path.join(out_dir, f"uncurated_{name.lower().replace('-', '')}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path
    
def plot_real_grid(images, split, n_rows, n_cols, out_dir, seed=0):
    """Random (seeded, unselected) real slices in the same layout as the
    generated grids, as a visual reference for fidelity."""
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(images), generator=gen)[: n_rows * n_cols]

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 1.6, n_rows * 0.9 + 0.6))
    for ax, i in zip(axes.flat, idx):
        ax.imshow(to_display(images[i]), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")

    fig.suptitle(f"Real {split} slices: random sample "
                 f"({n_rows * n_cols}, no selection)")
    plt.tight_layout()
    path = os.path.join(out_dir, f"real_{split}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path

# =============================================================================
# 2. FIDELITY: SSIM TO THE CLOSEST REAL SLICE
# =============================================================================

def gaussian_window(size=11, sigma=1.5):
    """2D Gaussian weighting window used by SSIM (standard 11x11, sigma 1.5)."""
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]          # [1, 1, size, size]


def ssim(x, y, window):
    """Mean SSIM per image pair (Wang et al., 2004).

    x, y: [N, 1, H, W] in [0, 1]. Compares local luminance (means),
    contrast (variances) and structure (covariance) in Gaussian-weighted
    windows, then averages over the image. 1 = identical.
    Note: zero padding at the borders slightly lowers edge values compared
    with implementations that only use full windows.
    """
    C1, C2 = 0.01 ** 2, 0.03 ** 2                         # stabilisers for data range 1
    pad = window.shape[-1] // 2

    mu_x = F.conv2d(x, window, padding=pad)
    mu_y = F.conv2d(y, window, padding=pad)
    var_x = F.conv2d(x * x, window, padding=pad) - mu_x ** 2
    var_y = F.conv2d(y * y, window, padding=pad) - mu_y ** 2
    cov_xy = F.conv2d(x * y, window, padding=pad) - mu_x * mu_y

    num = (2 * mu_x * mu_y + C1) * (2 * cov_xy + C2)
    den = (mu_x ** 2 + mu_y ** 2 + C1) * (var_x + var_y + C2)
    return (num / den).mean(dim=(1, 2, 3))                # one value per pair

# =============================================================================
# 3. MEMORISATION AUDIT: NEAREST-NEIGHBOUR DISTANCES
# =============================================================================

def to_unit(x):
    """[-1, 1] -> [0, 1], the range SSIM's constants assume."""
    return (x + 1) / 2

def nearest_neighbours(queries, reference, chunk=256):
    """For each query image, the index of and L2 pixel distance to its closest
    reference image. Chunked so the distance matrix fits in memory.
    Reused by the memorisation audit."""
    ref = reference.flatten(1)
    dists, idxs = [], []
    for q in queries.flatten(1).split(chunk):
        d = torch.cdist(q, ref)                           # [chunk, n_reference]
        best = d.min(dim=1)
        dists.append(best.values)
        idxs.append(best.indices)
    return torch.cat(dists), torch.cat(idxs)


def fidelity_ssim(samples, train, window, k=20):
    """SSIM between each sample and its most structurally similar training
    slice. The k nearest training slices by L2 distance are shortlisted
    (cheap), then the highest SSIM among them is kept. This approximates
    the SSIM-nearest slice without comparing against all 11,460 slices."""
    ref = train.flatten(1)
    best = []
    for q in samples.split(64):
        d = torch.cdist(q.flatten(1), ref)                    # [b, n_train]
        idx = d.topk(k, largest=False).indices                # [b, k] nearest by L2
        cand = train[idx.flatten()]                           # [b*k, 1, H, W]
        qq = q.repeat_interleave(k, dim=0)                    # pair each query with its k
        s = ssim(to_unit(qq), to_unit(cand), window).view(-1, k)
        best.append(s.max(dim=1).values)                      # best match per query
    return torch.cat(best)

def plot_distance_histogram(audit, out_dir):
    """Overlay nearest-neighbour distance distributions. A synthetic
    distribution shifted LEFT of the real reference suggests memorisation."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, d in audit.items():
        ax.hist(d.numpy(), bins=40, alpha=0.5, density=True, label=label)
    ax.set_xlabel("L2 distance to nearest training slice (lower = closer)")
    ax.set_ylabel("density")
    ax.set_title("Memorisation audit: nearest-neighbour distances")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, "memorisation_histogram.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path

def plot_nearest_pairs(samples, d_syn, idx_syn, train, name, out_dir, n=6):
    """The n generated slices CLOSEST to the training set (the most likely
    copies), each above its nearest training slice, for visual inspection."""
    order = d_syn.argsort()[:n]
    fig, axes = plt.subplots(2, n, figsize=(n * 1.8, 2.8))
    for c, i in enumerate(order):
        axes[0, c].imshow(to_display(samples[i]), cmap="gray", vmin=0, vmax=1)
        axes[0, c].set_title(f"d = {d_syn[i]:.1f}", fontsize=8)
        axes[1, c].imshow(to_display(train[idx_syn[i]]), cmap="gray", vmin=0, vmax=1)
        axes[0, c].axis("off")
        axes[1, c].axis("off")
    fig.suptitle(f"{name}: closest generated slices (top) vs nearest training slice (bottom)",
                 fontsize=10)
    plt.tight_layout()
    path = os.path.join(out_dir, f"nearest_pairs_{name.lower().replace('-', '')}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path

# =============================================================================
# 4. LATENT INTERPOLATION
# =============================================================================

@torch.no_grad()
def plot_interpolation(models, out_dir, n_steps=8, n_pairs=2, seed=7, device="cpu"):
    """Morph between pairs of random noise vectors, using the same endpoints
    for every model. Rows = (model, pair), columns = interpolation steps.

    Smooth, anatomically plausible transitions suggest a continuous learned
    space rather than memorised images; abrupt jumps or implausible
    intermediate frames are failure evidence. Linear interpolation is used;
    spherical interpolation (slerp) is a common alternative for Gaussian
    noise vectors.
    """
    gen = torch.Generator().manual_seed(seed)
    ends = torch.randn(n_pairs, 2, Z_DIM, generator=gen)    # [pair, start/end, z]
    t = torch.linspace(0, 1, n_steps)[:, None]              # [n_steps, 1]

    rows = []
    for name, G in models.items():
        for p in range(n_pairs):
            z = (1 - t) * ends[p, 0] + t * ends[p, 1]       # [n_steps, Z_DIM]
            rows.append((f"{name}, pair {p + 1}", G(z.to(device)).cpu()))

    fig, axes = plt.subplots(len(rows), n_steps,
                             figsize=(n_steps * 1.6, len(rows) * 0.9 + 0.6))
    for r, (label, imgs) in enumerate(rows):
        for c in range(n_steps):
            axes[r, c].imshow(to_display(imgs[c]), cmap="gray", vmin=0, vmax=1)
            axes[r, c].axis("off")
        axes[r, 0].set_title(label, fontsize=8, loc="left")

    fig.suptitle("Latent interpolation (same noise endpoints for both models)")
    plt.tight_layout()
    path = os.path.join(out_dir, "latent_interpolation.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path

# =============================================================================
# 5. FAILURE CASES
# =============================================================================

def plot_worst_samples(samples_by_model, ssim_by_model, out_dir, n=6):
    """Each model's n lowest-SSIM samples, selected by metric rather than by
    hand, as candidates for the failure autopsy. Labels give the sample index
    (seed 1234 draw) so every case can be regenerated exactly."""
    names = list(samples_by_model)
    fig, axes = plt.subplots(len(names), n,
                             figsize=(n * 1.8, len(names) * 1.2 + 0.6))
    for r, name in enumerate(names):
        s = ssim_by_model[name]
        for c, i in enumerate(s.argsort()[:n]):              # lowest SSIM first
            axes[r, c].imshow(to_display(samples_by_model[name][i]),
                              cmap="gray", vmin=0, vmax=1)
            axes[r, c].set_title(f"{name} #{i.item()}  SSIM {s[i]:.2f}", fontsize=7)
            axes[r, c].axis("off")
    fig.suptitle("Failure candidates: lowest-SSIM samples per model (selected by metric)")
    plt.tight_layout()
    path = os.path.join(out_dir, "failure_candidates.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path

# =============================================================================
# 6. GENERATION RUNTIME
# =============================================================================

@torch.no_grad()
def time_generation(G, n, device, repeats=5):
    """Mean wall-clock time for G to generate n slices.

    One warm-up pass runs first so one-off setup costs are excluded, then
    the average of several repeats is returned.
    """
    z = torch.randn(n, Z_DIM, device=device)
    G(z)
    start = time.perf_counter()
    for _ in range(repeats):
        G(z)
    return (time.perf_counter() - start) / repeats

# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()
    device = torch.device("cpu")      # evaluation is light; CPU keeps it simple
    os.makedirs(args.out_dir, exist_ok=True)

    # --- Real reference data (train = what the GANs learned from;
    #     test = unseen patients, used as the reference for the audit) -------
    real = {split: load_split(args.root, split)["images"]
            for split in ("train", "test")}
    print(f"Real slices: train {tuple(real['train'].shape)}, "
          f"test {tuple(real['test'].shape)}")
    path = plot_real_grid(real["test"], "test", len(args.seeds),
                          args.n_per_seed, args.out_dir)
    print(f"  saved {path}")

    models = {
        "DCGAN": load_generator(args.dcgan, device),
        "WGAN-GP": load_generator(args.wgan, device),
    }
    samples_by_model = {}
    for name, G in models.items():
        sample = generate(G, 2, seed=0, device=device)
        print(f"Loaded {name}: {count_params(G):,} params, "
              f"sample {tuple(sample.shape)}, range [{sample.min():.2f}, {sample.max():.2f}]")
        path = plot_uncurated_grid(G, name, args.seeds, args.n_per_seed,
                                   args.out_dir, device)
        print(f"  saved {path}")
    # --- 2. Fidelity: SSIM to the nearest real training slice ---------------
    window = gaussian_window()
    ref = fidelity_ssim(real["test"], real["train"], window)
    print(f"\nFidelity (SSIM to nearest training slice, higher = more similar)")
    print(f"  Real test patients (reference): {ref.mean():.3f} ± {ref.std():.3f}")

    samples_by_model = {}     # generated samples, reused by the memorisation audit
    ssim_by_model = {}        # per-sample SSIM scores, reused by the confound checks
    for name, G in models.items():
        samples = generate(G, args.n_eval, seed=1234, device=device)
        samples_by_model[name] = samples
        s = fidelity_ssim(samples, real["train"], window)
        ssim_by_model[name] = s
        print(f"  {name:8s} ({args.n_eval} samples):  {s.mean():.3f} ± {s.std():.3f}")

    # --- 3. Memorisation audit: distance to the nearest training slice -------
    # Real unseen patients show how close a genuinely NEW scan naturally sits
    # to the training set. Synthetic samples much closer than that suggest
    # copying. Threshold = 5th percentile of the real reference distances, so
    # ~5% of samples falling below it is what "new data" looks like.
    d_ref, _ = nearest_neighbours(real["test"], real["train"])
    threshold = torch.quantile(d_ref, 0.05)
    print(f"\nMemorisation audit (L2 distance to nearest training slice, lower = closer)")
    print(f"  Real test patients (reference): median {d_ref.median():.2f}, "
          f"min {d_ref.min():.2f}, 5th pct {threshold:.2f}")

    audit = {"Real test (unseen patients)": d_ref}
    nn_idx = {}
    for name, samples in samples_by_model.items():
        d_syn, idx_syn = nearest_neighbours(samples, real["train"])
        frac = (d_syn < threshold).float().mean().item()
        print(f"  {name:8s}: median {d_syn.median():.2f}, min {d_syn.min():.2f}, "
              f"{frac:.1%} of samples closer than real 5th pct (expected ~5%)")
        audit[name] = d_syn
        nn_idx[name] = idx_syn

    print(f"  saved {plot_distance_histogram(audit, args.out_dir)}")
    for name, samples in samples_by_model.items():
        path = plot_nearest_pairs(samples, audit[name], nn_idx[name],
                                  real["train"], name, args.out_dir)
        print(f"  saved {path}")

    # --- 3b. Confound checks: is "closeness" just blur? ----------------------
    # L2 distance favours smooth, low-contrast images (they sit near the
    # average of many slices without matching any). Check (1) whether close
    # samples are simply low-contrast ones, and (2) whether any sample is a
    # structural near-copy by SSIM, which unlike L2 does penalise blur.
    ssim_thr = torch.quantile(ref, 0.95)
    print(f"\nConfound checks")
    print(f"  Real test: 95th pct best-SSIM {ssim_thr:.3f}, max {ref.max():.3f}")
    for name, samples in samples_by_model.items():
        contrast = samples.flatten(1).std(dim=1)
        r = torch.corrcoef(torch.stack([contrast, audit[name]]))[0, 1].item()
        s = ssim_by_model[name]
        frac_hi = (s > ssim_thr).float().mean().item()
        print(f"  {name:8s}: corr(contrast, NN distance) = {r:+.2f} | "
              f"max SSIM {s.max():.3f} | {frac_hi:.1%} above real 95th pct SSIM (expected ~5%)")

    # --- 4. Latent interpolation ---------------------------------------------
    print(f"\nLatent interpolation")
    print(f"  saved {plot_interpolation(models, args.out_dir, device=device)}")

    # --- 5. Failure cases ----------------------------------------------------
    print(f"\nFailure cases")
    print(f"  saved {plot_worst_samples(samples_by_model, ssim_by_model, args.out_dir)}")

    # --- 6. Generation runtime -----------------------------------------------
    print(f"\nGeneration runtime ({args.n_eval} slices, {device}, mean of 5 runs)")
    for name, G in models.items():
        t = time_generation(G, args.n_eval, device)
        print(f"  {name:8s}: {t:.3f} s per {args.n_eval} slices "
              f"({t / args.n_eval * 1000:.2f} ms/slice, {args.n_eval / t:,.0f} slices/s)")

if __name__ == "__main__":
    main()