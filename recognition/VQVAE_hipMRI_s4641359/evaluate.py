# AI Declaration
#
# Tool: Claude (Anthropic), used conversationally over several requests.
#
# What the AI did in this file:
# - After I suggested, it wrote this generative evaluation after we checked the
#   spec and found it asks for: a baseline comparison on sample clarity,
#   diversity and fidelity; a distribution-fidelity metric (FID); and
#   latent-space interpolation between synthetic images and training scans.
#
# What I did:
# - Agreed to cover these remaining spec requirements before writing up,
#   and selected the models and checkpoints compared.
# - Ran it on Colab and interpreted the results, including the limits of
#   Inception FID for MRI and why the VAE's higher NN-SSIM reflects blur
#   rather than copying.


import argparse
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg

from audit import _gauss_1d, local_stats, nearest_neighbours, pairwise_ssim
from dataset import get_dataloaders, patient_id
from modules import VAE
from predict import load_prior, load_vqvae
from utils import load_checkpoint


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------

def frechet_distance(feat_a, feat_b):
    """Frechet distance between Gaussians fitted to two feature sets (N, D).

    FD = |mu_a - mu_b|^2 + tr(C_a) + tr(C_b) - 2 tr(sqrt(C_a C_b)).
    tr(sqrt(C_a C_b)) is computed as the sum of square roots of the eigenvalues of
    the symmetric matrix sqrt(C_a) C_b sqrt(C_a), which is numerically stable.
    """
    a, b = feat_a.double().numpy(), feat_b.double().numpy()
    mu_a, mu_b = a.mean(0), b.mean(0)
    cov_a, cov_b = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    w, v = linalg.eigh(cov_a)
    sqrt_a = (v * np.sqrt(np.clip(w, 0, None))) @ v.T
    eig = linalg.eigvalsh(sqrt_a @ cov_b @ sqrt_a)
    tr_covmean = np.sqrt(np.clip(eig, 0, None)).sum()
    return float(((mu_a - mu_b) ** 2).sum() + np.trace(cov_a) + np.trace(cov_b) - 2 * tr_covmean)


@torch.no_grad()
def vqvae_features(model, images, device, batch=64):
    """Per-slice features: channel-wise mean and std of the VQ-VAE encoder output."""
    feats = []
    for s in range(0, images.size(0), batch):
        z = model.encoder(images[s:s + batch].to(device))
        feats.append(torch.cat([z.mean(dim=(2, 3)), z.std(dim=(2, 3))], dim=1).cpu())
    return torch.cat(feats)


@torch.no_grad()
def inception_fid(groups, reference, device):
    """Inception FID of each group against the reference set, or None if unavailable."""
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except ImportError:
        print("  torchmetrics[image] not installed: skipping Inception FID")
        return {name: None for name in groups}
    results = {}
    for name, images in groups.items():
        try:
            fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
            for s in range(0, reference.size(0), 50):
                fid.update(reference[s:s + 50].repeat(1, 3, 1, 1).to(device), real=True)
            for s in range(0, images.size(0), 50):
                fid.update(images[s:s + 50].clamp(0, 1).repeat(1, 3, 1, 1).to(device), real=False)
            results[name] = fid.compute().item()
        except Exception as err:  # e.g. Inception weights could not be downloaded
            print(f"  Inception FID failed for {name}: {err}")
            results[name] = None
    return results


@torch.no_grad()
def mean_pairwise_ssim(images, device, chunk=32):
    """Mean SSIM over all distinct pairs within a set (lower = more diverse)."""
    g = _gauss_1d(device=device)
    n, total, count = images.size(0), 0.0, 0
    for i in range(0, n, chunk):
        q = images[i:i + chunk].to(device)
        q_stats = local_stats(q, g)
        for j in range(i, n, chunk):
            t = images[j:j + chunk].to(device)
            s = pairwise_ssim(q, *q_stats, t, *local_stats(t, g), g)
            if i == j:  # same block: keep the upper triangle only (no self-pairs)
                mask = torch.triu(torch.ones_like(s, dtype=torch.bool), diagonal=1)
                total += s[mask].sum().item(); count += int(mask.sum())
            else:
                total += s.sum().item(); count += s.numel()
    return total / count


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------

def save_comparison(vae_samples, vq_samples, real, out_path, n=8):
    """Rows: real training slices, VAE baseline samples, VQ-VAE + prior samples."""
    rows = [("Real (train)", real), ("VAE baseline", vae_samples), ("VQ-VAE + prior", vq_samples)]
    fig, axes = plt.subplots(3, n, figsize=(1.3 * n, 8))
    for r, (label, imgs) in enumerate(rows):
        for c in range(n):
            axes[r, c].imshow(imgs[c, 0], cmap="gray", vmin=0, vmax=1)
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
        axes[r, 0].set_ylabel(label, fontsize=10)
    fig.suptitle("Uncurated samples: baseline vs VQ-VAE (first 8 of each set)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


@torch.no_grad()
def save_interpolation(model, pairs, out_path, steps=8):
    """Decode linear blends of quantised latents for each (label, codes_a, codes_b) pair."""
    alphas = torch.linspace(0, 1, steps)
    fig, axes = plt.subplots(len(pairs), steps, figsize=(1.3 * steps, 2.9 * len(pairs)))
    for r, (label, codes_a, codes_b) in enumerate(pairs):
        z_a = model.vq.lookup(codes_a[None].long())
        z_b = model.vq.lookup(codes_b[None].long())
        for c, a in enumerate(alphas):
            img = model.decoder((1 - a) * z_a + a * z_b)[0, 0].cpu()
            axes[r, c].imshow(img, cmap="gray", vmin=0, vmax=1)
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(f"alpha={a:.2f}", fontsize=8)
        axes[r, 0].set_ylabel(label, fontsize=8)
    fig.suptitle("Latent interpolation in the VQ-VAE quantised space", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Generative evaluation: VAE vs VQ-VAE + prior.")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--vqvae_ckpt", required=True)
    parser.add_argument("--prior_ckpt", required=True)
    parser.add_argument("--vae_ckpt", required=True)
    parser.add_argument("--n_down", type=int, default=2,
                        help="Only needed for old-format VQ-VAE checkpoints")
    parser.add_argument("--num_embeddings", type=int, default=512)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--prior_channels", type=int, default=128)
    parser.add_argument("--prior_blocks", type=int, default=10)
    parser.add_argument("--latent_dim", type=int, default=256,
                        help="VAE latent size (only needed for old-format checkpoints)")
    parser.add_argument("--n_samples", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--out_dir", default=os.path.join("outputs", "evaluate"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Device: {device}")

    loaders = get_dataloaders(args.data_root, batch_size=64)
    train_set, test_set = loaders["train"].dataset, loaders["test"].dataset
    test_real = test_set.images

    vqvae = load_vqvae(args, device)
    prior = load_prior(args, device)
    state, hp = load_checkpoint(args.vae_ckpt, device)
    vae = VAE(latent_dim=hp["latent_dim"] if hp else args.latent_dim).to(device)
    vae.load_state_dict(state)
    vae.eval()

    n = args.n_samples
    print(f"Sampling {n} slices from each model...")
    start = time.time()
    with torch.no_grad():
        vae_samples = torch.cat([vae.sample(min(100, n - s), device).cpu()
                                 for s in range(0, n, 100)])
    grid = tuple(vqvae.encode_indices(test_real[:1].to(device)).shape[1:])
    vq_codes, vq_samples = [], []
    for s in range(0, n, 100):
        codes = prior.sample(min(100, n - s), grid, device, temperature=args.temperature)
        with torch.no_grad():
            vq_samples.append(vqvae.decode_indices(codes).cpu())
        vq_codes.append(codes.cpu())
    vq_samples, vq_codes = torch.cat(vq_samples), torch.cat(vq_codes)
    print(f"  done in {time.time() - start:.0f}s")

    # Real references: random training slices, and their VQ-VAE reconstructions.
    ref_idx = torch.randperm(len(train_set))[:n]
    train_real = train_set.images[ref_idx]
    with torch.no_grad():
        train_recon = torch.cat([vqvae(train_real[s:s + 32].to(device))[0].cpu()
                                 for s in range(0, n, 32)])

    groups = {"VAE baseline samples": vae_samples,
              "VQ-VAE + prior samples": vq_samples,
              "Real train slices (reference)": train_real,
              "VQ-VAE reconstructions of train (reference)": train_recon}

    print("Inception FID vs real test slices...")
    fid = inception_fid(groups, test_real, device)

    print("VQ-VAE feature Frechet distance vs real test slices...")
    test_feat = vqvae_features(vqvae, test_real, device)
    fd = {name: frechet_distance(vqvae_features(vqvae, imgs, device), test_feat)
          for name, imgs in groups.items()}

    print("Diversity (mean pairwise SSIM within each group)...")
    diversity = {name: mean_pairwise_ssim(imgs, device) for name, imgs in groups.items()}

    print("Nearest training slice for each group...")
    nn = {}
    for name, imgs in groups.items():
        if "Real train" in name:
            nn[name] = None  # these ARE training slices, so NN-SSIM is trivially 1
            continue
        best, _ = nearest_neighbours(imgs, train_set.images, device, q_chunk=25, t_chunk=256)
        nn[name] = {"mean": best.mean().item(), "max": best.max().item()}

    # Table for the README.
    def fmt(v, p=3):
        return "n/a" if v is None else f"{v:.{p}f}"
    lines = ["| Slices | FID (Inception) ↓ | FD (VQ-VAE features) ↓ | Diversity: mean pairwise SSIM ↓ "
             "| NN-SSIM to train (mean / max) |",
             "|---|---|---|---|---|"]
    for name in groups:
        nn_txt = "n/a (are training slices)" if nn[name] is None else \
            f"{nn[name]['mean']:.3f} / {nn[name]['max']:.3f}"
        lines.append(f"| {name} | {fmt(fid[name], 1)} | {fmt(fd[name], 2)} | "
                     f"{fmt(diversity[name])} | {nn_txt} |")
    lines.append(f"\nReference set for FID/FD: all {len(test_real)} real test slices. "
                 f"{n} slices per group; prior temperature {args.temperature:g}.")
    table = "\n".join(lines)
    with open(os.path.join(args.out_dir, "evaluation_table.md"), "w", encoding="utf-8") as f:
        f.write(table + "\n")
    with open(os.path.join(args.out_dir, "evaluation.json"), "w") as f:
        json.dump({"n_samples": n, "temperature": args.temperature, "fid_inception": fid,
                   "fd_vqvae_features": fd, "diversity_mean_pairwise_ssim": diversity,
                   "nn_ssim_to_train": nn}, f, indent=2)

    # Figures.
    save_comparison(vae_samples, vq_samples, train_real,
                    os.path.join(args.out_dir, "baseline_vs_vqvae_samples.png"))
    gen_best, gen_idx = nearest_neighbours(vq_samples[:1], train_set.images, device)
    nn_codes = vqvae.encode_indices(train_set.images[gen_idx[0]][None].to(device))[0].cpu()
    patients = [patient_id(f) for f in train_set.files]
    # Second training slice from a different patient (falls back to the last slice).
    j = next((k for k in range(len(train_set) - 1, -1, -1) if patients[k] != patients[0]),
             len(train_set) - 1)
    two_train = vqvae.encode_indices(train_set.images[[0, j]].to(device)).cpu()
    pairs = [("generated ->\nits nearest\ntraining slice", vq_codes[0], nn_codes),
             (f"train {patients[0]} ->\ntrain {patients[j]}", two_train[0], two_train[1]),
             ("generated ->\ngenerated", vq_codes[1], vq_codes[2])]
    save_interpolation(vqvae, [(l, a.to(device), b.to(device)) for l, a, b in pairs],
                       os.path.join(args.out_dir, "interpolation.png"))

    print("\n" + table)
    print(f"\nOutputs saved to {args.out_dir}")


if __name__ == "__main__":
    main()