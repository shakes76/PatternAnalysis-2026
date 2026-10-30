import sys

import matplotlib
matplotlib.use("Agg")  # headless-safe; figures are saved, not shown
import matplotlib.pyplot as plt
import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure

from dataset import _find, load_slices, pad_multiple, IMG_DIRS
from modules import BaseVAE, VQVAE, CodePrior

ROOT = "C:/Users/danfu/PatternAnalysis-2026/data/"
VAE_CKPT, VQ_CKPT, PRIOR_CKPT = "vae_best.pth", "vqvae_best.pth", "prior_best.pth"
VAE_LATENT = 8          # must match the trained VAE
GRID = (32, 16)         # VQ-VAE code grid for 256 x 128 inputs (H/8, W/8)
N_SHOW = 6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(kind):
    if kind == "vae":
        model, ckpt = BaseVAE(latent_dim=VAE_LATENT), VAE_CKPT
    elif kind == "vqvae":
        model, ckpt = VQVAE(), VQ_CKPT
    else:
        raise ValueError("kind must be 'vae' or 'vqvae'")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    return model.to(device).eval()


def reconstruct(kind, out=None):
    #Reconstruct real test slices (no sampling). Works for both models because both return (reconstruction, extra_loss).
    out = out or f"{kind}_predictions.png"
    model = load_model(kind)
    x = pad_multiple(torch.from_numpy(load_slices(_find(ROOT, IMG_DIRS["test"]))), 8)

    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    recons, scores = [], []
    with torch.no_grad():
        for i in range(0, len(x), 64):
            batch = x[i:i + 64].to(device)
            recon, _ = model(batch)
            scores += [ssim(recon[j:j + 1], batch[j:j + 1]).item() for j in range(len(batch))]
            recons.append(recon.cpu())
    recon, scores = torch.cat(recons), torch.tensor(scores)
    print(f"{kind}: {len(x)} test slices, mean SSIM {scores.mean():.3f} "
          f"(min {scores.min():.3f}, max {scores.max():.3f})")

    idx = torch.linspace(0, len(x) - 1, N_SHOW).long()  # evenly spaced, not cherry-picked
    fig, axes = plt.subplots(3, N_SHOW, figsize=(2 * N_SHOW, 7))
    for col, i in enumerate(idx):
        axes[0, col].imshow(x[i].squeeze(), cmap="gray", vmin=0, vmax=1)
        axes[1, col].imshow(recon[i].squeeze(), cmap="gray", vmin=0, vmax=1)
        axes[2, col].imshow((x[i] - recon[i]).abs().squeeze(), cmap="hot", vmin=0, vmax=0.5)
        axes[0, col].set_title(f"SSIM {scores[i]:.2f}", fontsize=9)
        for row in range(3):
            axes[row, col].axis("off")
    for row, label in enumerate(["Original", "Recon", "|Error|"]):
        axes[row, 0].text(-0.1, 0.5, label, transform=axes[row, 0].transAxes,
                          rotation=90, va="center", ha="right")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"saved {out}")


def _save_grid(imgs, path, title, cols=8):
    rows = (len(imgs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 4 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, img in zip(axes.ravel(), imgs):
        ax.imshow(img.squeeze(), cmap="gray", vmin=0, vmax=1)
    plt.suptitle(title)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"saved {path}")


@torch.no_grad()
def generate_vae(n=16, seed=0, match_data=False, out="vae_generated.png"):
    #Sample VAE latents (N(0,1), or a Gaussian fitted to encoded training slices).
    model = load_model("vae")
    torch.manual_seed(seed)
    h, w = GRID
    if match_data:
        paths = _find(ROOT, IMG_DIRS["train"])
        x = pad_multiple(torch.from_numpy(load_slices(paths[::max(1, len(paths) // 500)])), 8).to(device)
        mu, _ = model.encoder(x).chunk(2, dim=1)
        mean, std = mu.mean(dim=(0, 2, 3), keepdim=True), mu.std(dim=(0, 2, 3), keepdim=True)
        z = mean + std * torch.randn(n, model.latent_dim, h, w, device=device)
    else:
        z = torch.randn(n, model.latent_dim, h, w, device=device)
    _save_grid(model.decoder(z).cpu(), out, f"VAE samples (seed {seed})")


@torch.no_grad()
def generate_vqvae(n=16, seed=0, temperature=1.0, out=None):
    #Sample code grids from the trained prior, then decode with the VQ-VAE.
    out = out or f"vqvae_generated_t{temperature}.png"
    vq = load_model("vqvae")
    prior = CodePrior(num_codes=vq.quant.num_codes, seq_len=GRID[0] * GRID[1]).to(device)
    prior.load_state_dict(torch.load(PRIOR_CKPT, map_location=device))
    prior.eval()

    torch.manual_seed(seed)  # fixed seed -> reproducible, uncurated grid
    codes = prior.sample(n, temperature=temperature, device=device).view(n, *GRID)
    _save_grid(vq.decode_indices(codes).cpu(), out,
               f"VQ-VAE + prior (seed {seed}, temperature {temperature})")


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 2:
        sys.exit(__doc__)
    action, kind, extra = args[0], args[1], args[2:]
    if action == "recon":
        reconstruct(kind)
    elif action == "generate" and kind == "vae":
        generate_vae(match_data=bool(extra and extra[0] == "match"))
    elif action == "generate" and kind == "vqvae":
        generate_vqvae(temperature=float(extra[0]) if extra else 1.0)
    else:
        sys.exit(__doc__)