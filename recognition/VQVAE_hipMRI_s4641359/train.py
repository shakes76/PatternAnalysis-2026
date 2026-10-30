# AI Declaration
#
# Tool: Claude (Anthropic), used conversationally over several requests.
#
# What the AI did in this file:
# - Wrote the training, validation and test loop for the VAE and VQ-VAE,
#   with SSIM, codebook perplexity, peak VRAM and timing logged.
# - Added PixelCNN prior training on code grids from a trained VQ-VAE,
#   followed by sampling new slices.
# - After I suggested, added command-line options for prior width, depth and the position
#   embedding, and read the VQ-VAE's settings from its checkpoint.
#
# What I did:
# - Ran all training on my laptop GPU (VAE 20 epochs, VQ-VAE 64x32
#   15 epochs, VQ-VAE 32x16 12 epochs, and the 32x16 priors) and the
#   final 60-block prior on a Google Colab A100.
# - Chose the run settings, compared results between runs, and set up
#   the Rangpur and Colab environments.


import argparse
import json
import os
import time

import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from dataset import get_dataloaders
from modules import VAE, VQVAE, PixelCNNPrior, vae_loss, vqvae_loss
from utils import (count_params, load_checkpoint, plot_history, save_checkpoint,
                   save_image_grid, ssim)


def model_hparams(args):
    """Hyperparameters stored in each checkpoint so predict.py can rebuild the model."""
    if args.model == "vae":
        return {"model": "vae", "latent_dim": args.latent_dim}
    if args.model == "vqvae":
        return {"model": "vqvae", "num_embeddings": args.num_embeddings,
                "embedding_dim": args.embedding_dim, "n_down": args.n_down}
    return {"model": "prior", "num_embeddings": args.num_embeddings,
            "channels": args.prior_channels, "n_blocks": args.prior_blocks,
            "pos": args.prior_pos}


def build_model(args):
    """Create the model selected by --model."""
    if args.model == "vae":
        return VAE(latent_dim=args.latent_dim)
    return VQVAE(num_embeddings=args.num_embeddings, embedding_dim=args.embedding_dim,
                 n_down=args.n_down)


def forward_and_loss(model, x, args):
    """Run one batch through either model. Returns (loss, reconstruction, perplexity)."""
    if args.model == "vae":
        recon, mu, logvar = model(x)
        loss, _, _ = vae_loss(recon, x, mu, logvar, args.beta)
        return loss, recon, None
    recon, vq_loss, _, perplexity = model(x)
    loss, _ = vqvae_loss(recon, x, vq_loss)
    return loss, recon, perplexity


def run_epoch(model, loader, device, args, optimiser=None):
    """One pass over a loader. Trains if an optimiser is given, else evaluates.

    Returns (mean loss, mean SSIM, mean perplexity or None).
    """
    training = optimiser is not None
    model.train(training)
    total_loss, total_ssim, total_perp, n, batches = 0.0, 0.0, 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x in loader:
            x = x.to(device, non_blocking=True)
            loss, recon, perplexity = forward_and_loss(model, x, args)
            if training:
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
            total_loss += loss.item() * x.size(0)
            total_ssim += ssim(recon.detach(), x).sum().item()
            if perplexity is not None:
                total_perp += perplexity.item()
            n += x.size(0)
            batches += 1
    mean_perp = total_perp / batches if args.model == "vqvae" else None
    return total_loss / n, total_ssim / n, mean_perp


@torch.no_grad()
def save_examples(model, loaders, device, args, out_dir):
    """Save test reconstructions, plus samples (VAE) or random-code decodes (VQ-VAE)."""
    model.eval()
    x = next(iter(loaders["test"]))[:8].to(device)
    _, recon, _ = forward_and_loss(model, x, args)
    save_image_grid(torch.cat([x, recon]), os.path.join(out_dir, "reconstructions.png"),
                    title="Top: test input | Bottom: reconstruction")
    if args.model == "vae":
        save_image_grid(model.sample(16, device), os.path.join(out_dir, "samples.png"),
                        title="VAE random samples")
    else:
        # Without a learned prior, uniformly random codes decode to noise-like
        # texture. This motivates training the PixelCNN prior.
        grid = model.encode_indices(x[:1]).shape[1:]
        codes = torch.randint(0, model.num_embeddings, (16, *grid), device=device)
        save_image_grid(model.decode_indices(codes),
                        os.path.join(out_dir, "random_code_samples.png"),
                        title="Uniform random codes (no prior)")


@torch.no_grad()
def extract_codes(vqvae, loader, device):
    """Encode every slice in a loader into its VQ-VAE code grid (N, H', W')."""
    vqvae.eval()
    return torch.cat([vqvae.encode_indices(x.to(device)).cpu() for x in loader])


def run_prior_epoch(prior, loader, device, optimiser=None):
    """One pass of the prior. Returns mean cross-entropy loss (nats per code)."""
    training = optimiser is not None
    prior.train(training)
    total, n = 0.0, 0
    with torch.set_grad_enabled(training):
        for (codes,) in loader:
            codes = codes.to(device, non_blocking=True)
            loss = F.cross_entropy(prior(codes), codes)
            if training:
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
            total += loss.item() * codes.size(0)
            n += codes.size(0)
    return total / n


def train_prior(args, device, out_dir):
    """Train the PixelCNN prior on code grids from a frozen, trained VQ-VAE."""
    loaders = get_dataloaders(args.data_root, batch_size=args.batch_size)
    state, hp = load_checkpoint(args.vqvae_ckpt, device)
    if hp:  # newer checkpoints carry their own settings; override the CLI flags
        args.num_embeddings, args.embedding_dim, args.n_down = (
            hp["num_embeddings"], hp["embedding_dim"], hp["n_down"])
    vqvae = VQVAE(num_embeddings=args.num_embeddings, embedding_dim=args.embedding_dim,
                  n_down=args.n_down).to(device)
    vqvae.load_state_dict(state)
    vqvae.eval()

    codes = {split: extract_codes(vqvae, loader, device) for split, loader in loaders.items()}
    grid_shape = tuple(codes["train"].shape[1:])
    print(f"Code grid shape: {grid_shape}")
    code_loaders = {split: DataLoader(TensorDataset(c), batch_size=args.batch_size,
                                      shuffle=(split == "train"))
                    for split, c in codes.items()}

    prior = PixelCNNPrior(args.num_embeddings, args.prior_channels, args.prior_blocks,
                          use_pos=args.prior_pos).to(device)
    optimiser = torch.optim.Adam(prior.parameters(), lr=args.lr)
    print(f"Model: prior | trainable parameters: {count_params(prior):,}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    history = {"train_loss": [], "val_loss": []}
    best_val, epoch_times = float("inf"), []
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss = run_prior_epoch(prior, code_loaders["train"], device, optimiser)
        val_loss = run_prior_epoch(prior, code_loaders["validate"], device)
        epoch_times.append(time.time() - start)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"Epoch {epoch:3d} | train CE {train_loss:.4f} | val CE {val_loss:.4f} "
              f"({val_loss / math.log(2):.3f} bits/code) | {epoch_times[-1]:.1f}s")
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(prior, model_hparams(args), os.path.join(out_dir, "best.pt"))

    plot_history(history, os.path.join(out_dir, "training_curves.png"))
    prior.load_state_dict(load_checkpoint(os.path.join(out_dir, "best.pt"), device)[0])
    test_loss = run_prior_epoch(prior, code_loaders["test"], device)

    # Uncurated samples: sample code grids from the prior and decode them.
    start = time.time()
    sampled = prior.sample(args.n_samples, grid_shape, device)
    images = vqvae.decode_indices(sampled)
    sample_seconds = time.time() - start
    save_image_grid(images, os.path.join(out_dir, "prior_samples.png"),
                    title=f"{args.n_samples} uncurated prior samples")

    peak_vram = torch.cuda.max_memory_allocated() / 1024 ** 3 if device.type == "cuda" else 0.0
    results = {
        "model": "prior",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "params": count_params(prior),
        "best_val_ce": best_val,
        "test_ce": test_loss,
        "test_bits_per_code": test_loss / math.log(2),
        "seconds_to_sample": round(sample_seconds, 1),
        "n_samples": args.n_samples,
        "peak_vram_gb": round(peak_vram, 3),
        "mean_epoch_seconds": round(sum(epoch_times) / len(epoch_times), 1),
    }
    print(json.dumps(results, indent=2))
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Train a model on HipMRI 2D slices.")
    parser.add_argument("--model", default="vqvae", choices=["vae", "vqvae", "prior"])
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=None,
                        help="Default: 1e-3 for the VAE and prior, 2e-4 for the VQ-VAE")
    parser.add_argument("--beta", type=float, default=1.0, help="VAE KL weight")
    parser.add_argument("--latent_dim", type=int, default=256, help="VAE latent size")
    parser.add_argument("--num_embeddings", type=int, default=512, help="VQ-VAE codebook size K")
    parser.add_argument("--embedding_dim", type=int, default=64, help="VQ-VAE code dimension D")
    parser.add_argument("--n_down", type=int, default=2,
                        help="VQ-VAE downsampling steps (2 -> 64x32 code grid)")
    parser.add_argument("--vqvae_ckpt", default=None,
                        help="Trained VQ-VAE weights (required for --model prior)")
    parser.add_argument("--prior_channels", type=int, default=128)
    parser.add_argument("--prior_blocks", type=int, default=10)
    parser.add_argument("--prior_pos", action="store_true",
                        help="Add a learned row/column position embedding to the prior")
    parser.add_argument("--n_samples", type=int, default=16,
                        help="Prior samples to generate after training")
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.lr is None:
        args.lr = 2e-4 if args.model == "vqvae" else 1e-3

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = args.out_dir or os.path.join("outputs", args.model)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Device: {device}")

    if args.model == "prior":
        if not args.vqvae_ckpt:
            parser.error("--model prior needs --vqvae_ckpt (e.g. outputs/vqvae/best.pt)")
        train_prior(args, device, out_dir)
        return

    loaders = get_dataloaders(args.data_root, batch_size=args.batch_size)
    print({split: len(loader.dataset) for split, loader in loaders.items()})

    model = build_model(args).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"Model: {args.model} | trainable parameters: {count_params(model):,}")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    history = {"train_loss": [], "val_loss": [], "val_ssim": [], "val_perplexity": []}
    best_ssim, epoch_times = -1.0, []
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, _, _ = run_epoch(model, loaders["train"], device, args, optimiser)
        val_loss, val_ssim, val_perp = run_epoch(model, loaders["validate"], device, args)
        epoch_times.append(time.time() - start)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_ssim"].append(val_ssim)
        history["val_perplexity"].append(val_perp)
        perp_text = f" | perplexity {val_perp:6.1f}" if val_perp is not None else ""
        print(f"Epoch {epoch:3d} | train loss {train_loss:9.4f} | val loss {val_loss:9.4f} | "
              f"val SSIM {val_ssim:.4f}{perp_text} | {epoch_times[-1]:.1f}s")

        if val_ssim > best_ssim:
            best_ssim = val_ssim
            save_checkpoint(model, model_hparams(args), os.path.join(out_dir, "best.pt"))

    plot_history(history, os.path.join(out_dir, "training_curves.png"))

    # Test with the best weights.
    model.load_state_dict(load_checkpoint(os.path.join(out_dir, "best.pt"), device)[0])
    test_loss, test_ssim, test_perp = run_epoch(model, loaders["test"], device, args)
    save_examples(model, loaders, device, args, out_dir)

    peak_vram = torch.cuda.max_memory_allocated() / 1024 ** 3 if device.type == "cuda" else 0.0
    results = {
        "model": args.model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "params": count_params(model),
        "best_val_ssim": best_ssim,
        "test_loss": test_loss,
        "test_ssim": test_ssim,
        "test_perplexity": test_perp,
        "peak_vram_gb": round(peak_vram, 3),
        "mean_epoch_seconds": round(sum(epoch_times) / len(epoch_times), 1),
    }
    print(json.dumps(results, indent=2))
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()