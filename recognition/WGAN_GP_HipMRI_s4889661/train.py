"""
train.py — Training script for the HipMRI GANs

Trains either the DCGAN baseline (--model dcgan) or WGAN-GP (--model wgan-gp)
on the HipMRI 2D training split. Both use the same generator; only the
discriminator/critic and loss differ. Logs losses and model-specific
diagnostics, fixed-noise sample grids, checkpoints and resource usage
(time, peak VRAM, parameters) for the README.

Usage
-----
    # quick local smoke test
    python train.py --root ~/keras_slices_data --max-files 256 --epochs 2 \
        --batch-size 32 --run-name dcgan_smoke
    # full runs (Rangpur A100: see slurm/)
    python train.py --root <data> --model dcgan   --epochs 30 --run-name dcgan_baseline
    python train.py --root <data> --model wgan-gp --epochs 30 --run-name wgan_gp

Outputs (in runs/<run-name>/, gitignored)
-----------------------------------------
    samples/epoch_XXX.png   fixed-noise sample grid per epoch
    checkpoint.pt           full state (G, D, optimisers, epoch, args)
    generator.pt            generator weights only (used by predict.py)
    history.json            per-epoch metrics + run summary
    loss_curves.png         training loss / discriminator confidence plots
"""

import argparse
import json
import os
import random
import time

import matplotlib
matplotlib.use("Agg")                 # save plots to file; no display needed
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision.utils import save_image

from dataset import load_split, get_loader
from modules import Generator, Discriminator, weights_init, count_params, Z_DIM


# =============================================================================
# COMMAND-LINE ARGUMENTS (all hyperparameters explicit for reproducibility)
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Train a GAN on HipMRI 2D slices")
    p.add_argument("--root", required=True, help="path to keras_slices_data")
    p.add_argument("--model", default="dcgan", choices=["dcgan", "wgan-gp"])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=None,
                   help="default: 2e-4 (DCGAN) or 1e-4 (WGAN-GP)")
    p.add_argument("--n-critic", type=int, default=5, help="WGAN-GP critic steps per G step")
    p.add_argument("--gp-weight", type=float, default=10.0, help="WGAN-GP penalty weight (lambda)")
    p.add_argument("--z-dim", type=int, default=Z_DIM)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-files", type=int, default=None, help="subset for smoke tests")
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--run-name", default=None)
    return p.parse_args()

# =============================================================================
# REPRODUCIBILITY AND DEVICE
# =============================================================================

def set_seed(seed):
    """Seed Python and PyTorch RNGs (CUDA kernels may still be nondeterministic)."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    """Prefer CUDA (Colab), then Apple MPS (local Mac), then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# =============================================================================
# DCGAN TRAINING STEP (one batch)
# =============================================================================

def dcgan_step(G, D, real, opt_G, opt_D, bce, z_dim, device):
    """One DCGAN update: discriminator step, then generator step.

    Returns per-batch statistics for logging.
    """
    b = real.size(0)
    ones = torch.ones(b, 1, device=device)    # "real" targets
    zeros = torch.zeros(b, 1, device=device)  # "fake" targets

    # --- 1. Discriminator: real -> 1, fake -> 0 -------------------------------
    z = torch.randn(b, z_dim, device=device)
    fake = G(z)
    d_real = D(real)
    d_fake = D(fake.detach())     # detach: this step must not update G
    loss_D = bce(d_real, ones) + bce(d_fake, zeros)

    opt_D.zero_grad(set_to_none=True)
    loss_D.backward()
    opt_D.step()

    # --- 2. Generator: make D output 1 on fakes (non-saturating loss) ---------
    # Re-score the same fakes with the just-updated D; gradients flow into G.
    loss_G = bce(D(fake), ones)

    opt_G.zero_grad(set_to_none=True)
    loss_G.backward()   # also writes grads into D, but opt_D zeroes them next step
    opt_G.step()

    return {
        "loss_D": loss_D.item(),
        "loss_G": loss_G.item(),
        # Mean probability D assigns: ideal equilibrium is ~0.5 for both
        "D_real": torch.sigmoid(d_real).mean().item(),
        "D_fake": torch.sigmoid(d_fake).mean().item(),
    }

# =============================================================================
# WGAN-GP TRAINING STEP (several critic updates, then one generator update)
# =============================================================================

def gradient_penalty(critic, real, fake, device):
    """Gradient penalty (Gulrajani et al., 2017).

    Keeps the critic smooth (1-Lipschitz) by penalising the gradient norm of
    its score, at points blended between real and fake images, for
    deviating from 1.
    """
    b = real.size(0)

    # One random blend weight per image, broadcast over channels and pixels
    eps = torch.rand(b, 1, 1, 1, device=device)
    mixed = (eps * real + (1 - eps) * fake).requires_grad_(True)

    scores = critic(mixed)

    # Gradient of the scores w.r.t. the blended pixels. create_graph=True lets
    # the penalty itself be backpropagated into the critic's weights.
    grads = torch.autograd.grad(
        outputs=scores, inputs=mixed,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
    )[0]

    grads = grads.view(b, -1)                     # one gradient vector per image
    return ((grads.norm(2, dim=1) - 1) ** 2).mean()

def wgan_gp_step(G, C, real, opt_G, opt_C, z_dim, device, n_critic=5, gp_weight=10.0):
    """One WGAN-GP update: n_critic critic updates, then one generator update.

    Wasserstein critic loss (Arjovsky et al., 2017) with gradient penalty
    (Gulrajani et al., 2017).
    """
    b = real.size(0)

    # --- 1. Critic: maximise score(real) - score(fake), keep it smooth --------
    for _ in range(n_critic):
        z = torch.randn(b, z_dim, device=device)
        fake = G(z).detach()                       # critic step must not update G
        c_real = C(real).mean()
        c_fake = C(fake).mean()
        gp = gradient_penalty(C, real, fake, device)
        loss_C = c_fake - c_real + gp_weight * gp  # minimising this widens the gap

        opt_C.zero_grad(set_to_none=True)
        loss_C.backward()
        opt_C.step()

    # --- 2. Generator: make fakes score high ----------------------------------
    z = torch.randn(b, z_dim, device=device)
    loss_G = -C(G(z)).mean()

    opt_G.zero_grad(set_to_none=True)
    loss_G.backward()
    opt_G.step()

    return {
        "loss_C": loss_C.item(),
        "loss_G": loss_G.item(),
        # Estimated Wasserstein distance: should shrink as fakes improve
        "w_dist": (c_real - c_fake).item(),
        "gp": gp.item(),
    }

# =============================================================================
# OUTPUTS: SAMPLE GRIDS, CHECKPOINTS, PLOTS
# =============================================================================

@torch.no_grad()
def save_samples(G, fixed_z, path):
    """Save an 8x8 grid generated from the SAME noise every epoch."""
    G.eval()
    imgs = G(fixed_z)
    save_image(imgs, path, nrow=8, normalize=True, value_range=(-1, 1))
    G.train()


def save_checkpoint(G, D, opt_G, opt_D, epoch, args, out):
    """Full state for resuming, plus generator-only weights for predict.py."""
    torch.save({
        "epoch": epoch,
        "G": G.state_dict(), "D": D.state_dict(),
        "opt_G": opt_G.state_dict(), "opt_D": opt_D.state_dict(),
        "args": vars(args),
    }, os.path.join(out, "checkpoint.pt"))
    torch.save(G.state_dict(), os.path.join(out, "generator.pt"))


def plot_losses(history, out):
    """Plot every logged loss, plus the model-specific training diagnostic.

    DCGAN: discriminator confidence D(real) / D(G(z)).
    WGAN-GP: estimated Wasserstein distance (critic score gap).
    """
    epochs = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Left: all losses (loss_D / loss_C and loss_G)
    for key in [k for k in history[0] if k.startswith("loss_")]:
        ax1.plot(epochs, [h[key] for h in history], label=key)
    ax1.set_xlabel("epoch"); ax1.set_title("Training losses"); ax1.legend()

    # Right: the diagnostic that matters for this model type
    if "D_real" in history[0]:
        ax2.plot(epochs, [h["D_real"] for h in history], label="D(real)")
        ax2.plot(epochs, [h["D_fake"] for h in history], label="D(G(z))")
        ax2.axhline(0.5, ls="--", c="grey", lw=0.8)
        ax2.set_title("Discriminator confidence")
    else:
        ax2.plot(epochs, [h["w_dist"] for h in history], label="Wasserstein estimate")
        ax2.set_title("Critic score gap (Wasserstein estimate)")
    ax2.set_xlabel("epoch"); ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out, "loss_curves.png"), dpi=120)
    plt.close()

# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================

def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    run_name = args.run_name or f"{args.model}_seed{args.seed}"
    out = os.path.join(args.out_dir, run_name)
    os.makedirs(os.path.join(out, "samples"), exist_ok=True)

    # --- Data: TRAIN split only (validate/test reserved for the audit) -------
    data = load_split(args.root, "train", max_files=args.max_files,
                      cache_dir=args.cache_dir)
    loader = get_loader(data["images"], batch_size=args.batch_size)
    assert len(loader) > 0, "dataset smaller than one batch; raise --max-files"
    print(f"Device: {device} | train images: {len(data['images'])} "
          f"| batches/epoch: {len(loader)}")

    # --- Models and optimisers ------------------------------------------------
    # Same generator for both models; only the discriminator/critic and loss differ.
    wgan = args.model == "wgan-gp"
    G = Generator(args.z_dim).to(device)
    D = Discriminator(norm="layer" if wgan else "batch").to(device)
    G.apply(weights_init)
    D.apply(weights_init)

    # Optimiser settings from each model's paper:
    #   DCGAN   (Radford et al.):   lr 2e-4, betas (0.5, 0.999)
    #   WGAN-GP (Gulrajani et al.): lr 1e-4, betas (0.0, 0.9)
    lr = args.lr or (1e-4 if wgan else 2e-4)
    betas = (0.0, 0.9) if wgan else (0.5, 0.999)
    opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=betas)
    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=betas)
    bce = nn.BCEWithLogitsLoss()   # used by DCGAN only

    # Fixed noise: the same 64 vectors every epoch, to watch samples evolve
    fixed_z = torch.randn(64, args.z_dim, device=device)

    # --- Training --------------------------------------------------------------
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    history = []
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        t_epoch = time.time()
        sums, n = {}, 0

        for (real,) in loader:
            real = real.to(device)
            if wgan:
                stats = wgan_gp_step(G, D, real, opt_G, opt_D, args.z_dim, device,
                                     n_critic=args.n_critic, gp_weight=args.gp_weight)
            else:
                stats = dcgan_step(G, D, real, opt_G, opt_D, bce, args.z_dim, device)
            for k, v in stats.items():
                sums[k] = sums.get(k, 0.0) + v
            n += 1

        # Average batch statistics over the epoch
        record = {k: v / n for k, v in sums.items()}
        record["epoch"] = epoch
        record["epoch_time_s"] = time.time() - t_epoch
        history.append(record)
        metrics = " | ".join(f"{k} {v:.3f}" for k, v in record.items()
                             if k not in ("epoch", "epoch_time_s"))
        print(f"Epoch {epoch:3d}/{args.epochs} | {metrics} | {record['epoch_time_s']:.1f}s")

        save_samples(G, fixed_z, os.path.join(out, "samples", f"epoch_{epoch:03d}.png"))
        save_checkpoint(G, D, opt_G, opt_D, epoch, args, out)

    # --- Run summary: resource profiling for the README -----------------------
    summary = {
        "model": args.model,
        "total_time_s": time.time() - t_start,
        "peak_vram_gb": (torch.cuda.max_memory_allocated() / 1024**3
                         if device.type == "cuda" else None),
        "params_G": count_params(G),
        "params_D": count_params(D),
        "device": str(device),
        "args": vars(args),
    }
    with open(os.path.join(out, "history.json"), "w") as f:
        json.dump({"summary": summary, "history": history}, f, indent=2)
    plot_losses(history, out)
    print(f"Done in {summary['total_time_s']:.1f}s. Outputs in {out}/")

if __name__ == "__main__":
    main()