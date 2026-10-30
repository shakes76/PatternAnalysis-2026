# AI Declaration
#
# Tool: Claude (Anthropic), used conversationally over several requests.
#
# What the AI did in this file:
# - Wrote the SSIM metric, parameter counting, training-curve and image-grid plotting.
# - At my request, added saving hyperparameters inside checkpoints so
#   models can be reloaded with the settings they were trained with.
#
# What I did:
# - Used these helpers through train.py, predict.py, audit.py and
#   evaluate.py, and asked for the checkpoint change after reloading a
#   model with the wrong settings caused errors.


import os

import matplotlib
matplotlib.use("Agg")  # save figures without needing a display (works on Rangpur)
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


def _gaussian_window(size=11, sigma=1.5, device="cpu"):
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] @ g[None, :])[None, None]  # (1, 1, size, size)


def ssim(x, y, data_range=1.0):
    """Mean SSIM per image for batches of shape (N, 1, H, W). Returns (N,)."""
    window = _gaussian_window(device=x.device)
    pad = window.shape[-1] // 2
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2

    mu_x = F.conv2d(x, window, padding=pad)
    mu_y = F.conv2d(y, window, padding=pad)
    var_x = F.conv2d(x * x, window, padding=pad) - mu_x ** 2
    var_y = F.conv2d(y * y, window, padding=pad) - mu_y ** 2
    cov = F.conv2d(x * y, window, padding=pad) - mu_x * mu_y

    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / (
        (mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2))
    return ssim_map.flatten(1).mean(dim=1)


def count_params(model):
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_history(history, out_path):
    """Plot train/val loss, plus val SSIM when the history has it, to a PNG."""
    epochs = range(1, len(history["train_loss"]) + 1)
    has_ssim = bool(history.get("val_ssim"))
    fig, axes = plt.subplots(1, 2 if has_ssim else 1, figsize=(10 if has_ssim else 5, 4))
    ax1 = axes[0] if has_ssim else axes
    ax1.plot(epochs, history["train_loss"], label="train")
    ax1.plot(epochs, history["val_loss"], label="validate")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.legend(); ax1.set_title("Loss")
    if has_ssim:
        ax2 = axes[1]
        ax2.plot(epochs, history["val_ssim"], color="tab:green")
        ax2.axhline(0.6, linestyle="--", color="grey", label="target 0.6")
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("SSIM"); ax2.legend(); ax2.set_title("Validation SSIM")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def save_image_grid(images, out_path, ncols=8, title=None):
    """Save a grid of (N, 1, H, W) images in [0, 1] to a PNG."""
    images = images.detach().cpu()
    n = images.size(0)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.2, nrows * 2.2))
    for i, ax in enumerate(axes.flat if n > 1 else [axes]):
        ax.axis("off")
        if i < n:
            ax.imshow(images[i, 0], cmap="gray", vmin=0, vmax=1)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def save_checkpoint(model, hparams, path):
    """Save weights together with the hyperparameters needed to rebuild the model."""
    torch.save({"model": model.state_dict(), "hyperparameters": hparams}, path)


def load_checkpoint(path, device):
    """Load a checkpoint. Returns (state_dict, hyperparameters or None).

    Also accepts older checkpoints that hold only a state_dict, in which case
    the hyperparameters are None and the caller falls back to its arguments.
    """
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt and "hyperparameters" in ckpt:
        return ckpt["model"], ckpt["hyperparameters"]
    return ckpt, None