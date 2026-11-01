"""
train.py - Train the baseline autoencoder or VQ-VAE2 on HipMRI 2D slices (COMP3710, task 2.4).

Smoke test (a few hundred steps, prints the measured speed):
    python train.py --data-dir "<path>/keras_slices_data" --model vqvae2 --max-steps 200

Full run (keep the Mac awake):
    caffeinate -i python train.py --data-dir "<path>/keras_slices_data" --model vqvae2 --epochs 30

Outputs, in <out-dir>/<model>/:
    log.csv            losses and time per epoch (for the training curves)
    snapshots/         reconstructions of the same 8 validation slices every few epochs
    best.pt, last.pt   model state_dict at the best validation loss and at the end
"""
import argparse
import csv
import os
import random
import time

import matplotlib
matplotlib.use("Agg")  # save figures to files without opening windows
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import HipMRIDataset
from modules import BaselineAE, VQVAE2


# setup
def parse_args():
    p = argparse.ArgumentParser(description="Train baseline AE or VQ-VAE2 on HipMRI slices")
    p.add_argument("--data-dir", required=True, help="path to keras_slices_data")
    p.add_argument("--model", choices=["baseline", "vqvae2"], default="vqvae2")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--snapshot-every", type=int, default=2, help="epochs between snapshot grids")
    p.add_argument("--max-steps", type=int, default=None, help="stop after this many steps (smoke test)")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device):
    """Wait for the GPU to finish, so timings measure real work (GPUs run asynchronously)."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(name):
    return BaselineAE() if name == "baseline" else VQVAE2()


def fixed_snapshot_batch(dataset, n=8):
    """The same n validation slices every time, spread evenly through the split."""
    idx = np.linspace(0, len(dataset) - 1, n).astype(int)
    return torch.stack([dataset[i] for i in idx])

def model_forward(model, x):
    """
    Run either model and return (x_hat, vq_loss).
    BaselineAE returns just x_hat; VQVAE2 returns (x_hat, vq_loss).
    """

    output = model(x)

    if isinstance(output, tuple):
        x_hat, vq_loss = output
    else:
        x_hat = output
        vq_loss = torch.tensor(0.0, device=x.device)

    return x_hat, vq_loss

@torch.no_grad()
def evaluate(model, loader, device):
    """Average reconstruction loss and VQ loss over a whole loader."""
    model.eval()

    total_recon_loss = 0.0
    total_vq_loss = 0.0
    total_samples = 0

    for x in loader:
        x = x.to(device)
        batch_size = x.size(0)

        x_hat, vq_loss = model_forward(model, x)

        recon_loss = F.mse_loss(x_hat, x)

        total_recon_loss += recon_loss.item() * batch_size
        total_vq_loss += vq_loss.item() * batch_size

        total_samples += batch_size

    mean_recon = total_recon_loss / total_samples
    mean_vq = total_vq_loss / total_samples

    return mean_recon, mean_vq

# outputs
@torch.no_grad()
def save_snapshot(model, fixed_x, device, path, title):
    """Originals (top row) and reconstructions (bottom row) of the fixed slices."""
    was_training = model.training
    model.eval()
    x_hat, _ = model_forward(model, fixed_x.to(device))
    if was_training:
        model.train()
    x, x_hat = fixed_x.cpu().numpy(), x_hat.clamp(0, 1).cpu().numpy()
    n = x.shape[0]
    fig, axes = plt.subplots(2, n, figsize=(1.6 * n, 6.5))
    for i in range(n):
        axes[0, i].imshow(x[i, 0], cmap="gray", vmin=0, vmax=1)
        axes[1, i].imshow(x_hat[i, 0], cmap="gray", vmin=0, vmax=1)
        axes[0, i].axis("off")
        axes[1, i].axis("off")
    axes[0, 0].set_title("original", loc="left", fontsize=9)
    axes[1, 0].set_title("reconstruction", loc="left", fontsize=9)
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close(fig)


class CSVLog:
    FIELDS = ["epoch", "train_recon", "train_vq", "val_recon", "val_vq", "epoch_seconds"]

    def __init__(self, path):
        self.path = path
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(self.FIELDS)

    def write(self, **row):
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow([row[k] for k in self.FIELDS])


def save_checkpoint(model, args, epoch, val_recon, path):
    torch.save({"model": model.state_dict(), "model_name": args.model,
                "epoch": epoch, "val_recon": val_recon, "args": vars(args)}, path)

def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    out = os.path.join(args.out_dir, args.model)
    os.makedirs(os.path.join(out, "snapshots"), exist_ok=True)

    train_ds = HipMRIDataset(args.data_dir, "train")
    val_ds = HipMRIDataset(args.data_dir, "validate")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)
    fixed_x = fixed_snapshot_batch(val_ds)

    model = build_model(args.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {device} | model: {args.model} ({n_params:,} parameters) | "
          f"train {len(train_ds)} slices, {len(train_loader)} steps/epoch | val {len(val_ds)} slices")

    log = CSVLog(os.path.join(out, "log.csv"))
    save_snapshot(model, fixed_x, device, os.path.join(out, "snapshots", "epoch000.png"),
                  f"{args.model}: before training")

    best_val, global_step = float("inf"), 0
    timing_steps, timing_start = 50, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.time()
        sum_recon, sum_vq, n_batches = 0.0, 0.0, 0

        for x in train_loader:
            x = x.to(device)
            if global_step == 5:  # skip the first few (warm-up) steps when timing
                sync(device)
                timing_start = time.time()

            # One training step
            # 1. x_hat, vq_loss = model_forward(model, x)
            x_hat, vq_loss = model_forward(model, x)
            # 2. recon_loss = reconstruction loss between x_hat and x
            recon_loss = F.mse_loss(x_hat, x)
            # 3. combine loss
            loss = recon_loss + vq_loss
            # 4. zero the gradients, backpropagate, take an optimiser step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_recon += recon_loss.item()
            sum_vq += vq_loss.item()
            n_batches += 1
            global_step += 1

            if global_step == 5 + timing_steps:
                sync(device)
                sec = (time.time() - timing_start) / timing_steps
                print(f"Measured speed: {sec:.3f} s/step -> about "
                      f"{sec * len(train_loader) / 60:.1f} min per epoch on {device}")
            if global_step % 100 == 0:
                print(f"  step {global_step}: recon {recon_loss.item():.5f}, vq {vq_loss.item():.5f}")
            if args.max_steps and global_step >= args.max_steps:
                break

        train_recon, train_vq = sum_recon / max(n_batches, 1), sum_vq / max(n_batches, 1)
        val_recon, val_vq = evaluate(model, val_loader, device)
        epoch_seconds = time.time() - epoch_start
        log.write(epoch=epoch, train_recon=train_recon, train_vq=train_vq,
                  val_recon=val_recon, val_vq=val_vq, epoch_seconds=round(epoch_seconds, 1))
        print(f"Epoch {epoch}: train recon {train_recon:.5f} vq {train_vq:.5f} | "
              f"val recon {val_recon:.5f} vq {val_vq:.5f} | {epoch_seconds / 60:.1f} min")

        if val_recon < best_val:
            best_val = val_recon
            save_checkpoint(model, args, epoch, val_recon, os.path.join(out, "best.pt"))
        if epoch % args.snapshot_every == 0 or epoch == args.epochs:
            save_snapshot(model, fixed_x, device,
                          os.path.join(out, "snapshots", f"epoch{epoch:03d}.png"),
                          f"{args.model}: epoch {epoch}")
        if args.max_steps and global_step >= args.max_steps:
            print(f"Stopped after {global_step} steps (--max-steps).")
            break

    save_checkpoint(model, args, epoch, val_recon, os.path.join(out, "last.pt"))
    save_snapshot(model, fixed_x, device, os.path.join(out, "snapshots", "final.png"),
                  f"{args.model}: final")
    print(f"Done. Best val recon {best_val:.5f}. Outputs in {out}/")


if __name__ == "__main__":
    main()