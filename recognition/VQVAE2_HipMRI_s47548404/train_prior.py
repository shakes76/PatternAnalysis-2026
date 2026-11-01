"""
train_prior.py - Stage 2: train the PixelCNN priors on VQ-VAE2 code grids (COMP3710, task 2.4).

1. Encodes every train/validation slice with the trained VQ-VAE2 into integer code
   grids (top 32x16, bottom 64x32), caches them, and reports codebook usage.
2. Trains the top prior (unconditional) and/or the bottom prior (conditioned on top).

Smoke test:
    python train_prior.py --data-dir "<path>/keras_slices_data" --level top --max-steps 100
Full run:
    caffeinate -i python train_prior.py --data-dir "<path>/keras_slices_data" --level both --epochs 10

Outputs, in <out-dir>/:
    codes_train.pt, codes_validate.pt   cached code grids (int16)
    <level>_log.csv                     train/val cross-entropy per epoch
    <level>_best.pt, <level>_last.pt    prior checkpoints
"""
import argparse
import csv
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from dataset import HipMRIDataset
from modules import PixelCNNPrior, VQVAE2
from train import get_device, set_seed, sync


def parse_args():
    p = argparse.ArgumentParser(description="Train PixelCNN priors over VQ-VAE2 codes")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--vqvae-ckpt", default="runs/vqvae2/best.pt")
    p.add_argument("--level", choices=["top", "bottom", "both"], default="both")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--channels", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=10)
    p.add_argument("--out-dir", default="runs/prior")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-codes", type=int, default=512)
    return p.parse_args()


def load_vqvae(path, device):
    ckpt = torch.load(path, map_location=device)
    model = VQVAE2().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded VQ-VAE2 from {path} (epoch {ckpt['epoch']}, val recon {ckpt['val_recon']:.5f})")
    return model


@torch.no_grad()
def extract_codes(vqvae, data_dir, split, device, cache_path, batch_size=64):
    """Code grids for every slice in a split, cached to disk (int16 is plenty for 512 codes)."""
    if os.path.exists(cache_path):
        codes = torch.load(cache_path)
        print(f"[{split}] loaded cached codes from {cache_path}")
        return codes["top"], codes["bottom"]
    loader = DataLoader(HipMRIDataset(data_dir, split), batch_size=batch_size, shuffle=False)
    tops, bottoms = [], []
    for x in loader:
        _, _, _, id_t, id_b = vqvae.encode(x.to(device))
        tops.append(id_t.to(torch.int16).cpu())
        bottoms.append(id_b.to(torch.int16).cpu())
    top, bottom = torch.cat(tops), torch.cat(bottoms)
    torch.save({"top": top, "bottom": bottom}, cache_path)
    print(f"[{split}] extracted codes: top {tuple(top.shape)}, bottom {tuple(bottom.shape)}")
    return top, bottom


def report_usage(top, bottom, num_codes=512):
    """Codebook usage on this split: distinct codes used, out of num_codes."""
    print(f"Codebook usage: top {top.unique().numel()} / {num_codes}, "
          f"bottom {bottom.unique().numel()} / {num_codes}")


def prior_forward(prior, batch, level, device):
    """Return (logits, target) for either level.

    top:    predict the top grid from itself (masked, so only earlier positions are seen)
    bottom: predict the bottom grid, conditioned on the matching top grid
    """
    top = batch[0].to(device).long()
    if level == "top":
        return prior(top), top
    bottom = batch[1].to(device).long()
    return prior(bottom, cond_ids=top), bottom


@torch.no_grad()
def evaluate(prior, loader, level, device):
    prior.eval()
    total, n = 0.0, 0
    for batch in loader:
        logits, target = prior_forward(prior, batch, level, device)
        total += F.cross_entropy(logits, target).item() * target.size(0)
        n += target.size(0)
    prior.train()
    return total / n


def train_level(level, args, device, codes_train, codes_val):
    out = args.out_dir
    train_loader = DataLoader(TensorDataset(*codes_train), batch_size=args.batch_size,
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(*codes_val), batch_size=args.batch_size)
    prior = PixelCNNPrior(num_codes=args.num_codes, channels=args.channels, n_layers=args.n_layers,
                          cond_num_codes=args.num_codes if level == "bottom" else None).to(device)
    # Start at the code-frequency baseline: output bias = log p(code)
    target = codes_train[0] if level == "top" else codes_train[1]
    counts = torch.bincount(target.long().flatten(), minlength=args.num_codes).float()
    prior.head[-1].bias.data = (counts.clamp_min(1) / counts.sum()).log().to(device)
    optimizer = torch.optim.Adam(prior.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in prior.parameters())
    print(f"\n=== {level} prior: {n_params:,} parameters, {len(train_loader)} steps/epoch ===")

    log_path = os.path.join(out, f"{level}_log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_ce", "val_ce", "epoch_seconds"])

    best_val, global_step, timing_start = float("inf"), 0, None
    for epoch in range(1, args.epochs + 1):
        prior.train()
        epoch_start, sum_loss, n_batches = time.time(), 0.0, 0
        for batch in train_loader:
            if global_step == 5:
                sync(device)
                timing_start = time.time()

            # One training step
            # 1. Forward pass
            logits, target = prior_forward(prior, batch, level, device)

            # 2. Cross-entropy loss
            loss = F.cross_entropy(logits, target)

            # 3. Backpropagation and optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            n_batches += 1
            global_step += 1
            if global_step == 55:
                sync(device)
                sec = (time.time() - timing_start) / 50
                print(f"Measured speed: {sec:.3f} s/step -> about "
                      f"{sec * len(train_loader) / 60:.1f} min per epoch on {device}")
            if global_step % 100 == 0:
                print(f"  step {global_step}: cross-entropy {loss.item():.4f}")
            if args.max_steps and global_step >= args.max_steps:
                break

        train_ce = sum_loss / max(n_batches, 1)
        val_ce = evaluate(prior, val_loader, level, device)
        seconds = time.time() - epoch_start
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, train_ce, val_ce, round(seconds, 1)])
        print(f"[{level}] epoch {epoch}: train CE {train_ce:.4f} | val CE {val_ce:.4f} | "
              f"{seconds / 60:.1f} min")

        ckpt = {"model": prior.state_dict(), "level": level, "epoch": epoch, "val_ce": val_ce,
                "channels": args.channels, "n_layers": args.n_layers, "num_codes": args.num_codes}
        if val_ce < best_val:
            best_val = val_ce
            torch.save(ckpt, os.path.join(out, f"{level}_best.pt"))
        if args.max_steps and global_step >= args.max_steps:
            print(f"Stopped after {global_step} steps (--max-steps).")
            break
    torch.save(ckpt, os.path.join(out, f"{level}_last.pt"))
    print(f"[{level}] done. Best val CE {best_val:.4f}")


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    os.makedirs(args.out_dir, exist_ok=True)

    vqvae = load_vqvae(args.vqvae_ckpt, device)
    codes_train = extract_codes(vqvae, args.data_dir, "train", device,
                                os.path.join(args.out_dir, "codes_train.pt"))
    codes_val = extract_codes(vqvae, args.data_dir, "validate", device,
                              os.path.join(args.out_dir, "codes_validate.pt"))
    report_usage(*codes_train, num_codes=args.num_codes)
    del vqvae  # free memory before training the priors

    levels = ["top", "bottom"] if args.level == "both" else [args.level]
    for level in levels:
        train_level(level, args, device, codes_train, codes_val)


if __name__ == "__main__":
    main()