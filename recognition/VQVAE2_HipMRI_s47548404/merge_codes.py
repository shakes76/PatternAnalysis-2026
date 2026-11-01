"""
merge_codes.py - Plan B: merge near-twin codes so the PixelCNN priors have something learnable.

The dead-code reset left each 512-code codebook crowded (about 30% of codes have a near-twin),
so the encoder's choice between near-identical codes is close to random and the priors cannot
predict it. This script clusters each codebook with k-means into K groups, maps every code to
its group, and picks one REPRESENTATIVE code per group (the member nearest the group centre).
The VQ-VAE2 is NOT retrained: merged ids are decoded through their representative codes.

It reports, before anything is trained:
  - learnability: how often a position has the same code as its left neighbour, and code entropy,
    before vs after merging (higher agreement / lower entropy = easier for the prior)
  - cost: test-set SSIM when decoding with merged codes vs the original codes

Writes to --out-dir (ready for: train_prior.py --out-dir <out-dir> --num-codes K):
  codes_train.pt, codes_validate.pt   merged code grids
  merge.pt                            code -> group maps and group -> representative code

Usage:
    python merge_codes.py --data-dir "<path>/keras_slices_data" --k 128
"""
import argparse
import math
import os

import numpy as np
import torch
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader

from dataset import HipMRIDataset
from train import get_device
from train_prior import load_vqvae

EXACT = "donot_use_mm_for_euclid_dist"  # exact distances (no precision loss for close points)


def parse_args():
    p = argparse.ArgumentParser(description="Merge near-twin VQ-VAE2 codes with k-means")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--vqvae-ckpt", default="runs/vqvae2/best.pt")
    p.add_argument("--codes-dir", default="runs/prior", help="where the original cached codes are")
    p.add_argument("--out-dir", default="runs/prior_merged")
    p.add_argument("--k", type=int, default=128, help="number of merged codes per level")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def kmeans(x, k, seed=0, iters=100):
    """k-means with k-means++ initialisation. Returns (assignment (N,), centres (k, D))."""
    g = torch.Generator().manual_seed(seed)
    centres = x[torch.randint(len(x), (1,), generator=g)]
    for _ in range(1, k):  # k-means++: pick far-away points as new centres
        d2 = torch.cdist(x, centres, compute_mode=EXACT).min(1).values ** 2
        centres = torch.cat([centres, x[torch.multinomial(d2 / d2.sum(), 1, generator=g)]])
    for _ in range(iters):
        assign = torch.cdist(x, centres, compute_mode=EXACT).argmin(1)
        new = torch.stack([x[assign == j].mean(0) if (assign == j).any() else centres[j]
                           for j in range(k)])
        if torch.allclose(new, centres):
            break
        centres = new
    return torch.cdist(x, centres, compute_mode=EXACT).argmin(1), centres


def merge_level(codebook, k, seed):
    """code -> group map (512,), and group -> representative original code (k,)."""
    assign, centres = kmeans(codebook, k, seed)
    reps = torch.empty(k, dtype=torch.long)
    dist = torch.cdist(centres, codebook, compute_mode=EXACT)  # (k, 512)
    for j in range(k):
        members = (assign == j).nonzero().flatten()
        pool = members if len(members) else torch.arange(len(codebook))
        reps[j] = pool[dist[j, pool].argmin()]
    return assign, reps


def learnability(ids, n_codes):
    ids = ids.long()
    p = torch.bincount(ids.flatten(), minlength=n_codes).float()
    p = p[p > 0] / p.sum()
    entropy = -(p * p.log()).sum().item()
    same_left = (ids[:, :, 1:] == ids[:, :, :-1]).float().mean().item()
    return entropy, same_left


@torch.no_grad()
def test_ssim(vqvae, data_dir, maps, device):
    """Mean test SSIM decoding with the original codes vs with merged (representative) codes."""
    loader = DataLoader(HipMRIDataset(data_dir, "test"), batch_size=64)
    s_orig, s_merged = [], []
    for x in loader:
        x = x.to(device)
        _, _, _, id_t, id_b = vqvae.encode(x)
        orig = vqvae.decode_code(id_t, id_b).clamp(0, 1)
        merged = vqvae.decode_code(maps["top_rep"][maps["top_map"][id_t]],
                                   maps["bottom_rep"][maps["bottom_map"][id_b]]).clamp(0, 1)
        xs, o, m = x[:, 0].cpu().numpy(), orig[:, 0].cpu().numpy(), merged[:, 0].cpu().numpy()
        s_orig += [structural_similarity(xs[i], o[i], data_range=1.0) for i in range(len(xs))]
        s_merged += [structural_similarity(xs[i], m[i], data_range=1.0) for i in range(len(xs))]
    return float(np.mean(s_orig)), float(np.mean(s_merged)), float(np.min(s_merged))


def main():
    args = parse_args()
    device = get_device()
    os.makedirs(args.out_dir, exist_ok=True)
    vqvae = load_vqvae(args.vqvae_ckpt, device)
    sd = {k: v.cpu() for k, v in vqvae.state_dict().items()}

    maps = {}
    for level, key in [("top", "quantize_t.embedding.weight"), ("bottom", "quantize_b.embedding.weight")]:
        code_map, reps = merge_level(sd[key], args.k, args.seed)
        maps[f"{level}_map"], maps[f"{level}_rep"] = code_map, reps
        print(f"[{level}] 512 codes -> {args.k} groups "
              f"(largest group {torch.bincount(code_map).max().item()} codes)")

    train = torch.load(os.path.join(args.codes_dir, "codes_train.pt"))
    val = torch.load(os.path.join(args.codes_dir, "codes_validate.pt"))
    print(f"\nLearnability on training codes (max entropy: 512 codes {math.log(512):.2f}, "
          f"{args.k} codes {math.log(args.k):.2f}):")
    for level in ["top", "bottom"]:
        e0, s0 = learnability(train[level], 512)
        e1, s1 = learnability(maps[f"{level}_map"][train[level].long()], args.k)
        print(f"  {level}: same-as-left {s0:.1%} -> {s1:.1%} | entropy {e0:.2f} -> {e1:.2f}")

    maps_dev = {k: v.to(device) for k, v in maps.items()}
    s_orig, s_merged, s_min = test_ssim(vqvae, args.data_dir, maps_dev, device)
    print(f"\nTest SSIM: original codes {s_orig:.4f} -> merged codes {s_merged:.4f} (min {s_min:.4f})")

    for split, codes in [("train", train), ("validate", val)]:
        merged = {lvl: maps[f"{lvl}_map"][codes[lvl].long()].to(torch.int16) for lvl in ["top", "bottom"]}
        torch.save(merged, os.path.join(args.out_dir, f"codes_{split}.pt"))
    torch.save({**maps, "k": args.k}, os.path.join(args.out_dir, "merge.pt"))
    print(f"\nSaved merged codes and merge.pt to {args.out_dir}/")
    print(f"Next: python train_prior.py --data-dir ... --out-dir {args.out_dir} --num-codes {args.k} ...")


if __name__ == "__main__":
    main()