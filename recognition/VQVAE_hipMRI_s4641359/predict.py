# AI Declaration
#
# Tool: Claude (Anthropic), used conversationally over several requests.
#
# What the AI did in this file:
# - Wrote the initial script: loading the trained VQ-VAE and PixelCNN
#   checkpoints, computing test-set SSIM, and saving reconstruction and
#   code-map figures.
# - Added prior sampling with an adjustable temperature.
# - At my request, extended it to compute SSIM for every test slice,
#   save the scores to CSV, plot their distribution, and show the
#   five lowest- and highest-SSIM slices with error maps.
# - At my request, added a per-patient SSIM summary and a plot of SSIM
#   against slice position, to check whether failures cluster in
#   particular patients or parts of the scan.
#
# What I did:
# - Decided which analyses the report needed and requested each one.
# - Ran it on my trained models on both my laptop and Colab, and
#   checked that it reproduced the test SSIM from training (0.9174).
# - Ran the temperature sweep (T = 1.0, 0.8, 0.7) to test my hypothesis
#   that rare codes caused the smeared regions.


import argparse
import json
import os
import re
import time
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from dataset import get_dataloaders, patient_id
from modules import VQVAE, PixelCNNPrior
from utils import load_checkpoint, save_image_grid, ssim


def load_vqvae(args, device):
    """Build a VQ-VAE with its training settings and load its weights."""
    state, hp = load_checkpoint(args.vqvae_ckpt, device)
    if hp:
        args.num_embeddings, args.embedding_dim, args.n_down = (
            hp["num_embeddings"], hp["embedding_dim"], hp["n_down"])
    model = VQVAE(num_embeddings=args.num_embeddings, embedding_dim=args.embedding_dim,
                  n_down=args.n_down).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def load_prior(args, device):
    """Build a PixelCNN prior with its training settings and load its weights."""
    state, hp = load_checkpoint(args.prior_ckpt, device)
    if hp:
        args.prior_channels, args.prior_blocks = hp["channels"], hp["n_blocks"]
    use_pos = bool(hp.get("pos", False)) if hp else False
    prior = PixelCNNPrior(args.num_embeddings, args.prior_channels, args.prior_blocks,
                          use_pos=use_pos).to(device)
    prior.load_state_dict(state)
    prior.eval()
    return prior


@torch.no_grad()
def per_slice_ssim(model, loader, device):
    """SSIM of every test slice vs its reconstruction, in dataset order. Returns (N,)."""
    scores = []
    for x in loader:  # the test loader is not shuffled, so order matches the file list
        x = x.to(device)
        scores.append(ssim(model(x)[0], x).cpu())
    return torch.cat(scores)


def save_ssim_report(scores, files, out_dir):
    """Write per-slice SSIM (worst first) to CSV and plot its histogram."""
    order = torch.argsort(scores)
    with open(os.path.join(out_dir, "per_slice_ssim.csv"), "w") as f:
        f.write("file,ssim\n")
        for i in order.tolist():
            f.write(f"{os.path.basename(files[i])},{scores[i].item():.4f}\n")

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(scores.numpy(), bins=40, color="tab:blue")
    ax.axvline(0.6, linestyle="--", color="grey", label="target 0.6")
    ax.axvline(scores.mean().item(), color="tab:red", label=f"mean {scores.mean():.3f}")
    ax.set_xlabel("SSIM (reconstruction vs input)")
    ax.set_ylabel("Test slices")
    ax.set_title("Per-slice test SSIM")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "ssim_histogram.png"), dpi=120)
    plt.close(fig)
    return order


@torch.no_grad()
def save_ranked_cases(model, dataset, scores, idx, title, out_path):
    """Input | reconstruction | absolute error for the given slice indices.

    Used for both the lowest- and highest-SSIM slices. The error maps share one
    colour scale within a figure so rows can be compared directly.
    """
    n = len(idx)
    x = torch.stack([dataset[i] for i in idx]).to(next(model.parameters()).device)
    recon = model(x)[0]
    x, recon = x.cpu(), recon.cpu()
    error = (x - recon).abs()
    fig, axes = plt.subplots(n, 3, figsize=(6, 2.6 * n))
    if n == 1:
        axes = axes[None]
    for row, i in enumerate(idx):
        name = os.path.basename(dataset.files[i]).replace(".nii.gz", "").replace(".nii", "")
        axes[row, 0].imshow(x[row, 0], cmap="gray", vmin=0, vmax=1)
        axes[row, 1].imshow(recon[row, 0], cmap="gray", vmin=0, vmax=1)
        axes[row, 2].imshow(error[row, 0], cmap="magma", vmin=0, vmax=max(error.max().item(), 1e-6))
        axes[row, 0].set_ylabel(f"{name}\nSSIM {scores[i]:.3f}", fontsize=7)
        for j, col in enumerate(["Input", "Reconstruction", "|Error|"]):
            axes[row, j].set_xticks([]); axes[row, j].set_yticks([])
            if row == 0:
                axes[row, j].set_title(col, fontsize=9)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def slice_info(path):
    """'case_040_week_0_slice_29.nii.gz' -> ('040', 'case_040_week_0', 29).

    Returns (patient, scan, slice number); the slice number is None if the
    filename has no 'slice_<n>' part.
    """
    name = os.path.basename(path).split(".")[0]
    match = re.search(r"slice_?(\d+)", name, flags=re.IGNORECASE)
    scan = name[:match.start()].rstrip("_") if match else name
    return patient_id(path), scan, int(match.group(1)) if match else None


def save_patient_report(scores, files, out_dir):
    """Per-patient SSIM summary (CSV) and SSIM vs slice position for each scan."""
    by_patient, by_scan = defaultdict(list), defaultdict(list)
    for path, score in zip(files, scores.tolist()):
        patient, scan, slice_no = slice_info(path)
        by_patient[patient].append(score)
        if slice_no is not None:
            by_scan[scan].append((slice_no, score))

    summary = []
    for patient, vals in sorted(by_patient.items()):
        v = torch.tensor(vals)
        summary.append({"patient": patient, "slices": len(vals),
                        "mean": v.mean().item(), "std": v.std().item() if len(vals) > 1 else 0.0,
                        "min": v.min().item(), "below_0.6": int((v < 0.6).sum())})
    with open(os.path.join(out_dir, "per_patient_ssim.csv"), "w") as f:
        f.write("patient,slices,mean_ssim,std_ssim,min_ssim,slices_below_0.6\n")
        for row in summary:
            f.write(f"{row['patient']},{row['slices']},{row['mean']:.4f},{row['std']:.4f},"
                    f"{row['min']:.4f},{row['below_0.6']}\n")

    if by_scan:
        fig, ax = plt.subplots(figsize=(7, 3.8))
        for scan, points in sorted(by_scan.items()):
            points.sort()
            ax.plot([p[0] for p in points], [p[1] for p in points], marker=".", markersize=3,
                    linewidth=1, label=scan)
        ax.axhline(0.6, linestyle="--", color="grey", linewidth=1)
        ax.set_xlabel("Slice number within scan")
        ax.set_ylabel("SSIM")
        ax.set_title("Reconstruction SSIM by slice position")
        ax.legend(fontsize=7, loc="lower right")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "ssim_by_slice.png"), dpi=120)
        plt.close(fig)
    return summary


@torch.no_grad()
def save_code_maps(model, x, out_path):
    """For each slice: input | code-index map | reconstruction, one row per slice."""
    recon = model(x)[0].cpu()
    codes = model.encode_indices(x).cpu()
    x = x.cpu()
    n = x.size(0)
    fig, axes = plt.subplots(n, 3, figsize=(6, 2.6 * n))
    if n == 1:
        axes = axes[None]
    titles = ["Input", f"Code map {tuple(codes.shape[1:])}", "Reconstruction"]
    for i in range(n):
        axes[i, 0].imshow(x[i, 0], cmap="gray", vmin=0, vmax=1)
        axes[i, 1].imshow(codes[i], cmap="nipy_spectral", vmin=0, vmax=model.num_embeddings - 1,
                          interpolation="nearest")
        axes[i, 2].imshow(recon[i, 0], cmap="gray", vmin=0, vmax=1)
        for j in range(3):
            axes[i, j].axis("off")
            if i == 0:
                axes[i, j].set_title(titles[j], fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Inference and figures for trained models.")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--vqvae_ckpt", required=True)
    parser.add_argument("--n_down", type=int, default=2)
    parser.add_argument("--num_embeddings", type=int, default=512)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--prior_ckpt", default=None, help="Optional: also sample from the prior")
    parser.add_argument("--prior_channels", type=int, default=128)
    parser.add_argument("--prior_blocks", type=int, default=10)
    parser.add_argument("--n_samples", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Prior sampling temperature (<1 = safer, less varied)")
    parser.add_argument("--n_show", type=int, default=8, help="Test slices to show in figures")
    parser.add_argument("--n_worst", type=int, default=5, help="Lowest-SSIM slices to plot")
    parser.add_argument("--n_best", type=int, default=5, help="Highest-SSIM slices to plot")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--out_dir", default=os.path.join("outputs", "predict"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Device: {device}")

    loaders = get_dataloaders(args.data_root, batch_size=args.batch_size)
    model = load_vqvae(args, device)

    test_set = loaders["test"].dataset
    scores = per_slice_ssim(model, loaders["test"], device)
    mean_ssim = scores.mean().item()
    print(f"Test SSIM: mean {mean_ssim:.4f} | min {scores.min():.4f} | "
          f"below 0.6: {(scores < 0.6).sum().item()} of {len(scores)}")
    order = save_ssim_report(scores, test_set.files, args.out_dir)
    worst = order[:args.n_worst].tolist()
    best = order.flip(0)[:args.n_best].tolist()
    save_ranked_cases(model, test_set, scores, worst, f"{len(worst)} lowest-SSIM test slices",
                      os.path.join(args.out_dir, "worst_cases.png"))
    save_ranked_cases(model, test_set, scores, best, f"{len(best)} highest-SSIM test slices",
                      os.path.join(args.out_dir, "best_cases.png"))

    patients = save_patient_report(scores, test_set.files, args.out_dir)
    print("Per patient:")
    for row in patients:
        print(f"  {row['patient']}: {row['slices']} slices | mean {row['mean']:.4f} | "
              f"min {row['min']:.4f} | below 0.6: {row['below_0.6']}")

    # Fixed, evenly spaced test slices so figures are the same every run.
    idx = torch.linspace(0, len(test_set) - 1, args.n_show).long()
    x = torch.stack([test_set[int(i)] for i in idx]).to(device)
    with torch.no_grad():
        recon = model(x)[0]
    save_image_grid(torch.cat([x, recon]), os.path.join(args.out_dir, "reconstructions.png"),
                    ncols=args.n_show, title="Top: test input | Bottom: VQ-VAE reconstruction")
    save_code_maps(model, x[:4], os.path.join(args.out_dir, "code_maps.png"))

    results = {"vqvae_ckpt": args.vqvae_ckpt, "n_down": args.n_down,
               "test_ssim_mean": mean_ssim, "test_ssim_std": scores.std().item(),
               "test_ssim_min": scores.min().item(),
               "slices_below_0.6": int((scores < 0.6).sum()),
               "worst_slices": [os.path.basename(test_set.files[i]) for i in worst],
               "best_slices": [os.path.basename(test_set.files[i]) for i in best],
               "per_patient": patients}

    if args.prior_ckpt:
        prior = load_prior(args, device)
        grid_shape = tuple(model.encode_indices(x[:1]).shape[1:])
        start = time.time()
        codes = prior.sample(args.n_samples, grid_shape, device, temperature=args.temperature)
        samples = model.decode_indices(codes)
        seconds = time.time() - start
        name = f"prior_samples_T{args.temperature:g}.png"
        save_image_grid(samples, os.path.join(args.out_dir, name),
                        title=f"{args.n_samples} uncurated prior samples (T={args.temperature:g})")
        torch.save(samples.cpu(), os.path.join(args.out_dir, f"samples_T{args.temperature:g}.pt"))
        print(f"Sampled {args.n_samples} slices in {seconds:.1f}s -> {name}")
        results.update({"prior_ckpt": args.prior_ckpt, "temperature": args.temperature,
                        "n_samples": args.n_samples, "seconds_to_sample": round(seconds, 1)})

    with open(os.path.join(args.out_dir, "predict_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Figures saved to {args.out_dir}")


if __name__ == "__main__":
    main()