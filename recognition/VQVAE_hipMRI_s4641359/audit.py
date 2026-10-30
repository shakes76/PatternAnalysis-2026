# AI Declaration
#
# Tool: Claude (Anthropic), used conversationally over several requests.
#
# What the AI did in this file:
# - Wrote this memorisation audit to my specification (sample count,
#   SSIM nearest-neighbour search over the whole training set, nearest
#   patient ID, latent code overlap, a real test-slice baseline, the
#   similarity histogram, and the top-3 closest cases with difference maps).
# - Added a reconstructed-test baseline and the maximum code overlap over
#   all training slices, so the comparison is not biased by decoder
#   smoothing, and a plot of which training patients are matched.
#
# What I did:
# - Specified the audit steps listed above and chose the comparisons.
# - Ran it on Colab, inspected the closest sample (83) against its
#   training neighbour by eye, and analysed the patient distribution
#   (checking slice counts before attributing the skew to the model).
# - 


import argparse
import json
import os
import time
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from dataset import get_dataloaders, patient_id
from predict import load_prior, load_vqvae
from utils import save_image_grid

GROUPS = ["generated", "test_recon", "test_real"]
LABELS = {"generated": "Generated -> Train",
          "test_recon": "Test (VQ-VAE reconstructed) -> Train",
          "test_real": "Test (raw) -> Train"}


# ----------------------------------------------------------------------
# Pairwise SSIM. Same formula and Gaussian window (11x11, sigma 1.5, zero
# padding) as utils.ssim, applied as two 1D passes for speed so every query
# can be compared with all 11,460 training slices.
# ----------------------------------------------------------------------

def _gauss_1d(size=11, sigma=1.5, device="cpu"):
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def blur(x, g):
    """Separable Gaussian blur of (N, 1, H, W) images with zero padding."""
    k = g.numel()
    x = F.conv2d(x, g.view(1, 1, 1, k), padding=(0, k // 2))
    return F.conv2d(x, g.view(1, 1, k, 1), padding=(k // 2, 0))


def local_stats(x, g):
    """Local mean and variance maps used by SSIM."""
    mu = blur(x, g)
    return mu, blur(x * x, g) - mu ** 2


def pairwise_ssim(q, q_mu, q_var, t, t_mu, t_var, g, c1=0.01 ** 2, c2=0.03 ** 2):
    """Mean SSIM for every (query, train) pair. q: (Q,1,H,W), t: (B,1,H,W) -> (Q, B)."""
    Q, B, H, W = q.size(0), t.size(0), q.size(2), q.size(3)
    cross = blur((q[:, None] * t[None]).view(Q * B, 1, H, W), g).view(Q, B, H, W)
    mq, vq = q_mu[:, None, 0], q_var[:, None, 0]
    mt, vt = t_mu[None, :, 0], t_var[None, :, 0]
    cov = cross - mq * mt
    ssim_map = ((2 * mq * mt + c1) * (2 * cov + c2)) / ((mq ** 2 + mt ** 2 + c1) * (vq + vt + c2))
    return ssim_map.flatten(2).mean(dim=2)


@torch.no_grad()
def nearest_neighbours(queries, train_images, device, q_chunk=8, t_chunk=128):
    """For each query slice, the max SSIM over all training slices and its index."""
    g = _gauss_1d(device=device)
    n_q = queries.size(0)
    best = torch.full((n_q,), -2.0)
    best_idx = torch.zeros(n_q, dtype=torch.long)
    q_stats = []
    for s in range(0, n_q, q_chunk):
        q = queries[s:s + q_chunk].to(device)
        q_stats.append((q, *local_stats(q, g)))
    n_t = train_images.size(0)
    start = time.time()
    for ts in range(0, n_t, t_chunk):
        t = train_images[ts:ts + t_chunk].to(device, non_blocking=True)
        t_mu, t_var = local_stats(t, g)
        for qi, (q, q_mu, q_var) in enumerate(q_stats):
            scores = pairwise_ssim(q, q_mu, q_var, t, t_mu, t_var, g).cpu()
            top, arg = scores.max(dim=1)
            sl = slice(qi * q_chunk, qi * q_chunk + q.size(0))
            better = top > best[sl]
            best[sl] = torch.where(better, top, best[sl])
            best_idx[sl] = torch.where(better, arg + ts, best_idx[sl])
        done = min(ts + t_chunk, n_t)
        if ts == 0 or done == n_t or (ts // t_chunk) % 20 == 0:
            print(f"  NN search: {done}/{n_t} training slices ({time.time() - start:.0f}s)")
    return best, best_idx


@torch.no_grad()
def encode_all(model, images, device, batch=64):
    """Code grids (N, H', W') for a stack of images, stored compactly."""
    out = [model.encode_indices(images[s:s + batch].to(device)).cpu().to(torch.int16)
           for s in range(0, images.size(0), batch)]
    return torch.cat(out)


@torch.no_grad()
def max_code_overlap(query_codes, train_codes, device, chunk=1024):
    """For each query grid, the highest fraction of identical codes with any training grid."""
    q = query_codes.flatten(1).to(device)
    best = torch.zeros(q.size(0), device=device)
    best_idx = torch.zeros(q.size(0), dtype=torch.long, device=device)
    for s in range(0, train_codes.size(0), chunk):
        t = train_codes[s:s + chunk].flatten(1).to(device)
        overlap = (q[:, None, :] == t[None, :, :]).float().mean(dim=2)
        top, arg = overlap.max(dim=1)
        better = top > best
        best = torch.where(better, top, best)
        best_idx = torch.where(better, arg + s, best_idx)
    return best.cpu(), best_idx.cpu()


def describe(x):
    """Summary statistics of a 1D tensor."""
    return {"mean": x.mean().item(), "std": x.std().item(), "median": x.median().item(),
            "min": x.min().item(), "max": x.max().item(),
            "p95": torch.quantile(x, 0.95).item()}


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------

def plot_histogram(results, out_path):
    """Overlaid NN-SSIM histograms for each group."""
    fig, ax = plt.subplots(figsize=(7, 4))
    colours = {"generated": "tab:red", "test_recon": "tab:blue", "test_real": "tab:grey"}
    all_vals = torch.cat([results[g]["nn_ssim"] for g in GROUPS])
    bins = torch.linspace(all_vals.min().item(), all_vals.max().item(), 41).numpy()
    for g in GROUPS:
        vals = results[g]["nn_ssim"]
        ax.hist(vals.numpy(), bins=bins, alpha=0.5, color=colours[g],
                label=f"{LABELS[g]} (mean {vals.mean():.3f})")
    ref = results["test_recon"]["nn_ssim"].max().item()
    ax.axvline(ref, color="tab:blue", linestyle="--", linewidth=1,
               label=f"max of reconstructed-test baseline ({ref:.3f})")
    ax.set_xlabel("SSIM to nearest training slice")
    ax.set_ylabel("Slices")
    ax.set_title("Memorisation audit: nearest-neighbour similarity to the training set")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_top_closest(gen_images, train_set, res, train_codes, gen_codes, k, out_path):
    """Generated | nearest training slice | |difference| for the k closest generated slices."""
    order = torch.argsort(res["nn_ssim"], descending=True)[:k].tolist()
    fig, axes = plt.subplots(k, 3, figsize=(6.5, 2.9 * k))
    if k == 1:
        axes = axes[None]
    for row, i in enumerate(order):
        j = res["nn_idx"][i].item()
        gen, nn = gen_images[i, 0], train_set[j][0]
        diff = (gen - nn).abs()
        overlap = (gen_codes[i] == train_codes[j]).float().mean().item()
        name = os.path.basename(train_set.files[j]).split(".")[0]
        axes[row, 0].imshow(gen, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].imshow(nn, cmap="gray", vmin=0, vmax=1)
        axes[row, 2].imshow(diff, cmap="magma", vmin=0, vmax=1)
        axes[row, 0].set_ylabel(f"sample {i}\nSSIM {res['nn_ssim'][i]:.3f}\n"
                                f"code overlap {overlap:.1%}", fontsize=7)
        axes[row, 1].set_xlabel(name, fontsize=6)
        for c, title in enumerate(["Generated", "Nearest training slice", "|Difference|"]):
            axes[row, c].set_xticks([]); axes[row, c].set_yticks([])
            if row == 0:
                axes[row, c].set_title(title, fontsize=9)
    fig.suptitle(f"{k} generated slices closest to the training set", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_patients(counts, out_path):
    """How often each training patient is the nearest neighbour of a generated slice."""
    items = counts.most_common()
    fig, ax = plt.subplots(figsize=(max(5, 0.35 * len(items)), 3.5))
    ax.bar([p for p, _ in items], [c for _, c in items], color="tab:red")
    ax.set_xlabel("Training patient (case ID)")
    ax.set_ylabel("Generated slices with this NN")
    ax.set_title("Nearest-neighbour patients of generated slices")
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Memorisation audit for prior samples.")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--vqvae_ckpt", required=True)
    parser.add_argument("--prior_ckpt", required=True)
    parser.add_argument("--n_down", type=int, default=2,
                        help="Only needed for old-format VQ-VAE checkpoints")
    parser.add_argument("--num_embeddings", type=int, default=512)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--prior_channels", type=int, default=128)
    parser.add_argument("--prior_blocks", type=int, default=10)
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--n_test", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=3, help="Closest cases to display")
    parser.add_argument("--q_chunk", type=int, default=8,
                        help="Queries per SSIM batch (raise on a big GPU)")
    parser.add_argument("--t_chunk", type=int, default=128,
                        help="Training slices per SSIM batch (raise on a big GPU)")
    parser.add_argument("--out_dir", default=os.path.join("outputs", "audit"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Device: {device}")

    loaders = get_dataloaders(args.data_root, batch_size=64)
    train_set, test_set = loaders["train"].dataset, loaders["test"].dataset
    model = load_vqvae(args, device)
    prior = load_prior(args, device)

    # 1. Generated slices (keep their sampled codes for the overlap measure).
    grid = tuple(model.encode_indices(test_set[0][None].to(device)).shape[1:])
    print(f"Sampling {args.n_samples} slices (code grid {grid}, T={args.temperature})...")
    start = time.time()
    gen_codes = prior.sample(args.n_samples, grid, device, temperature=args.temperature)
    with torch.no_grad():
        gen_images = model.decode_indices(gen_codes).cpu()
    gen_codes = gen_codes.cpu().to(torch.int16)
    print(f"  done in {time.time() - start:.0f}s")
    save_image_grid(gen_images, os.path.join(args.out_dir, "audit_samples.png"),
                    title=f"{args.n_samples} audited samples (T={args.temperature:g})")

    # 4. Baseline: evenly spaced real test slices, raw and reconstructed.
    test_idx = torch.linspace(0, len(test_set) - 1, args.n_test).long().tolist()
    test_real = torch.stack([test_set[i] for i in test_idx])
    with torch.no_grad():
        test_recon = torch.cat([model(test_real[s:s + 32].to(device))[0].cpu()
                                for s in range(0, len(test_real), 32)])
    test_codes = encode_all(model, test_real, device)

    # 3. Code grids for every training slice.
    print("Encoding training slices...")
    train_codes = encode_all(model, train_set.images, device)

    # 2. SSIM nearest neighbour against all training slices, for all groups at once.
    queries = torch.cat([gen_images, test_recon, test_real])
    print(f"SSIM nearest-neighbour search: {len(queries)} queries x {len(train_set)} training slices")
    nn_ssim, nn_idx = nearest_neighbours(queries, train_set.images, device,
                                         args.q_chunk, args.t_chunk)
    query_codes = torch.cat([gen_codes, test_codes, test_codes])
    overlap_any, overlap_any_idx = max_code_overlap(query_codes, train_codes, device)

    results, rows = {}, []
    offsets = {"generated": 0, "test_recon": args.n_samples,
               "test_real": args.n_samples + args.n_test}
    sizes = {"generated": args.n_samples, "test_recon": args.n_test, "test_real": args.n_test}
    for g in GROUPS:
        sl = slice(offsets[g], offsets[g] + sizes[g])
        codes_g = query_codes[sl]
        idx_g = nn_idx[sl]
        overlap_nn = (codes_g == train_codes[idx_g]).flatten(1).float().mean(dim=1)
        results[g] = {"nn_ssim": nn_ssim[sl], "nn_idx": idx_g, "overlap_nn": overlap_nn,
                      "overlap_any": overlap_any[sl]}
        for k in range(sizes[g]):
            j = idx_g[k].item()
            source = (f"sample_{k}" if g == "generated"
                      else os.path.basename(test_set.files[test_idx[k]]))
            rows.append([g, source, f"{nn_ssim[sl][k].item():.4f}",
                         os.path.basename(train_set.files[j]), patient_id(train_set.files[j]),
                         f"{overlap_nn[k].item():.4f}", f"{overlap_any[sl][k].item():.4f}",
                         os.path.basename(train_set.files[overlap_any_idx[sl][k].item()])])

    with open(os.path.join(args.out_dir, "audit_per_slice.csv"), "w") as f:
        f.write("group,slice,nn_ssim,nn_train_file,nn_patient,"
                "code_overlap_with_nn,max_code_overlap_any_train,max_overlap_train_file\n")
        for r in rows:
            f.write(",".join(r) + "\n")

    # 5-6. Figures.
    plot_histogram(results, os.path.join(args.out_dir, "nn_ssim_hist.png"))
    plot_top_closest(gen_images, train_set, results["generated"], train_codes, gen_codes,
                     args.top_k, os.path.join(args.out_dir, "top3_closest.png"))
    gen_patients = Counter(patient_id(train_set.files[j])
                           for j in results["generated"]["nn_idx"].tolist())
    plot_patients(gen_patients, os.path.join(args.out_dir, "nn_patients.png"))

    # Memorisation flags, judged against the fair (reconstructed) baseline.
    gen = results["generated"]
    ref_ssim = results["test_recon"]["nn_ssim"]
    ref_overlap = results["test_recon"]["overlap_any"]
    summary = {
        "n_samples": args.n_samples, "n_test": args.n_test, "temperature": args.temperature,
        "vqvae_ckpt": args.vqvae_ckpt, "prior_ckpt": args.prior_ckpt,
        "nn_ssim": {g: describe(results[g]["nn_ssim"]) for g in GROUPS},
        "code_overlap_with_nn": {g: describe(results[g]["overlap_nn"]) for g in GROUPS},
        "max_code_overlap_any_train": {g: describe(results[g]["overlap_any"]) for g in GROUPS},
        "generated_above_baseline_max_ssim":
            int((gen["nn_ssim"] > ref_ssim.max()).sum()),
        "generated_above_baseline_p95_ssim":
            int((gen["nn_ssim"] > torch.quantile(ref_ssim, 0.95)).sum()),
        "generated_above_baseline_max_overlap":
            int((gen["overlap_any"] > ref_overlap.max()).sum()),
        "generated_nn_patients": dict(gen_patients.most_common()),
        "distinct_nn_patients_generated": len(gen_patients),
    }
    with open(os.path.join(args.out_dir, "audit_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nNearest-neighbour SSIM to training set (mean | max):")
    for g in GROUPS:
        s = summary["nn_ssim"][g]
        o = summary["max_code_overlap_any_train"][g]
        print(f"  {LABELS[g]:<40} {s['mean']:.4f} | {s['max']:.4f}   "
              f"max code overlap: mean {o['mean']:.1%} | max {o['max']:.1%}")
    print(f"Generated slices above the reconstructed-test max SSIM: "
          f"{summary['generated_above_baseline_max_ssim']} of {args.n_samples}")
    print(f"Generated slices above its 95th percentile: "
          f"{summary['generated_above_baseline_p95_ssim']} of {args.n_samples}")
    print(f"Generated slices above its max code overlap: "
          f"{summary['generated_above_baseline_max_overlap']} of {args.n_samples}")
    print(f"Distinct training patients among generated NNs: {len(gen_patients)}")
    print(f"Outputs saved to {args.out_dir}")


if __name__ == "__main__":
    main()