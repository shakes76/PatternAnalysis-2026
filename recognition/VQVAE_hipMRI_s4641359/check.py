# AI Declaration
#
# Tool: Claude (Anthropic), used conversationally over several requests.
#
# What the AI did in this file:
# - Generated the seven verification checks after I asked how to verify the
#   pipeline and proposed to for the checks, and wrote all of the code in this file.
#
# What I did:
# - Reviewed what each check tests and which README claim it supports.
# - Ran every check on my own machine, interpreted the results, and recorded
#   them in the README (Section 9, "How I verified").
# - Proposed to generate verification checks for the project

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

import torch

from modules import VAE, VQVAE, PixelCNNPrior


def report(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{': ' + detail if detail else ''}")
    return ok


# ----------------------------------------------------------------------
# 1. Tensor shapes
# ----------------------------------------------------------------------

def check_shapes():
    """A 256x128 slice must come out as 256x128, with the expected code grids."""
    print("Checking tensor shapes for a batch of 4 slices of size 1x256x128...\n")
    x = torch.rand(4, 1, 256, 128)
    ok = True

    vae = VAE().eval()
    recon, mu, logvar = vae(x)
    print(f"  VAE: input {tuple(x.shape)} -> latent mean {tuple(mu.shape)} -> output {tuple(recon.shape)}")
    ok &= report("VAE output matches input", recon.shape == x.shape)
    ok &= report("VAE latent is 256-d", mu.shape == (4, 256))

    for n_down, grid in [(2, (64, 32)), (3, (32, 16))]:
        model = VQVAE(n_down=n_down).eval()
        z_e = model.encoder(x)
        recon, _, codes, _ = model(x)
        print(f"\n  VQ-VAE n_down={n_down}: input {tuple(x.shape)} -> encoder {tuple(z_e.shape)} "
              f"-> codes {tuple(codes.shape)} -> output {tuple(recon.shape)}")
        ok &= report(f"VQ-VAE n_down={n_down} output matches input", recon.shape == x.shape)
        ok &= report(f"VQ-VAE n_down={n_down} code grid is {grid}", tuple(codes.shape[1:]) == grid)
        ok &= report("codes are valid indices 0..511", codes.min() >= 0 and codes.max() < 512)
        ok &= report("decoding codes gives a full slice",
                     model.decode_indices(codes).shape == x.shape)

    prior = PixelCNNPrior(512, 32, 4, use_pos=True).eval()
    logits = prior(torch.randint(0, 512, (2, 64, 32)))
    print(f"\n  PixelCNN: codes (2, 64, 32) -> logits {tuple(logits.shape)}")
    ok &= report("prior gives 512 logits per position", logits.shape == (2, 512, 64, 32))
    return ok


# ----------------------------------------------------------------------
# 2. PixelCNN causality
# ----------------------------------------------------------------------

@torch.no_grad()
def check_causal():
    """Changing the code at (i, j) must not change predictions at (i, j) or before it."""
    print("Changing one code at row 20, column 10 and comparing all predictions...\n")
    ok = True
    for use_pos in (False, True):
        torch.manual_seed(0)
        prior = PixelCNNPrior(512, 32, 6, use_pos=use_pos).double().eval()
        codes = torch.randint(0, 512, (2, 64, 32))
        changed = codes.clone()
        i, j = 20, 10
        changed[:, i, j] = (changed[:, i, j] + 5) % 512
        diff = (prior(changed) - prior(codes)).abs().amax(dim=1)  # (B, H, W)
        before = diff[:, :i].max().item()                          # rows above
        same_row_left = diff[:, i, :j + 1].max().item()           # left of / at (i, j)
        after = diff[:, i, j + 1:].max().item()                   # right of it, same row
        label = "with position embedding" if use_pos else "without position embedding"
        print(f"  {label}: max change above = {before:.2e}, at/left = {same_row_left:.2e}, "
              f"right = {after:.2e}")
        ok &= report(f"no information leaks backwards ({label})",
                     before == 0 and same_row_left == 0)
        ok &= report(f"later positions do use it ({label})", after > 0)

    # Receptive field: with n blocks the prior should see 3 + n rows above.
    torch.manual_seed(0)
    n_blocks = 6
    prior = PixelCNNPrior(512, 16, n_blocks).double().eval()
    codes = torch.randint(0, 512, (1, 32, 16))
    base = prior(codes)[0, :, 31, 8]
    rows = [r for r in range(31)
            if (prior(torch.where(torch.arange(32)[:, None] == r, (codes + 7) % 512, codes))
                [0, :, 31, 8] - base).abs().max() > 0]
    seen = 31 - min(rows)
    print(f"\n  Receptive field with {n_blocks} blocks: sees {seen} rows up (theory 3 + {n_blocks} = {3 + n_blocks})")
    ok &= report("receptive field matches 3 + n_blocks", seen == 3 + n_blocks)
    return ok


# ----------------------------------------------------------------------
# 3. Frechet distance
# ----------------------------------------------------------------------

def check_frechet():
    """evaluate.frechet_distance must match SciPy's direct matrix square-root formula."""
    import numpy as np
    from scipy import linalg
    from evaluate import frechet_distance

    torch.manual_seed(0)
    a = torch.randn(400, 8)
    b = torch.randn(400, 8) * 1.5 + 2
    A, B = a.double().numpy(), b.double().numpy()
    ca, cb = np.cov(A, rowvar=False), np.cov(B, rowvar=False)
    reference = ((A.mean(0) - B.mean(0)) ** 2).sum() + np.trace(ca + cb - 2 * linalg.sqrtm(ca @ cb).real)
    ours = frechet_distance(a, b)
    same = frechet_distance(a, torch.randn(400, 8))
    print(f"  ours = {ours:.4f}, SciPy reference = {reference:.4f}, same distribution = {same:.3f}")
    ok = report("matches SciPy", abs(ours - reference) < 1e-6 * max(1, abs(reference)))
    ok &= report("near 0 for samples from the same distribution", same < 0.5)
    return ok


# ----------------------------------------------------------------------
# 4. Smoke test on fake data
# ----------------------------------------------------------------------

def make_fake_dataset(root, per_split=24):
    """Random-noise Nifti slices named and sized like HipMRI (one odd 256x144 slice)."""
    import nibabel as nib
    import numpy as np
    rng = np.random.default_rng(0)
    for split, case in [("train", 4), ("validate", 36), ("test", 40)]:
        folder = os.path.join(root, f"keras_slices_{split}")
        os.makedirs(folder, exist_ok=True)
        for k in range(per_split):
            width = 144 if k == 0 else 128
            img = (rng.random((256, width)) * 600).astype(np.float32)
            nib.save(nib.Nifti1Image(img, np.eye(4)),
                     os.path.join(folder, f"case_{case:03d}_week_0_slice_{k}.nii.gz"))


def run(cmd):
    print("  $ python " + " ".join(cmd))
    result = subprocess.run([sys.executable] + cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout[-2000:], result.stderr[-2000:])
    return result.returncode == 0


def check_smoke():
    """Every script runs end to end on a tiny fake dataset (outputs go to a temp folder)."""
    tmp = tempfile.mkdtemp(prefix="hipmri_smoke_")
    data, out = os.path.join(tmp, "data"), os.path.join(tmp, "out")
    print(f"Fake data and outputs in {tmp} (deleted afterwards)\n")
    try:
        make_fake_dataset(data)
        steps = [
            ("dataset audit", ["dataset.py", "--data_root", data]),
            ("train VAE", ["train.py", "--model", "vae", "--data_root", data, "--epochs", "1",
                           "--batch_size", "8", "--out_dir", f"{out}/vae"]),
            ("train VQ-VAE", ["train.py", "--model", "vqvae", "--data_root", data, "--epochs", "1",
                              "--batch_size", "8", "--n_down", "3", "--out_dir", f"{out}/vqvae"]),
            ("train prior", ["train.py", "--model", "prior", "--data_root", data,
                             "--vqvae_ckpt", f"{out}/vqvae/best.pt", "--epochs", "1",
                             "--batch_size", "8", "--prior_channels", "32", "--prior_blocks", "4",
                             "--prior_pos", "--n_samples", "2", "--out_dir", f"{out}/prior"]),
            ("predict", ["predict.py", "--data_root", data, "--vqvae_ckpt", f"{out}/vqvae/best.pt",
                         "--prior_ckpt", f"{out}/prior/best.pt", "--n_samples", "2",
                         "--out_dir", f"{out}/predict"]),
            ("audit", ["audit.py", "--data_root", data, "--vqvae_ckpt", f"{out}/vqvae/best.pt",
                       "--prior_ckpt", f"{out}/prior/best.pt", "--n_samples", "4",
                       "--n_test", "6", "--out_dir", f"{out}/audit"]),
            ("evaluate", ["evaluate.py", "--data_root", data, "--vqvae_ckpt", f"{out}/vqvae/best.pt",
                          "--prior_ckpt", f"{out}/prior/best.pt", "--vae_ckpt", f"{out}/vae/best.pt",
                          "--n_samples", "10", "--out_dir", f"{out}/evaluate"]),
        ]
        ok = True
        for name, cmd in steps:
            ok &= report(name, run(cmd))
        expected = ["vae/best.pt", "vqvae/best.pt", "prior/best.pt", "prior/prior_samples.png",
                    "predict/per_slice_ssim.csv", "predict/worst_cases.png",
                    "audit/nn_ssim_hist.png", "audit/top3_closest.png",
                    "evaluate/evaluation_table.md", "evaluate/interpolation.png"]
        missing = [f for f in expected if not os.path.exists(os.path.join(out, f))]
        ok &= report("all expected output files created", not missing,
                     "missing: " + ", ".join(missing) if missing else f"{len(expected)} files")
        return ok
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ----------------------------------------------------------------------
# 5. Planted copy (positive control for the memorisation audit)
# ----------------------------------------------------------------------

def check_planted(data_root, n_train=500):
    """Hide real training slices among the queries: the audit search must find them."""
    from audit import nearest_neighbours
    from dataset import HipMRIDataset, list_split_files

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    files = list_split_files(data_root, "train")[:n_train]
    train = HipMRIDataset(files).images
    exact, altered = len(train) // 4, (3 * len(train)) // 4
    torch.manual_seed(0)
    queries = torch.rand(4, 1, 256, 128)
    queries[1] = train[exact]                                   # exact copy
    queries[3] = (train[altered] * 0.9 + 0.05).clamp(0, 1)      # slightly altered copy
    best, idx = nearest_neighbours(queries, train, device, q_chunk=4, t_chunk=128)
    print(f"\n  exact copy   -> found slice {idx[1].item()} (planted {exact}), SSIM {best[1]:.4f}")
    print(f"  altered copy -> found slice {idx[3].item()} (planted {altered}), SSIM {best[3]:.4f}")
    print(f"  random noise -> best SSIM {best[0]:.4f} (should be low)")
    ok = report("exact copy found with SSIM ~1", idx[1].item() == exact and best[1] > 0.999)
    ok &= report("altered copy found", idx[3].item() == altered and best[3] > 0.9)
    ok &= report("noise is not matched", best[0] < 0.3)
    return ok


# ----------------------------------------------------------------------
# 7. SSIM implementation vs an independent library
# ----------------------------------------------------------------------

def check_ssim():
    """utils.ssim must agree with scikit-image's SSIM (same Gaussian window settings).

    Small differences (< 0.02) are expected: scikit-image ignores a border strip,
    while utils.ssim zero-pads the image edges.
    """
    from skimage.metrics import structural_similarity
    from utils import ssim

    torch.manual_seed(0)
    ok = True
    x = torch.rand(1, 1, 256, 128)
    for mix in (0.9, 0.7, 0.4):
        y = x * mix + (1 - mix) * torch.rand(1, 1, 256, 128)
        ours = ssim(x, y).item()
        ref = structural_similarity(x[0, 0].numpy(), y[0, 0].numpy(), data_range=1.0,
                                    gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
        print(f"  similarity level {mix}: ours {ours:.4f} | scikit-image {ref:.4f} | diff {abs(ours - ref):.4f}")
        ok &= report(f"agrees within 0.02 (level {mix})", abs(ours - ref) < 0.02)
    ok &= report("identical images give SSIM 1", abs(ssim(x, x).item() - 1) < 1e-5)
    print("  (differences come from edge handling: scikit-image skips a border strip,\n"
          "   utils.ssim zero-pads; they grow slightly when images are less similar)")
    return ok


# ----------------------------------------------------------------------
# 6. Reproducibility of test SSIM
# ----------------------------------------------------------------------

def check_repro(train_json, predict_json):
    """predict.py's test SSIM must equal what train.py reported for the same checkpoint."""
    with open(train_json) as f:
        trained = json.load(f)["test_ssim"]
    with open(predict_json) as f:
        predicted = json.load(f)["test_ssim_mean"]
    print(f"  train.py   ({train_json}): {trained:.4f}")
    print(f"  predict.py ({predict_json}): {predicted:.4f}")
    return report("test SSIM reproduced", abs(trained - predicted) < 1e-3)


def main():
    parser = argparse.ArgumentParser(description="Verification checks.")
    parser.add_argument("check", choices=["shapes", "causal", "frechet", "smoke", "planted", "repro", "ssim"])
    parser.add_argument("--data_root", help="Needed for 'planted'")
    parser.add_argument("--train_json", default=os.path.join("outputs", "vqvae_final_15ep", "results.json"))
    parser.add_argument("--predict_json", default=os.path.join("outputs", "predict_d2", "predict_results.json"))
    args = parser.parse_args()

    if args.check == "shapes":
        ok = check_shapes()
    elif args.check == "causal":
        ok = check_causal()
    elif args.check == "frechet":
        ok = check_frechet()
    elif args.check == "smoke":
        ok = check_smoke()
    elif args.check == "planted":
        if not args.data_root:
            parser.error("planted needs --data_root")
        ok = check_planted(args.data_root)
    elif args.check == "ssim":
        ok = check_ssim()
    else:
        ok = check_repro(args.train_json, args.predict_json)
    print("\nALL PASSED" if ok else "\nSOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()