import torch
import json
import argparse
from pathlib import Path
from utils import load_generator, nearest_train, invert_ssim
from utils import BEST_CHECKPOINTS
from train import setup_out_dir, CONFIGS
from dataset import load_split
import torchvision.utils as vutils

BASE = Path(__file__).resolve().parent

def get_sample_grids(name, generator, device, nz, out_dir):
    """
    Saves sample grids from three pre-chosen seeds to analyse generator
    output.
    """
    for seed in [1, 2, 3]:
        torch.manual_seed(seed)

        with torch.no_grad():
            samples = generator(torch.randn(32, nz, 1, 1, device=device))

        vutils.save_image(samples, out_dir / f"grid_{name}_seed{seed}.png",
                                  nrow=8, padding=2, normalize=True, value_range=(-1, 1))


def latent_interpolation(generator, device, nz, out_path, steps=10, seed=0):
    """
    Generate images along a straight line between two random latent vectors.
    """
    g = torch.Generator().manual_seed(seed)
    z0, z1 = torch.randn(2, nz, 1, 1, generator=g).to(device)
    alphas = torch.linspace(0, 1, steps, device=device).view(-1, 1, 1, 1)

    with torch.no_grad():
        imgs = generator((1 - alphas) * z0 + alphas * z1)

    vutils.save_image(imgs, out_path, nrow=steps, padding=2, normalize=True, value_range=(-1, 1))

def generate_results(cfg, name):
    """
    Generate results and visualisations for saved models based on checkpoint entry
    in utils.py BEST_CHECKPOINTS.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = setup_out_dir(name)

    train = load_split("train", downsample=cfg["downsample"]).to(device)
    train_flat = train.flatten(1)
    ref = load_split("test", downsample=cfg["downsample"]).to(device)

    # Get a set of real images
    torch.manual_seed(0)
    idx = torch.randperm(ref.size(0))[:32]
    vutils.save_image(ref[idx], out_dir / "grid_real_test.png",
                        nrow=8, padding=2, normalize=True, value_range=(-1, 1))

    netG = load_generator(out_dir / "checkpoints" / BEST_CHECKPOINTS[name],
                              device=device, nz=cfg["nz"], ngf=cfg["ngf"])

    torch.manual_seed(0)
    with torch.no_grad():
        # Want to generate for size of test set
        gen = netG(torch.randn(ref.size(0), cfg["nz"], 1, 1, device=device))
    
    get_sample_grids(name, netG, device, cfg["nz"], out_dir)
    latent_interpolation(netG, device, cfg["nz"], out_dir / f"interpolation_{name}.png")

    # Figure: test slices (top) vs their reconstructions (bottom)
    pick = torch.arange(0, ref.size(0), ref.size(0) // 8)[:8]
    recon, inv_scores = invert_ssim(netG, ref[pick], cfg["nz"])

    print("Inversion SSIM of displayed test slices (left to right): "
          + ", ".join(f"{s:.3f}" for s in inv_scores.tolist()))
    print(f"Mean over these 8: {inv_scores.mean():.3f}")

    vutils.save_image(torch.cat([ref[pick], recon]),
                        out_dir / f"inversion_{name}.png",
                        nrow=8, padding=2, normalize=True, value_range=(-1, 1))

    gen_d, gen_idx = nearest_train(gen, train_flat)
    order = gen_d.argsort()[:8]
    vutils.save_image(torch.cat([gen[order], train[gen_idx[order]]]),
                        out_dir / f"nn_pairs_{name}.png",
                        nrow=8, padding=2, normalize=True, value_range=(-1, 1))

    # Print out results from the previous test
    results_path = out_dir / "test_results.json"

    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f) # read the file back into a Python dict
        print(json.dumps(results, indent=2)) # print it neatly formatted
    else:
        print(f"No test results yet, run: python train.py --run {name} --mode test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate samples and visualisations from" \
    "a trained model.")
    parser.add_argument("--run", required=True, choices=BEST_CHECKPOINTS.keys())

    args = parser.parse_args()
    generate_results(CONFIGS[args.run], args.run)