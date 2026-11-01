"""
predict.py
Authored by: Gemini 3.8

Image generation pipeline and class conditioning check for ADNI brain MRI slices.
- python predict.py [n_per_class]: Synthesizes cohorts to 'ldm_out/{NC,AD}/' and 'gan/'
- python predict.py --check [seed]: Generates side-by-side counterfactual plot (class_check.png)
"""

import os
import sys
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt

from modules import LDMVAE, LDMUNet
from gan import Generator

SCALE = 0.82
Z_DIM = 256
LDM_DIR = "ldm_out"
GAN_DIR = "gan_out"

def load_models(dev):
    """Loads trained VAE and UNet models in eval mode."""
    vae = LDMVAE(in_c=1, latent_c=4).to(dev)
    vae.load_state_dict(torch.load("vae_weights.pt", map_location=dev))
    vae.eval()

    unet = LDMUNet(in_c=4, out_c=4, cond=256, timesteps=1000).to(dev)
    unet.load_state_dict(torch.load("unet_weights.pt", map_location=dev))
    unet.eval()
    return vae, unet

@torch.no_grad()
def p_sample(unet, vae, class_val, dev, z_init=None, seed=None):
    """Reverse DDPM sampling for LDM conditioned on class_val (0=NC, 1=AD)."""
    if seed is not None:
        torch.manual_seed(seed)
        if dev.type == "cuda":
            torch.cuda.manual_seed(seed)

    z = z_init.clone() if z_init is not None else torch.randn((1, 4, 32, 32), device=dev)
    c = torch.tensor([class_val], device=dev, dtype=torch.long)

    for step in reversed(range(1000)):
        t = torch.tensor([step], device=dev, dtype=torch.long)
        a_bar_t = 1.0 - (step / 1000.0)
        a_bar_prev = 1.0 - ((step - 1) / 1000.0) if step > 0 else 1.0
        a_t = a_bar_t / a_bar_prev
        beta_t = 1.0 - a_t

        eps = unet(z, t, c)
        mu = (1.0 / (a_t ** 0.5)) * (z - (beta_t / ((1.0 - a_bar_t + 1e-8) ** 0.5)) * eps)
        var = ((1.0 - a_bar_prev) / (1.0 - a_bar_t + 1e-8)) * beta_t if step > 0 else 0.0
        z = mu + (var ** 0.5) * torch.randn_like(z) if step > 0 else mu

    return vae.decode(z / SCALE).clamp(0.0, 1.0).squeeze().cpu().numpy()

@torch.no_grad()
def gan_sample(gen, dev):
    """Samples from DCGAN generator."""
    noise = torch.randn(1, Z_DIM, 1, 1, device=dev)
    return ((gen(noise) + 1.0) / 2.0).clamp(0.0, 1.0).squeeze().cpu().numpy()

def generate_cohort(dev, n_per_class=25):
    """Generates synthetic slices to disk folders: 'ldm_out/{NC,AD}/' and 'gan/'."""
    print(f"Generating LDM ({n_per_class} NC + {n_per_class} AD) to '{LDM_DIR}/'...")
    os.makedirs(f"{LDM_DIR}/NC", exist_ok=True)
    os.makedirs(f"{LDM_DIR}/AD", exist_ok=True)

    vae, unet = load_models(dev)

    for c_idx, c_name in [(0, "NC"), (1, "AD")]:
        for i in range(n_per_class):
            img = p_sample(unet, vae, c_idx, dev)
            Image.fromarray((img * 255).astype(np.uint8)).save(f"{LDM_DIR}/{c_name}/ldm_{i}.png")

    if os.path.exists("generator_weights.pt"):
        print(f"Generating DCGAN ({n_per_class * 2} samples) to '{GAN_DIR}/'...")
        os.makedirs(GAN_DIR, exist_ok=True)
        gen = Generator(z_dim=Z_DIM).to(dev)
        gen.load_state_dict(torch.load("generator_weights.pt", map_location=dev))
        gen.eval()
        for i in range(n_per_class * 2):
            img = gan_sample(gen, dev)
            Image.fromarray((img * 255).astype(np.uint8)).save(f"{GAN_DIR}/gan_{i}.png")

    print("[Done] Generation complete. Saved directly to folders.")

def class_check(dev, seed=42, out_file="class_check.png"):
    """Fixed-seed counterfactual class conditioning comparison (NC vs AD)."""
    vae, unet = load_models(dev)

    torch.manual_seed(seed)
    if dev.type == "cuda":
        torch.cuda.manual_seed(seed)
    z_init = torch.randn((1, 4, 32, 32), device=dev)

    img_nc = p_sample(unet, vae, class_val=0, dev=dev, z_init=z_init, seed=seed)
    img_ad = p_sample(unet, vae, class_val=1, dev=dev, z_init=z_init, seed=seed)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), dpi=300)
    ax1.imshow(img_nc, cmap="gray", vmin=0.0, vmax=1.0)
    ax1.set_title("Normal Control (NC)", fontsize=12, fontweight="bold")
    ax1.axis("off")

    ax2.imshow(img_ad, cmap="gray", vmin=0.0, vmax=1.0)
    ax2.set_title("CAlzheimer's Disease (AD)", fontsize=12, fontweight="bold")
    ax2.axis("off")

    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device :: {device}")

    # Mode 1: Counterfactual class check (e.g. python predict.py --check [seed])
    if "--check" in sys.argv or "--class-check" in sys.argv:
        seed = 42
        for arg in sys.argv[1:]:
            if arg.isdigit():
                seed = int(arg)
                break
        class_check(device, seed=seed)
        return

    # Mode 2: Full cohort generation (e.g. python predict.py [n_per_class])
    n_per_class = 25
    for arg in sys.argv[1:]:
        if arg.isdigit():
            n_per_class = int(arg)
            break

    generate_cohort(device, n_per_class=n_per_class)

if __name__ == "__main__":
    main()
