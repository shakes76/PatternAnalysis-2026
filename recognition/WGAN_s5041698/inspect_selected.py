#!/usr/bin/env python3
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from predict import load_generator
from modules import Critic
from dataset import HipMRIDataset



def load_latent_vectors(z_path='./plots/WGAN_synth/latent_vectors.pt'):
    """Load latent vectors from saved PyTorch file."""
    if z_path and os.path.exists(z_path):
        z = torch.load(z_path)
        print(f"[+] Loaded latent vectors from {z_path}: shape {z.shape}")
        return z
    else:
        raise FileNotFoundError(f"Latent vector file not found: {z_path}")


def perturb_latent_space(generator, z_base, num_perturbations=5, noise_scale=0.5, device='cuda'):
    """Add small random noise to a latent vector to check for mode collapse."""
    generator.eval()
    perturbed_images = []
    
    with torch.no_grad():
        # Original image
        if hasattr(generator, "decode"):
            orig_img = generator.decode(z_base.unsqueeze(0).to(device))
        else:
            orig_img = generator(z_base.unsqueeze(0).to(device))
        orig_img = orig_img * 0.5 + 0.5
        perturbed_images.append(orig_img)
        
        # Perturbed samples
        for i in range(num_perturbations):
            noise = torch.randn_like(z_base) * noise_scale
            z_perturbed = z_base + noise
            if hasattr(generator, "decode"):
                pert_img = generator.decode(z_perturbed.unsqueeze(0).to(device))
            else:
                pert_img = generator(z_perturbed.unsqueeze(0).to(device))
            pert_img = pert_img * 0.5 + 0.5
            perturbed_images.append(pert_img)
    
    return torch.cat(perturbed_images, dim=0)


def check_mode_collapse(generator, z_selected, valid_indices, num_perturbations=5,
                        noise_scales=[0.1, 0.3, 0.5], device='cuda',
                        output_dir='./plots/WGAN_synth/selected_inspection/mode_collapse_check'):
    """For each selected index, add small random noise to latent vectors and plot."""
    os.makedirs(output_dir, exist_ok=True)
    generator.eval()
    
    for idx, img_idx in enumerate(valid_indices):
        z_base = z_selected[idx]
        print(f"[*] Checking mode collapse for index {img_idx}...")
        
        fig, axes = plt.subplots(len(noise_scales), num_perturbations + 1,
                                 figsize=((num_perturbations + 1) * 3, len(noise_scales) * 3))
        if len(noise_scales) == 1:
            axes = axes.reshape(1, -1)
        
        for scale_idx, noise_scale in enumerate(noise_scales):
            perturbed_batch = perturb_latent_space(generator, z_base,
                                                   num_perturbations=num_perturbations,
                                                   noise_scale=noise_scale,
                                                   device=device)
            for p_idx in range(num_perturbations + 1):
                ax = axes[scale_idx, p_idx]
                img = perturbed_batch[p_idx].cpu().squeeze(0).numpy()
                ax.imshow(img, cmap='gray', vmin=0, vmax=1)
                ax.axis('off')
                if p_idx == 0:
                    ax.set_title(f'Noise={noise_scale}\nOriginal', fontsize=7)
                elif scale_idx == 0:
                    ax.set_title(f'Perturb {p_idx}', fontsize=8)
        
        plt.suptitle(f'Mode Collapse Check - Index {img_idx:03d} (Latent Perturbations)',
                     fontsize=12, y=0.98)
        plt.tight_layout()
        out_path = os.path.join(output_dir, f'mode_collapse_index_{img_idx:03d}.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[+] Saved {out_path}")
    
    # Summary
    fig, axes = plt.subplots(len(valid_indices), num_perturbations + 1,
                             figsize=((num_perturbations + 1) * 3, len(valid_indices) * 3))
    if len(valid_indices) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, img_idx in enumerate(valid_indices):
        z_base = z_selected[idx]
        perturbed_batch = perturb_latent_space(generator, z_base,
                                               num_perturbations=num_perturbations,
                                               noise_scale=0.3,
                                               device=device)
        for p_idx in range(num_perturbations + 1):
            ax = axes[idx, p_idx]
            img = perturbed_batch[p_idx].cpu().squeeze(0).numpy()
            ax.imshow(img, cmap='gray', vmin=0, vmax=1)
            ax.axis('off')
            if p_idx == 0:
                ax.set_title(f'Idx {img_idx:03d}\nOriginal', fontsize=7)
            elif idx == 0:
                ax.set_title(f'Perturb {p_idx}', fontsize=8)
    
    plt.suptitle('Mode Collapse Summary (noise_scale=0.3)', fontsize=12, y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mode_collapse_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved summary to {output_dir}/mode_collapse_summary.png")


def generate_selected_images(selected_indices, generator, device, z=None,
                             run_mode_collapse_check=True, num_perturbations=5,
                             noise_scales=[0.1, 0.3, 0.5]):
    """Generate images from selected indices with optional mode collapse check."""
    print(f"Using device: {device}")
    
    if z is None:
        z = load_latent_vectors(z_path='./plots/WGAN_synth/latent_vectors.pt')
    
    valid_indices = [i for i in selected_indices if 0 <= i < z.shape[0]]
    if len(valid_indices) != len(selected_indices):
        print(f"[!] Some indices out of range. Using valid indices: {valid_indices}")
    
    if not valid_indices:
        print("[!] No valid indices selected.")
        return
    
    print(f"[*] Generating images for indices: {valid_indices}")
    
    z_selected = z[valid_indices].to(device)
    
    generator.eval()
    with torch.no_grad():
        if hasattr(generator, "decode"):
            fake_imgs = generator.decode(z_selected)
        else:
            fake_imgs = generator(z_selected)
        fake_imgs = fake_imgs * 0.5 + 0.5
    
    os.makedirs('./plots/WGAN_synth/selected_inspection', exist_ok=True)
    
    grid = make_grid(fake_imgs, nrow=min(4, len(fake_imgs)), normalize=False)
    grid_np = grid.permute(1, 2, 0).cpu().numpy()
    plt.figure(figsize=(12, 12))
    plt.imshow(grid_np, cmap='gray' if grid_np.shape[2] == 1 else None)
    plt.axis('off')
    plt.title(f'Selected Indices: {valid_indices}')
    plt.savefig('./plots/WGAN_synth/selected_inspection/selected_grid.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved overview grid to ./plots/WGAN_synth/selected_inspection/selected_grid.png")
    
    if run_mode_collapse_check:
        print("\n[*] Running mode collapse check...")
        check_mode_collapse(generator, z_selected, valid_indices,
                            num_perturbations=num_perturbations,
                            noise_scales=noise_scales,
                            device=device)
        print("[+] Mode collapse check complete!")


def check_critic_functionality(selected_indices, generator, critic, device,
                               real_data_dir="/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_validate"):
    """Compare critic outputs on generated images against real slices."""
    generator.eval()
    critic.eval()
    
    z = load_latent_vectors(z_path='./plots/WGAN_synth/latent_vectors.pt')

    valid_indices = [i for i in selected_indices if 0 <= i < z.shape[0]]
    if len(valid_indices) != len(selected_indices):
        print(f"[!] Some indices out of range. Using valid indices: {valid_indices}")
    
    if not valid_indices:
        print("[!] No valid indices selected.")
        return

    with torch.no_grad():
        if hasattr(generator, "decode"):
            fake_imgs = generator.decode(z)
        else:
            fake_imgs = generator(z)

        critic_outputs = critic(fake_imgs)
        critic_outputs_selected = critic_outputs[valid_indices]
        print(f"[*] Critic outputs for bad looking generated images:\n{critic_outputs_selected.cpu().numpy()}")
        print(f"[*] Generated: mean={critic_outputs.mean().item():.4f}, "
              f"std={critic_outputs.std().item():.4f} (n={critic_outputs.shape[0]})")

        if not os.path.isdir(real_data_dir):
            print(f"[!] Real data dir not found: {real_data_dir}")
            print("[!] Skipping real-image comparison.")
            return

        real_dataset = HipMRIDataset(data_dir=real_data_dir)
        real_loader = DataLoader(real_dataset, batch_size=min(64, len(real_dataset)), shuffle=True)
        real_imgs = next(iter(real_loader)).to(device)
        real_outputs = critic(real_imgs)

        print(f"[*] Real: mean={real_outputs.mean().item():.4f}, "
              f"std={real_outputs.std().item():.4f} (n={real_outputs.shape[0]})")
        wasserstein_estimate = real_outputs.mean().item() - critic_outputs.mean().item()
        print(f"[*] Wasserstein estimate (real - fake): {wasserstein_estimate:.4f}")


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_file = './checkpoints/wgan_gp_epoch_49.pth'
    generator = load_generator(checkpoint_file, latent_dim=128, device=device, type="WGAN-GP")
    
    selected_indices = [8, 10, 14, 17, 18]
    
    generate_selected_images(selected_indices=selected_indices, generator=generator, device=device)

    critic = Critic(img_channels=1, base_filters=64).to(device)
    critic.load_state_dict(torch.load(checkpoint_file, map_location=device)['critic_state_dict'])
    critic.eval()

    check_critic_functionality(selected_indices, generator, critic, device=device)
