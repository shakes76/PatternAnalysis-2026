import os
import csv
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from dataset import HipMRIDataset
from modules import WGANGP

def save_epoch_samples(model, fixed_z, epoch, sample_dir):
    """
    Generates and saves a grid of sample images using fixed noise at the end of an epoch.
    """
    model.generator.eval()
    with torch.no_grad():
        gen_samples = model.generator(fixed_z)
        # Rescale from [-1, 1] to [0, 1] for image saving
        gen_samples = gen_samples * 0.5 + 0.5
        filename = os.path.join(sample_dir, f"epoch_{epoch + 1:03d}.png")
        vutils.save_image(
            gen_samples.cpu(), 
            filename, 
            nrow=4, 
            normalize=False
        )
    model.generator.train()

def train_wgan_gp(
    train_data_dir,
    val_data_dir,
    epochs=50,
    batch_size=64,
    latent_dim=128,
    lr=1e-4,
    b1=0.0,
    b2=0.9,
    n_critic=5,
    lambda_gp=10.0,
    checkpoint_dir="./checkpoints",
    sample_dir="./samples",
    logs_dir="./logs"
):
    """
    Training loop for WGAN-GP on medical imaging slices with GPU VRAM monitoring.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    metrics_file = os.path.join(logs_dir, "training_metrics.csv")
    with open(metrics_file, "w", newline="") as csv_file:
        csv.writer(csv_file).writerow([
            "epoch", "train_critic_loss", "train_gen_loss", 
            "val_critic_loss", "val_gen_loss", "peak_vram_gb"
        ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on device: {device}")

    # 1. Initialize Dataset & DataLoader
    train_dataset = HipMRIDataset(data_dir=train_data_dir)
    val_dataset = HipMRIDataset(data_dir=val_data_dir)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)
    print(f"[*] Loaded dataset with {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")

    # 2. Initialize Model
    model = WGANGP(latent_dim=latent_dim, img_channels=1, feature_map_size=64).to(device)

    # WGAN-GP uses Adam with beta1=0.0 and beta2=0.9
    optimizer_G = torch.optim.Adam(model.generator.parameters(), lr=lr, betas=(b1, b2))
    optimizer_C = torch.optim.Adam(model.critic.parameters(), lr=lr, betas=(b1, b2))

    # Fixed noise vector for tracking generation progress across epochs
    fixed_z = torch.randn(16, latent_dim, device=device)

    # 3. Training Loop
    total_steps = 0
    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Reset CUDA peak memory tracking for this epoch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)

        epoch_critic_loss = 0.0
        epoch_generator_loss = 0.0
        epoch_gradient_penalty = 0.0
        epoch_batches = 0
        
        for i, real_imgs in enumerate(train_dataloader):
            # Handle dataset outputs if they return tuples (e.g., image + label)
            if isinstance(real_imgs, (tuple, list)):
                real_imgs = real_imgs[0]
                
            real_imgs = real_imgs.to(device)
            batch_size_curr = real_imgs.size(0)

            # ---------------------
            #  Train Critic
            # ---------------------
            for _ in range(n_critic):
                optimizer_C.zero_grad()

                # Sample random latent vectors
                z = torch.randn(batch_size_curr, latent_dim, device=device)

                # Generate fake medical slices
                fake_imgs = model.generator(z)

                # Compute critic outputs
                real_validity = model.critic(real_imgs)
                fake_validity = model.critic(fake_imgs.detach())

                # Compute gradient penalty
                gradient_penalty = model.compute_gradient_penalty(real_imgs, fake_imgs.detach(), device)

                # WGAN-GP loss: Wasserstein loss + gradient penalty
                loss_critic = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

                loss_critic.backward()
                optimizer_C.step()

            # ---------------------
            #  Train Generator
            # ---------------------
            optimizer_G.zero_grad()

            # Generate a batch of images
            z = torch.randn(batch_size_curr, latent_dim, device=device)
            fake_imgs = model.generator(z)

            # Generator loss tries to maximize critic score on fake images
            loss_gen = -torch.mean(model.critic(fake_imgs))
            loss_gen.backward()
            optimizer_G.step()

            total_steps += 1
            epoch_critic_loss += loss_critic.item()
            epoch_generator_loss += loss_gen.item()
            epoch_gradient_penalty += gradient_penalty.item()
            epoch_batches += 1

        avg_train_C = epoch_critic_loss / epoch_batches
        avg_train_G = epoch_generator_loss / epoch_batches
        avg_GP = epoch_gradient_penalty / epoch_batches

        # Calculate validation losses
        model.eval()
        val_critic_loss = 0.0
        val_generator_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for val_imgs in val_dataloader:
                if isinstance(val_imgs, (tuple, list)):
                    val_imgs = val_imgs[0]

                val_imgs = val_imgs.to(device)
                val_batch_size = val_imgs.size(0)

                z_val = torch.randn(val_batch_size, latent_dim, device=device)
                fake_val_imgs = model.generator(z_val)

                real_validity = model.critic(val_imgs)
                fake_validity = model.critic(fake_val_imgs)

                # Standard Wasserstein evaluation loss (without gradient penalty)
                val_critic_loss += (-torch.mean(real_validity) + torch.mean(fake_validity)).item()
                val_generator_loss += (-torch.mean(fake_validity)).item()
                val_batches += 1

        avg_val_C = val_critic_loss / val_batches if val_batches > 0 else 0.0
        avg_val_G = val_generator_loss / val_batches if val_batches > 0 else 0.0

        # Capture Peak GPU VRAM Usage for this Epoch
        peak_vram_gb = 0.0
        if torch.cuda.is_available():
            peak_vram_bytes = torch.cuda.max_memory_allocated(device)
            peak_vram_gb = peak_vram_bytes / (1024 ** 3)  # Convert bytes to GB

        epoch_duration = time.time() - epoch_start
        print(
            f"[Epoch {epoch+1}/{epochs}] "
            f"[Train C: {avg_train_C:.4f}] [Train G: {avg_train_G:.4f}] [GP: {avg_GP:.4f}] "
            f"[Val C: {avg_val_C:.4f}] [Val G: {avg_val_G:.4f}] "
            f"[Peak VRAM: {peak_vram_gb:.2f} GB] "
            f"({epoch_duration:.2f}s)"
        )

        # Log metrics to CSV
        with open(metrics_file, "a", newline="") as csv_file:
            csv.writer(csv_file).writerow([
                epoch + 1, avg_train_C, avg_train_G, 
                avg_val_C, avg_val_G, f"{peak_vram_gb:.4f}"
            ])

        model.train()

        if epoch >= 20:
            # Save checkpoint and sample grid
            torch.save({
                'epoch': epoch + 1,
                'generator_state_dict': model.generator.state_dict(),
                'critic_state_dict': model.critic.state_dict(),
            }, os.path.join(checkpoint_dir, f"wgan_gp_epoch_{epoch+1}.pth"))

        save_epoch_samples(model, fixed_z, epoch, sample_dir)

    print(f"[*] Training complete. Total time: {(time.time() - start_time) / 3600:.2f} hours.")


if __name__ == "__main__":
    train_data_dir = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_train"
    val_data_dir = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_validate"
    train_wgan_gp(train_data_dir=train_data_dir, val_data_dir=val_data_dir, epochs=50)
