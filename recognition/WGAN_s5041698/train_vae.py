import os
import time
import torch
import torch.optim as optim
from modules import ConvVAE
from dataset import HipMRIDataset


def train_vae(train_dataloader, epochs=100, checkpoint_path="./checkpoints/oasis_vae.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ConvVAE(in_channels=1, latent_dim=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_loss = float("inf")
    best_epoch = -1
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    start_time = time.time()

    print("\n--- Starting VAE Training ---")
    for epoch in range(1, epochs + 1):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)

        model.train()
        train_loss, total_bce, total_kld = 0.0, 0.0, 0.0

        for batch in train_dataloader:
            if isinstance(batch, (tuple, list)):
                batch = batch[0]
            batch = batch.to(device)
            # Convert from [-1,1] to [0,1] for BCE + Sigmoid
            batch_01 = (batch + 1.0) / 2.0
            optimizer.zero_grad()

            recon_batch, mu, logvar = model(batch_01)
            loss, bce, kld = model.vae_loss_function(recon_batch, batch_01, mu, logvar)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            total_bce += bce.item()
            total_kld += kld.item()

        num_batches = len(train_dataloader)
        avg_train_loss = train_loss / num_batches
        avg_train_bce = total_bce / num_batches
        avg_train_kld = total_kld / num_batches

        peak_vram_gb = 0.0
        if torch.cuda.is_available():
            peak_vram_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

        print(f"Epoch {epoch:02d}/{epochs:02d} | "
                f"Loss: {avg_train_loss:.2f} | "
                f"BCE: {avg_train_bce:.2f} | "
                f"KLD: {avg_train_kld:.2f} | "
                f"Peak VRAM: {peak_vram_gb:.2f} GB")

        current_loss = avg_train_loss
        loss_label = "train loss"

        # Save checkpoint whenever a new best loss is achieved
        if current_loss < best_loss:
            best_loss = current_loss
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint_path)
            print(f"[*] Best model saved at epoch {epoch:02d} with {loss_label}: {best_loss:.2f} -> {checkpoint_path}")

    elapsed_time = time.time() - start_time
    elapsed_minutes = elapsed_time / 60
    
    if best_epoch != -1:
        print(f"\n[+] Training complete in {elapsed_minutes:.2f} minutes. Best model saved from epoch {best_epoch:02d} ({loss_label}: {best_loss:.2f}) to {checkpoint_path}")
    else:
        print("\n[!] Training complete. No model checkpoint was saved.")


if __name__ == "__main__":
    train_data_dir = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_train"

    train_dataset = HipMRIDataset(data_dir=train_data_dir)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    train_vae(train_dataloader, epochs=100)