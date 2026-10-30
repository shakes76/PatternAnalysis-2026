import time
import torch
import yaml
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from baseline.modules import Encoder, Decoder
from baseline.utils import save_generated_samples, plot_training_losses, save_validation_comparison
from dataset import HipMRIDataset
from torchvision.utils import save_image
from metrics import check_gradients
# Run from the project root.

IMAGE_SIZE = (256, 128)
CONFIG_PATH = Path("config.yaml")

# Set random seed for reproducibility.
def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Load configuration from the YAML file.
def load_config():
    with open(CONFIG_PATH, encoding="utf-8") as file:
        raw = yaml.safe_load(file)

    profile = "hipmri_vae" if raw["defaults"]["rangpur"] else "hipmri_vae_cpu"
    return {**raw["defaults"], **raw[profile]}

# Configure the device for training (CPU or GPU).
def configure_device():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        print(torch.cuda.get_device_name(0))

    return device

# Initialize the training dataloader.
def initialize_dataloader(config):
    print("Starting dataset initialisation...", flush=True)
    dataset = HipMRIDataset(
        data_dir = Path(config["data_dir"]),
        max_images = config.get("max_train_images"),
        norm_image = True,
        week = config["week"]
    )

    print("Dataset initialised. Loading first batch...", flush=True)
    train_dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0, # Set to 0 while using the CPU.
        pin_memory=torch.cuda.is_available(),
    )

    images = next(iter(train_dataloader))
    print("First batch loaded.", flush=True)
    print("Number of 2D images:", len(dataset))
    print("Batch shape:", images.shape)
    print("Image range:", images.min().item(), images.max().item())
    assert images.ndim == 4
    assert images.shape[1:] == (1, 256, 128), (
        f"Expected [B, 1, 256, 128], got {images.shape}"
    )

    return train_dataloader

# Initialize the validation dataloader.
def initialize_validation_dataloader(config):
    print("Starting dataset initialisation...", flush=True)
    dataset = HipMRIDataset(
        data_dir = Path(config["validate_dir"]),
        max_images = None,
        norm_image = True,
        week = config["week"]
    )

    print("Dataset initialised. Loading first batch...", flush=True)
    validation_dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0, # Set to 0 while using the CPU.
        pin_memory=torch.cuda.is_available(),
    )

    images = next(iter(validation_dataloader))
    print("First batch loaded.", flush=True)
    print("Number of 2D images:", len(dataset))
    print("Batch shape:", images.shape)
    print("Image range:", images.min().item(), images.max().item())
    assert images.ndim == 4
    assert images.shape[1:] == (1, 256, 128), (
        f"Expected [B, 1, 256, 128], got {images.shape}"
    )

    return validation_dataloader

# Initialize the VAE model for training.
def initialize_training_model(config, device):
    """Initialize the generator and discriminator models for training."""
    latent_dim = config["latent_dim"]
    encoder = Encoder(
        latent_dim=latent_dim,
        channels=1,
        image_size=(256, 128),
    ).to(device)

    decoder = Decoder(
        latent_dim=latent_dim,
        channels=1,
        image_size=(256, 128),
    ).to(device)

    with torch.no_grad():
        z = torch.randn(2, latent_dim, device=device)
        fake = decoder(z)
        mu, logvar = encoder(fake)

        print("Decoder output:", fake.shape)
        print("Encoder mean:", mu.shape)
        print("Encoder logvar:", logvar.shape)


    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=config["learning_rate"],
        betas=tuple(config["betas"]),
    )

    return encoder, decoder, optimizer


def train_one_epoch(
    train_dataloader: DataLoader,
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    beta: float,
):
    """Train the VAE for one epoch.

    Args:
        train_dataloader (DataLoader): DataLoader for the training dataset.
        encoder (torch.nn.Module): The encoder model of the VAE.
        decoder (torch.nn.Module): The decoder model of the VAE.
        optimizer (torch.optim.Optimizer): Optimizer for the VAE parameters.
        device (torch.device): Device to run the training on.
        epoch (int): Current epoch number.
        beta (float): Weight for the KL divergence term.

    Returns:
        tuple: (total_loss, total_reconstruction, total_kl)
    """
    encoder.train()
    decoder.train()
    total_loss = 0.0
    total_reconstruction = 0.0
    total_kl = 0.0
    epoch_grad_encoder = 0.0
    epoch_grad_decoder = 0.0
    num_batches = 0

    for batch_idx, real_images in enumerate(train_dataloader):
        # Ensure real_images is a tensor, not a tuple or list.
        if isinstance(real_images, (tuple, list)):
            real_images = real_images[0]
        # Move real_images to the correct device and type.
        real_images = real_images.to(
            device=device,
            dtype=torch.float32,
        )
        
        # Set gradients to zero.
        optimizer.zero_grad(set_to_none=True)

        # Encode into a latent distribution.
        mu, logvar = encoder(real_images)
        if not torch.isfinite(mu).all() or not torch.isfinite(logvar).all():
            raise FloatingPointError(
                f"Non-finite latent parameters at epoch {epoch}, batch {batch_idx}"
            )

        # Reparameterization trick: sample latent vector z from the distribution defined by mu and logvar.
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std

        # Reconstruct the input image.
        reconstructed = decoder(z)
        
        # Check if the reconstructed image contains finite values.
        if not torch.isfinite(reconstructed).all():
            raise FloatingPointError(
                f"Non-finite reconstruction at epoch {epoch}, batch {batch_idx}"
            )
        # Reconstruction loss; this is a function of mse and SSIM losses.
        reconstruction_loss = F.mse_loss(
            reconstructed,
            real_images,
            reduction="sum",
        ) / real_images.size(0)

        # KL divergence against N(0, I).
        kl_loss = -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp()
        ) / real_images.size(0)

        # Combined VAE objective.
        loss = reconstruction_loss + beta * kl_loss
        # Check if loss is finite before back-propagation.
        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite loss encountered at batch {batch_idx}.")

        # Backpropagate the loss.
        loss.backward()
        grad_norm_encoder = check_gradients(encoder, "encoder", epoch, batch_idx)
        grad_norm_decoder = check_gradients(decoder, "decoder", epoch, batch_idx)
        optimizer.step()

        total_loss += loss.item()
        total_reconstruction += reconstruction_loss.item()
        total_kl += kl_loss.item()
        epoch_grad_encoder += grad_norm_encoder
        epoch_grad_decoder += grad_norm_decoder
        num_batches += 1

    if num_batches == 0:
        raise ValueError("Training dataloader contains no batches.")

    return (
        total_loss / num_batches,
        total_reconstruction / num_batches,
        total_kl / num_batches,
        epoch_grad_encoder / num_batches,
        epoch_grad_decoder / num_batches,
    )

def validate_one_epoch(
    validate_dataloader: DataLoader,
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    device: torch.device,
    beta: float ,
):
    encoder.eval()
    decoder.eval()

    total_loss = 0.0
    total_reconstruction = 0.0
    total_kl = 0.0
    num_batches = 0

    with torch.no_grad():
        for real_images in validate_dataloader:
            if isinstance(real_images, (tuple, list)):
                real_images = real_images[0]

            real_images = real_images.to(
                device=device,
                dtype=torch.float32,
            )

            mu, logvar = encoder(real_images)

            std = torch.exp(0.5 * logvar)
            z = mu + torch.randn_like(std) * std

            reconstructed = decoder(z)

            reconstruction_loss = F.mse_loss(
                reconstructed,
                real_images,
                reduction="sum",
            ) / real_images.size(0)

            kl_loss = -0.5 * torch.sum(
                1 + logvar - mu.pow(2) - logvar.exp()
            ) / real_images.size(0)

            loss = reconstruction_loss + beta * kl_loss

            total_loss += loss.item()
            total_reconstruction += reconstruction_loss.item()
            total_kl += kl_loss.item()
            num_batches += 1

    if num_batches == 0:
        raise ValueError("Validation dataloader contains no batches.")

    return (
        total_loss / num_batches,
        total_reconstruction / num_batches,
        total_kl / num_batches,
    )
    

def save_checkpoint(
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    latent_dim: int,
    results_dir: Path,
):
    # Save the current state of the VAE model and optimizer as a checkpoint.
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "latent_dim": latent_dim,
        },
        results_dir / "checkpoint.pth",
    )

def train_loop(
    config,
):
    # Setup training components: dataloaders, device, models, and optimizers.
    latent_dim = config["latent_dim"]
    train_dataloader = initialize_dataloader(config)
    device = configure_device()
    if config["validate"]:
        validate_dataloader = initialize_validation_dataloader(config)

    encoder, decoder, optimizer = (
        initialize_training_model(config, device)
    )

    num_epochs = config["num_epochs"]

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

    print("Training started!")
    train_start = time.perf_counter()

    # Main training loop.
    total_losses = []
    reconstruction_losses = []
    kl_losses = []
    grad_encoder_losses = []
    grad_decoder_losses = []
    epoch_times = []
    beta = config.get("beta", 1.0)

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.perf_counter()
        mean_loss, mean_reconstruction, mean_kl, mean_grad_encoder, mean_grad_decoder = \
        train_one_epoch(
            train_dataloader,
            encoder,
            decoder,
            optimizer,
            device,
            epoch=epoch,
            beta = beta
        )

        total_losses.append(mean_loss)
        reconstruction_losses.append(mean_reconstruction)
        kl_losses.append(mean_kl)
        grad_encoder_losses.append(mean_grad_encoder)
        grad_decoder_losses.append(mean_grad_decoder)
        epoch_times.append(time.perf_counter() - epoch_start)

        # Save generated samples for every [sample_interval] epochs.
        if epoch % config["sample_interval"] == 0 or epoch == 1:
            save_generated_samples(
                decoder,
                epoch,
                latent_dim,
                device,
                Path(config["results_dir"]),
                64,
            )

        if config["validate"] and epoch % config["validate_every"] == 0:
            val_loss, val_reconstruction, val_kl = validate_one_epoch(
                validate_dataloader,  # type: ignore
                encoder,
                decoder,
                device,
                beta=beta,
            )

            save_validation_comparison(
                encoder,
                decoder,
                validate_dataloader,  # type: ignore
                epoch,
                device,
                Path(config["results_dir"]),
                count=8,
            )

            validation_message = (
                f"Validation - Epoch [{epoch}/{num_epochs}] "
                f"Loss: {val_loss:.4f} "
                f"Reconstruction: {val_reconstruction:.4f} "
                f"KL: {val_kl:.4f}"
            )
            print(validation_message)
            
        message = (
            f"Epoch [{epoch}/{num_epochs}] "
            f"Loss: {mean_loss:.4f} "
            f"Reconstruction: {mean_reconstruction:.4f} "
            f"KL: {mean_kl:.4f} "
            f"Grad Encoder: {grad_encoder_losses[-1]:.4f} "
            f"Grad Decoder: {grad_decoder_losses[-1]:.4f} "
            f"Epoch Time: {epoch_times[-1]:.2f} s"
        )
        # Log peak memory
        if device.type == "cuda":
            allocated = torch.cuda.max_memory_allocated(device) / 1024**2
            reserved = torch.cuda.max_memory_reserved(device) / 1024**2
            message += (
                f"Peak Allocated: {allocated:.1f} MB, "
                f"Peak Reserved: {reserved:.1f} MB"
            )
        print(message)
    # Final statistics.
    print("Training finished!")
    print(f"Total Losses: {total_losses}")
    print(f"Reconstruction Losses: {reconstruction_losses}")
    print(f"KL Losses: {kl_losses}")
    print(f"Grad Encoder Losses: {grad_encoder_losses}")
    print(f"Grad Decoder Losses: {grad_decoder_losses}")
    if device.type == "cuda":
        torch.cuda.synchronize()
        peak_memory = torch.cuda.max_memory_allocated(device)
        print(f"Peak GPU memory: {peak_memory / 1024**3:.2f} GB")

    elapsed = time.perf_counter() - train_start
    print(f"Training time: {elapsed:.2f} seconds")

    # Save final checkpoint as a pth file.
    save_checkpoint(
        encoder,
        decoder,
        optimizer,
        num_epochs,
        latent_dim,
        Path(config["results_dir"]),
    )

    return encoder, decoder, total_losses, reconstruction_losses, kl_losses, grad_encoder_losses, grad_decoder_losses

def create_plot(decoder, results_dir, total_losses, reconstruction_losses, kl_losses, grad_encoder_losses, grad_decoder_losses, latent_dim, device):
    # Generate final samples.
    decoder.eval()

    with torch.no_grad():
        noise = torch.randn(64, latent_dim, device=device)
        generated = decoder(noise).cpu()

    print("Generated shape:", generated.shape)
    print("Generated pixel std:", generated.std().item())
    print(
        "Difference between first two images:",
        (generated[0] - generated[1]).abs().mean().item()
    )
    print(
        "Average variation across images:",
        generated.std(dim=0).mean().item()
    )

    # Convert [-1, 1] to [0, 1] for saving.
    generated = ((generated + 1) / 2).clamp(0, 1)

    save_image(
        generated,
        results_dir / "generated_grid.png",
        nrow=4,
    )

    # Plot generated images.
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))

    for i, ax in enumerate(axes.flat):
        ax.imshow(generated[i, 0], cmap="gray", vmin=0, vmax=1)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(results_dir / "generated_samples.png", dpi=150)
    plt.close(fig)

    plot_training_losses(
        total_losses,
        reconstruction_losses,
        kl_losses,
        grad_encoder_losses,
        grad_decoder_losses,
        results_dir,
    )
        
def main():
    config = load_config()
    set_seed(config.get("seed"))
    results_dir = Path(config["results_dir"])
    device = configure_device()
    results_dir.mkdir(parents=True, exist_ok=True)
    encoder, decoder, total_losses, reconstruction_losses, kl_losses, grad_encoder_losses, grad_decoder_losses = \
        train_loop(config)
    create_plot(
        decoder, 
        results_dir, 
        total_losses, 
        reconstruction_losses, 
        kl_losses, 
        grad_encoder_losses,
        grad_decoder_losses,
        config["latent_dim"],
        device
    )
    return encoder, decoder, total_losses, reconstruction_losses, kl_losses, grad_encoder_losses, grad_decoder_losses

if __name__ == "__main__":
    main()