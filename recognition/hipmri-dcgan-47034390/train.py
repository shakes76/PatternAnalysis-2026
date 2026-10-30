import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pathlib import Path
from modules import Generator, Discriminator
import yaml
from utils import save_generated_samples, plot_training_losses, save_validation_comparison
from dataset import HipMRIDataset
from torchvision.utils import save_image
from metrics import check_gradients

IMAGE_SIZE = (256, 128)
IMAGE_CHANNELS = 1
SAMPLE_COUNT = 64
SAMPLE_INTERVAL = 3
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

    profile = "hipmri_dcgan" if raw["defaults"]["rangpur"] else "hipmri_dcgan_cpu"
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
        data_dir = config["data_dir"],
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
        shuffle=True,
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

# Initialize the GAN model for training.
def initialize_training_model(config, latent_dim, device):
    """Initialize the generator and discriminator models for training.
    Define the generator and discriminator models, and their optimizers.
    """
    G = Generator(
        latent_dim=latent_dim,
        channels=1,
        image_size=(256, 128),
    ).to(device)

    D = Discriminator(
        channels=1,
        image_size=(256, 128),
    ).to(device)

    with torch.no_grad():
        fake = G(torch.randn(2, latent_dim, 1, 1, device=device))
        print("Generator output:", fake.shape)
        print("Discriminator output:", D(fake).shape)

    optimizer_G = torch.optim.Adam(
        G.parameters(),
        lr=config["learning_rate_G"],
        betas=tuple(config["betas"]),
    )

    optimizer_D = torch.optim.Adam(
        D.parameters(),
        lr=config["learning_rate_D"],
        betas=tuple(config["betas"]),
    )

    return G, D, optimizer_G, optimizer_D

def train_one_epoch(
    train_dataloader: DataLoader,
    G: torch.nn.Module,
    D: torch.nn.Module,
    optimizer_G: torch.optim.Optimizer,
    optimizer_D: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    latent_dim: int,
    device: torch.device,
    epoch:int
):
    """Train the GAN for one epoch.

    Args:
        train_dataloader (DataLoader): DataLoader for the training dataset.
        G (torch.nn.Module): Generator model.
        D (torch.nn.Module): Discriminator model.
        optimizer_G (torch.optim.Optimizer): Optimizer for the generator.
        optimizer_D (torch.optim.Optimizer): Optimizer for the discriminator.
        criterion (torch.nn.Module): Loss function for GAN training.
        latent_dim (int): Dimensionality of the latent space.
        device (torch.device): Device to run the training on.

    Raises:
        ValueError: If the training dataloader contains no batches.

    Returns:
        tuple: (epoch_loss_G, epoch_loss_D) representing the average losses for the epoch.
    """

    # Set the models to training mode, and initialize epoch losses and batch counter.
    G.train()
    D.train()
    epoch_loss_G = 0.0
    epoch_loss_D = 0.0
    epoch_grad_G = 0.0
    epoch_grad_D = 0.0
    num_batches = 0

    # Model architecture goes here.
    for batch_idx, real_images in enumerate(train_dataloader):
        # Ensure real_images is a tensor of shape [B, 1, 256, 128] before moving to device.
        if isinstance(real_images, (tuple, list)):
            real_images = real_images[0]
            
        # Move the images to the device. 
        real_images = real_images.to(
            device=device,
            dtype=torch.float32,
        )
        batch_size = real_images.size(0)
        # Train discriminator on real and fake images. 
        # Zero gradients first, so that previous updates do not affect the current step.
        optimizer_D.zero_grad(set_to_none=True)
        real_output = D(real_images).reshape(-1)
        loss_real = criterion(
            real_output,
            torch.ones_like(real_output),
        )
        # Generate fake images using the generator.
        noise = torch.randn(
            batch_size, latent_dim, 1, 1, device=device
        )
        fake_images = G(noise)
        fake_output = D(fake_images.detach()).reshape(-1)
        loss_fake = criterion(
            fake_output,
            torch.zeros_like(fake_output),
        )
        # Compute total discriminator loss and update its parameters.
        loss_D = loss_real + loss_fake
        # Check if the loss is finite before backpropagating.
        if not torch.isfinite(loss_D):
            raise RuntimeError(f"Discriminator loss is not finite; batch index: {batch_idx}")
        # Backpropagate the discriminator loss.
        loss_D.backward()
        grad_norm_D = check_gradients(D, "discriminator", epoch, batch_idx)

        # Check if gradient is finite before updating the discriminator's parameters.
        for param in D.parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                raise RuntimeError(f"Discriminator gradient is not finite; batch index: {batch_idx}")
        # Update the discriminator's parameters.
        optimizer_D.step()
        # Train generator.
        optimizer_G.zero_grad(set_to_none=True)
        
        # Generate fake images for the generator update step.
        fake_output = D(fake_images).reshape(-1)
        loss_G = criterion(
            fake_output,
            torch.ones_like(fake_output),
        )
        # Check if the generator loss is finite before backpropagating. 
        if not torch.isfinite(loss_G):
            raise RuntimeError(f"Generator loss is not finite; batch index: {batch_idx}")
        # Backpropagate the generator loss.
        loss_G.backward()
        grad_norm_G = check_gradients(G, "generator", epoch, batch_idx)
        # Check if the generator's gradients are finite before updating its parameters.
        for param in G.parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                raise RuntimeError(f"Generator gradient is not finite; batch index: {batch_idx}")
        # Update the generator's parameters.
        optimizer_G.step()
        # Get the epoch loss and update the number of processed batches.
        epoch_loss_G += loss_G.item()
        epoch_loss_D += loss_D.item()
        epoch_grad_G += grad_norm_G
        epoch_grad_D += grad_norm_D
        num_batches += 1

    if num_batches == 0:
        raise ValueError("Training dataloader contains no batches.")
    # Return the average losses for the epoch.
    return (
        epoch_loss_G / num_batches,
        epoch_loss_D / num_batches,
        epoch_grad_G/ num_batches,
        epoch_grad_D/ num_batches
    )


def validate_one_epoch(
    validate_dataloader: DataLoader,
    G: torch.nn.Module,
    D: torch.nn.Module,
    criterion: torch.nn.Module,
    latent_dim: int,
    device: torch.device,
):
    "Validate the GAN for one epoch and return the average generator and discriminator losses."
    # Set the models to evaluation mode and initialize epoch statistics.
    G.eval()
    D.eval()
    epoch_loss_G = 0.0
    epoch_loss_D = 0.0
    num_batches = 0
    
    # Iterate over the validation dataloader without computing gradients.
    with torch.no_grad():
        for real_images in validate_dataloader:
            if isinstance(real_images, (tuple, list)):
                real_images = real_images[0]

            real_images = real_images.to(
                device=device,
                dtype=torch.float32,
            )
            batch_size = real_images.size(0)

            real_output = D(real_images).reshape(-1)
            loss_real = criterion(
                real_output,
                torch.ones_like(real_output),
            )

            noise = torch.randn(
                batch_size, latent_dim, 1, 1, device=device
            )
            fake_images = G(noise)

            fake_output = D(fake_images).reshape(-1)
            loss_fake = criterion(
                fake_output,
                torch.zeros_like(fake_output),
            )

            loss_D = loss_real + loss_fake

            fake_output_for_G = D(fake_images).reshape(-1)
            loss_G = criterion(
                fake_output_for_G,
                torch.ones_like(fake_output_for_G),
            )

            epoch_loss_G += loss_G.item()
            epoch_loss_D += loss_D.item()
            num_batches += 1

    if num_batches == 0:
        raise ValueError("Validation dataloader contains no batches.")

    return (
        epoch_loss_G / num_batches,
        epoch_loss_D / num_batches,
    )
    

def save_checkpoint(
    G: torch.nn.Module,
    D: torch.nn.Module,
    optimizer_G: torch.optim.Optimizer,
    optimizer_D: torch.optim.Optimizer,
    epoch: int,
    latent_dim: int,
    results_dir: Path,
):
    # Save the current state of the GAN model and optimizers as a checkpoint.
    torch.save(
        {
            "generator": G.state_dict(),
            "discriminator": D.state_dict(),
            "optimizer_G": optimizer_G.state_dict(),
            "optimizer_D": optimizer_D.state_dict(),
            "epoch": epoch,
            "latent_dim": latent_dim,
        },
        results_dir / "checkpoint.pth",
    )

def train_loop(config):
    # Setup training components: dataloaders, device, models, and optimizers.
    latent_dim = config["latent_dim"]
    train_dataloader = initialize_dataloader(config)
    device = configure_device()
    if config["validate"]:
        validate_dataloader = initialize_validation_dataloader(config)
    # Initialize the GAN models and optimizers.
    G, D, optimizer_G, optimizer_D = (
        initialize_training_model(config, latent_dim, device)
    )
    # Initialize the loss criterion and number of epochs.
    criterion = nn.BCEWithLogitsLoss()
    num_epochs = config["num_epochs"]
    # Initialize lists to store the generator and discriminator losses for each epoch.
    generator_losses = []
    discriminator_losses = []
    epoch_times = []
    grad_G_norms = []
    grad_D_norms = []
    
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

    print("Training started!")
    train_start = time.perf_counter()
    # Main training loop.
    for epoch in range(1, num_epochs + 1):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_start = time.perf_counter()
        mean_G, mean_D, grad_G, grad_D = train_one_epoch(
            train_dataloader,
            G,
            D,
            optimizer_G,
            optimizer_D,
            criterion,
            latent_dim,
            device,
            epoch
        )

        # Append the mean losses for this epoch to the respective lists, then log the epoch statistics.
        generator_losses.append(mean_G)
        discriminator_losses.append(mean_D)
        grad_G_norms.append(grad_G)
        grad_D_norms.append(grad_D)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        epoch_times.append(time.perf_counter() - epoch_start)

        # Save generated samples for every [sample_interval] epochs.
        if epoch % config["sample_interval"] == 0 or epoch == 1:
            save_generated_samples(
                G,
                epoch,
                latent_dim,
                device,
                Path(config["results_dir"]),
                64,
            )
            
        if config["validate"] and epoch % config["validate_every"] == 0:
            val_G, val_D = validate_one_epoch(
                validate_dataloader, # type:ignore
                G,
                D,
                criterion,
                latent_dim,
                device,
            )
            save_validation_comparison(
                    G,
                    validate_dataloader, #type:ignore
                    epoch,
                    latent_dim,
                    device,
                    Path(config["results_dir"]),
                    count=8,
                )
            validation_message = (
                f"Validation - Epoch [{epoch}/{num_epochs}] "
                f"Loss D: {val_D:.4f} "
                f"Loss G: {val_G:.4f} "
            )
            print(validation_message)
            
        message = (
            f"Epoch [{epoch}/{num_epochs}] "
            f"Loss D: {mean_D:.4f} "
            f"Loss G: {mean_G:.4f} "
            f"Grad G: {grad_G_norms[-1]:.4f} "
            f"Grad D: {grad_D_norms[-1]:.4f} "
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
    if device.type == "cuda":
        torch.cuda.synchronize()
        peak_memory = torch.cuda.max_memory_allocated(device)
        print(f"Peak GPU memory: {peak_memory / 1024**3:.2f} GB")

    elapsed = time.perf_counter() - train_start
    print(f"Training time: {elapsed:.2f} seconds")

    # Save final checkpoint as a pth file.
    save_checkpoint(
        G,
        D,
        optimizer_G,
        optimizer_D,
        num_epochs,
        latent_dim,
        Path(config["results_dir"]),
    )

    return G, D, generator_losses, discriminator_losses, grad_G_norms, grad_D_norms

def create_plot(G, results_dir, generator_losses, discriminator_losses, grad_G_norms, grad_D_norms, latent_dim):
    # Generate final samples.
    G.eval()
    device = next(G.parameters()).device

    # Generate a batch of images from random noise, don't compute gradients.
    with torch.no_grad():
        noise = torch.randn(64, latent_dim, 1, 1, device=device)
        generated = G(noise).cpu()
        
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
        generator_losses,
        discriminator_losses,
        grad_G_norms,
        grad_D_norms,
        results_dir,
    )
        
def main():
    config = load_config()
    set_seed(config.get("seed"))
    results_dir = Path(config["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    G, D, generator_losses, discriminator_losses, grad_G_norms, grad_D_norms = train_loop(config)
    create_plot(
        G, 
        results_dir, 
        generator_losses, 
        discriminator_losses, 
        grad_G_norms,
        grad_D_norms,
        config["latent_dim"])
    return G, D, generator_losses, discriminator_losses, grad_G_norms, grad_D_norms

if __name__ == "__main__":
    main()