import torch
import matplotlib.pyplot as plt
from pathlib import Path
import torch.nn.functional as F
from torchvision.transforms.functional import resize
from torchmetrics.image.fid import FrechetInceptionDistance

def save_generated_samples(
    G: torch.nn.Module,
    epoch: int,
    latent_dim: int,
    device: torch.device,
    results_dir: Path,
    batch_size: int):
    """
    Save generated samples from the generator.

    Args:
        G (torch.nn.Module): The generator model.
        epoch (int): The current epoch number.
        latent_dim (int): The dimensionality of the latent space.
        device (torch.device): The device to run the model on.
        results_dir (Path): Directory to save the generated samples.
        batch_size (int): Number of samples to generate.
    """
    G.eval()

    with torch.no_grad():
        z = torch.randn(batch_size, latent_dim, 1, 1, device=device)
        generated = G(z).cpu()

    G.train()

    fig, axes = plt.subplots(
        4, 4,
        figsize=(6, 6),
        gridspec_kw={"wspace": 0, "hspace": 0}
    )

    for i, ax in enumerate(axes.flat):
        ax.imshow(
            generated[i, 0] * 0.5 + 0.5,
            cmap="gray",
            vmin=0,
            vmax=1
        )
        ax.axis("off")

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(
        results_dir / f"epoch_{epoch:02d}.png",
        bbox_inches="tight",
        pad_inches=0
    )
    plt.close(fig)

def plot_training_losses(
    generator_losses: list,
    discriminator_losses: list,
    grad_G_norms: list,
    grad_D_norms: list,
    results_dir: Path
):
    """
    Plot the training losses and gradient norms for the DCGAN.

    Args:
        generator_losses (list): List of generator losses per iteration.
        discriminator_losses (list): List of discriminator losses per iteration.
        grad_G_norms (list): List of gradient norms for the generator per iteration.
        grad_D_norms (list): List of gradient norms for the discriminator per iteration.
        results_dir (Path): Directory to save the plots.
    """
    # Plots the training losses.
    plt.figure()
    plt.plot(
        generator_losses,
        label="Generator"
    )

    plt.plot(
        discriminator_losses,
        label="Discriminator"
    )

    plt.xlabel("Training iteration")
    plt.ylabel("Loss")
    plt.legend()

    plt.savefig(
        f"{results_dir}/training_losses.png",
        dpi=150,
        bbox_inches="tight"
    )

    fig, ax = plt.subplots()

    ax.plot(generator_losses, label="Generator")
    ax.plot(discriminator_losses, label="Discriminator")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("DCGAN Training Loss")
    ax.legend()

    fig.savefig(
        results_dir / "training_losses.png",
        bbox_inches="tight"
    )

    plt.close(fig)
    # Plot norms.
    fig, ax = plt.subplots()
    ax.plot(grad_G_norms, label="Grad G Norms")
    ax.plot(grad_D_norms, label="Grad D Norms")
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Gradient Norm")
    ax.set_title("Gradient Norms During Training")
    ax.legend()

    fig.savefig(
        results_dir / "gradient_norms.png",
        bbox_inches="tight"
    )
    plt.close(fig)


def save_validation_comparison(
    G: torch.nn.Module,
    validate_dataloader: torch.utils.data.DataLoader,
    epoch: int,
    latent_dim: int,
    device: torch.device,
    results_dir: Path,
    count: int = 8,
):
    """
    Save a comparison of real and generated images from the validation set.

    Args:
        G (torch.nn.Module): The generator model.
        validate_dataloader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        epoch (int): Current epoch number.
        latent_dim (int): The dimensionality of the latent space.
        device (torch.device): Device to run the model on.
        results_dir (Path): Directory to save the comparison images.
        count (int, optional): Number of images to display. Defaults to 8.
    """
    G.eval()
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Get real validation images.
    real_images = next(iter(validate_dataloader))
    if isinstance(real_images, (tuple, list)):
        real_images = real_images[0]

    real_images = real_images[:count].to(device=device, dtype=torch.float32)

    # Generate fake images.
    with torch.no_grad():
        noise = torch.randn(count, latent_dim, 1, 1, device=device)
        fake_images = G(noise)

    real_images = ((real_images.cpu() + 1) / 2).clamp(0, 1)
    fake_images = ((fake_images.cpu() + 1) / 2).clamp(0, 1)

    fig, axes = plt.subplots(2, count, figsize=(2 * count, 5))

    for i in range(count):
        axes[0, i].imshow(real_images[i, 0], cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title(f"Real {i + 1}")
        axes[0, i].axis("off")

        axes[1, i].imshow(fake_images[i, 0], cmap="gray", vmin=0, vmax=1)
        axes[1, i].set_title(f"Generated {i + 1}")
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Validation", fontsize=11)
    axes[1, 0].set_ylabel("Generated", fontsize=11)

    fig.suptitle(f"Real vs Generated MRI — Epoch {epoch}")
    plt.tight_layout()
    plt.savefig(results_dir / f"comparison_epoch_{epoch:02d}.png", dpi=150)
    plt.close(fig)
