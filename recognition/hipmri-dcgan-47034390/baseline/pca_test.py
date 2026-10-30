
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from baseline.modules import Encoder, Decoder
from dataset import HipMRIDataset

def get_latent_vectors(
    encoder: torch.nn.Module, 
    dataloader: DataLoader, 
    device: torch.device):
    """Extract latent vectors from a dataset using the encoder.

    Args:
        encoder (torch.nn.Module): The encoder model to extract latent vectors.
        dataloader (torch.utils.data.DataLoader): DataLoader for the dataset.
        device (torch.device): Device to perform computations on.

    Returns:
        np.ndarray: Array of latent vectors of shape (num_samples, latent_dim).
    """
    encoder.eval()
    latent_vectors = []

    with torch.no_grad():
        for images in dataloader:
            if isinstance(images, (tuple, list)):
                images = images[0]

            images = images.to(device, dtype=torch.float32)
            mu, _ = encoder(images)
            latent_vectors.append(mu.cpu())

    return torch.cat(latent_vectors).numpy()


def plot_latent_pca(latent_vectors, output_path):
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(latent_vectors)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(latent_2d[:, 0], latent_2d[:, 1], s=12, alpha=0.65)

    ax.set_title("VAE Latent Space — PCA")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(
        "Variance explained by first two components:",
        f"{pca.explained_variance_ratio_.sum():.2%}",
    )

    return pca


def plot_latent_grid(
    decoder, latent_vectors, pca, device, output_path, grid_size=8
):
    decoder.eval()

    # Use the mean of the encoded latent vectors as the grid centre.
    latent_mean = torch.tensor(
        latent_vectors.mean(axis=0),
        dtype=torch.float32,
        device=device,
    )

    # Principal directions in the original latent space.
    components = torch.tensor(
        pca.components_,
        dtype=torch.float32,
        device=device,
    )

    # Span each direction across the observed encoded distribution.
    projected = pca.transform(latent_vectors)
    x_values = np.linspace(
        projected[:, 0].min(), projected[:, 0].max(), grid_size
    )
    y_values = np.linspace(
        projected[:, 1].min(), projected[:, 1].max(), grid_size
    )

    latent_grid = []
    for y in reversed(y_values):
        for x in x_values:
            z = (
                latent_mean
                + x * components[0]
                + y * components[1]
            )
            latent_grid.append(z)

    latent_grid = torch.stack(latent_grid)

    with torch.no_grad():
        generated = decoder(latent_grid).cpu()

    fig, axes = plt.subplots(
        grid_size, grid_size, figsize=(12, 13)
    )

    for i, ax in enumerate(axes.flat):
        ax.imshow(
            generated[i, 0].numpy(),
            cmap="gray",
            vmin=-1,
            vmax=1,
        )
        ax.axis("off")

    fig.suptitle(
        "VAE Generated Images Across the First Two PCA Directions",
        fontsize=14,
    )
    fig.supxlabel("Principal Component 1 (low to high)")
    fig.supylabel("Principal Component 2 (high to low)")

    fig.tight_layout(rect=(0.03, 0.03, 1, 0.96))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    results_dir = Path("./baseline_results")
    results_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = results_dir / "checkpoint.pth"
    config_path = Path("config.yaml")
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    settings = {
        **config["defaults"],
        **config["hipmri_vae_cpu"],
    }
    latent_dim = settings["latent_dim"]

    checkpoint = torch.load(checkpoint_path, map_location=device)

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

    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])

    dataset = HipMRIDataset(
        data_dir=settings["test_dir"],
        max_images=None,
        norm_image=True,
        week=settings["week"],
    )

    dataloader = DataLoader(
        dataset,
        batch_size=settings["batch_size"],
        shuffle=False,
    )

    latent_vectors = get_latent_vectors(
        encoder, dataloader, device
    )

    pca = plot_latent_pca(
        latent_vectors,
        results_dir / "pca_plot.png",
    )

    plot_latent_grid(
        decoder,
        latent_vectors,
        pca,
        device,
        results_dir / "latent_grid.png",
    )

    print(f"Saved PCA plot to {results_dir / 'pca_plot.png'}")
    print(f"Saved latent grid to {results_dir / 'latent_grid.png'}")


if __name__ == "__main__":
    main()
