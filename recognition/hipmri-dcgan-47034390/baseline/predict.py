import random
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import yaml
from torch.utils.data import DataLoader
from baseline.modules import Encoder, Decoder

# Prediction script for generating and visualizing images using a trained DCGAN generator.
# A different predict must be used for the baseline.

# Path to the configuration file and other constants for prediction
CONFIG_PATH = Path("config.yaml")
INCLUDE_REFERENCE_DATASET = True
GENERATE_ROWS = 4
GENERATE_COLS = 4
COUNT = GENERATE_ROWS * GENERATE_COLS

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

def load_config():
    with open(CONFIG_PATH, encoding="utf-8") as file:
        raw = yaml.safe_load(file)

    profile = "hipmri_vae" if raw["defaults"]["rangpur"] else "hipmri_vae_cpu"
    return {**raw["defaults"], **raw[profile]}

def load_checkpoint(checkpoint_path, device, config):
    """Load the generator model from a checkpoint.

    Args:
        checkpoint_path (Path): Path to the checkpoint file.
        device (torch.device): Device to load the model onto.
        config (dict): Configuration dictionary containing model parameters.

    Returns:
        Generator: The loaded generator model.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    G = Decoder(latent_dim=config["latent_dim"], channels=1).to(device)
    G.load_state_dict(checkpoint["decoder"])
    G.eval()
    return G

def resolve_path(value):
    """Resolve a file path relative to the configuration file.

    Args:
        value (str or Path): The file path to resolve.

    Returns:
        Path: The resolved absolute file path.
    """
    path = Path(value)
    return path if path.is_absolute() else CONFIG_PATH.parent / path

def load_real_images(config):
    """Load a subset of real images from the dataset.

    Args:
        config (dict): Configuration dictionary containing dataset parameters.

    Returns:
        torch.Tensor: A batch of real images.
    """
    from dataset import HipMRIDataset
    from torch.utils.data import DataLoader, Subset

    dataset = HipMRIDataset(
        data_dir=config.get("test_dir"),
        max_images=config.get("max_train_images"),
        norm_image=config.get("norm_image", True),
        week=config.get("week"),
    )
    indices = random.sample(range(len(dataset)), COUNT)
    return next(iter(DataLoader(Subset(dataset, indices), batch_size=COUNT)))

def plot_generated_images_comparison(
    device: torch.device, 
    results_dir: Path, 
    generated: torch.Tensor, 
    real_images: torch.Tensor):
    
    """Plot a comparison of generated and real images.

    Args:
        config (dict): Configuration dictionary containing model and dataset parameters.
        device (torch.device): Device to perform computations on.
        results_dir (Path): Directory to save the generated images.
        generated (torch.Tensor): Batch of generated images.
        real_images (torch.Tensor): Batch of real images for comparison.
    """
    
    if INCLUDE_REFERENCE_DATASET:
        fig, axes = plt.subplots(
            GENERATE_ROWS,
            GENERATE_COLS * 2,
            figsize=(GENERATE_COLS * 2 * 0.9, GENERATE_ROWS * 1.5),
        )
        print("Real images shape:", real_images.shape)
        print("Generated images shape:", generated.shape)
        for i in range(COUNT):
            row = i // GENERATE_COLS
            real_col = i % GENERATE_COLS
            generated_col = real_col + GENERATE_COLS

            for col, images in (
                (real_col, real_images),
                (generated_col, generated),
            ):
                ax = axes[row, col]
                ax.imshow(images[i, 0].numpy(), cmap="gray", vmin=-1, vmax=1)
                ax.axis("off")

        fig.subplots_adjust(
            left=0.01, right=0.99,
            top=0.97, bottom=0.01,
            wspace=0.01, hspace=0.01,
        )
        axes[0, 0].set_title("REAL")
        axes[0, GENERATE_COLS].set_title("GENERATED")
    else:
        fig, axes = plt.subplots(GENERATE_ROWS, GENERATE_COLS, figsize=(10, 10))

        for ax, image in zip(axes.flat, generated):
            ax.imshow(image[0].numpy(), cmap="gray", vmin=-1, vmax=1)
            ax.axis("off")

    plt.tight_layout()
    output_path = results_dir / "generated_images_baseline.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.02)
    plt.close()

    print(f"Generated shape: {generated.shape}")
    print(f"Device: {device}")
    print(f"Saved images to: {output_path}")

def plot_reconstructions(
    encoder,
    decoder,
    dataloader,
    device,
    results_dir,
    num_rows=4,
):
    encoder.eval()
    decoder.eval()

    real_images = []

    with torch.no_grad():
        images = next(iter(DataLoader(
            dataloader.dataset,
            batch_size=len(dataloader.dataset),
            shuffle=True,
        )))

        real_images = images[:num_rows].to(
            device=device,
            dtype=torch.float32,
        )

        mu, _ = encoder(real_images)
        reconstructed = decoder(mu)

    real_images = real_images.cpu()
    reconstructed = reconstructed.cpu()

    fig, axes = plt.subplots(
        2,
        num_rows,
        figsize=(3 * num_rows, 6),
    )

    for i in range(num_rows):
        axes[0, i].imshow(real_images[i, 0], cmap="gray", vmin=-1, vmax=1)
        axes[0, i].set_title(f"Real {i + 1}")
        axes[0, i].axis("off")

        axes[1, i].imshow(
            reconstructed[i, 0],
            cmap="gray",
            vmin=-1,
            vmax=1,
        )
        axes[1, i].set_title(f"Reconstruction {i + 1}")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig(results_dir / "reconstruction_comparison.png", dpi=150)
    print(f"Saved reconstruction comparison to: {results_dir / 'reconstruction_comparison.png'}")
    plt.close(fig)

def main():
    from dataset import HipMRIDataset
    from metrics import compute_ssim, compute_fid, diversity_stats, evaluate_reconstructions
    from baseline.modules import Encoder, Decoder
    import pprint

    config = load_config()
    pprint.pprint(config)
    set_seed(config.get("seed", 42))

    results_dir = resolve_path(config["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_dim = config["latent_dim"]
    checkpoint_path = results_dir / "checkpoint.pth"

    print(f"RANGPUR: {config.get('rangpur', False)}")
    print(f"Results directory: {results_dir}")
    print(f"Checkpoint path: {checkpoint_path}")

    real_images = load_real_images(config)
    real_images = real_images.to(device, dtype=torch.float32)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    encoder = Encoder(latent_dim=latent_dim, channels=1).to(device)
    decoder = Decoder(latent_dim=latent_dim, channels=1).to(device)

    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])

    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        noise = torch.randn(COUNT, latent_dim, device=device)
        generated = decoder(noise).cpu()


    dataset = HipMRIDataset(
        data_dir=config["test_dir"],
        max_images=config.get("max_train_images"),
        norm_image=config.get("norm_image", True),
        week=config.get("week"),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
    )

    # Plot generated images comparison between real and generated images. 
    # This uses the images generated from random noise and the real images from the dataset.
    plot_generated_images_comparison(
        device=device,
        results_dir=results_dir,
        generated=generated,
        real_images=real_images,
    )
    
    # Plot reconstructions of real images using the encoder and decoder. This does not use PCA since the images are being
    # reconstructed directly from the encoder's latent representation, therefore PCA is not involved in this step.
    # This takes images from the test dataset.
    plot_reconstructions(
        encoder=encoder,
        decoder=decoder,
        dataloader=dataloader,
        device=device,
        results_dir=results_dir,
        num_rows=GENERATE_ROWS,
    )
        
    # Reconstruction evaluation.
    reconstruction_ssim = compute_ssim(
        *evaluate_reconstructions(
            encoder,
            decoder,
            dataloader,
            device,
        )
    )

    # Generation evaluation.
    fid = compute_fid(real_images, generated)
    diversity = diversity_stats(generated)

    print(f"Reconstruction SSIM: {reconstruction_ssim:.4f}")
    print(f"Generation FID: {fid:.4f}")
    print("Diversity:")
    print(f"  Mean pairwise distance: {diversity['mean_pairwise_distance']:.4f}")
    print(f"  Mean nearest neighbour distance: {diversity['mean_nearest_neighbour_distance']:.4f}")

if __name__ == "__main__":
    main()