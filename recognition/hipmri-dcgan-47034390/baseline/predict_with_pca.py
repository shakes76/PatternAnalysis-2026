import random
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import numpy as np
import yaml
from torch.utils.data import DataLoader
from baseline.modules import Encoder, Decoder
from sklearn.decomposition import PCA
# Prediction script for generating and visualizing images using a trained DCGAN generator.
# A different predict must be used for the baseline.

# Path to the configuration file and other constants for prediction
CONFIG_PATH = Path("config.yaml")
INCLUDE_REFERENCE_DATASET = True
GENERATE_ROWS = 4
GENERATE_COLS = 4
COUNT = GENERATE_ROWS * GENERATE_COLS
PCA_COMPONENTS = 10

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


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

def load_config():
    with open(CONFIG_PATH, encoding="utf-8") as file:
        raw = yaml.safe_load(file)

    profile = "hipmri_vae" if raw["defaults"]["rangpur"] else "hipmri_vae_cpu"
    config = {**raw["defaults"], **raw[profile]}
    set_seed(config.get("seed", 42))
    return config


def resolve_path(value):
    """Resolve a file path relative to the configuration file."""
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
        max_images=None,
        norm_image=config.get("norm_image", True),
        week=config.get("week"),
    )
    indices = random.sample(range(len(dataset)), COUNT)
    return next(iter(DataLoader(Subset(dataset, indices), batch_size=COUNT)))

def generate_random_pca_samples(
    latent_vectors,
    decoder,
    device,
):
    pca = PCA(n_components=PCA_COMPONENTS)
    projected = pca.fit_transform(latent_vectors)

    latent_mean = torch.tensor(
        latent_vectors.mean(axis=0),
        dtype=torch.float32,
        device=device,
    )

    components = torch.tensor(
        pca.components_,
        dtype=torch.float32,
        device=device,
    )

    coordinates = torch.tensor(
        np.column_stack([
            np.random.uniform(
                projected[:, i].min(),
                projected[:, i].max(),
                size=COUNT,
            )
            for i in range(PCA_COMPONENTS)
        ]),
        dtype=torch.float32,
        device=device,
    )

    # Map each 2D coordinate back into the full latent space.
    z = latent_mean + coordinates @ components

    with torch.no_grad():
        generated = decoder(z).cpu()

    return generated

def init():
    import pprint
    config = load_config()
    pprint.pprint(config)
    
    data_dir = resolve_path(config["data_dir"])
    results_dir = resolve_path(config["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_dim = config["latent_dim"]
    checkpoint_path = results_dir / "checkpoint.pth"
    
    print(f"RANGPUR: {config.get('rangpur', False)}")
    print(f"Data directory: {data_dir}, Results directory: {results_dir}, Checkpoint path: {checkpoint_path}")
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return config, data_dir, results_dir, device, latent_dim, checkpoint_path

def plot_generated_images_comparison(
    device: torch.device, 
    results_dir: Path, 
    generated: torch.Tensor, 
    real_images: torch.Tensor):

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
    output_path = results_dir / "generated_images_pca.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.02)
    plt.close()

    print(f"Generated shape: {generated.shape}")
    print(f"Device: {device}")
    print(f"Saved images to: {output_path}")


def main():
    from dataset import HipMRIDataset
    from metrics import compute_ssim, compute_fid, diversity_stats, evaluate_reconstructions
    from baseline.predict import plot_reconstructions

    config, data_dir, results_dir, device, latent_dim, checkpoint_path = init()
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    encoder = Encoder(latent_dim=config["latent_dim"], channels=1).to(device)
    decoder = Decoder(latent_dim=config["latent_dim"], channels=1).to(device)
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])
    encoder.eval()
    decoder.eval()

    dataset = HipMRIDataset(
        data_dir=config["test_dir"],
        max_images=config.get("max_train_images"),
        norm_image=config.get("norm_image", True),
        week=config.get("week"),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
    )

    latent_vectors = get_latent_vectors(
        encoder, dataloader, device
    )

    with torch.no_grad():
        generated = generate_random_pca_samples(
            latent_vectors, decoder, device
        )

    real_images = load_real_images(config)
    real_images = real_images.to(device, dtype=torch.float32)

    plot_generated_images_comparison(
        device=device,
        results_dir=results_dir,
        generated=generated,
        real_images=real_images
    )
    
    # Plot reconstructions of real images using the encoder and decoder. This does not use PCA since the images are being
    # reconstructed directly from the encoder's latent representation, therefore PCA is not involved in this step.
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

    # Generate images using PCA latent sampling.
    with torch.no_grad():
        generated = generate_random_pca_samples(
            latent_vectors,
            decoder,
            device,
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