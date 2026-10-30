import random
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import yaml
from modules import Generator

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

def load_config(path=CONFIG_PATH):
    with open(path, encoding="utf-8") as file:
        raw = yaml.safe_load(file)

    profile = "hipmri_dcgan" if raw["defaults"]["rangpur"] else "hipmri_dcgan_cpu"
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
    G = Generator(latent_dim=config["latent_dim"], channels=1).to(device)
    G.load_state_dict(checkpoint["generator"])
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
    """Load a subset of real images from the dataset; this is using the HipMRIDataset class.

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
    output_path = results_dir / "generated_images.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.02)
    plt.close()

    print(f"Generated shape: {generated.shape}")
    print(f"Device: {device}")
    print(f"Saved images to: {output_path}")


def main():
    # Load AFTER to prevent import.
    from metrics import compute_fid, diversity_stats, compute_ssim
    import pprint
    config = load_config()
    set_seed(config.get("seed", 42))
    pprint.pprint(config)
    results_dir = resolve_path(config["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_dim = config["latent_dim"]
    checkpoint_path = results_dir / "checkpoint.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    G = load_checkpoint(checkpoint_path, device, config)
    with torch.no_grad():
        z = torch.randn(COUNT, latent_dim, 1, 1, device=device)
        generated = G(z).cpu()

    real_images=load_real_images(config)

    plot_generated_images_comparison(
        device=device,
        results_dir=results_dir,
        generated=generated,
        real_images=real_images
    )
    
    print("Metrics for DC-GAN:")
    print("Real:", diversity_stats(real_images))
    print("Generated:", diversity_stats(generated))
    # This SSIM isn't directly comparing corresponding real and generated images, 
    # but gives an overall similarity measure; therefore, it won't be a high value.
    print(f"Generation SSIM: {compute_ssim(real_images, generated):.4f}")
    compute_fid(real_images, generated)
    print("Metrics computation completed.")

    
if __name__ == "__main__":
    main()