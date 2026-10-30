from dataset import HipMRIDataset
import torch
from pathlib import Path
from metrics import compute_fid, compute_ssim, diversity_stats
from baseline.modules import Encoder, Decoder
from baseline.predict import load_config, resolve_path, set_seed

# Path to the configuration file and other constants for prediction
CONFIG_PATH = Path("config.yaml")
# Whether to include real images from the reference dataset in the visualization.
INCLUDE_REFERENCE_DATASET = True
# Number of images to generate and visualize.

# Path to the configuration file and other constants for prediction
CONFIG_PATH = Path("config.yaml")
# Whether to include real images from the reference dataset in the visualization.
INCLUDE_REFERENCE_DATASET = True
# Number of images to generate and visualize.
COUNT = 32  # Number of images to generate and visualize.

def main():
    print("Metrics for model VAE:")
    config = load_config()
    set_seed(config.get("seed", 42))
    print(f"week: {config.get('week')}, seed: {config.get('seed', 42)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = resolve_path(config["results_dir"])
    checkpoint_path = results_dir / "checkpoint.pth"

    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )

    encoder = Encoder(latent_dim=config["latent_dim"], channels=1).to(device)
    decoder = Decoder(latent_dim=config["latent_dim"], channels=1).to(device)
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])
    encoder.eval()
    decoder.eval()

    dataset = HipMRIDataset(
        data_dir=resolve_path(config["test_dir"]),
        max_images=None,
        norm_image=config.get("norm_image", True),
        week=config.get("week"),
    )
    real_images = torch.stack([
        dataset[i] for i in range(len(dataset))
    ]).float()
    real_images_device = real_images.to(device)

    with torch.no_grad():
        # Reconstruct each real image using its encoded latent mean.
        mu, _ = encoder(real_images_device)
        reconstructions = decoder(mu).cpu()

        # Generate new images by sampling the VAE's standard normal prior.
        noise = torch.randn(
            len(real_images), config["latent_dim"], device=device
        )
        generated = decoder(noise).cpu()

    print(f"Mean reconstruction SSIM: "
          f"{compute_ssim(real_images, reconstructions):.4f}")
    print("Diversity:", diversity_stats(generated))
    
    compute_fid(real_images, generated)

if __name__ == "__main__":
    main()