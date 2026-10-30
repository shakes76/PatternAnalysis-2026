from torchvision.transforms.functional import resize
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image import StructuralSimilarityIndexMeasure
from dataset import HipMRIDataset
import torch
from pathlib import Path
from predict import load_config, resolve_path, load_checkpoint, set_seed

# Path to the configuration file and other constants for prediction
CONFIG_PATH = Path("config.yaml")
# Whether to include real images from the reference dataset in the visualization.
INCLUDE_REFERENCE_DATASET = True
# Number of images to generate and visualize.

def compute_fid(reference_images: torch.Tensor, generated: torch.Tensor) -> float:
    """Compute the Frechet Inception Distance (FID) between reference and generated images.

    Args:
        reference_images (torch.Tensor): A batch of real images.
        generated (torch.Tensor): A batch of generated images.

    Returns:
        float: The computed FID score.
    """
    # Compute the Frechet Inception Distance (FID) between reference and generated images.
    
    def prepare_for_fid(images: torch.Tensor) -> torch.Tensor:
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)

        images = resize(images, [299, 299])

        images = ((images + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)

        return images

    fid = FrechetInceptionDistance(
        feature=2048,
        normalize=False
    )
    batch_size = 8

    for i in range(0, len(reference_images), batch_size):

        real_batch = reference_images[i:i + batch_size]

        real_batch = prepare_for_fid(real_batch)

        fid.update(
            real_batch,
            real=True
        )


    for i in range(0, len(generated), batch_size):

        fake_batch = generated[i:i + batch_size]

        fake_batch = prepare_for_fid(fake_batch)

        fid.update(
            fake_batch,
            real=False
        )


    fid_score = fid.compute()

    print(f"FID: {fid_score.item():.4f}")
    return fid_score.item()

def compute_ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
) -> float:
    """Compute the Structural Similarity Index Measure (SSIM) between two images; 
    this is more useful for reconstruction than for generative quality.

    Args:
        img1 (torch.Tensor): A batch of real images.
        img2 (torch.Tensor): A batch of generated images.

    Returns:
        float: The computed SSIM score.
    """
    img1 = ((img1.float() + 1) / 2).clamp(0, 1)
    img2 = ((img2.float() + 1) / 2).clamp(0, 1)

    # Use Torch SSIM instead of implementing it manually.
    metric = StructuralSimilarityIndexMeasure(
        data_range=1.0
    ).to(img1.device)

    return metric(img1, img2).item()

def check_gradients(model: torch.nn.Module, model_name: str, epoch: int, batch_idx: int) -> float:
    """Check and clip gradients of a model if they are finite.

    Args:
        model (torch.nn.Module): The model whose gradients are to be checked.
        model_name (str): Name of the model for logging purposes.
        epoch (int): Current epoch number.
        batch_idx (int): Current batch index within the epoch.

    Raises:
        FloatingPointError: Raised if the gradients contain non-finite values.

    Returns:
        float: The norm of the gradients after clipping.
    """
    try:
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=float("inf"),
            error_if_nonfinite=True,
        )
    except RuntimeError as exc:
        raise FloatingPointError(
            f"Invalid {model_name} gradients at "
            f"epoch {epoch}, batch {batch_idx}"
        ) from exc

    return norm.item()

def diversity_stats(images):
    """Compute diversity statistics for a batch of images.

    Args:
        images (torch.Tensor): A batch of images with shape (N, C, H, W) and range [-1, 1].

    Returns:
        dict: A dictionary containing mean pairwise distance and mean nearest neighbour distance.
    """
    # Assumes images have shape (N, C, H, W) and range [-1, 1].
    images = ((images.detach().float().cpu() + 1) / 2).clamp(0, 1)
    features = images.flatten(start_dim=1)

    # Normalised Euclidean distances between all image pairs.
    distances = torch.cdist(features, features, p=2)
    distances /= features.shape[1] ** 0.5

    # Exclude each image's distance to itself.
    distances.fill_diagonal_(float("inf"))

    return {
        "mean_pairwise_distance": distances[distances.isfinite()].mean().item(),
        "mean_nearest_neighbour_distance": distances.min(dim=1).values.mean().item(),
    }
    
def evaluate_reconstructions(encoder, decoder, dataloader, device):
    encoder.eval()
    decoder.eval()

    real_images = []
    reconstructed_images = []

    with torch.no_grad():
        for images in dataloader:
            if isinstance(images, (tuple, list)):
                images = images[0]

            images = images.to(device, dtype=torch.float32)

            mu, _ = encoder(images)
            reconstructed = decoder(mu)

            real_images.append(images.cpu())
            reconstructed_images.append(reconstructed.cpu())

    real_images = torch.cat(real_images)
    reconstructed_images = torch.cat(reconstructed_images)

    return (real_images, reconstructed_images)

    
def main():
    config = load_config()
    set_seed(config.get("seed", 42))
    print("Metrics for model DCGAN:")

    results_dir = resolve_path(config["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_dim = config["latent_dim"]
    checkpoint_path = results_dir / "checkpoint.pth"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    G = load_checkpoint(checkpoint_path, device, config)
    G.eval()

    dataset = HipMRIDataset(
        config.get("test_dir"),
        max_images=None,
        norm_image=True,
        week=config.get("week")
    )

    real_images = torch.stack(
        [dataset[i] for i in range(len(dataset))]
    ).float()

    with torch.no_grad():
        noise = torch.randn(
            len(real_images), latent_dim, 1, 1, device=device
        )
        generated_images = G(noise).cpu()

    print(f"Test images: {len(real_images)}")
    print("Real:", diversity_stats(real_images))
    print("Diversity:", diversity_stats(generated_images))

    compute_fid(real_images, generated_images)


if __name__ == "__main__":
    main()