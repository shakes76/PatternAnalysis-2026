import torch

from dataset import HipMRI3DDataset, find_volume_pairs
from modules import UNet3D


def smoke_test_3d():
    """
    Test one HipMRI volume through the 3D U-Net.
    """

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # Find the available 3D MRI/segmentation pairs
    pairs = find_volume_pairs()

    # Load the 3D dataset
    dataset = HipMRI3DDataset(pairs)

    # Load one volume
    image, mask, mask_one_hot = dataset[0]

    # Add batch dimension
    image = image.unsqueeze(0).to(device)

    print(f"Input shape: {image.shape}")

    # Create the model
    model = UNet3D(
        in_channels=1,
        num_classes=6
    ).to(device)

    model.eval()

    # Forward pass only
    with torch.no_grad():
        output = model(image)

    print(f"Output shape: {output.shape}")

    expected_shape = (
        1,
        6,
        64,
        128,
        128
    )

    assert output.shape == expected_shape

    print("3D U-Net smoke test passed")


if __name__ == "__main__":
    smoke_test_3d()