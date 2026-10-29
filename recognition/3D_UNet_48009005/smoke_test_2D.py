import torch

from dataset import HipMRI2DDataset, SLICE_DATA_DIR
from modules import UNet2D


def test_2d_unet():
    """
    Test one real HipMRI slice through the 2D U-Net.
    """

    # Load the training dataset
    dataset = HipMRI2DDataset(
        SLICE_DATA_DIR / "keras_slices_train",
        SLICE_DATA_DIR / "keras_slices_seg_train"
    )

    # Load one real MRI slice
    image, mask, mask_one_hot = dataset[0]

    # Add batch dimension
    image = image.unsqueeze(0)

    # Create model
    model = UNet2D(
        in_channels=1,
        num_classes=6
    )

    model.eval()

    # Forward pass
    with torch.no_grad():
        output = model(image)

    print(f"Input shape: {image.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"One-hot mask shape: {mask_one_hot.shape}")
    print(f"Output shape: {output.shape}")

    # Check expected output dimensions
    assert output.shape == (
        1,
        6,
        image.shape[2],
        image.shape[3]
    )

    print("\n2D U-Net smoke test passed")

if __name__ == "__main__":
    test_2d_unet()