"""
Author: Lawrence Tiang  
Student ID: s4892210  

Description:
    Calculate the mean and standard deviation of pixel intensities
    across the ADNI training dataset. Images from the AD and NC
    classes are converted to grayscale and their pixel intensities
    are scaled to [0, 1]. The calculated statistics are used to
    normalise images during model training and evaluation.

Usage:
    python util.py
"""

from pathlib import Path
from PIL import Image
import numpy as np

ROOT = Path("/home/groups/comp3710/ADNI/AD_NC/train")

def calculate_mean_std(root):
    """
    Calculate the mean and standard deviation of the training dataset.

    Args:
        root (Path): Path to the root directory containing
        the AD and NC training image folders.

    Returns:
        tuple[float, float]: The mean and standard deviation of all
        pixel intensities after scaling them to the range [0, 1].
    """
    # Accumulate the pixel intensities and number of
    # pixels across the entire training dataset.
    pixel_sum = 0.0
    pixel_squared_sum = 0.0
    pixel_count = 0

    # Process images from both diagnostic classes.
    for class_name in ["NC", "AD"]:
        class_dir = root / class_name

        # Process each supported image file in a deterministic order.
        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue

            # Convert the image to single-channel grayscale.
            image = Image.open(image_path).convert("L")

            # Scale pixel intensities from [0, 255] to [0, 1].
            pixels = (np.asarray(image, dtype=np.float64) / 255.0)
            pixel_sum += pixels.sum()
            pixel_squared_sum += (pixels ** 2).sum()
            pixel_count += pixels.size

    # Calculate the mean and standard deviation pixel intensity.
    mean = pixel_sum / pixel_count
    variance = (pixel_squared_sum / pixel_count - mean ** 2)
    std = np.sqrt(variance)

    return mean, std


def main():
    """
    Calculate the mean and standard deviation
    of the ADNI training dataset.
    """
    mean, std = calculate_mean_std(ROOT)
    print(f"Mean: {mean:.6f}")
    print(f"Std : {std:.6f}")


if __name__ == "__main__":
    main()