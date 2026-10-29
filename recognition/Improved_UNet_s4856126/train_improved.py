"""Train Improved U-Net using the same pipeline as the baseline."""

import train
from improved_modules import ImprovedUNet2D

# Replace the baseline architecture with the residual U-Net.
# Keep the same dataset, optimizer, loss, and evaluation logic.
train.UNet2D = ImprovedUNet2D

if __name__ == "__main__":
    train.main()