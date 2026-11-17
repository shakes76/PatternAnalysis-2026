import os
import sys

# Add the parent directory (PatternAnalysis-2026) to Python's search path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(PARENT_DIR)

import matplotlib.pyplot as plt
import numpy as np
from dataset import get_oasis_dataloaders

# Setup path to your OASIS dataset
DATA_DIR = os.path.join(PARENT_DIR, "OASIS")

def show_colored_brain():
    # Load dataloader with batch size 1 to grab a single brain
    _, val_loader, _, _ = get_oasis_dataloaders(DATA_DIR, batch_size=1)
    
    images, targets = next(iter(val_loader))
    brain_image = images[0, 0].cpu().numpy()  # Shape: [160, 192]
    brain_mask = targets[0].cpu().numpy()     # Shape: [160, 192]
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # 1. Plot the grayscale brain MRI underneath
    ax.imshow(brain_image, cmap='gray')
    
    # 2. Hide the background (class 0) so only the tissue regions show colors
    masked_mask = np.ma.masked_where(brain_mask == 0, brain_mask)
    
    # 3. Overlay the color-coded mask with 50% transparency (alpha)
    ax.imshow(masked_mask, cmap='nipy_spectral', alpha=0.5)
    
    # ax.set_title("Single Brain MRI with Segmentation Overlay")
    ax.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    show_colored_brain()