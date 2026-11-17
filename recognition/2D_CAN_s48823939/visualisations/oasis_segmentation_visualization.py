import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

from modules import ContextAggregationModule, Simple2DUNet
from dataset import get_oasis_dataloaders

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "OASIS")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "visualisations")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Custom color palette for OASIS 4-class segmentation
# 0: Background (Transparent), 1: CSF / Class 1, 2: Gray Matter / Class 2, 3: White Matter / Class 3
CLASS_COLORS = ['none', '#e6194b', '#3cb44b', '#4363d8']
CLASS_NAMES = ['Background', 'Class 1', 'Class 2', 'Class 3']


def create_overlay_cmap():
    """Returns a ListedColormap for discrete class segmentation masks."""
    return ListedColormap(CLASS_COLORS)


def visualise_samples(model=None, dataloader=None, device='cpu', num_samples=3, save_filename='mri_segmentations.png'):
    """
    Extracts samples from the dataloader and generates side-by-side plots of:
    1. Raw Grayscale Input MRI
    2. Ground Truth Segmentation Mask
    3. Model Predicted Segmentation Mask (if model is provided)
    """
    images, targets = next(iter(dataloader))
    num_samples = min(num_samples, images.size(0))

    if model is not None:
        model.eval()
        with torch.no_grad():
            outputs = model(images[:num_samples].to(device))
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
        ncols = 3
    else:
        preds = None
        ncols = 2

    fig, axes = plt.subplots(num_samples, ncols, figsize=(4 * ncols, 4 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    cmap = create_overlay_cmap()

    for idx in range(num_samples):
        img = images[idx, 0].numpy()
        gt = targets[idx].numpy()

        # 1. Input MRI Slice
        axes[idx, 0].imshow(img, cmap='gray')
        axes[idx, 0].set_title(f"Sample {idx + 1}: Raw MRI Input")
        axes[idx, 0].axis('off')

        # 2. Ground Truth Overlay
        axes[idx, 1].imshow(img, cmap='gray')
        axes[idx, 1].imshow(gt, cmap=cmap, alpha=0.5, vmin=0, vmax=3)
        axes[idx, 1].set_title(f"Sample {idx + 1}: Ground Truth Mask")
        axes[idx, 1].axis('off')

        # 3. Model Prediction Overlay (if available)
        if model is not None and preds is not None:
            pred = preds[idx]
            axes[idx, 2].imshow(img, cmap='gray')
            axes[idx, 2].imshow(pred, cmap=cmap, alpha=0.5, vmin=0, vmax=3)
            axes[idx, 2].set_title(f"Sample {idx + 1}: Model Prediction")
            axes[idx, 2].axis('off')

    # Add legend
    patches = [mpatches.Patch(color=CLASS_COLORS[i], label=CLASS_NAMES[i]) for i in range(1, len(CLASS_COLORS))]
    fig.legend(handles=patches, loc='lower center', ncol=len(patches), bbox_to_anchor=(0.5, -0.02), frameon=True)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, save_filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"--> Saved visual output to: {save_path}")


def load_model_weights(model, weights_path, device='cpu'):
    """Loads state dictionary into model if file exists."""
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=device)
        if 'model_state' in state_dict:
            model.load_state_dict(state_dict['model_state'])
        else:
            model.load_state_dict(state_dict)
        print(f"Loaded model weights from {weights_path}")
    else:
        print(f"Weights path {weights_path} not found. Running with uninitialized weights.")
    return model


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataloaders
    _, val_loader, _, num_classes = get_oasis_dataloaders(DATA_DIR, batch_size=8)

    # Initialize model (CAN model or U-Net)
    model = ContextAggregationModule(in_channels=1, num_classes=num_classes, C=64).to(device)

    # Optionally load weights if available
    weights_file = os.path.join(SCRIPT_DIR, "best_can_model.pth")
    model = load_model_weights(model, weights_file, device=device)

    # Render visualisations
    visualise_samples(
        model=model,
        dataloader=val_loader,
        device=device,
        num_samples=4,
        save_filename="oasis_can_predictions.png"
    )