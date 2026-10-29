import torch
import matplotlib.pyplot as plt
import numpy as np
from dataset import get_dataloaders
from modules import UNet2D, UNet3D

def calculate_dice(y_true, y_pred, class_index, smooth=1e-5):
    """Calculate Dice coefficient for a specific class."""
    y_true_c = (y_true == class_index).astype(np.float32)
    y_pred_c = (y_pred == class_index).astype(np.float32)

    intersection = np.sum(y_true_c * y_pred_c)
    denominator = np.sum(y_true_c) + np.sum(y_pred_c)

    return (2.0 * intersection + smooth) / (denominator + smooth)

def evaluate_and_visualize(base_data_dir, weights_path):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load test data and model
    _, _, test_loader = get_dataloaders(base_data_dir, batch_size=4)
    model = UNet2D(in_channels=1, out_classes=6).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"2D U-Net Total Trainable Parameters: {total_params:,}")

    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()

    # Fetch one test batch
    images, masks = next(iter(test_loader))

    # Crop the tensors to 256x128 BEFORE passing them to the model
    images = images[:, :, :, :128].to(device)
    masks = masks[:, :, :, :128].to(device)

    images = images.to(device)

    with torch.no_grad():
        outputs = model(images)
        # Convert logits to class predictions (argmax across channel dimension)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()

    images = images.cpu().numpy()
    masks = torch.argmax(masks, dim=1).cpu().numpy()

    # Visualize the first sample in the batch[cite: 5, 13]
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("MRI Input")
    plt.imshow(images[0, 0], cmap='gray')

    plt.subplot(1, 3, 2)
    plt.title("Ground Truth Mask")
    plt.imshow(masks[0], cmap='nipy_spectral')

    plt.subplot(1, 3, 3)
    plt.title("Model Prediction")
    plt.imshow(preds[0], cmap='nipy_spectral')

    plt.tight_layout()
    plt.savefig("2d_prediction_output.png", bbox_inches='tight', dpi=300)
    plt.show()

def evaluate_patient_volume(patient_loader, weights_path, target_class_idx=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = UNet2D(in_channels=1, out_classes=6).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()

    all_preds = []
    all_masks = []

    with torch.no_grad():
        for images, masks in patient_loader:  # Assumes loader yields ordered slices for 1 patient
            images = images[:, :, :, :128].to(device)
            masks = masks[:, :, :, :128].numpy()

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            all_preds.append(preds)
            masks_argmax = np.argmax(masks, axis=1)
            all_masks.append(masks_argmax)

    # Stack 2D slices into a 3D volume: (Z, H, W)
    volume_preds = np.concatenate(all_preds, axis=0)
    volume_masks = np.concatenate(all_masks, axis=0)

    # Global 3D Dice Score
    global_dice = calculate_dice(volume_masks, volume_preds, class_index=target_class_idx)
    print(f"Global 3D Target Dice Score: {global_dice:.4f}")

    # Per-Slice Dice Score (for Base, Mid, Apex analysis)
    slice_dices = []
    for z in range(volume_preds.shape[0]):
        # Only evaluate slices where the ground truth contains the target class
        if np.any(volume_masks[z] == target_class_idx):
            sdice = calculate_dice(volume_masks[z], volume_preds[z], class_index=target_class_idx)
            slice_dices.append((z, sdice))

    if slice_dices:
        print(f"Base Slice Dice: {slice_dices[0][1]:.4f}")
        print(f"Mid-Volume Slice Dice: {slice_dices[len(slice_dices) // 2][1]:.4f}")
        print(f"Apex Slice Dice: {slice_dices[-1][1]:.4f}")

    return volume_preds, volume_masks


def evaluate_patient_volume_3d(test_loader_3d, weights_path, target_class_idx=5):
    """
    Evaluates a 3D U-Net on a full patient volume, extracting global and per-slice metrics
    to investigate the base/apex boundary dilemma.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = UNet3D(in_channels=1, out_classes=6, base_filters=16).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"3D U-Net Total Trainable Parameters: {total_params:,}")

    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()

    with torch.no_grad():
        for images, masks in test_loader_3d:
            images = images.to(device)
            masks = masks.numpy()

            outputs = model(images)

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            masks_argmax = np.argmax(masks, axis=1)

            volume_preds = preds[0]
            volume_masks = masks_argmax[0]

            global_dice = calculate_dice(volume_masks, volume_preds, class_index=target_class_idx)
            print(f"Global 3D Target Dice Score: {global_dice:.4f}")

            slice_dices = []
            for z in range(volume_preds.shape[0]):
                if np.any(volume_masks[z] == target_class_idx):
                    sdice = calculate_dice(volume_masks[z], volume_preds[z], class_index=target_class_idx)
                    slice_dices.append((z, sdice))

            if slice_dices:
                print(f"Base Slice Dice (Superior): {slice_dices[0][1]:.4f}")
                print(f"Mid-Volume Slice Dice (Central): {slice_dices[len(slice_dices) // 2][1]:.4f}")
                print(f"Apex Slice Dice (Inferior): {slice_dices[-1][1]:.4f}")

            mid_z = slice_dices[len(slice_dices) // 2][0] if slice_dices else volume_preds.shape[0] // 2

            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.title(f"3D MRI Input (Slice {mid_z})")
            plt.imshow(images[0, 0, mid_z].cpu().numpy(), cmap='gray')

            plt.subplot(1, 3, 2)
            plt.title(f"Ground Truth Mask (Slice {mid_z})")
            plt.imshow(volume_masks[mid_z], cmap='nipy_spectral')

            plt.subplot(1, 3, 3)
            plt.title(f"3D Model Prediction (Slice {mid_z})")
            plt.imshow(volume_preds[mid_z], cmap='nipy_spectral')
            
            plt.tight_layout()
            plt.savefig(f"3d_prediction_slice_{mid_z}.png", bbox_inches='tight', dpi=300)
            plt.show()
            break

    return volume_preds, volume_masks

if __name__ == "__main__":
    data_dir = './training data/keras_slices_data'
    evaluate_and_visualize(data_dir, "unet2d_baseline.pth")
    evaluate_patient_volume_3d('./training data', 'unet3d_best.pth')