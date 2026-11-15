import os
import torch
import argparse
import time
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from dataset import get_dataloaders, HipMRIDataset, HipMRI3DDataset
from modules import UNet2D, UNet3D
from train import get_file_paths

def calculate_dice(y_true, y_pred, class_index, smooth=1e-5):
    """Calculate Dice coefficient for a specific class."""
    y_true_c = (y_true == class_index).astype(np.float32)
    y_pred_c = (y_pred == class_index).astype(np.float32)

    intersection = np.sum(y_true_c * y_pred_c)
    denominator = np.sum(y_true_c) + np.sum(y_pred_c)

    return (2.0 * intersection + smooth) / (denominator + smooth)


def calculate_per_class_metrics(y_true, y_pred, num_classes=6, smooth=1e-5):
    """Calculates and returns Dice and IoU arrays for every class."""
    dices, ious = [], []
    for c in range(num_classes):
        y_true_c = (y_true == c).astype(np.float32)
        y_pred_c = (y_pred == c).astype(np.float32)

        intersection = np.sum(y_true_c * y_pred_c)
        sum_both = np.sum(y_true_c) + np.sum(y_pred_c)
        union = sum_both - intersection

        dice = (2.0 * intersection + smooth) / (sum_both + smooth)
        iou = (intersection + smooth) / (union + smooth)
        dices.append(dice)
        ious.append(iou)

    return np.array(dices), np.array(ious)

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
    images, masks, _ = next(iter(test_loader))

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


def evaluate_patient_volume(test_loader, weights_path, target_class_idx=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = UNet2D(in_channels=1, out_classes=6).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()

    patient_preds = {}
    patient_masks = {}

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    total_inference_time = 0.0

    with torch.no_grad():
        for images, masks, filenames in test_loader:
            images = images[:, :, :, :128].to(device)
            masks = masks[:, :, :, :128].numpy()

            start_time = time.time()
            outputs = model(images)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_inference_time += (time.time() - start_time)

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            masks_argmax = np.argmax(masks, axis=1)

            # Group slices by patient ID using the filename
            for i, filename in enumerate(filenames):
                patient_id = "_".join(filename.split("_")[:2])
                if patient_id not in patient_preds:
                    patient_preds[patient_id] = []
                    patient_masks[patient_id] = []

                patient_preds[patient_id].append(preds[i])
                patient_masks[patient_id].append(masks_argmax[i])

    # Tracking variables for final averages
    all_patient_dices = []
    all_patient_ious = []
    all_base = []
    all_mid = []
    all_apex = []

    # Evaluate each patient individually
    for patient_id in patient_preds.keys():
        volume_preds = np.stack(patient_preds[patient_id], axis=0)
        volume_masks = np.stack(patient_masks[patient_id], axis=0)

        # Get 6-class metrics and store them
        dices, ious = calculate_per_class_metrics(volume_masks, volume_preds, num_classes=6)
        all_patient_dices.append(dices)
        all_patient_ious.append(ious)

        # Calculate Base, Mid, and Apex scores and store them
        slice_dices = []
        for z in range(volume_preds.shape[0]):
            if np.any(volume_masks[z] == target_class_idx):
                sdice = calculate_dice(volume_masks[z], volume_preds[z], class_index=target_class_idx)
                slice_dices.append((z, sdice))

        if slice_dices:
            all_base.append(slice_dices[0][1])
            all_mid.append(slice_dices[len(slice_dices) // 2][1])
            all_apex.append(slice_dices[-1][1])

    # --- Print Final Averages ---
    avg_dices = np.mean(all_patient_dices, axis=0)
    avg_ious = np.mean(all_patient_ious, axis=0)
    class_names = ["Background", "Body Outline", "Bone", "Bladder", "Rectum", "Prostate"]

    print("\n=== FINAL 2D TEST SET AVERAGES ===")
    print(f"{'Anatomy (Class)':<18} | {'Avg Dice':<8} | {'Avg IoU':<8}")
    print("-" * 40)
    for c in range(6):
        print(f"{class_names[c]:<18} | {avg_dices[c]:<8.4f} | {avg_ious[c]:<8.4f}")

    print("\n--- Boundary Metrics ---")
    if all_base: print(f"Average Base Slice Dice: {np.mean(all_base):.4f}")
    if all_mid: print(f"Average Mid-Volume Slice Dice: {np.mean(all_mid):.4f}")
    if all_apex: print(f"Average Apex Slice Dice: {np.mean(all_apex):.4f}")

    print("\n--- 2D Global Resource Profiling ---")
    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        print(f"2D Peak VRAM Utilized: {peak_vram:.2f}GB")

    num_patients = len(patient_preds)
    latency_per_volume = total_inference_time / num_patients if num_patients > 0 else 0
    print(f"2D Inference Latency: {latency_per_volume:.4f}s per patient volume")

    return patient_preds, patient_masks


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

    total_global_dice = 0.0
    num_volumes = 0

    torch.cuda.reset_peak_memory_stats(device)  # Reset VRAM tracker
    all_latencies = []

    with torch.no_grad():
        for images, masks in test_loader_3d:
            images = images.to(device)
            masks = masks.numpy()

            start_time = time.time()  # Start latency timer
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            inference_time = time.time() - start_time  # End latency timer
            all_latencies.append(inference_time)
            
            masks_argmax = np.argmax(masks, axis=1)

            volume_preds = preds[0]
            volume_masks = masks_argmax[0]

            global_dice = calculate_dice(volume_masks, volume_preds, class_index=target_class_idx)
            total_global_dice += global_dice
            num_volumes += 1

            if num_volumes == 1 :
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

    # Calculate and print the true test set average
    avg_test_dice = total_global_dice / num_volumes
    avg_latency = np.mean(all_latencies)
    peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

    print("\n--- Final 3D Test Set Evaluation ---")
    print(f"Average 3D Test Set Dice Score: {avg_test_dice:.4f}")
    print(f"Average Inference Latency: {avg_latency:.4f}s per volume")
    print(f"Peak VRAM Utilized: {peak_vram:.2f}GB")

    return volume_preds, volume_masks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate 2D or 3D UNet models.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["2d", "3d", "both"],
        required=True,
        help="Evaluation mode: '2d', '3d', or 'both'",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to the data directory. Defaults to './training data/keras_slices_data' for 2d and './training data' for 3d.",
    )
    args = parser.parse_args()

    if args.mode in ["2d", "both"]:
        data_dir_2d = (
            args.data_dir
            if args.data_dir
            else "./training data/keras_slices_data"
        )
        weights_file = "unet2d_baseline.pth"

        print("--- Running 2D Visualisation ---")
        evaluate_and_visualize(data_dir_2d, weights_file)

        print("\n--- Running 2D Volumetric Evaluation ---")
        _, _, test_loader = get_dataloaders(data_dir_2d, batch_size=4)

        evaluate_patient_volume(test_loader, weights_file, target_class_idx=5)

    if args.mode in ["3d", "both"]:
        data_dir_3d = args.data_dir if args.data_dir else "./training data"
        img_dir = os.path.join(data_dir_3d, "semantic_MRs")
        mask_dir = os.path.join(data_dir_3d, "semantic_labels_only")

        all_imgs = get_file_paths(img_dir)
        all_masks = get_file_paths(mask_dir)

        test_imgs = all_imgs[189:]
        test_masks = all_masks[189:]

        test_dataset_3d = HipMRI3DDataset(test_imgs, test_masks)
        test_loader_3d = DataLoader(test_dataset_3d, batch_size=1, shuffle=False)

        evaluate_patient_volume_3d(test_loader_3d, "unet3d_best.pth")