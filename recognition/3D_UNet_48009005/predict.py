import time
import numpy as np

import torch
from torch.utils.data import DataLoader

from dataset import *
from modules import UNet2D, UNet3D
from metrics import calculate_dice_iou

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def evaluate_2d(model_path=None, slice_batch_size=8):
    """
    Evaluate the trained 2D U-Net on the held-out HipMRI test set.

    Reports per-class Dice and IoU, average inference time per slice,
    and peak GPU memory usage.
    """

    # Use the saved best model by default
    if model_path is None:
        model_path = PROJECT_DIR / "best_2d_unet_aligned.pth"

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    # Use the same test patients as the 3D U-Net
    pairs = find_volume_pairs()
    _, _, test_patients = split_patients(pairs)

    test_pairs = [
        pair for pair in pairs
        if pair["patient"] in test_patients
    ]

    # Load the same preprocessed volumes as the 3D model
    test_dataset = HipMRI3DDataset(test_pairs)

    # Batch size 1 is required because slice sizes vary
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False
    )

    print(f"Test patients: {len(test_patients)}")
    print(f"Test volumes: {len(test_dataset)}")

    # Create the same model architecture used during training
    model = UNet2D(
        in_channels=1,
        num_classes=6
    ).to(device)

    # Load the trained model weights
    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )

    model.eval()

    # Count actual and predicted voxels across all test volumes
    confusion_total = torch.zeros((6, 6), dtype=torch.long)

    inference_times = []

    # Warm up the GPU before timing
    if len(test_dataset) > 0:
        image, _, _ = test_dataset[0]
        
        # Convert the volume into a batch of 2D slices
        image = image.permute(1, 0, 2, 3)[:slice_batch_size].to(device)

        with torch.no_grad():
            model(image)

        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

    # Evaluate without calculating gradients
    with torch.no_grad():

        for images, masks, _ in test_loader:

            # Convert the 3D volume into individual 2D slices
            images = images[0].permute(1, 0, 2, 3).to(device)
            masks = masks[0].to(device)

            # Process the slices in smaller batches
            for start in range(0, len(images), slice_batch_size):
                batch_images = images[start:start + slice_batch_size]
                batch_masks = masks[start:start + slice_batch_size]
                # Wait for previous GPU operations before timing
                if device.type == "cuda":
                    torch.cuda.synchronize()

                start_time = time.perf_counter()

                # Forward pass
                outputs = model(batch_images)

                if device.type == "cuda":
                    torch.cuda.synchronize()

                end_time = time.perf_counter()

                inference_times.append(
                    end_time - start_time
                )

                # Choose the most likely class at each pixel
                predictions = torch.argmax(
                    outputs,
                    dim=1
                )

                # Count actual and predicted labels for this batch
                confusion = torch.bincount((batch_masks * 6 + predictions).reshape(-1),  minlength=36).reshape(6, 6).cpu()

                # Add the counts to the whole test dataset
                confusion_total += confusion

    print("\n2D U-Net Test Results")

    dice_scores = []
    iou_scores = []

    # Calculate Dice and IoU across all test voxels
    for class_id in range(1, 6):

        intersection = confusion_total[class_id, class_id].item()
        true_count = confusion_total[class_id, :].sum().item()
        pred_count = confusion_total[:, class_id].sum().item()

        # Skip classes absent from both prediction and ground truth
        if true_count + pred_count == 0:
            continue

        dice = 2 * intersection / (true_count + pred_count)

        union = true_count + pred_count - intersection
        iou = intersection / union

        dice_scores.append(dice)
        iou_scores.append(iou)

        print(
            f"Class {class_id}: "
            f"Dice = {dice:.4f}, "
            f"IoU = {iou:.4f}"
        )

    # Mean across foreground classes
    if dice_scores:
        mean_dice = sum(dice_scores) / len(dice_scores)
        mean_iou = sum(iou_scores) / len(iou_scores)

        print(f"\nMean Dice: {mean_dice:.4f}")
        print(f"Mean IoU: {mean_iou:.4f}")
        
    # Average inference time for one 2D slice
    if inference_times:

        average_time = (sum(inference_times) / (len(test_dataset) * test_dataset.target_size[0]))

        print(
            f"Average inference time per slice: "
            f"{average_time:.6f} seconds"
        )

    # Report peak GPU memory
    if device.type == "cuda":

        peak_memory = (
            torch.cuda.max_memory_allocated()
            / (1024 ** 3)
        )

        print(
            f"Peak GPU memory: "
            f"{peak_memory:.3f} GB"
        )


def visualise_3d():
    """
    Visualise the weighted 3D U-Net prediction
    using a validation MRI volume.
    """

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Load the saved weighted model
    model = UNet3D(in_channels=1, num_classes=6).to(device)

    checkpoint = PROJECT_DIR / "best_3d_unet_weighted.pth"

    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True)
    )

    model.eval()

    # Use validation patients, not the held-out test patients
    pairs = find_volume_pairs()
    _, val_patients, _ = split_patients(pairs)

    val_pairs = [pair for pair in pairs if pair["patient"] in val_patients]

    dataset = HipMRI3DDataset(val_pairs)

    # Find a validation slice containing Class 5
    selected = None

    for index in range(len(dataset)):
        image, mask, _ = dataset[index]

        class_counts = (mask == 5).sum(dim=(1, 2))

        if class_counts.max().item() > 0:
            slice_index = class_counts.argmax().item()
            selected = (image, mask, val_pairs[index])
            break

    if selected is None:
        raise ValueError("No Class 5 validation slice found")

    image, mask, pair = selected

    # Predict the segmentation for the full 3D volume
    with torch.no_grad():
        output = model(image.unsqueeze(0).to(device))
        prediction = output.argmax(dim=1).squeeze(0).cpu()

    # Show the same slice across all three images
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(
        image[0, slice_index].numpy(),
        cmap="gray"
    )
    axes[0].set_title("MRI Image")

    axes[1].imshow(
        mask[slice_index].numpy(),
        cmap="tab10",
        vmin=-0.5,
        vmax=5.5
    )
    axes[1].set_title("Ground Truth")

    seg_image = axes[2].imshow(
        prediction[slice_index].numpy(),
        cmap="tab10",
        vmin=-0.5,
        vmax=5.5
    )
    axes[2].set_title("3D U-Net Prediction")

    for ax in axes:
        ax.axis("off")

    fig.colorbar(
        seg_image,
        ax=axes[1:],
        ticks=range(6),
        shrink=0.75
    )

    fig.suptitle(
        f"{pair['mr'].name} - Slice {slice_index}"
    )

    # Save the visualisation
    output_dir = PROJECT_DIR / "outputs"
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "3d_validation_prediction.png"

    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Visualisation saved: {output_path}")

def visualise_3d_overlay():
    """
    Visualise the weighted 3D U-Net prediction as an overlay
    on top of the MRI image using a validation MRI volume.
    """

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Load the saved weighted 3D U-Net
    model = UNet3D(
        in_channels=1,
        num_classes=6
    ).to(device)

    checkpoint = PROJECT_DIR / "best_3d_unet_weighted.pth"

    model.load_state_dict(
        torch.load(
            checkpoint,
            map_location=device,
            weights_only=True
        )
    )

    model.eval()

    # Use validation patients only
    pairs = find_volume_pairs()
    _, val_patients, _ = split_patients(pairs)

    val_pairs = [
        pair for pair in pairs
        if pair["patient"] in val_patients
    ]

    dataset = HipMRI3DDataset(val_pairs)

    # Choose a validation volume with Class 5 present
    selected = None

    for index in range(len(dataset)):
        image, mask, _ = dataset[index]

        class5_counts = (mask == 5).sum(dim=(1, 2))

        if class5_counts.max().item() > 0:
            slice_index = class5_counts.argmax().item()
            selected = (image, mask, val_pairs[index], slice_index)
            break

    if selected is None:
        raise ValueError("No validation slice containing Class 5 was found")

    image, mask, pair, slice_index = selected

    # Predict segmentation
    with torch.no_grad():
        output = model(image.unsqueeze(0).to(device))
        prediction = torch.argmax(output, dim=1).squeeze(0).cpu()

    # MRI slice
    mri_slice = image[0, slice_index].numpy()
    mask_slice = mask[slice_index].numpy()
    pred_slice = prediction[slice_index].numpy()

    # Make label 0 transparent, colours for labels 1-5
    overlay_cmap = ListedColormap([
        (0, 0, 0, 0),      # class 0 transparent
        (0, 1, 0, 0.45),   # class 1 green
        (0.6, 0.4, 1, 0.45), # class 2 purple
        (1, 0.4, 0.8, 0.45), # class 3 pink
        (0.7, 0.7, 0.7, 0.45), # class 4 grey
        (0, 1, 1, 0.45)    # class 5 cyan
    ])

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Ground truth overlay
    axes[0].imshow(mri_slice, cmap="gray")
    axes[0].imshow(
        mask_slice,
        cmap=overlay_cmap,
        vmin=0,
        vmax=5,
        interpolation="nearest"
    )
    axes[0].set_title("MRI + Ground Truth")
    axes[0].axis("off")

    # Prediction overlay
    axes[1].imshow(mri_slice, cmap="gray")
    axes[1].imshow(
        pred_slice,
        cmap=overlay_cmap,
        vmin=0,
        vmax=5,
        interpolation="nearest"
    )
    axes[1].set_title("MRI + 3D U-Net Prediction")
    axes[1].axis("off")

    fig.suptitle(
        f"{pair['mr'].name} - Slice {slice_index}"
    )

    output_dir = PROJECT_DIR / "outputs"
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "3d_validation_overlay.png"

    plt.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight"
    )
    plt.close(fig)

    print(f"Overlay visualisation saved: {output_path}")


def evaluate_3d(model_path=None):
    """
    Evaluate the weighted 3D U-Net on held-out test patients.

    Reports Dice, IoU, inference time, GPU memory
    and model parameter count.
    """

    # Use the saved weighted model by default
    if model_path is None:
        model_path = PROJECT_DIR / "best_3d_unet_weighted.pth"

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # Select test patients using the original split
    pairs = find_volume_pairs()
    _, _, test_patients = split_patients(pairs)

    test_pairs = [
        pair for pair in pairs
        if pair["patient"] in test_patients
    ]

    if not test_pairs:
        raise ValueError("No test volumes found")

    # Create the test dataset
    test_dataset = HipMRI3DDataset(test_pairs)

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False
    )

    print(f"Test patients: {len(test_patients)}")
    print(f"Test volumes: {len(test_dataset)}")
    print(f"Patient IDs: {test_patients}")

    # Load the trained 3D U-Net
    model = UNet3D(
        in_channels=1,
        num_classes=6
    ).to(device)

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device,
            weights_only=True
        )
    )

    model.eval()

    # Count all model parameters
    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(f"Model parameters: {parameter_count:,}")

    # Store prediction results for all test volumes
    confusion_total = torch.zeros(
        (6, 6), dtype=torch.long
    )

    inference_times = []
    case_results = []

    # Warm up the GPU before measuring inference time
    if device.type == "cuda":
        with torch.inference_mode():
            warmup = torch.zeros(
                (1, 1, *test_dataset.target_size),
                device=device
            )
            model(warmup)
            torch.cuda.synchronize()
            del warmup

        torch.cuda.reset_peak_memory_stats()

    # Evaluate without updating model weights
    with torch.inference_mode():
        for index, (images, masks, _) in enumerate(test_loader):

            images = images.to(device)
            masks = masks.to(device)

            # Synchronise GPU before timing
            if device.type == "cuda":
                torch.cuda.synchronize()

            start_time = time.perf_counter()

            # Forward pass
            outputs = model(images)

            if device.type == "cuda":
                torch.cuda.synchronize()

            inference_times.append(
                time.perf_counter() - start_time
            )

            # Choose the predicted class at each voxel
            predictions = torch.argmax(outputs, dim=1)

            # Count predicted and true class combinations
            confusion = torch.bincount(
                masks.reshape(-1) * 6 + predictions.reshape(-1),
                minlength=36
            ).reshape(6, 6).cpu()

            confusion_total += confusion

            # Calculate mean Dice for this test volume
            case_dice = []

            for class_id in range(1, 6):
                intersection = confusion[class_id, class_id].item()
                true_count = confusion[class_id, :].sum().item()
                pred_count = confusion[:, class_id].sum().item()

                if true_count + pred_count > 0:
                    dice = (
                        2 * intersection / (true_count + pred_count)
                    )
                    case_dice.append(dice)

            mean_case_dice = (
                sum(case_dice) / len(case_dice)
                if case_dice else 0.0
            )

            filename = test_pairs[index]["mr"].name
            case_results.append((mean_case_dice, filename))

            print(
                f"{filename}: Mean Dice = {mean_case_dice:.4f}"
            )

    print("\n3D U-Net Test Results")

    dice_scores = []
    iou_scores = []

    # Calculate metrics across all test voxels
    for class_id in range(1, 6):
        intersection = confusion_total[class_id, class_id].item()
        true_count = confusion_total[class_id, :].sum().item()
        pred_count = confusion_total[:, class_id].sum().item()

        if true_count + pred_count == 0:
            print(f"Class {class_id}: Absent from test predictions and labels")
            continue

        dice = 2 * intersection / (true_count + pred_count)

        union = true_count + pred_count - intersection
        iou = intersection / union

        dice_scores.append(dice)
        iou_scores.append(iou)

        print(
            f"Class {class_id}: "
            f"Dice = {dice:.4f}, "
            f"IoU = {iou:.4f}"
        )

    # Mean across foreground classes
    if dice_scores:
        mean_dice = sum(dice_scores) / len(dice_scores)
        mean_iou = sum(iou_scores) / len(iou_scores)

        print(f"\nMean Dice: {mean_dice:.4f}")
        print(f"Mean IoU: {mean_iou:.4f}")

    # Average forward-pass time per 3D volume
    average_time = sum(inference_times) / len(inference_times)

    print(
        f"Average inference time per volume: "
        f"{average_time:.4f} seconds"
    )

    # Peak GPU memory during evaluation
    if device.type == "cuda":
        peak_memory = (
            torch.cuda.max_memory_allocated() / (1024 ** 3)
        )

        print(f"Peak GPU memory: {peak_memory:.3f} GB")

    # Identify volumes for later error analysis
    print("\nLowest-scoring test volumes:")

    for dice, filename in sorted(case_results)[:5]:
        print(f"{filename}: Mean Dice = {dice:.4f}")


def visualise_boundary_errors(
    case_name="C032_Week0_LFOV.nii.gz",
    class_id=5,
    slice_index=None,
    slice_batch_size=8,
    slice_mode="max_error"):
    """
    Compare 2D and 3D segmentation errors on the same MRI slice.
    Red shows false positives and blue shows false negatives.
    
    Assisted by ChatGPT model 6 with matplot code
    """

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Select one volume from the held-out test patients
    pairs = find_volume_pairs()
    _, _, test_patients = split_patients(pairs)

    selected_pairs = [
        pair for pair in pairs
        if pair["patient"] in test_patients
        and pair["mr"].name == case_name
    ]

    if not selected_pairs:
        raise ValueError(f"Test volume not found: {case_name}")

    image, mask, _ = HipMRI3DDataset(selected_pairs)[0]

    # Load both trained models
    model_2d = UNet2D(in_channels=1, num_classes=6).to(device)
    model_3d = UNet3D(in_channels=1, num_classes=6).to(device)

    model_2d.load_state_dict(
        torch.load(
            PROJECT_DIR / "best_2d_unet_aligned.pth",
            map_location=device,
            weights_only=True
        )
    )

    model_3d.load_state_dict(
        torch.load(
            PROJECT_DIR / "best_3d_unet_weighted.pth",
            map_location=device,
            weights_only=True
        )
    )

    model_2d.eval()
    model_3d.eval()

    with torch.inference_mode():

        # 2D model: predict each slice in small batches
        slices = image.permute(1, 0, 2, 3)
        predictions_2d = []

        for start in range(0, len(slices), slice_batch_size):
            batch = slices[start:start + slice_batch_size].to(device)

            output = model_2d(batch)
            predictions_2d.append(output.argmax(dim=1).cpu())

        prediction_2d = torch.cat(predictions_2d, dim=0)

        # 3D model: predict the whole volume
        output = model_3d(image.unsqueeze(0).to(device))
        prediction_3d = output.argmax(dim=1).squeeze(0).cpu()

    
    # Select a slice for boundary analysis
    if slice_index is None:
        target_class = mask == class_id

        # Find slices containing the selected class
        present_slices = torch.where(target_class.any(dim=(1, 2)))[0]

        if len(present_slices) == 0:
            raise ValueError(f"Class {class_id} is absent from this volume")

        if slice_mode == "first":
            slice_index = present_slices[0].item()

        elif slice_mode == "last":
            slice_index = present_slices[-1].item()

        elif slice_mode == "max_error":
            # Original behaviour: choose the slice with most errors
            errors_2d = ((prediction_2d == class_id) != target_class).sum(dim=(1, 2))

            errors_3d = ((prediction_3d == class_id) != target_class).sum(dim=(1, 2))

            combined_errors = errors_2d + errors_3d

            slice_index = present_slices[combined_errors[present_slices].argmax()].item()

        else:
            raise ValueError(
                "slice_mode must be 'first', 'last', or 'max_error'"
            )

    if not 0 <= slice_index < mask.shape[0]:
        raise ValueError("Slice index is outside the volume")

    # Select the same slice from all four volumes
    mri_slice = image[0, slice_index].numpy()
    ground_truth = mask[slice_index].numpy()
    pred_2d = prediction_2d[slice_index].numpy()
    pred_3d = prediction_3d[slice_index].numpy()

    # Use the same colours as the earlier 3D overlay
    overlay_cmap = ListedColormap([
        (0, 0, 0, 0),
        (0, 1, 0, 0.45),
        (0.6, 0.4, 1, 0.45),
        (1, 0.4, 0.8, 0.45),
        (0.7, 0.7, 0.7, 0.45),
        (0, 1, 1, 0.45)
    ])

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Original MRI
    axes[0, 0].imshow(mri_slice, cmap="gray")
    axes[0, 0].set_title("Original MRI")

    # Show ground truth and both predictions on the MRI
    panels = [
        (axes[0, 1], ground_truth, "Ground Truth"),
        (axes[0, 2], pred_2d, "2D U-Net Prediction"),
        (axes[1, 0], pred_3d, "3D U-Net Prediction")
    ]

    for ax, labels, title in panels:
        ax.imshow(mri_slice, cmap="gray")
        ax.imshow(
            labels,
            cmap=overlay_cmap,
            vmin=0,
            vmax=5,
            interpolation="nearest"
        )
        ax.set_title(title)

    # Highlight false positives and false negatives
    error_panels = [
        (axes[1, 1], pred_2d, "2D U-Net Errors"),
        (axes[1, 2], pred_3d, "3D U-Net Errors")
    ]

    for ax, prediction, title in error_panels:

        # False positive: predicted class is not in ground truth
        false_positive = (
            (prediction == class_id)
            & (ground_truth != class_id)
        )

        # False negative: ground-truth class was missed
        false_negative = (
            (prediction != class_id)
            & (ground_truth == class_id)
        )

        # Create a transparent coloured error image
        error_overlay = np.zeros(
            (*ground_truth.shape, 4),
            dtype=np.float32
        )

        error_overlay[false_positive] = [1, 0, 0, 0.65]
        error_overlay[false_negative] = [0, 0.5, 1, 0.65]

        ax.imshow(mri_slice, cmap="gray")
        ax.imshow(error_overlay, interpolation="nearest")
        ax.set_title(title)

        # Report the errors for this slice
        print(
            f"{title}: "
            f"FP = {false_positive.sum()}, "
            f"FN = {false_negative.sum()}"
        )

    for ax in axes.flat:
        ax.axis("off")

    fig.suptitle(
        f"{case_name} | Slice {slice_index} | Class {class_id}\n"
        "Red = False Positive | Blue = False Negative"
    )

    fig.tight_layout()

    # Save the comparison for the README
    output_dir = PROJECT_DIR / "outputs"
    output_dir.mkdir(exist_ok=True)

    short_name = case_name.replace("_LFOV.nii.gz", "")

    output_path = (
        output_dir /
        f"errors_{short_name}_class{class_id}_slice{slice_index}.png"
    )

    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Boundary comparison saved: {output_path}")

if __name__ == "__main__":
    evaluate_2d()