from pathlib import Path

import nibabel as nib
import numpy as np

from scipy.ndimage import binary_erosion, distance_transform_edt

import torch

#Part 1 - Data Pipeline Verification
# Verify patient-level separation to prevent data leakage
# between training, validation, and test datasets.
def extract_patient_id(filename):
    """
    Extract patient ID from a HipMRI filename.

    Example:
        case_040_week_0_slice_38.nii.gz -> case_040
    """
    parts = Path(filename).name.split("_")

    if len(parts) < 6 or parts[0] != "case":
        raise ValueError(f"Unexpected filename format: {filename}")

    return f"{parts[0]}_{parts[1]}"


def verify_patient_split(train_dir, val_dir, test_dir):
    """
    Verify that patients do not overlap between dataset splits.
    """

    def get_patient_ids(directory):
        files = Path(directory).glob("case_*.nii.gz")
        return {extract_patient_id(file.name) for file in files}

    train_patients = get_patient_ids(train_dir)
    val_patients = get_patient_ids(val_dir)
    test_patients = get_patient_ids(test_dir)

    overlaps = {
        "Train-Val": train_patients & val_patients,
        "Train-Test": train_patients & test_patients,
        "Val-Test": val_patients & test_patients,
    }

    print("Patient-level split verification")
    print(f"Training patients: {len(train_patients)}")
    print(f"Validation patients: {len(val_patients)}")
    print(f"Test patients: {len(test_patients)}")

    for split, patients in overlaps.items():
        print(f"{split} overlap: {len(patients)}")

    return overlaps


def verify_image_mask_pairs(image_dir, mask_dir):
    """
    Verify that every MRI image has a corresponding segmentation mask
    and that no unmatched masks exist.

    Expected naming:
        Image: case_040_week_0_slice_38.nii.gz
        Mask:  seg_040_week_0_slice_38.nii.gz
    """

    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)

    image_files = list(image_dir.glob("case_*.nii.gz"))
    mask_files = list(mask_dir.glob("seg_*.nii.gz"))

    image_keys = {
        path.name.replace("case_", "", 1)
        for path in image_files
    }

    mask_keys = {
        path.name.replace("seg_", "", 1)
        for path in mask_files
    }

    missing_masks = image_keys - mask_keys
    missing_images = mask_keys - image_keys

    print("\nImage-mask pairing verification")
    print(f"MRI images: {len(image_files)}")
    print(f"Segmentation masks: {len(mask_files)}")
    print(f"Missing masks: {len(missing_masks)}")
    print(f"Missing images: {len(missing_images)}")

    return {
        "missing_masks": missing_masks,
        "missing_images": missing_images,
    }

def verify_spatial_alignment(image_dir, mask_dir):
    """
    Verify spatial consistency between MRI images and segmentation masks.

    Checks:
        1. Image and mask dimensions
        2. Affine transformation matrices
        3. Anatomical orientation
    """

    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)

    shape_mismatches = []
    affine_mismatches = []
    orientation_mismatches = []

    image_files = sorted(image_dir.glob("case_*.nii.gz"))

    for image_path in image_files:
        mask_name = image_path.name.replace("case_", "seg_", 1)
        mask_path = mask_dir / mask_name

        if not mask_path.exists():
            continue

        image = nib.load(str(image_path))
        mask = nib.load(str(mask_path))

        # Check whether image and mask have identical dimensions.
        if image.shape != mask.shape:
            shape_mismatches.append(image_path.name)

        # Check whether image and mask use matching spatial transforms.
        if not np.allclose(image.affine, mask.affine):
            affine_mismatches.append(image_path.name)

        # Check whether image and mask have matching orientations.
        image_orientation = nib.aff2axcodes(image.affine)
        mask_orientation = nib.aff2axcodes(mask.affine)

        if image_orientation != mask_orientation:
            orientation_mismatches.append(image_path.name)

    print("\nSpatial alignment verification")
    print(f"Image-mask pairs checked: {len(image_files)}")
    print(f"Shape mismatches: {len(shape_mismatches)}")
    print(f"Affine mismatches: {len(affine_mismatches)}")
    print(f"Orientation mismatches: {len(orientation_mismatches)}")

    return {
        "shape_mismatches": shape_mismatches,
        "affine_mismatches": affine_mismatches,
        "orientation_mismatches": orientation_mismatches,
    }

def verify_mask_labels(mask_dir, num_classes=6):
    """
    Verify that segmentation masks contain valid integer class labels.

    Expected labels: 0, 1, 2, 3, 4, 5.
    """

    mask_dir = Path(mask_dir)
    mask_files = sorted(mask_dir.glob("seg_*.nii.gz"))

    valid_labels = set(range(num_classes))
    observed_labels = set()
    invalid_masks = []

    for mask_path in mask_files:
        mask = nib.load(str(mask_path)).get_fdata()

        unique_labels = np.unique(mask)
        observed_labels.update(unique_labels.tolist())

        # Detect unexpected or non-integer segmentation labels.
        if not set(unique_labels).issubset(valid_labels):
            invalid_masks.append(mask_path.name)

    print("\nMask label verification")
    print(f"Masks checked: {len(mask_files)}")
    print(f"Observed labels: {sorted(observed_labels)}")
    print(f"Masks with invalid labels: {len(invalid_masks)}")

    return {
        "observed_labels": sorted(observed_labels),
        "invalid_masks": invalid_masks,
    }


#Part 2 - Boundary precision evaluation

def calculate_boundary_precision(
    prediction,
    ground_truth,
    class_id,
    tolerance=2
):
    """
    Calculate boundary precision for a segmentation class.

    A predicted boundary pixel is considered correct if it lies
    within the specified distance of the ground-truth boundary.

    Args:
        prediction: Predicted segmentation mask [H, W].
        ground_truth: Ground-truth segmentation mask [H, W].
        class_id: Segmentation class to evaluate.
        tolerance: Maximum allowed boundary distance in pixels.

    Returns:
        Boundary precision between 0 and 1.
        Returns None when the predicted boundary is empty.
    """

    prediction = np.asarray(prediction)
    ground_truth = np.asarray(ground_truth)

    if prediction.shape != ground_truth.shape:
        raise ValueError("Prediction and ground truth must have matching shapes.")

    pred_mask = prediction == class_id
    gt_mask = ground_truth == class_id

    # Extract boundaries using binary erosion.
    pred_boundary = pred_mask & ~binary_erosion(pred_mask)
    gt_boundary = gt_mask & ~binary_erosion(gt_mask)

    # No predicted boundary: precision is undefined.
    if not pred_boundary.any():
        return None

    # Predicted boundary exists, but GT boundary is absent.
    if not gt_boundary.any():
        return 0.0

    # Distance of every pixel to the nearest ground-truth boundary.
    distances = distance_transform_edt(~gt_boundary)

    # Count predicted boundary pixels within the tolerance.
    correct_pixels = np.sum(
        pred_boundary & (distances <= tolerance)
    )

    total_predicted_pixels = np.sum(pred_boundary)

    return float(correct_pixels / total_predicted_pixels)


#Part-3 Computational efficiency benchmarking
#Counting trained parameters
def count_trainable_parameters(model):
    """
    Count the number of trainable parameters in a model.

    This helps compare the complexity of Baseline U-Net
    and Improved U-Net.
    """
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

def measure_gpu_memory(model, input_tensor, device):
    """
    Measure peak GPU memory allocated during model inference.

    Args:
        model: Trained segmentation model.
        input_tensor: Input MRI tensor [B, 1, H, W].
        device: CUDA device.

    Returns:
        Peak GPU memory usage in MB.
    """

    if device.type != "cuda":
        raise ValueError("GPU memory measurement requires CUDA.")

    model.eval()
    model = model.to(device)
    input_tensor = input_tensor.to(device)

    # Complete any previously queued GPU operations.
    torch.cuda.synchronize(device)

    # Reset peak memory statistics before measurement.
    torch.cuda.reset_peak_memory_stats(device)

    # Perform inference without calculating gradients.
    with torch.inference_mode():
        _ = model(input_tensor)

    # Wait until GPU execution finishes.
    torch.cuda.synchronize(device)

    peak_memory_mb = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    )

    return peak_memory_mb

def measure_inference_time(model, input_tensor, device, warmup=10, runs=100):
    """
    Measure average GPU inference time using CUDA events.

    Args:
        model: Trained segmentation model.
        input_tensor: MRI tensor [B, 1, H, W].
        device: CUDA device.
        warmup: Number of warm-up iterations.
        runs: Number of timed iterations.

    Returns:
        Average inference time in milliseconds per slice.
    """

    if device.type != "cuda":
        raise ValueError("Inference benchmarking requires CUDA.")

    if warmup < 0 or runs <= 0:
        raise ValueError("warmup must be non-negative and runs must be positive.")

    model.eval()
    model = model.to(device)
    input_tensor = input_tensor.to(device)

    batch_size = input_tensor.shape[0]

    # Warm up the GPU before measuring execution time.
    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(input_tensor)

    torch.cuda.synchronize(device)

    timings = []

    # Measure repeated inference using CUDA events.
    with torch.inference_mode():
        for _ in range(runs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            _ = model(input_tensor)
            end.record()

            end.synchronize()

            timings.append(start.elapsed_time(end))

    average_batch_time = sum(timings) / len(timings)

    # Convert batch inference time to average time per slice.
    average_slice_time = average_batch_time / batch_size

    return average_slice_time