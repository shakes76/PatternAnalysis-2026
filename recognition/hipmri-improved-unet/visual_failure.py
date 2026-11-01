import torch
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

#We are going to compare in case of base model- 2D U-Net
from dataset import HipMRIDataset
from modules import ImprovedUNet
from baseline_unet.modules import UNet


def visualize_segmentation(image, ground_truth, baseline_prediction,
                           improved_prediction, filename):

    image = image.squeeze().cpu().numpy()
    ground_truth = ground_truth.cpu().numpy()
    baseline_prediction = baseline_prediction.cpu().numpy()
    improved_prediction = improved_prediction.cpu().numpy()

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original MRI")

    axes[1].imshow(ground_truth, cmap="tab10", vmin=0, vmax=5)
    axes[1].set_title("Ground Truth")

    axes[2].imshow(baseline_prediction, cmap="tab10", vmin=0, vmax=5)
    axes[2].set_title("Baseline U-Net")

    axes[3].imshow(improved_prediction, cmap="tab10", vmin=0, vmax=5)
    axes[3].set_title("Improved U-Net")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(filename)
    plt.tight_layout()

    output_dir = Path("failure_visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"comparison_{Path(filename).name}.png"

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved comparison: {output_path}")


def visualize_error_maps(ground_truth, baseline_prediction,
                         improved_prediction, filename):

    ground_truth = ground_truth.cpu().numpy()
    baseline_prediction = baseline_prediction.cpu().numpy()
    improved_prediction = improved_prediction.cpu().numpy()

    # Error maps:
    # 0 = correct prediction
    # 1 = false positive (background predicted as foreground)
    # 2 = false negative (foreground predicted as background)
    # 3 = incorrect foreground class

    def create_error_map(prediction):
        error_map = np.zeros_like(ground_truth, dtype=np.uint8)

        error_map[(ground_truth == 0) & (prediction != 0)] = 1
        error_map[(ground_truth != 0) & (prediction == 0)] = 2
        error_map[
            (ground_truth != 0) &
            (prediction != 0) &
            (ground_truth != prediction)
        ] = 3

        return error_map

    baseline_errors = create_error_map(baseline_prediction)
    improved_errors = create_error_map(improved_prediction)

    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    colors = ["white", "red", "blue", "orange"]
    cmap = ListedColormap(colors)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(baseline_errors, cmap=cmap, vmin=0, vmax=3)
    axes[0].set_title("Baseline U-Net Errors")

    axes[1].imshow(improved_errors, cmap=cmap, vmin=0, vmax=3)
    axes[1].set_title("Improved U-Net Errors")

    for ax in axes:
        ax.axis("off")

    legend_elements = [
        Patch(facecolor="white", label="Correct"),
        Patch(facecolor="red", label="False Positive"),
        Patch(facecolor="blue", label="False Negative"),
        Patch(facecolor="orange", label="Wrong Foreground Class")
    ]

    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=4
    )

    fig.suptitle(filename)
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])

    output_dir = Path("failure_visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"errors_{Path(filename).name}.png"

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved error map: {output_path}")


#Loading specific MRI slice and generating predictions
def load_mri_slice(dataset, filename):
    for index, image_path in enumerate(dataset.image_files):
        if image_path.name == filename:
            image, mask = dataset[index]
            return image, mask

    raise FileNotFoundError(f"MRI slice not found: {filename}")

#for base model and improved model
def load_trained_model(model_class, checkpoint_path, device):
    model = model_class(in_channels=1, num_classes=6)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(device)
    model.eval()

    return model

def predict_segmentation(model, image, device):
    # Add a batch dimension: [1, 256, 144]
    # becomes [1, 1, 256, 144]
    input_tensor = image.unsqueeze(0).to(device)

    # Disable gradient calculations during inference
    with torch.no_grad():
        output = model(input_tensor)

        # Select the class with the highest score for each pixel
        prediction = output.argmax(dim=1)

    # Remove batch dimension and return to CPU
    return prediction.squeeze(0).cpu()

def calculate_class_dice(prediction, ground_truth, num_classes=6):
    results = {}

    for class_id in range(1, num_classes):
        pred_mask = prediction == class_id
        true_mask = ground_truth == class_id

        intersection = (pred_mask & true_mask).sum().item()
        pred_pixels = pred_mask.sum().item()
        true_pixels = true_mask.sum().item()

        denominator = pred_pixels + true_pixels

        if denominator == 0:
            dice = None  # Class absent in both masks
        else:
            dice = 2 * intersection / denominator

        results[class_id] = {
            "dice": dice,
            "pred_pixels": pred_pixels,
            "true_pixels": true_pixels
        }

    return results

#Finding cases where improved Unet works better in case of finding class 5
def find_improvement_cases(dataset, baseline_model, improved_model, device):
    improvements = []

    for index in range(len(dataset)):
        image, ground_truth = dataset[index]

        # Only examine slices containing Class 5
        if not (ground_truth == 5).any().item():
            continue

        baseline_prediction = predict_segmentation(
            baseline_model, image, device
        )

        improved_prediction = predict_segmentation(
            improved_model, image, device
        )

        baseline_dice = calculate_class_dice(
            baseline_prediction, ground_truth
        )[5]["dice"]

        improved_dice = calculate_class_dice(
            improved_prediction, ground_truth
        )[5]["dice"]

        improvement = improved_dice - baseline_dice

        if improvement > 0:
            improvements.append((
                improvement,
                dataset.image_files[index].name,
                baseline_dice,
                improved_dice
            ))

    # Rank cases by Class 5 Dice improvement
    improvements.sort(reverse=True)

    return improvements[:2]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_root = Path(
        "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"
    )

    dataset = HipMRIDataset(
        image_dir=dataset_root / "keras_slices_test",
        mask_dir=dataset_root / "keras_slices_seg_test"
    )

    # Load both trained models
    baseline_model = load_trained_model(
        UNet,
        "checkpoints_unet_20/best_unet.pth",
        device
    )

    improved_model = load_trained_model(
        ImprovedUNet,
        "checkpoints_improved_20/best_improved_unet.pth",
        device
    )

    filenames = [
        "case_040_week_0_slice_38.nii.gz",
        "case_040_week_1_slice_20.nii.gz",
        "case_040_week_2_slice_51.nii.gz"
    ]

    for filename in filenames:
        image, ground_truth = load_mri_slice(dataset, filename)

        baseline_prediction = predict_segmentation(
            baseline_model, image, device
        )

        improved_prediction = predict_segmentation(
            improved_model, image, device
        )

        baseline_scores = calculate_class_dice(
            baseline_prediction, ground_truth
        )

        improved_scores = calculate_class_dice(
            improved_prediction, ground_truth
        )

        print(f"\nSlice: {filename}")
        print("Class | Baseline Dice | Improved Dice | GT Pixels")

        for class_id in range(1, 6):
            baseline = baseline_scores[class_id]["dice"]
            improved = improved_scores[class_id]["dice"]
            gt_pixels = baseline_scores[class_id]["true_pixels"]

            baseline_text = "N/A" if baseline is None else f"{baseline:.4f}"
            improved_text = "N/A" if improved is None else f"{improved:.4f}"

            print(
                f"{class_id:5d} | "
                f"{baseline_text:13} | "
                f"{improved_text:13} | "
                f"{gt_pixels}"
            )

        visualize_segmentation(
            image,
            ground_truth,
            baseline_prediction,
            improved_prediction,
            filename
        )

        visualize_error_maps(
            ground_truth,
            baseline_prediction,
            improved_prediction,
            filename
        )

        # Find two cases where Improved U-Net performs better
    improvement_cases = find_improvement_cases(
        dataset,
        baseline_model,
        improved_model,
        device
    )

    print("\nTop Class 5 improvement cases:")

    for improvement, filename, baseline_dice, improved_dice in improvement_cases:
        print(f"\nSlice: {filename}")
        print(f"Baseline Class 5 Dice: {baseline_dice:.4f}")
        print(f"Improved Class 5 Dice: {improved_dice:.4f}")
        print(f"Dice improvement: {improvement:.4f}")

        image, ground_truth = load_mri_slice(dataset, filename)

        baseline_prediction = predict_segmentation(
            baseline_model, image, device
        )

        improved_prediction = predict_segmentation(
            improved_model, image, device
        )

        visualize_segmentation(
            image,
            ground_truth,
            baseline_prediction,
            improved_prediction,
            filename
        )

        visualize_error_maps(
            ground_truth,
            baseline_prediction,
            improved_prediction,
            filename
        )


if __name__ == "__main__":
    main()
