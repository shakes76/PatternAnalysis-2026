import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import time
import matplotlib.pyplot as plt
from modules import UNetClassifier
from dataset import Dataset
from dataset import get_dataloaders
import numpy as np
import nibabel as nib
import torch.nn.functional as F



''' Function Declarations '''
def get_weights_path():
    """
    Returns the path to the weights file.
    """
    # Training Path
    #return Path(__file__).resolve().parent / "unet_normal_model.pth"
    # Modification 1 Path
    #return Path(__file__).resolve().parent / "Improved_UNET_MOD_1.pth"
    # Modification 1 and 2 Path
    #return Path(__file__).resolve().parent / "Improved_UNET_MOD_12.pth"
    # Modification 1, 2 and 3 Path
    return Path(__file__).resolve().parent / "Improved_UNET_MOD_123.pth"
    # Basic UNET PATH
    #return Path(__file__).resolve().parent / "UNET.pth"

def load_model(model, path, device):
    """
    Loads the model's state dictionary from the specified path.
    """
    model.load_state_dict(torch.load(path, map_location=device))
    print(f"Model loaded from {path}")

def show_predictions(image, mask, prediction):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(image[0], cmap='gray')
    plt.title('Input Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap='nipy_spectral')
    plt.title('Ground Truth')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(prediction, cmap='nipy_spectral')
    plt.title('Prediction')
    plt.axis('off')

def show_failure_case(case, class_id):
    image = case["image"].squeeze().numpy()
    mask = case["mask"].numpy()
    prediction = case["prediction"].numpy()

    ground_truth = mask == class_id
    predicted = prediction == class_id

    true_positive = predicted & ground_truth
    false_positive = predicted & ~ground_truth
    false_negative = ~predicted & ground_truth

    # RGB overlay: green=TP, red=FP, blue=FN
    overlay = np.zeros((*mask.shape, 4), dtype=np.float32)

    overlay[true_positive] = [0, 1, 0, 0.65]
    overlay[false_positive] = [1, 0, 0, 0.8]
    overlay[false_negative] = [0, 0.4, 1, 0.8]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original MRI")

    axes[1].imshow(mask, cmap="tab10", vmin=0, vmax=5)
    axes[1].set_title("Ground Truth")

    axes[2].imshow(prediction, cmap="tab10", vmin=0, vmax=5)
    axes[2].set_title("Prediction")

    axes[3].imshow(image, cmap="gray")
    axes[3].imshow(overlay)
    axes[3].set_title(
        f"Class {class_id} errors\n"
        f"Dice: {case['dice']:.4f}"
    )

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

def run_weights_on_data(dataloader, device, model, num_classes, intersections, pred_counts, mask_counts):
    model.eval()
    total_inference_time = 0.0
    total_images = 0
    with torch.inference_mode():
        for i, (images, masks) in enumerate(dataloader):

            images = images.to(device).float()
            masks = masks.to(device).long()

            # Ensure previous GPU operations are complete
            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            # Forward pass
            logits = model(images)

            if device.type == "cuda":
                torch.cuda.synchronize()

            end = time.perf_counter()

            total_inference_time += end - start
            total_images += images.size(0)

            # Convert logits to predicted classes
            predictions = torch.argmax(logits, dim=1)

            # Accumulate Dice statistics
            for class_id in range(num_classes):
                pred_class = predictions == class_id
                mask_class = masks == class_id

                intersections[class_id] += (
                    pred_class & mask_class
                ).sum()

                pred_counts[class_id] += pred_class.sum()
                mask_counts[class_id] += mask_class.sum()

            # Visualize only the first two batches
            if i < 2:
                images_np = images.cpu().numpy()
                masks_np = masks.cpu().numpy()
                predictions_np = predictions.cpu().numpy()

                show_predictions(
                    images_np[0],
                    masks_np[0],
                    predictions_np[0]
                )

    # Average inference latency per image
    average_time = total_inference_time / total_images

    print(
        f"Average inference time per image: "
        f"{average_time * 1000:.2f} ms"
    )


def find_worst_cases(model, dataloader, device, class_id=4, top_k=5):
    model.eval()
    cases = []
    image_index = 0

    with torch.inference_mode():
        for images, masks in dataloader:
            images = images.to(device).float()
            masks = masks.to(device).long()

            logits = model(images)
            predictions = torch.argmax(logits, dim=1)

            for i in range(images.size(0)):
                pred = predictions[i]
                mask = masks[i]

                pred_class = pred == class_id
                mask_class = mask == class_id

                intersection = (pred_class & mask_class).sum().item()
                denominator = (
                    pred_class.sum().item()
                    + mask_class.sum().item()
                )

                # Ignore slices where the class is absent
                # from both prediction and ground truth.
                if denominator == 0:
                    image_index += 1
                    continue

                dice = (2 * intersection) / denominator

                cases.append({
                    "index": image_index,
                    "dice": dice,
                    "image": images[i].cpu().clone(),
                    "mask": mask.cpu().clone(),
                    "prediction": pred.cpu().clone()
                })

                image_index += 1

    cases.sort(key=lambda case: case["dice"])

    return cases[:top_k]

'''
############ Main Code ############
'''
''' Variable Declaration'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 6
model = UNetClassifier(in_channels=1, out_channels=6).to(device)

model.load_state_dict(torch.load(get_weights_path(), map_location=device, weights_only=True))
model.eval()

hip_train_loader, hip_val_loader, hip_test_loader, hip_train_set, hip_val_set, hip_test_set = get_dataloaders()
# Statistics declarations
intersections = torch.zeros(num_classes, device=device)
pred_counts = torch.zeros(num_classes, device=device)
mask_counts = torch.zeros(num_classes, device=device)
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()

''' 
Running Weights 
'''
with torch.inference_mode():
    run_weights_on_data(hip_test_loader, device, model, num_classes, intersections, pred_counts, mask_counts)

plt.show()

'''
Statistics Section
'''
denominators = pred_counts + mask_counts
dice_scores = torch.full((num_classes,), float("nan"), device=device)
valid_classes = denominators > 0
dice_scores[valid_classes] = (2 * intersections[valid_classes] / denominators[valid_classes])

''' Dice Scores '''
# Print individual class scores
print("\n===== Dice Scores =====")
for class_id, score in enumerate(dice_scores):
    print(f"Dice {class_id}: {score.item():.4f}")

# Calculate average Dice score
average_dice = dice_scores[valid_classes].mean().item()
print(f"\nAverage Dice Score: {average_dice:.4f}")
print(f"Average Dice Percentage: {average_dice * 100:.2f}%")

''' VRam Usage '''
torch.cuda.synchronize()
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024**2

print(f"Peak VRAM: {peak_vram_mb:.2f} MB")

''' Parameters Ran '''
total_parameters = sum(para.numel() for para in model.parameters())
print(f"Parameters ran on validation: {total_parameters}")

worst_cases = find_worst_cases(
    model,
    hip_val_loader,
    device,
    class_id=4,
    top_k=5
)

for case in worst_cases:
    print(
        f"Image {case['index']}: "
        f"Dice = {case['dice']:.4f}"
    )
for case in worst_cases:
    show_failure_case(case, class_id=4)