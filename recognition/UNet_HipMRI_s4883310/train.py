import os
import time
import copy
import csv
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import random
import numpy as np

from dataset import (
    train_dataset,
    validation_dataset,
    train_loader,
    validation_loader
)

from model import UNet


# REPRODUCIBILITY
seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# DICE SCORE

def calculate_dice(predictions, masks, num_classes=6):
    """Calculate the Dice score for each foreground class."""

    intersections = torch.zeros(num_classes)
    totals = torch.zeros(num_classes)

    # Exclude background (class 0) from Dice evaluation
    for class_id in range(1, num_classes):

        predicted_class = predictions == class_id
        actual_class = masks == class_id

        intersections[class_id] = (
            predicted_class & actual_class
        ).sum().item()

        totals[class_id] = (
            predicted_class.sum().item()
            + actual_class.sum().item()
        )

    return intersections, totals

def initialise_weights(module):
    """Initialise convolution weights using He normal initialisation."""

    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(
            module.weight,
            mode="fan_in",
            nonlinearity="relu"
        )

        if module.bias is not None:
            nn.init.zeros_(module.bias)


# DEVICE CONFIGURATION

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# HYPERPARAMETERS

num_epochs = 40
learning_rate = 0.001
num_classes = 6


# CREATE MODEL

model = UNet(
    in_channels=1,
    num_classes=num_classes
)

# Apply Kaiming initialisation before moving the model to the device
model.apply(initialise_weights)

model = model.to(device)


# LOSS AND OPTIMIZER

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate
)


# TRAINING

print("> Training")

# Reset GPU memory statistics before measuring training resources
if torch.cuda.is_available():
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

start = time.time()

# Store epoch metrics for training curves and model comparison
training_losses = []
validation_losses = []
validation_dice_scores = []

# Track the checkpoint with the highest mean foreground validation Dice
best_validation_dice = -1.0
best_model_state = None
best_epoch = 0

for epoch in range(num_epochs):

    model.train()
    total_training_loss = 0

    for images, masks in train_loader:

        images = images.to(device)
        masks = masks.to(device)

        # Clear old gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)

        # Calculate segmentation loss
        loss = criterion(outputs, masks)

        # Backpropagation
        loss.backward()

        # Update model parameters
        optimizer.step()

        total_training_loss += (
            loss.item() * images.size(0)
        )

    average_training_loss = (
        total_training_loss / len(train_dataset)
    )

    training_losses.append(average_training_loss)


    # VALIDATION

    model.eval()

    total_validation_loss = 0
    dice_intersections = torch.zeros(num_classes)
    dice_totals = torch.zeros(num_classes)

    with torch.no_grad():

        for images, masks in validation_loader:

            images = images.to(device)
            masks = masks.to(device)

            # Forward pass
            outputs = model(images)

            # Calculate validation loss
            loss = criterion(outputs, masks)

            total_validation_loss += (
                loss.item() * images.size(0)
            )

            # Predict the segmentation class for each pixel
            predictions = outputs.argmax(dim=1)

            # Calculate intersection and total counts for each batch
            intersections, totals = calculate_dice(
                predictions,
                masks,
                num_classes
            )

            dice_intersections += intersections
            dice_totals += totals


    average_validation_loss = (
        total_validation_loss / len(validation_dataset)
    )

    validation_losses.append(average_validation_loss)


    # CALCULATE VALIDATION DICE

    class_dice_scores = []

    # Calculate Dice for each foreground class
    for class_id in range(1, num_classes):

        if dice_totals[class_id] > 0:

            dice = (
                2 * dice_intersections[class_id]
                / dice_totals[class_id]
            )

            class_dice_scores.append(dice.item())

    # Average Dice across foreground classes
    average_validation_dice = (
        sum(class_dice_scores) / len(class_dice_scores)
        if class_dice_scores else 0.0
    )

    validation_dice_scores.append(average_validation_dice)

    
    # SAVE BEST MODEL

    # Select the checkpoint with the highest mean foreground validation Dice
    if average_validation_dice > best_validation_dice:

        best_validation_dice = average_validation_dice
        best_epoch = epoch + 1

        best_model_state = copy.deepcopy(
            model.state_dict()
        )

        os.makedirs("checkpoints", exist_ok=True)

        torch.save(
            best_model_state,
            "checkpoints/unet_baseline_best.pth"
        )

    print(
        "Epoch [{}/{}], Training Loss: {:.5f}, "
        "Validation Loss: {:.5f}, Validation Dice: {:.4f}"
        .format(
            epoch + 1,
            num_epochs,
            average_training_loss,
            average_validation_loss,
            average_validation_dice
        )
    )


# TRAINING TIME AND VRAM

if torch.cuda.is_available():
    torch.cuda.synchronize()

end = time.time()

elapsed = end - start

print(
    "Training took "
    + str(elapsed)
    + " secs or "
    + str(elapsed / 60)
    + " mins in total"
)

if torch.cuda.is_available():
    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print("Peak GPU VRAM:", peak_vram, "GB")

print("Best epoch:", best_epoch)


# RESTORE BEST TRAINED MODEL

os.makedirs("checkpoints", exist_ok=True)

# Restore the checkpoint with the highest validation Dice
model.load_state_dict(best_model_state)

print(
    "Best Validation Dice: {:.4f}".format(
        best_validation_dice
    )
)


# SAVE TRAINING HISTORY

# Export epoch metrics for plotting learning curves
with open(
    "checkpoints/unet_baseline_history.csv",
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "epoch",
        "training_loss",
        "validation_loss",
        "validation_dice"
    ])

    for i in range(num_epochs):
        writer.writerow([
            i + 1,
            training_losses[i],
            validation_losses[i],
            validation_dice_scores[i]
        ])

# SAVE TRAINING SUMMARY

# Record best validation performance and computational cost
with open("checkpoints/unet_baseline_summary.txt", "w") as file:
    file.write(f"Best Epoch: {best_epoch}\n")
    file.write(f"Best Validation Dice: {best_validation_dice:.4f}\n")
    file.write(f"Training Time: {elapsed / 60:.2f} minutes\n")
    file.write(f"Parameters: {sum(p.numel() for p in model.parameters())}\n")

    if torch.cuda.is_available():
        file.write(f"Peak VRAM: {peak_vram:.3f} GB\n")