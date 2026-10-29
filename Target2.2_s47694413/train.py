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
''' 
Function Definitions 
'''

''' save_model saves the model's state dictionary to the specified path. '''
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

''' evaluate_dice evaluates the model's performance on a given dataset using the Dice coefficient metric. It returns the Dice scores for each class. '''
@torch.no_grad()
def evaluate_dice(model, loader, device, num_classes=6):
    model.eval()

    intersections = torch.zeros(num_classes, device=device)
    pred_counts = torch.zeros(num_classes, device=device)
    true_counts = torch.zeros(num_classes, device=device)

    for images, masks in loader:
        images = images.to(device).float()
        masks = masks.to(device).long()

        logits = model(images)
        predictions = torch.argmax(logits, dim=1)

        for c in range(num_classes):
            predicted = predictions == c
            actual = masks == c

            intersections[c] += (predicted & actual).sum()
            pred_counts[c] += predicted.sum()
            true_counts[c] += actual.sum()
    dice = ((2 * intersections) / (pred_counts + true_counts).clamp(min=1))
    return dice.cpu().tolist()

''' 
Training main section 
'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available():
    print("CUDA is not available. Using CPU.")
print("Using device:", device)
#Configuration
num_epochs = 10
learning_rate = 0.001
num_classes = 6

# Model Initialisation
model = UNetClassifier().to(device)
# Optimiser Initialisation
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# Loss function
criterion = nn.CrossEntropyLoss()
    
print("Loading dataset...")
HipTrainLoader, HipValLoader, HipTestLoader, HipTrainset, HipValset, HipTestset = get_dataloaders()

print(f"Number of samples: Training = {len(HipTrainset)}, Validation = {len(HipValset)}, Test = {len(HipTestset)}")
print(f"Batch size: {HipTrainLoader.batch_size}")
print(f"Number of batches in: training set = {len(HipTrainLoader)}, validation set = {len(HipValLoader)}, test set = {len(HipTestLoader)}")
print(f"Image shape: {HipTrainset[0][0].shape}, Mask shape: {HipTrainset[0][1].shape}")


#
# Training Loop
#
train_losses = []
train_dice_scores = []
val_losses = []
val_dice_mean_scores = []
best_val_dice = 0.0
class_scores = [[] for _ in range(6)]

print("Starting training...")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    time_start = time.time()

    # TRAINING
    for images, masks in HipTrainLoader:
        images = images.to(device).float()
        masks = masks.to(device).long()

        optimizer.zero_grad()

        # Standard U-Net
        predictions = model(images)
        loss = criterion(predictions, masks)

        # Deep supervision version instead:
        #predictions, aux1, aux2 = model(images)
        #loss_main = criterion(predictions, masks)
        #loss_aux1 = criterion(aux1, masks)
        #loss_aux2 = criterion(aux2, masks)
        #loss = loss_main + 0.3 * loss_aux1 + 0.2 * loss_aux2

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    time_end = time.time()

    train_loss = total_loss / len(HipTrainLoader)
    train_losses.append(train_loss)

    print(
        f"Epoch {epoch + 1}/{num_epochs}, "
        f"Training Loss: {train_loss:.4f}, "
        f"Time: {time_end - time_start:.2f}s"
    )

    # VALIDATION DICE
    val_dice_scores = evaluate_dice(
        model, HipValLoader, device, num_classes=6
    )

    val_mean_dice = sum(val_dice_scores) / len(val_dice_scores)
    val_dice_mean_scores.append(val_mean_dice)

    print("Validation Dice:")
    for class_id, score in enumerate(val_dice_scores):
        class_scores[class_id].append(score)
        print(f"  Class {class_id}: {score:.4f}")

    print(f"Mean Validation Dice: {val_mean_dice:.4f}")

    # VALIDATION LOSS
    model.eval()
    total_val_loss = 0.0

    with torch.inference_mode():
        for images, masks in HipValLoader:
            images = images.to(device).float()
            masks = masks.to(device).long()

            predictions = model(images)
            loss = criterion(predictions, masks)

            total_val_loss += loss.item()

    val_loss = total_val_loss / len(HipValLoader)
    val_losses.append(val_loss)

    # SAVE BEST MODEL BASED ON VALIDATION DICE
    if val_mean_dice > best_val_dice:
        best_val_dice = val_mean_dice

        save_model(
            model,
            Path(__file__).resolve().parent / "unet_normal_model.pth"
        )

torch.save((train_losses, val_losses, train_dice_scores, val_dice_mean_scores, class_scores), Path(__file__).resolve().parent / "training_stats.pth")
''' Graphs '''
# Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# Plot validation Dice scores
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), val_dice_mean_scores, label='Validation Dice')
plt.xlabel('Epochs')
plt.ylabel('Dice Score')
plt.title('Validation Dice Scores')
plt.legend()

plt.show()

# Plot Dice scores for each class
plt.figure(figsize=(10, 5))
for class_id in range(num_classes):
    plt.plot(range(1, num_epochs + 1), class_scores[class_id], label=f'Class {class_id}')
plt.xlabel('Epochs')
plt.ylabel('Dice Score')
plt.title('Dice Scores for Each Class')
plt.legend()
plt.show()

''' Parameters Ran '''
total_parameters = sum(para.numel() for para in model.parameters())
print(f"Parameters ran on training: {total_parameters}")
