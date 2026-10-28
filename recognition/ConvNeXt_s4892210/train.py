"""
Author: Lawrence Tiang
Student ID: s4892210

Description:
    Train the selected model for binary classification. Model performance
    is evaluated on the validation dataset after each epoch and the
    checkpoint with the lowest validation loss is saved. Training
    and validation loss and accuracy curves are also generated to
    support performance analysis.

Usage:
    python train.py --model baseline
    python train.py --model convnext
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import build_dataloader
from modules import build_model


# Training hyperparameters and output configuration
EPOCHS = 400
LEARNING_RATE = 1e-3
BATCH_SIZE = 256
WEIGHT_DECAY = 0.01
LABEL_SMOOTHING = 0.1
SEED = 42
OUTPUT_DIR = "assets"

# Set random seeds for reproducibility of experiments
torch.manual_seed(SEED)
np.random.seed(SEED)

# Set random seed for CUDA devices when GPU is available
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Use GPU when available, otherwise run model on CPU
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


def train_one_epoch(model, dataloader, criterion, optimizer):
    """
    Train the model for one complete epoch.

    Args:
        model: Neural network being trained.
        dataloader: DataLoader containing the training images and labels.
        criterion: Loss function used to measure prediction errors.
        optimizer: Optimiser used to update model parameters.

    Returns:
        tuple[float, float]: Average training loss and classification
        accuracy for the epoch.
    """
    # Enable training behaviour
    model.train()
    running_loss = 0.0
    correct = 0
    num_samples = 0

    # Process the training dataset one batch at a time
    for images, targets in dataloader:
        # Move the current batch to the selected device
        images = images.to(DEVICE)
        targets = targets.to(DEVICE)

        # Update model parameters
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        # Accumulate loss and accuracy statistics
        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        _, predicted = logits.max(1)
        correct += predicted.eq(targets).sum().item()
        num_samples += batch_size

    # Calculate average loss and accuracy across all training samples
    average_loss = running_loss / num_samples
    accuracy = correct / num_samples
    return average_loss, accuracy


def validate(model, dataloader, criterion):
    """
    Evaluate the model on the validation
    dataset without updating its parameters.

    Args:
        model: Neural network being evaluated.
        dataloader: DataLoader containing validation images and labels.
        criterion: Loss function used to measure prediction errors.

    Returns:
        tuple[float, float]: Average validation loss and classification
        accuracy for the epoch.
    """
    # Enable evaluation behaviour
    model.eval()
    running_loss = 0.0
    correct = 0
    num_samples = 0

    # Evaluate the model without computing gradients
    with torch.inference_mode():
        for images, targets in dataloader:
            # Move the current batch to the selected device
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            # Generate predictions and calculate validation loss
            logits = model(images)
            loss = criterion(logits, targets)

            # Accumulate loss weighted by number of images
            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            _, predicted = logits.max(1)
            correct += predicted.eq(targets).sum().item()
            num_samples += batch_size

    # Calculate average validation loss and accuracy
    average_loss = running_loss / num_samples
    accuracy = correct / num_samples
    return average_loss, accuracy


def plot_training_curves(
    train_losses,
    val_losses,
    train_accuracies,
    val_accuracies,
    model_name
):
    """
    Plot training and validation loss and accuracy curves.
    Save plots in assets folder.

    Args:
        train_losses: Average training loss recorded for each epoch.
        val_losses: Average validation loss recorded for each epoch.
        train_accuracies: Training accuracy recorded for each epoch.
        val_accuracies: Validation accuracy recorded for each epoch.
        model_name: Name of the selected model architecture.
    
    Returns:
        None: Saves the loss and accuracy plots as PNG files.
    """
    # Create the output directory if it does not already exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    epochs = range(1, len(train_losses) + 1)
    display_name = "Baseline CNN" if model_name == "baseline" else "ConvNeXt"

    # Plot training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"{display_name} Training and Validation Loss")
    plt.legend()
    plt.tight_layout()

    # Save the loss plot
    loss_path = os.path.join(OUTPUT_DIR, f"{model_name}_loss.png")
    plt.savefig(loss_path)
    plt.close()

    # Plot training and validation accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accuracies, label="Training Accuracy")
    plt.plot(epochs, val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(f"{display_name} Training and Validation Accuracy")
    plt.legend()
    plt.tight_layout()

    # Save the accuracy plot
    accuracy_path = os.path.join(OUTPUT_DIR, f"{model_name}_accuracy.png")
    plt.savefig(accuracy_path)
    plt.close()


def train():
    """
    Configure and execute model training.
    Parse the selected model architecture, prepare DataLoaders,
    initialise the model and train for the configured number of epochs.
    Save the checkpoint with the lowest validation loss and generate
    training history plots after training is complete.

    Args:
        None: Configuration is read from command-line arguments.

    Returns:
        None: Saves the best model checkpoint and training history plots.
    """
    # Configure command-line arguments for selecting the model
    parser = argparse.ArgumentParser(
        description="Train an ADNI image classification model."
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["baseline", "convnext"],
        default="convnext",
        help="Model architecture."
    )

    args = parser.parse_args()
    print(f"Using device: {DEVICE}")
    print(f"Model: {args.model}\n")

    # Load the training and validation datasets
    train_loader, val_loader = build_dataloader(is_train=True, batch_size=BATCH_SIZE)
    
    # Initialise the selected model and move it to the device
    model = build_model(args.model).to(DEVICE)

    # Define the loss function, optimiser and learning rate scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=EPOCHS
    )

    # Store loss and accuracy values for plotting
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # Track the checkpoint with the lowest validation loss
    best_val_loss = float("inf")
    best_val_accuracy = 0.0
    best_epoch = 0

    # Repeat training and validation for the configured number of epochs
    for epoch in range(1, EPOCHS + 1):
        # Update model parameters using the training dataset
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        # Evaluate the updated model on the validation dataset
        val_loss, val_acc = validate(model, val_loader, criterion)

        # Update the learning rate for the next epoch
        scheduler.step()

        # Record metrics for the training history plots
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        # Print the current epoch's training and validation performance
        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        # Save a checkpoint whenever validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_accuracy = val_acc
            best_epoch = epoch

            os.makedirs(OUTPUT_DIR, exist_ok=True)
            model_path = os.path.join(OUTPUT_DIR, f"best_{args.model}.pth")
            torch.save(model.state_dict(), model_path)

            print(f"=== New Best Model (Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}) =====\n")

    # Print the best validation results achieved during training
    print(f"\n=== Best {args.model} Model ===")
    print(f"Best Epoch: {best_epoch}")
    print(f"Validation Loss: {best_val_loss:.4f}")
    print(f"Validation Accuracy: {best_val_accuracy:.4f}")

    # Generate plots summarising training and validation performance
    plot_training_curves(
        train_losses=train_losses,
        val_losses=val_losses,
        train_accuracies=train_accuracies,
        val_accuracies=val_accuracies,
        model_name=args.model
    )


if __name__ == "__main__":
    train()