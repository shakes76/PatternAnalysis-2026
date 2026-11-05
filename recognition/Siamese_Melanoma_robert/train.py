from params import (
    MODEL_PATH, IMAGE_PATH, TRIPLET_MARGIN, 
    BATCH_TRIPLET, BATCH_TEST,
    LR_SIAMESE, EPOCHS_SIAMESE,
    PATIENCE, MIN_DELTA, SCHED_FACTOR, SCHED_PATIENCE,
    SIAMESE_LOSS_NAME, SAVE_SAMPLE_NAME, OUT_DIM
)

import os
import torch

torch.backends.cudnn.enabled = False # bandaid fix

import torch.nn as nn

#from utils import ensure_dir, plot_lines, save_sample_input, extract_features

from torch.utils.data import DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score
from torch.utils.data import DataLoader

from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt

from dataset import get_data_loaders
from modules import SiameseNet, TripletLoss
from predict import results_siamese_network, results_baseline

def train_epoch(
    model : nn.Module,
    train_loader,
    triplet_loss: nn.Module, 
    classifier_loss: nn.Module, 
    optimizer: torch.optim.Optimizer, 
    device: str | torch.device, 
    scaler: torch.cuda.amp.GradScaler, 
    epoch_idx: int
) -> Tuple[float, float, float]:
    """
    One full training epoch for a Siamese/Triplet model.
    Runs mixed-precision forward/backward, computes Triplet loss + classifier loss,
    applies gradient scaling & clipping, and tracks running metrics.
    Args:
        model (nn.Module): The Siamese Network model.
        train_loader (DataLoader): DataLoader for the training data.
        triplet_loss (nn.Module): The triplet loss function.
        classifier_loss (nn.Module): The classifier loss function.
        optimizer (torch.optim.Optimizer): optimizer for model parameters.
        device (str | torch.device): training device.
        scaler (torch.cuda.amp.GradScaler): gradient scaler for AMP.
        epoch_idx (int): current epoch index.
    Returns:
        tuple[float, float, float]:
            - avg_loss: mean total loss over the epoch
            - final_acc: accuracy computed over all seen samples
            - auc_roc: ROC-AUC over all seen samples
    """

    model.train()
    running_loss, all_labels, all_probs, all_preds = [], [], [], []

    for batch_idx, (anchor, positive, negative, labels) in enumerate(train_loader):
        # Move to device
        anchor, positive, negative, labels = (
            anchor.to(device), positive.to(device), negative.to(device), labels.to(device)
        )

        optimizer.zero_grad()

        with autocast(device_type='cuda'):
            # Embeddings for triplet loss
            anchor_out, positive_out, negative_out = model(anchor, positive, negative)
            # Triplet loss
            t_loss = triplet_loss(anchor_out, positive_out, negative_out)

            # Logits for classification loss 
            logits = model.classify(anchor)
            c_loss = classifier_loss(logits, labels.long())

            # Total loss 
            loss = t_loss + c_loss

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        # Gradient clipping to improve stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()

        running_loss.append(loss.item())
        # Convert logits to class probabilities along the class dimension and take the prob of class 1 
        probs = torch.softmax(logits, dim=1)[:, 1]
        # Predict class index for each sample
        preds = logits.argmax(dim=1)

        all_labels.extend(labels.tolist())
        all_probs.extend(probs.detach().cpu().numpy())
        all_preds.extend(preds.detach().cpu().numpy())

        running_acc = accuracy_score(all_labels, all_preds)
        if (batch_idx + 1) % 100 == 0:   # adjust frequency as you like
            print(f"Epoch [{epoch_idx}] Step [{batch_idx+1}/{len(train_loader)}] Acc [{running_acc}]")


    avg_loss = float(np.mean(running_loss))
    final_acc = accuracy_score(all_labels, all_preds)
    auc_roc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else float("nan")
    return avg_loss, final_acc, auc_roc

def validate(
    model: nn.Module,
    val_loader: DataLoader,
    triplet_loss: nn.Module,
    classifier_loss: nn.Module,
    device: torch.device | str,
    epoch_idx: int,
) -> Tuple[float, float, float]:
    """
    Validate the model on the validation set.
    Args:
        model (nn.Module): The Siamese Network model.
        val_loader (DataLoader): DataLoader for the validation data.
        triplet_loss (nn.Module): The triplet loss function.
        classifier_loss (nn.Module): The classifier loss function.
        device (torch.device): The device to validate on (CPU or GPU).
    Returns:
        tuple: A tuple containing average loss, final accuracy, and AUC-ROC for the validation set.
    """
    model.eval() 
    running_loss, all_labels, all_probs, all_preds = [], [], [], []

    with torch.no_grad():
        for batch_idx, (anchor, positive, negative, labels) in enumerate(val_loader):
            # Move data to the specified device for faster computation
            anchor, positive, negative, labels = (
                anchor.to(device), positive.to(device), negative.to(device), labels.to(device)
            )

            # Embeddings for triplet loss
            anchor_out, positive_out, negative_out = model(anchor, positive, negative)

            # Triplet loss
            t_loss = triplet_loss(anchor_out, positive_out, negative_out)

            # Logits for classification loss
            classifier_out = model.classify(anchor)
            c_loss = classifier_loss(classifier_out, labels.long())

            # Total loss
            loss = t_loss + c_loss

            running_loss.append(loss.item())

            # Convert logits to class probabilities along the class dimension and take the prob of class 1 
            probs = torch.softmax(classifier_out, dim=1)[:, 1]
            # Predict class index for each sample
            _, preds = classifier_out.max(1)

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

            running_acc = accuracy_score(all_labels, all_preds)
            if (batch_idx + 1) % 70 == 0:   # adjust frequency as you like
                print(f"Epoch [{epoch_idx}] Step [{batch_idx+1}/{len(val_loader)}] Acc [{running_acc}]")

    avg_loss = float(np.mean(running_loss)) if running_loss else 0.0
    final_acc = accuracy_score(all_labels, all_preds) if all_labels else 0.0
    auc_roc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else float("nan")
    return avg_loss, final_acc, auc_roc

def train_siamese_network(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: SiameseNet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    triplet_loss: TripletLoss,
    classifier_loss: nn.Module,
    epochs: int,
    scaler,
    device: str
):

    """
    Train a Siamese/Triplet network.
    Runs for the specified number of epochs, performing:
      - Train the model on the training dataset .
      - Validate model on the validation dataset.
      - Learning rate scheduling based on validation AUC-ROC
    Returns metric of training and validation dataset.
     
    Args:
        train_loader (DataLoader): DataLoader for the training data.
        val_loader (DataLoader): DataLoader for the validation data.
        model (SiameseNet): The Siamese Network model.
        optimizer (torch.optim.Optimizer): optimizer for model parameters.
        scheduler (torch.optim.lr_scheduler._LRScheduler | ReduceLROnPlateau): LR scheduler stepped with val AUC.
        triplet_loss (TripletLoss): The triplet loss function.
        classifier_loss (nn.Module): The classifier loss function.
        epochs (int): number of training epochs.
        scaler: AMP gradient scaler for mixed precision training.
        device (str): compute device.
    Returns:
        tuple[list[float], list[float], list[float], list[float], list[float], list[float]]:
            (train_loss_per_epoch, train_acc_per_epoch, train_auc_per_epoch,
             val_loss_per_epoch, val_acc_per_epoch, val_auc_per_epoch)
    """

    train_loss_per_epoch, train_acc_per_epoch, train_auc_per_epoch = [], [], []
    val_loss_per_epoch, val_acc_per_epoch, val_auc_per_epoch = [], [], []

    best_val_auc = 0.0

    for epoch in range(epochs):

        # Train the model for one epoch
        train_loss, train_acc, train_auc = train_epoch(model, train_loader, triplet_loss, classifier_loss, optimizer, device, scaler, epoch)

        # Validate the model
        val_loss, val_acc, val_auc = validate(model, val_loader, triplet_loss, classifier_loss, device, epoch)

        # Store metrics
        train_loss_per_epoch.append(train_loss)
        train_acc_per_epoch.append(train_acc)
        train_auc_per_epoch.append(train_auc)
        val_loss_per_epoch.append(val_loss)
        val_acc_per_epoch.append(val_acc)
        val_auc_per_epoch.append(val_auc)

        # Adjust learning rate based on validation AUC-ROC
        scheduler.step(val_auc)

        # Save the model if it performs with better auc roc score on validation set
        if val_auc > best_val_auc:
            torch.save(model.state_dict(), os.path.join(MODEL_PATH, "siamese.pth"))
            best_val_auc = val_auc

    return train_loss_per_epoch, train_acc_per_epoch, train_auc_per_epoch, val_loss_per_epoch, val_acc_per_epoch, val_auc_per_epoch

def train_baseline_epoch(
    model: nn.Module,
    train_loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str | torch.device,
    scaler: GradScaler,
    epoch_idx: int
) -> Tuple[float, float, float]:
    model.train()
    running_loss, all_labels, all_probs, all_preds = [], [], [], []

    for batch_idx, (anchor, _, _, labels) in enumerate(train_loader):
        anchor, labels = anchor.to(device), labels.to(device)
        optimizer.zero_grad()

        with autocast(device_type='cuda'):
            logits = model(anchor)
            loss = criterion(logits, labels.long())

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss.append(loss.item())
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = logits.argmax(dim=1)

        all_labels.extend(labels.tolist())
        all_probs.extend(probs.detach().cpu().numpy())
        all_preds.extend(preds.detach().cpu().numpy())

    avg_loss = float(np.mean(running_loss))
    final_acc = accuracy_score(all_labels, all_preds)
    auc_roc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else float("nan")
    return avg_loss, final_acc, auc_roc

def validate_baseline(
    model: nn.Module,
    val_loader,
    criterion: nn.Module,
    device: str | torch.device,
) -> Tuple[float, float, float]:
    model.eval()
    running_loss, all_labels, all_probs, all_preds = [], [], [], []

    with torch.no_grad():
        for anchor, _, _, labels in val_loader:
            anchor, labels = anchor.to(device), labels.to(device)
            logits = model(anchor)
            loss = criterion(logits, labels.long())

            running_loss.append(loss.item())
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    avg_loss = float(np.mean(running_loss)) if running_loss else 0.0
    final_acc = accuracy_score(all_labels, all_preds) if all_labels else 0.0
    auc_roc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else float("nan")
    return avg_loss, final_acc, auc_roc

def train_baseline(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: ResNetBaseline,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    epochs: int,
    scaler,
    device: str
):
    for epoch in range(epochs):
        train_loss, train_acc, train_auc = train_baseline_epoch(
            model, train_loader, criterion, optimizer, device, scaler, epoch
        )
        val_loss, val_acc, val_auc = validate_baseline(
            model, val_loader, criterion, device
        )

        print(f"Epoch [{epoch+1}/{epochs}] | "
              f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} - Train AUC: {train_auc:.4f} | "
              f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f} - Val AUC: {val_auc:.4f}")

        scheduler.step(val_auc)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            os.makedirs(MODEL_PATH, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(MODEL_PATH, "resnet_baseline.pth"))
    return best_val_auc

def plot_training_graphs(
    train_loss_per_epoch: list[float],
    val_loss_per_epoch: list[float],
    train_acc_per_epoch: list[float],
    val_acc_per_epoch: list[float],
    train_aucroc_per_epoch: list[float],
    val_aucroc_per_epoch: list[float],
    out_prefix: str = "train_val_progress"
) -> dict:
    """
    Plot train and validation metrics across epochs and save separate figures
    Returns dict of saved file paths
    """
    saved = {}

    def _plot_and_save(y_tr, y_val, title, ylabel, fname):
        plt.figure(figsize=(7, 5))
        plt.plot(range(1, len(y_tr)+1), y_tr, label="Train")
        plt.plot(range(1, len(y_val)+1), y_val, label="Validation")
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(fname, bbox_inches="tight", dpi=150)
        plt.close()
        saved[title] = fname

    _plot_and_save(train_loss_per_epoch, val_loss_per_epoch, "Loss over Epochs", "Loss", f"{out_prefix}_loss.png")
    _plot_and_save(train_acc_per_epoch,  val_acc_per_epoch,  "Accuracy over Epochs", "Accuracy", f"{out_prefix}_accuracy.png")
    _plot_and_save(train_aucroc_per_epoch, val_aucroc_per_epoch, "AUROC over Epochs", "AUROC", f"{out_prefix}_auroc.png")

    return saved

def main() -> None:
    """
    Run the full training pipeline: build dataloaders, train the model, 
    evaluate on the test set, and plot metrics.
    """

    # Determine device that we are training on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get the data loaders
    train_loader, val_loader, test_loader = get_data_loaders(
        train_batch_size=BATCH_TRIPLET,
        test_val_batch_size=BATCH_TEST
    )

    # Initalise Model
    model = SiameseNet(OUT_DIM).to(device)

    # Initialise loss functions
    triplet_loss = TripletLoss().to(device)
    classifier_loss = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)

    # Initialise Optimiser and Scheduler
    optimizer = optim.Adam(model.parameters(), lr=LR_SIAMESE)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=SCHED_FACTOR, patience=SCHED_PATIENCE)
    scaler = GradScaler()

    # Train the model
    train_loss_per_epoch, train_acc_per_epoch, train_auc_per_epoch, val_loss_per_epoch, val_acc_per_epoch, val_auc_per_epoch = train_siamese_network(
        train_loader,
        val_loader,
        model,
        optimizer,
        scheduler,
        triplet_loss,
        classifier_loss,
        EPOCHS_SIAMESE,
        scaler,
        device
    )

    # Plot train and validation metrics
    plot_training_graphs(
        train_loss_per_epoch=train_loss_per_epoch,
        val_loss_per_epoch=val_loss_per_epoch,
        train_acc_per_epoch=train_acc_per_epoch,
        val_acc_per_epoch=val_acc_per_epoch,
        train_aucroc_per_epoch=train_auc_per_epoch,
        val_aucroc_per_epoch=val_auc_per_epoch,
    )

    model.load_state_dict(torch.load(os.path.join(MODEL_PATH, "siamese.pth")))

    # Test the model with test dataset
    results_siamese_network(test_loader, model, device)

    # Now do baseline
    model = ResNetBaseline().to(device)

    best_val_auc_base = train_siamese_network(
        train_loader,
        val_loader,
        model,
        optimizer,
        scheduler,
        EPOCHS_SIAMESE,
        scaler,
        device
    )

    model.load_state_dict(torch.load(os.path.join(MODEL_PATH, "resnet_baseline.pth")))
    results_baseline(test_loader, model, device)

if __name__ == '__main__':
    main()
