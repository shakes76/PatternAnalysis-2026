"""
Contains source code for  for training, validating, testing, and saving model for ADNI brain classification. 
The model is imported from modules.py and the dataloader is imported from dataset.py. Training metrics and losses are plotted.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import os
import time
import argparse

# Training for Baseline CNN

from modules import BaselineCNN, convnext_base, convnext_tiny
from dataset import build_dataloaders

CLASS_NAMES = ["NC", "AD"]

def get_device():
    """Returns CUDA device if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    print("Warning: CUDA not found. Operating on CPU.")
    return torch.device("cpu")

device = get_device()

def create_model(model_name, num_classes=2, in_channels=1):
    """Creates and returns the model on the device."""

    model_name = model_name.lower()

    if model_name == "cnn":
        model = BaselineCNN(num_classes=num_classes, in_channels=in_channels)
    elif model_name == "convnext":
        model = convnext_tiny(num_classes=num_classes, in_chans=in_channels)
    else:
        raise ValueError(f"Unknown model name '{model_name}'. Choose 'cnn' or 'convnext'")

    return model.to(device)

def train_epoch(model, loader, criterion, optimizer, dev):
    """Trains model for one epoch and returns (loss, accuracy)"""
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels, _ in loader:
        images = images.to(dev, non_blocking=True)
        labels = labels.to(dev, dtype=torch.long, non_blocking=True)
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = 100 * correct/total

    return epoch_loss, epoch_accuracy

def validate(model, loader, criterion, dev):
    """Validates the model and returns (loss, accuracy)"""
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(dev, non_blocking=True)
            labels = labels.to(dev, dtype=torch.long, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)
            
            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += batch_size

    loss = running_loss / total
    accuracy = 100.0 * correct / total
    return loss, accuracy

def test(model, loader, criterion, dev, n_mis_per_class=4, n_correct_per_class=4):
    """Eeal model performance on the test set.
    Reports accuracy, per-class precision/recall/F1, confusion matrix, and AUROC."""
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_predictions = []
    all_probabilities = []
    all_patient_ids = []
    misclassified_samples = {0: [], 1: []}   # keyed by TRUE class (0 = false positives, 1 = false negatives)
    correct_samples = {0: [], 1: []}

    # Latency tracking
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    inference_start = time.perf_counter()

    with torch.no_grad():
        for images, labels, pids in loader:
            images = images.to(dev, non_blocking=True)
            labels = labels.to(dev, dtype=torch.long, non_blocking=True)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Get predicted classes
            predictions = outputs.argmax(dim=1) # predicted class index for each sample
            probs = torch.softmax(outputs, dim=1) # probability distribution across all classes for each sample
            confidences = probs.max(dim=1).values #  probability assigned to the predicted class for each sample

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            correct += (predictions == labels).sum().item()
            total += batch_size

            labels_l = labels.cpu().tolist()
            preds_l = predictions.cpu().tolist()
            confs_l = confidences.cpu().tolist()

            all_labels.extend(labels_l)
            all_predictions.extend(preds_l)
            all_probabilities.extend(probs.cpu().tolist())
            all_patient_ids.extend(pids)

            # Collect example slices (copy to CPU only when actually kept)
            for i in range(batch_size):
                t, p = labels_l[i], preds_l[i]
                is_correct = (p == t)
                bucket = correct_samples if is_correct else misclassified_samples
                cap = n_correct_per_class if is_correct else n_mis_per_class

                if len(bucket[t]) < cap:
                    bucket[t].append({"image": images[i].cpu(), "pred": p,
                                      "true": t, "confidence": confs_l[i]})


    if dev.type == "cuda":
        torch.cuda.synchronize(dev)

    inference_elapsed = time.perf_counter() - inference_start
    latency_ms = (inference_elapsed / total) * 1000.0 if total > 0 else 0.0

    test_loss = running_loss / total
    test_accuracy = 100.0 * correct / total

    # Compute AUROC for class 1 (AD)
    ad_scores = [p[1] for p in all_probabilities]
    try:
        test_auroc = roc_auc_score(all_labels, ad_scores)
    except Exception as e:
        print(f"Warning: Could not compute AUROC: {e}")
        test_auroc = float("nan")
    print("\n" + "=" * 50)
    print("--- Test Evaluation Results ---")
    print("=" * 50)
    print(f"Test Loss:             {test_loss:.4f}")
    print(f"Test Accuracy:         {test_accuracy:.2f}%")
    print(f"Test AUROC:            {test_auroc:.4f}")
    print(f"Inference Latency:     {latency_ms:.2f} ms / sample ({total} test slices)")

    print("\nClassification Report:")
    print(classification_report(
        all_labels,
        all_predictions,
        labels=[0, 1],
        target_names=CLASS_NAMES,
        zero_division=0,
        digits=4
    ))
    
    print("Confusion Matrix (rows: true, cols: pred):")
    print(confusion_matrix(all_labels, all_predictions, labels=[0, 1]))
    print("=" * 50 + "\n")

    return {
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "test_auroc": test_auroc,
        "latency_ms": latency_ms,
        "all_labels": all_labels,
        "all_predictions": all_predictions,
        "all_probabilities": all_probabilities,
        "misclassified_samples": misclassified_samples[0] + misclassified_samples[1],
        "correct_samples": correct_samples[0] + correct_samples[1],
        }

def plot_training_curves(
    train_losses,
    train_accuracies,
    val_losses,
    val_accuracies,
    val_epochs,
    test_loss=None,
    test_accuracy=None,
    save_path="training_curves.png"
):
    """Plots and saves loss and accuracy curves 
    """
    train_epochs = list(range(1, len(train_losses) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss Curve
    axes[0].plot(train_epochs, train_losses, label="Train Loss", color="#1f77b4", lw=2)
    if val_losses:
        axes[0].plot(val_epochs, val_losses, label="Val Loss", color="#ff7f0e", marker="o", lw=2)
    if test_loss is not None:
        axes[0].axhline(y=test_loss, color="crimson", linestyle="--", alpha=0.8, label=f"Test Loss ({test_loss:.4f})")
    axes[0].set_title("Cross-Entropy Loss vs. Epochs", fontsize=13)
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Loss", fontsize=11)
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # Accuracy Curve
    axes[1].plot(train_epochs, train_accuracies, label="Train Accuracy", color="#1f77b4", lw=2)
    if val_accuracies:
        axes[1].plot(val_epochs, val_accuracies, label="Val Accuracy", color="#ff7f0e", marker="o", lw=2)
    if test_accuracy is not None:
        axes[1].axhline(y=test_accuracy, color="crimson", linestyle="--", alpha=0.8, label=f"Test Acc ({test_accuracy:.2f}%)")
    axes[1].set_title("Classification Accuracy vs. Epochs", fontsize=13)
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Accuracy (%)", fontsize=11)
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Training and validation curves saved to: {save_path}")

def pipeline(
    model_name: str = "convnext",
    data_root: str = "/home/groups/comp3710/ADNI/AD_NC",
    num_epochs: int = 100,
    batch_size: int = 16,
    lr: float = 1e-4,
    weight_decay: float = 0.05,
    warmup_epochs: int = 3,
    min_lr: float = 1e-6,
    num_workers: int = 4,
    val_split: float = 0.2
):
    print(f"\nInitializing {model_name.upper()} training pipeline on {device}...")

    model = create_model(model_name=model_name, num_classes=2, in_channels=1)
    
    # Trainable parameter counting
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
    
    # Loss function with label smoothing for calibrated probabilities
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1) # training
    eval_criterion = nn.CrossEntropyLoss() # val + test
    
    # AdamW with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Sequential Warmup + Cosine Annealing learning rate scheduler
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.01,  # Start at 1% of the target LR
        end_factor=1.0,     # Reach the target LR
        total_iters=warmup_epochs
    )

    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, num_epochs - warmup_epochs),
        eta_min=min_lr
    )

    # Combine the two phases
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )

    # Build DataLoaders
    train_loader, val_loader, test_loader = build_dataloaders(
        in_channels=1,
        data_root=data_root,
        val_split=val_split,
        batch_size=batch_size,
        num_workers=num_workers
    )


    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    val_epochs = []

    best_val_accuracy = -1.0


    checkpoint_path = f"{model_name.lower()}_best.pth"
    curves_path = f"{model_name.lower()}_training_curves.png"

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    print("\nTraining model...")
    
    start_time = time.perf_counter()

    for epoch in range(num_epochs):
        train_loss, train_accuracy = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )
        scheduler.step()

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f}, "
            f"Train Accuracy: {train_accuracy:.2f}%"
        )

        # validate every 2nd epoch or final epoch
        if (epoch + 1) % 2 == 0 or (epoch + 1) == num_epochs:
            val_loss, val_accuracy = validate(
                model,
                val_loader,
                eval_criterion,
                device,
            )

            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            val_epochs.append(epoch + 1)

            print(
                f"           Epoch [{epoch + 1}/{num_epochs}] "
                f"Val Loss: {val_loss:.4f}, "
                f"Val Accuracy: {val_accuracy:.2f}%"
            )

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(model.state_dict(), checkpoint_path)
                print(f"                 --> Best checkpoint saved to '{checkpoint_path}' (Val Acc: {val_accuracy:.2f}%)")

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        
    end_time = time.perf_counter()
    total_training_time = end_time - start_time
    print(f"Total training and validation time: {total_training_time:.2f} seconds")

    # Profile Peak GPU VRAM
    peak_vram_mb = (torch.cuda.max_memory_allocated(device) / (1024 ** 2)) if device.type == "cuda" else 0.0
    print(f"Peak GPU VRAM allocated: {peak_vram_mb:.2f} MB")

    # Testing
    print("Testing model")

    # load best model
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    results = test(
        model,
        test_loader,
        eval_criterion,
        device,
    )

    # Plot and save training/validation curves
    plot_training_curves(
        train_losses,
        train_accuracies,
        val_losses,
        val_accuracies,
        val_epochs,
        test_loss=results["test_loss"],
        test_accuracy=results["test_accuracy"],
        save_path=curves_path
    )

    # Print Computational Resource Profiling Summary Table
    print("\n" + "=" * 65)
    print("--- COMPUTATIONAL RESOURCE PROFILING SUMMARY ---")
    print("=" * 65)
    print(f"Model Architecture:          {model_name.upper()}")
    print(f"Total Trainable Parameters:  {total_params:,}")
    print(f"Peak GPU VRAM Allocated:     {peak_vram_mb:.2f} MB")
    print(f"Total Training Runtime:      {total_training_time:.2f} seconds ({total_training_time / num_epochs:.2f} s/epoch)")
    print(f"Inference Latency:           {results['latency_ms']:.2f} ms / sample")
    print(f"Final Test Accuracy:         {results['test_accuracy']:.2f}%")
    print(f"Final Test AUROC:            {results['test_auroc']:.4f}")
    print("=" * 65 + "\n")
    return {
        "model_name": model_name,
        "total_params": total_params,
        "peak_vram_mb": peak_vram_mb,
        "training_time": total_training_time,
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies,
        "val_epochs": val_epochs,
        **results,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BaselineCNN or ConvNeXt on ADNI Brain MRI dataset.")
    parser.add_argument("--model", type=str, default="convnext", choices=["convnext", "cnn"],
                        help="Model architecture: 'convnext' or 'cnn' (default: convnext)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs (default: 100)")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training/eval (default: 128)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Peak learning rate (default: 1e-4)")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="AdamW weight decay (default: 0.05)")
    parser.add_argument("--data_root", type=str, default="/home/groups/comp3710/ADNI/AD_NC",
                        help="Root folder of the ADNI AD_NC dataset (default: /home/groups/comp3710/ADNI/AD_NC)")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader num_workers (default: 4)")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation fraction of patients (default: 0.2)")
    args = parser.parse_args()
    pipeline(
        model_name=args.model,
        data_root=args.data_root,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        val_split=args.val_split
    ) 
