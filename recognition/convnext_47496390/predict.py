"""
Shows example usage of trained model by loading saved model weights, 
runs inference on test cases.
Prints test accuracy, classification report, confusion matrix, and AUROC.
Produces visualisations of:
    - Test Confusion Matrix ({model}_test_cm.png)
    - Receiver Operating Characteristic (ROC) curve ({model}_test_roc.png)
    - Failure Case Autopsy Grid ({model}_misclassified_slices.png)
    - Successful Diagnostic Sample Predictions ({model}_sample_predictions.png)
"""

from modules import BaselineCNN, convnext_tiny
from dataset import build_dataloaders
from train import test, get_device, CLASS_NAMES

import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, roc_auc_score



def load_trained_model(model_name: str, weights_path: str, device: torch.device) -> nn.Module:
    """
    Instantiates the model and loads pre-trained weights.

    Args:
        model_name (str): 'convnext' or 'cnn'.
        weights_path (str): File path to the saved .pth weights.
        device (torch.device): Device to map model weights onto.
    Returns:
        nn.Module: Model in eval mode on target device.
    """
    model_name = model_name.lower()
    if model_name == "convnext":
        model = convnext_tiny(num_classes=2, in_chans=1)
    elif model_name == "cnn":
        model = BaselineCNN(num_classes=2, in_channels=1)
    else:
        raise ValueError(f"Unknown model name '{model_name}'. Choose 'convnext' or 'cnn'.")
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Weights file '{weights_path}' was not found!\n"
            f"Please train the model first by running: python train.py --model {model_name}"
        )

    print(f"Loading weights from: {weights_path}")
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model



# https://share.gemini.google/0E2OlinoIE2n
def plot_training_and_test_curves(
    train_losses,
    train_accuracies,
    val_losses,
    val_accuracies,
    val_epochs,
    test_loss=None,
    test_accuracy=None,
    save_path="training_test_curves.png",
):
  """Plots training and validation curves alongside final test performance markers."""
  train_epochs = list(range(1, len(train_losses) + 1))
  total_epochs = len(train_losses)

  fig, axes = plt.subplots(1, 2, figsize=(14, 5))

  # 1. Loss Curve
  axes[0].plot(train_epochs, train_losses, label="Train Loss", marker="o")
  axes[0].plot(val_epochs, val_losses, label="Val Loss", marker="s")
  if test_loss is not None:
    axes[0].axhline(
        y=test_loss,
        color="crimson",
        linestyle="--",
        alpha=0.8,
        label=f"Test Loss ({test_loss:.4f})",
    )
    axes[0].scatter(
        [total_epochs],
        [test_loss],
        color="crimson",
        s=80,
        zorder=5,
        marker="*",
    )

  axes[0].set_title("Loss vs. Epochs", fontsize=14)
  axes[0].set_xlabel("Epoch", fontsize=12)
  axes[0].set_ylabel("Loss", fontsize=12)
  axes[0].set_xticks(train_epochs)
  axes[0].legend()
  axes[0].grid(True, linestyle="--", alpha=0.6)

  # 2. Accuracy Curve
  axes[1].plot(
      train_epochs, train_accuracies, label="Train Accuracy", marker="o"
  )
  axes[1].plot(val_epochs, val_accuracies, label="Val Accuracy", marker="s")
  if test_accuracy is not None:
    axes[1].axhline(
        y=test_accuracy,
        color="crimson",
        linestyle="--",
        alpha=0.8,
        label=f"Test Acc ({test_accuracy:.2f}%)",
    )
    axes[1].scatter(
        [total_epochs],
        [test_accuracy],
        color="crimson",
        s=80,
        zorder=5,
        marker="*",
    )

  axes[1].set_title("Accuracy vs. Epochs", fontsize=14)
  axes[1].set_xlabel("Epoch", fontsize=12)
  axes[1].set_ylabel("Accuracy (%)", fontsize=12)
  axes[1].set_xticks(train_epochs)
  axes[1].legend()
  axes[1].grid(True, linestyle="--", alpha=0.6)

  plt.tight_layout()
  if save_path:
    plt.savefig(save_path, dpi=300)
    print(f"Curves with test results saved to {save_path}")



def plot_confusion_matrix(y_true, y_pred, class_names=("NC", "AD"), save_path="test_cm.png"):
    """Plots and saves an annotated confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=11)
    ax.set_yticklabels(class_names, fontsize=11)
    ax.set_xlabel("Predicted Label", labelpad=10, fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Test Confusion Matrix", pad=15, fontsize=13, fontweight="bold")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, f"{cm[i, j]:,}",
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=13, fontweight="bold"
            )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")

def plot_roc_curve(y_true, y_scores, pos_label=1, save_path="test_roc.png"):
    """Computes and plots Receiver Operating Characteristic (ROC) curve with AUC."""
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#d95f02", lw=2.5, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Random Classifier (AUC = 0.5000)")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    ax.set_title("Receiver Operating Characteristic (ROC)", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)

    plt.close()
    print(f"ROC curve saved to: {save_path}")

def _render_image_grid(samples, class_names, title_prefix, save_path, max_samples=8, ncols=4):
    """Internal helper to un-normalize and plot grids of MRI slice predictions."""
    if isinstance(samples, dict):
        flat_samples = []
        for k in sorted(samples.keys()):
            flat_samples.extend(samples[k])
        samples = flat_samples

    n_samples = min(len(samples), max_samples)
    if n_samples == 0:
        print(f"No samples found to plot for {save_path}.")
        return
    
    nrows = int(np.ceil(n_samples / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.8 * ncols, 4.0 * nrows))
    axes = np.array([axes]).flatten() if n_samples == 1 else axes.flatten()

    for idx in range(n_samples):
        sample = samples[idx]
        img = sample["image"]
        true_label = class_names[sample["true"]]
        pred_label = class_names[sample["pred"]]
        confidence = sample["confidence"] * 100.0
        
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()

        # Un-normalize single-channel (mean=0.5, std=0.5)
        if img.ndim == 3 and img.shape[0] == 1:
            img = img.squeeze(0) * 0.5 + 0.5
        elif img.ndim == 3 and img.shape[0] == 3:
            mean = np.array([0.485, 0.456, 0.406])[:, None, None]
            std = np.array([0.229, 0.224, 0.225])[:, None, None]
            img = np.transpose(img * std + mean, (1, 2, 0))

        img = np.clip(img, 0.0, 1.0)
        axes[idx].imshow(img, cmap="gray")
        axes[idx].axis("off")

        is_error = (pred_label != true_label)
        if is_error:
            error_type = "False Positive" if pred_label == "AD" else "False Negative"
            color = "crimson"
            title_text = f"Pred: {pred_label} ({confidence:.1f}%)\nTrue: {true_label} [{error_type}]"
        else:
            color = "#2ca02c"
            title_text = f"Pred: {pred_label} ({confidence:.1f}%)\nTrue: {true_label} [Correct]"

        axes[idx].set_title(title_text, fontsize=10, color=color, fontweight="bold")

    for idx in range(n_samples, len(axes)):
        axes[idx].axis("off")

    plt.suptitle(title_prefix, fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Sample grid saved to: {save_path}")


def plot_misclassified_grid(misclassified_samples, class_names=("NC", "AD"), max_samples=8, save_path="misclassified_slices.png"):
    """Plots failure case autopsy grid displaying misclassifications with confidence scores."""
    _render_image_grid(
        misclassified_samples,
        class_names,
        title_prefix="Failure Case Autopsy (False Positives & False Negatives)",
        save_path=save_path,
        max_samples=max_samples,
        ncols=4
    )

def plot_predictions_grid(correct_samples, class_names=("NC", "AD"), max_samples=8, save_path="sample_predictions.png"):
    """Plots correctly classified representative MRI test cases."""
    _render_image_grid(
        correct_samples,
        class_names,
        title_prefix="Representative Correct Diagnostic Predictions",
        save_path=save_path,
        max_samples=max_samples,
        ncols=4
    )

def evaluate_patient_level(test_dataset, all_probabilities, all_labels):
    """
    Aggregates slice-level probabilities by patient_id using soft-voting.
    """
    patient_probs = defaultdict(list)
    patient_true = {}

    for idx, pid in enumerate(test_dataset.patient_ids):
        patient_probs[pid].append(all_probabilities[idx][1]) # Probability of AD
        patient_true[pid] = all_labels[idx]

    patient_preds = []
    patient_targets = []

    for pid, probs_list in patient_probs.items():
        # Soft-voting: average AD probability across all slices for this patient
        mean_prob = sum(probs_list) / len(probs_list)
        pred_label = 1 if mean_prob >= 0.50 else 0
        patient_preds.append(pred_label)
        patient_targets.append(patient_true[pid])

    correct = sum(p == t for p, t in zip(patient_preds, patient_targets))
    patient_acc = 100.0 * correct / len(patient_targets)

    print("\n" + "=" * 55)
    print("--- PATIENT-LEVEL EVALUATION (450 PATIENTS) ---")
    print("=" * 55)
    print(f"Total Unique Patients: {len(patient_targets)}")
    print(f"Patient-Level Accuracy: {patient_acc:.2f}%")
    print("=" * 55 + "\n")
    return patient_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference demonstration on test cases with saved model.")
    parser.add_argument("--model", type=str, default="convnext", choices=["convnext", "cnn"],
                        help="Model architecture: 'convnext' or 'cnn' (default: convnext)")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to pre-trained .pth checkpoint (default: {model}_best.pth)")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader num_workers (default: 4)")
    parser.add_argument("--data_root", type=str, default="/home/groups/comp3710/ADNI/AD_NC",
                        help="Path to ADNI AD_NC dataset (default: /home/groups/comp3710/ADNI/AD_NC)")
    parser.add_argument("--batch_size", type=int, default=16, help="Inference batch size (default: 16)")
    parser.add_argument("--save_dir", type=str, default=".", help="Directory to save diagnostic plots (default: .)")

    args = parser.parse_args()

    # Determine weights path
    weights_path = args.weights if args.weights else f"{args.model.lower()}_best.pth"
    device = get_device()

    # Load pre-trained model
    model = load_trained_model(args.model, weights_path, device)
    
    # Load test DataLoader (using in_channels=1 native grayscale)
    _, _, test_loader = build_dataloaders(
        in_channels=1,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Run inference evaluation
    results = test(model, test_loader, nn.CrossEntropyLoss(), device)

    # Generate diagnostic plots
    os.makedirs(args.save_dir, exist_ok=True)
    cm_path = os.path.join(args.save_dir, f"{args.model.lower()}_test_cm.png")
    roc_path = os.path.join(args.save_dir, f"{args.model.lower()}_test_roc.png")
    autopsy_path = os.path.join(args.save_dir, f"{args.model.lower()}_misclassified_slices.png")
    samples_path = os.path.join(args.save_dir, f"{args.model.lower()}_sample_predictions.png")
    
    plot_confusion_matrix(results["all_labels"], results["all_predictions"], class_names=CLASS_NAMES, save_path=cm_path)
    ad_probs = [p[1] for p in results["all_probabilities"]]
    plot_roc_curve(results["all_labels"], ad_probs, pos_label=1, save_path=roc_path)
    plot_misclassified_grid(results["misclassified_samples"], class_names=CLASS_NAMES, max_samples=8, save_path=autopsy_path)
    plot_predictions_grid(results["correct_samples"], class_names=CLASS_NAMES, max_samples=8, save_path=samples_path)
   
    evaluate_patient_level(test_loader.dataset, results["all_probabilities"], results["all_labels"])

    print("\nInference demonstration completed successfully! All diagnostic plots generated.")
