import os
import torch
import torch.nn as nn
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, roc_curve
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from typing import Tuple
from modules import SiameseNet, ResNetBaseline

from params import (
    DATA_PATH, MODEL_PATH, IMAGE_PATH, TRIPLET_MARGIN,
)

def produce_evaluation_metrics(
    test_pred: list,
    test_probs: list,
    test_true: list
) -> None:
    """
    Compute and report evaluation metrics for a binary classifier
    Calculates and prints:
      - Testing overall accuracy.
      - ROC-AUC score.
      - Testing Sensitivity. 
      - Testing Specificity.
    Args:
        test_pred (list): predicted labels for each sample.
        test_probs (list): predicted probability for the positive class for each sample.
        test_true (list): ground-truth labels.
    """

    test_accuracy = accuracy_score(test_true, test_pred)
    test_auc_roc = roc_auc_score(test_true, test_probs)
    print(f"Testing Accuracy: {test_accuracy}")
    print(f"Testing AUR ROC: {test_auc_roc}")

    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(test_true, test_pred)
    tn, fp, fn, tp = conf_matrix.ravel()

    sensitivity = tp / (tp + fn) # Calculate Sensitivity (True Positive Rate)
    specificity = tn / (tn + fp) # Calculate Specificity (True Negative Rate)
    print(f"Testing Sensitivity (Recall): {sensitivity:.3f}")
    print(f"Testing Specificity: {specificity:.3f}")

def plot_tsne_from_embeddings(
    embeddings: np.ndarray | torch.Tensor, 
    test_pred: list,
    out_path: str = os.path.join(IMAGE_PATH, "testing_tsne_embeddings.png"), 
    title: str = "t-SNE visualization of embeddings"
) -> None:
    """
    Create a 2D t-SNE plot from high-dimensional embeddings and save it to an image file
    Args:
        embeddings (np.ndarray | torch.Tensor): contain embeddings of dimension
        test_pred (list): predicted labels for each sample.
        out_path (str): path to save the output image file
        title (str): figure title to display on the plot
    """

    # Set up t-SNE to reduce D-dim embeddings to 2D 
    tsne = TSNE(n_components=2, init="pca", random_state=42, max_iter=2000, learning_rate=200, perplexity=30)

    # Run t-SNE and get the 2D coordinates for each embedding
    X2 = tsne.fit_transform(embeddings)

    # plot with a single color since no labels provided
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(X2[:, 0], X2[:, 1], s=10, alpha=0.8, c=test_pred, cmap="cividis")
    plt.colorbar(sc)
    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_confusion_matrix(
    test_true: list,
    test_pred: list,
    out_path: str = os.path.join(IMAGE_PATH, "testing_confusion_matrix.png"), 
    title: str ="Confusion Matrix (Percentages)" 
) -> None:
    """
    Plot the confusion matrix and save it to an image file
    Args:
        test_true (list): ground-truth labels.
        test_pred (list): predicted labels for each sample.
        out_path (str): path to save the output image file.
        title (str): figure title to display on the plot.
    """

    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(test_true, test_pred)

    # Normalize the confusion matrix by rows (i.e., by the actual class counts)
    conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    # Plot the normalized confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix_normalized,
        annot=True,
        fmt='.2%',
        cmap='Greens', 
        xticklabels=['Normal (0)', 'Melanoma (1)'],
        yticklabels=['Normal (0)', 'Melanoma (1)']
    )
    plt.title(title)
    plt.ylabel('Actual Labels')
    plt.xlabel('Predicted Labels')
    plt.savefig(out_path)
    plt.close()

def plot_roc_curve(
    test_true: list, 
    test_probs: list,
    out_path: list =os.path.join(IMAGE_PATH, "roc_curve.png"), 
    title: list ="ROC Curve"
):
    """
    Plot the roc curve and save it to an image file
    Args:
        test_true (list): ground-truth labels.
        test_probs (list): predicted probability for the positive class for each sample.
        out_path (str): path to save the output image file.
        title (str): figure title to display on the plot.
    """
    y_true = np.asarray(test_true)
    y_prob = np.asarray(test_probs, dtype=float)

    # Need both classes present to compute ROC
    if np.unique(y_true).size < 2:
        print("[plot_roc_curve] Only one class present in y_true; skipping ROC.")
        return None

    # Compute false positive rate and true positive rate across thresholds
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    # Plot roc curve 
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label="ROC curve")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.0])
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path

def test_siamese_network(
    test_loader: DataLoader,
    model: SiameseNet,
    device: str
)-> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate the model on the test set and return predictions, probabilities, true labels, and embeddings
    Uses the classifier head on anchor images and also collects their embeddings
    Args:
        test_loader (DataLoader): Dataloader for test dataset.
        model (SiameseNet): Trained model.
        device (str): Compute device.
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            (preds, probs, labels, embeddings_2d) 
    """

    model.eval() 
    all_labels, all_probs, all_preds, all_embeddings = [], [], [], []

    with torch.no_grad():
        for batch_idx, (anchor, _, _, labels) in enumerate(test_loader):
            # Move data to the specified device for faster computation
            anchor = anchor.to(device).float()

            embeddings = model.get_embedding(anchor)
            classifier_out = model.classify(anchor) 

            # Probability of the positive class (class index 1)           
            probs = torch.softmax(classifier_out, dim=1)[:, 1]
            # Predicted class index via argmax over logits
            _, preds = classifier_out.max(1)

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_embeddings.append(embeddings.cpu().numpy())

    return np.array(all_preds), np.array(all_probs), np.array(all_labels), np.concatenate(all_embeddings)

def results_siamese_network(
    test_loader: DataLoader,
    model: SiameseNet,
    device: str
):
    """
    Run the full evaluation pipeline on the test dataset and generate metrics/plots
    Executes inference to get predictions, probabilities, true labels, and embeddings,
    then prints summary metrics:
      - evaluation metrics (accuracy, precision, recall, F1, ROC-AUC)
      - confusion matrix
      - t-SNE visualization of embeddings
      - ROC curve
    Args:
        test_loader (DataLoader): Dataloader for training dataset.
        model (SiameseNet): Trained model.
        device (str): compute device.
    """
    # Inference
    test_pred, test_probs, test_true, test_embeddings = test_siamese_network(test_loader=test_loader, model=model, device=device)

    # Metrics + visualizations
    produce_evaluation_metrics(test_pred, test_probs, test_true)
    plot_confusion_matrix(test_true, test_pred)
    plot_tsne_from_embeddings(test_embeddings, test_pred)
    plot_roc_curve(test_true, test_probs)

def test_baseline_network(test_loader, model, device):
    model.eval()
    all_labels, all_probs, all_preds = [], [], []

    with torch.no_grad():
        for anchor, _, _, labels in test_loader:
            anchor = anchor.to(device).float()
            logits = model(anchor)
            probs = torch.softmax(logits, dim=1)[:, 1]
            _, preds = logits.max(1)

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    return np.array(all_preds), np.array(all_probs), np.array(all_labels)

def results_baseline(
    test_loader: DataLoader,
    model: ResNetBaseline,
    device: str
):
    test_pred, test_probs, test_true = test_baseline_network(test_loader, model, device)
    print("\n--- Baseline ResNet-50 Test Results ---")
    produce_evaluation_metrics(test_pred.tolist(), test_probs.tolist(), test_true.tolist())
    plot_confusion_matrix(test_true.tolist(), test_pred.tolist(), out_path=os.path.join(IMAGE_PATH, "baseline_confusion_matrix.png"))
    plot_roc_curve(test_true.tolist(), test_probs.tolist(), out_path=os.path.join(IMAGE_PATH, "baseline_roc_curve.png"))