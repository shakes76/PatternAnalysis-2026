"""
Author: Lawrence Tiang
Student ID: s4892210

Description:
    Evaluate both the baseline CNN and ConvNeXt models using their
    saved checkpoints. Calculates classification metrics, including
    accuracy, F1-score, AUROC, precision, recall and confusion matrix.
    It also measures inference latency and peak GPU memory usage.
    Generates visualisations to support model evaluation, including ROC curve,
    accuracy vs confidence plot and examples of correct and failure predictions.

Usage:
    python predict.py
"""

import os
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay
)
from dataset import ADNI_MEAN, ADNI_STD, build_dataloader, print_dataset_summary
from modules import build_model

# Configuration settings and checkpoint paths
OUTPUT_DIR = "assets"
BATCH_SIZE = 256
BASELINE_CHECKPOINT = os.path.join(OUTPUT_DIR, "best_baseline.pth")
CONVNEXT_CHECKPOINT = os.path.join(OUTPUT_DIR, "best_convnext.pth")
SEED = 42

# Set random seeds for reproducibility
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


def load_model(model_name, checkpoint_path):
    """
    Load a trained model and its saved parameters.

    Args:
        model_name (str): Name of the model architecture to construct.
        checkpoint_path (str): Path to the saved model checkpoint.

    Returns:
        torch.nn.Module: The trained model in evaluation mode.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
    """
    # Ensure the trained model checkpoint exists
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Build requested model and load its saved weights
    model = build_model(model_name)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    model = model.to(DEVICE)
    model.eval()

    return model


def count_parameters(model):
    """
    Count the trainable parameters in a model.

    Args:
        model (torch.nn.Module): Model whose parameters are counted.

    Returns:
        int: Total number of trainable parameters.
    """
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def run_inference(model, dataloader):
    """
    Generate predictions and measure inference resource usage.

    Args:
        model (torch.nn.Module): Trained model to evaluate.
        dataloader (DataLoader): DataLoader containing evaluation images
            and their corresponding labels.

    Returns:
        tuple: Predicted labels, true labels, class probabilities,
            input images, inference latency in milliseconds per image
            and peak GPU memory allocated in bytes.
    """
    # Set the model to evaluation mode and initialise results
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    all_images = []

    # Clear cached GPU memory and reset peak allocation statistics
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(DEVICE)

    warmup_complete = False
    with torch.inference_mode():
        # Transfer the current batch to the selected device
        for images, labels in dataloader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # Warm up CUDA before measuring latency
            if DEVICE.type == "cuda" and not warmup_complete:
                for _ in range(5):
                    _ = model(images)
                torch.cuda.synchronize()
                warmup_complete = True

            # Convert and select model outputs with
            # highest predicted class probabilities
            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1)
            predictions = probabilities.argmax(dim=1)

            # Collect predictions, labels and probabilities on the CPU
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.append(probabilities.cpu().numpy())

            # Store CPU images for example figures.
            all_images.append(images.cpu().numpy())

    # Combine batch results into arrays for metric calculation and plotting
    probabilities = np.concatenate(all_probabilities, axis=0)
    images = np.concatenate(all_images, axis=0)

    # Measure the average model forward-pass latency per image
    latency = measure_inference_latency(model, dataloader)

    # Record peak GPU memory allocation
    if DEVICE.type == "cuda":
        peak_vram = torch.cuda.max_memory_allocated(DEVICE)
    else:
        peak_vram = 0

    return (
        np.array(all_predictions),
        np.array(all_labels),
        probabilities,
        images,
        latency,
        peak_vram
    )


def measure_inference_latency(model, dataloader):
    """
    Measure average model forward-pass latency per image.

    Args:
        model (torch.nn.Module): Model whose inference speed is measured.
        dataloader (DataLoader): DataLoader containing evaluation images.

    Returns:
        float: Average forward-pass latency in milliseconds per image.

    """
    # Enable evaluation behaviour
    model.eval()
    total_time = 0.0
    total_images = 0

    # Warm up the model before timing inference
    with torch.inference_mode():
        for batch_index, (images, _) in enumerate(dataloader):
            images = images.to(DEVICE)
            if batch_index >= 5:
                break

            _ = model(images)

    # Ensure warm-up GPU operations completed before timing
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    # Measure the forward-pass time for each batch
    with torch.inference_mode():
        for images, _ in dataloader:
            images = images.to(DEVICE)

            # Synchronise before timing
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()

            start_time = time.perf_counter()
            _ = model(images)

            # Wait for the forward pass to finish
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()

            end_time = time.perf_counter()

            # Accumulate elapsed time and the number of processed images
            total_time += end_time - start_time
            total_images += images.size(0)

    # Convert elapsed seconds into average milliseconds per image
    latency_ms_per_image = (total_time / total_images) * 1000.0
    return latency_ms_per_image


def calculate_metrics(predictions, labels, probabilities):
    """
    Calculate classification metrics for AD and NC predictions.

    Args:
        predictions (np.ndarray): Predicted class indices.
        labels (np.ndarray): Ground-truth class indices.
        probabilities (np.ndarray): Predicted probabilities for each class.

    Returns:
        dict: Accuracy, F1-score, AUROC, class-specific precision
            and recall and the confusion matrix.
    """
    # Calculate overall classification accuracy and macro-averaged F1-score
    accuracy = accuracy_score(labels, predictions)
    macro_f1 = f1_score(labels, predictions, average="macro")

    # Calculate precision and recall separately for AD (class 0) and NC (class 1)
    ad_precision = precision_score(labels, predictions, pos_label=0, zero_division=0)
    ad_recall = recall_score(labels, predictions, pos_label=0, zero_division=0)
    nc_precision = precision_score(labels, predictions, pos_label=1, zero_division=0)
    nc_recall = recall_score(labels, predictions, pos_label=1, zero_division=0)

    # Convert labels and probabilities so that AD is treated as positive
    ad_labels = (labels == 0).astype(int)
    ad_probabilities = probabilities[:, 0]
    auroc = roc_auc_score(ad_labels, ad_probabilities)

    # Create confusion matrix
    cm = confusion_matrix(labels, predictions)

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "auroc": auroc,
        "ad_precision": ad_precision,
        "ad_recall": ad_recall,
        "nc_precision": nc_precision,
        "nc_recall": nc_recall,
        "confusion_matrix": cm,
    }


def print_results(model_name, model, metrics, latency_ms, peak_vram):
    """
    Print classification performance and computational resource metrics.

    Args:
        model_name (str): Model identifier, either 'baseline' or 'convnext'.
        model (torch.nn.Module): Evaluated model.
        metrics (dict): Classification metrics returned by calculate_metrics.
        latency_ms (float): Average forward-pass latency per image.
        peak_vram (int): Peak allocated GPU memory in bytes.

    Returns:
        None: Prints a summary of performance and computational resource metrics
    """

    display_name = "Baseline CNN" if model_name == "baseline" else "ConvNeXt"
    parameter_count = count_parameters(model)

    # Print classification performance
    print(f"\n=== {display_name} Test Results ===")
    print(f"Test Accuracy       : {metrics['accuracy']:.4f}")
    print(f"F1-Score            : {metrics['macro_f1']:.4f}")
    print(f"AUROC               : {metrics['auroc']:.4f}")
    print(f"AD Precision        : {metrics['ad_precision']:.4f}")
    print(f"AD Recall           : {metrics['ad_recall']:.4f}")
    print(f"NC Precision        : {metrics['nc_precision']:.4f}")
    print(f"NC Recall           : {metrics['nc_recall']:.4f}")

    # Print model size and average forward-pass latency
    print(f"Parameters          : {parameter_count:,}")
    print(f"Inference Latency   : {latency_ms:.3f} ms/image")

    # Print peak GPU memory only when CUDA device is available
    if DEVICE.type == "cuda":
        print(f"Peak GPU VRAM      : {peak_vram / (1024 ** 3):.3f} GB")
    else:
        print("Peak GPU VRAM       : N/A (CPU)")


def plot_confusion_matrix(model_name, confusion_matrix_values):
    """
    Plot and save test confusion matrix.

    Args:
        model_name (str): Model identifier used in the plot title and filename.
        confusion_matrix_values (np.ndarray): Confusion matrix for AD and NC.

    Returns:
        None: Saves the confusion matrix as a PNG image.
    """

    display_name = "Baseline CNN" if model_name == "baseline" else "ConvNeXt"

    # Create confusion matrix visualisation
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix_values,
        display_labels=["AD", "NC"]
    )
    disp.plot(ax=ax, cmap="Blues", colorbar=True)
    ax.set_title(f"{display_name} Test Confusion Matrix")
    plt.tight_layout()

    # Save the confusion matrix
    path = os.path.join(OUTPUT_DIR, f"{model_name}_confusion_matrix.png")
    plt.savefig(path, dpi=200)
    plt.close()


def plot_roc_curve(labels, probabilities):
    """
    Plot and save ROC curve for ConvNeXt.

    Args:
        labels (np.ndarray): Ground-truth class indices.
        probabilities (np.ndarray): Predicted probabilities for each class.

    Returns:
        None: Saves the ROC curve as a PNG image.
    """
    # Convert labels and probabilities so that AD is treated as positive
    ad_labels = (labels == 0).astype(int)
    ad_probabilities = probabilities[:, 0]

    # Calculate ROC curve and area under the curve
    false_positive_rate, true_positive_rate, _ = roc_curve(ad_labels, ad_probabilities)
    auroc = roc_auc_score(ad_labels, ad_probabilities)

    # Plot model performance against random-classifier baseline
    plt.figure(figsize=(8, 6))
    plt.plot(
        false_positive_rate,
        true_positive_rate,
        label=f"ConvNeXt (AUROC = {auroc:.4f})"
    )
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ConvNeXt ROC Curve")
    plt.legend()
    plt.tight_layout()

    # Save the ROC curve
    path = os.path.join(OUTPUT_DIR, "convnext_roc_curve.png")
    plt.savefig(path, dpi=200)
    plt.close()


def plot_accuracy_vs_confidence(labels, predictions, probabilities):
    """
    Plot prediction accuracy across confidence intervals for ConvNeXt.

    Args:
        labels (np.ndarray): Ground-truth class indices.
        predictions (np.ndarray): Predicted class indices.
        probabilities (np.ndarray): Predicted probabilities for each class.

    Returns:
        None: Saves the accuracy vs confidence plot as a PNG image.
    """
    # Define each prediction's confidence as its highest class probability
    confidence = probabilities.max(axis=1)
    num_bins = 10
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    bin_centers = []
    bin_accuracies = []

    # Calculate classification accuracy within each confidence bin
    for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
        mask = ((confidence >= lower) & (confidence < upper))

        if upper == 1.0:
            mask = ((confidence >= lower) & (confidence <= upper))

        if np.sum(mask) == 0:
            continue

        bin_centers.append((lower + upper) / 2)
        bin_accuracies.append(np.mean(predictions[mask] == labels[mask]))

    # Plot empirical accuracy against midpoint of each confidence bin
    plt.figure(figsize=(8, 6))
    plt.plot(bin_centers, bin_accuracies, marker="o")
    plt.xlabel("Prediction Confidence")
    plt.ylabel("Accuracy")
    plt.title("ConvNeXt Accuracy vs Confidence")
    plt.xlim(0.5, 1.0)
    plt.ylim(0.0, 1.0)
    plt.tight_layout()

    # Save the accuracy vs confidence plot
    path = os.path.join(OUTPUT_DIR, "convnext_accuracy_vs_confidence.png")
    plt.savefig(path, dpi=200)
    plt.close()


def select_examples(
    labels,
    predictions,
    probabilities,
    correct=True,
    num_examples=9
):
    """
    Select correct or failure predictions across confidence intervals.

    Args:
        labels (np.ndarray): Ground-truth class indices.
        predictions (np.ndarray): Predicted class indices.
        probabilities (np.ndarray): Predicted probabilities for each class.
        correct (bool): If True, select correct predictions; otherwise,
            select failure predictions.
        num_examples (int): Maximum number of examples to select.

    Returns:
        list: Indices of the selected examples.
    """
    # Calculate each prediction's confidence
    confidence = probabilities.max(axis=1)

    # Find indices of correct or failure predictions
    if correct:
        candidate_indices = np.where(predictions == labels)[0]
    else:
        candidate_indices = np.where(predictions != labels)[0]

    # Return empty list when no predictions match requested category
    if len(candidate_indices) == 0:
        return []

    # Define confidence ranges
    confidence_ranges = [
        (0.50, 0.60),
        (0.60, 0.70),
        (0.70, 0.80),
        (0.80, 0.90),
        (0.90, 1.01),
    ]
    selected = []

    # Distribute nine examples across the confidence ranges
    if num_examples == 9:
        target_per_range = [2, 2, 2, 2, 1]
    else:
        target_per_range = [1] * len(confidence_ranges)

    # Select examples with lower confidence first within each range
    for ((lower, upper), target_count) in zip(confidence_ranges, target_per_range):
        candidates = candidate_indices[
            (confidence[candidate_indices] >= lower) &
            (confidence[candidate_indices] < upper)
        ]

        candidates = candidates[np.argsort(confidence[candidates])]

        for index in candidates[:target_count]:
            selected.append(index)

    # Fill remaining places using unused candidates with lowest confidence
    if len(selected) < num_examples:
        remaining = [index for index in candidate_indices if index not in selected]
        remaining = sorted(remaining, key=lambda index: confidence[index])

        for index in remaining:
            selected.append(index)
            if len(selected) >= num_examples:
                break

    return selected[:num_examples]


def unnormalize_image(image):
    """
    Reverse image normalisation for visualisation.

    Args:
        image (np.ndarray): Normalised image with three grayscale channels.

    Returns:
        np.ndarray: A single-channel image with pixel values clipped to [0, 1].
    """
    image = image[0]
    image = image * ADNI_STD + ADNI_MEAN
    image = np.clip(image, 0.0, 1.0)
    return image


def plot_prediction_examples(
    model_name,
    labels,
    predictions,
    probabilities,
    images,
    correct=True
):
    """
    Plot selected correct or failure model predictions.

    Args:
        model_name (str): Model identifier used in the figure title and filename.
        labels (np.ndarray): Ground-truth class indices.
        predictions (np.ndarray): Predicted class indices.
        probabilities (np.ndarray): Predicted probabilities for each class.
        images (np.ndarray): Normalised input images.
        correct (bool): Whether to plot correct or failure predictions.

    Returns:
        None: Saves the prediction examples as a PNG image.
    """
    # Configure the number of examples and output filename
    if correct:
        num_examples = 9
        title = "Correct Predictions"
        filename = (f"{model_name}_correct_predictions.png")
    else:
        num_examples = 5
        title = "Failure Predictions"
        filename = (f"{model_name}_failure_predictions.png")

    # Select examples based on correctness and prediction confidence
    indices = select_examples(
        labels=labels,
        predictions=predictions,
        probabilities=probabilities,
        correct=correct,
        num_examples=num_examples
    )

    # Skip plotting when no suitable examples are available
    if len(indices) == 0:
        print(f"No {'correct' if correct else 'failure'} examples found for {model_name}.")
        return

    # Create a 3 x 3 grid
    rows = 3
    cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
    axes = np.array(axes).reshape(rows, cols)

    for ax in axes.flat:
        ax.axis("off")

    # Display selected image alongside its prediction and confidence
    class_names = {0: "AD", 1: "NC"}
    for plot_index, data_index in enumerate(indices):
        row = plot_index // cols
        col = plot_index % cols
        ax = axes[row, col]
        
        # Restore image intensity values for display
        image = unnormalize_image(images[data_index])

        prediction = class_names[predictions[data_index]]
        true_label = class_names[labels[data_index]]
        confidence = probabilities[data_index].max()

        ax.imshow(image, cmap="gray")
        ax.set_title(
            f"Prediction: {prediction}\n"
            f"True: {true_label}\n"
            f"Confidence: {confidence * 100:.1f}%",
            fontsize=11
        )
        ax.axis("off")

    display_name = "Baseline CNN" if model_name == "baseline" else "ConvNeXt"

    fig.suptitle(f"{display_name} - {title}", fontsize=16)
    plt.tight_layout()

    # Save the prediction examples
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def evaluate_model(model_name, checkpoint_path, test_loader):
    """
    Evaluate a trained model and save its evaluation visualisations.

    Args:
        model_name (str): Model identifier, either 'baseline' or 'convnext'.
        checkpoint_path (str): Path to the trained model checkpoint.
        test_loader (DataLoader): DataLoader for the test dataset.

    Returns:
        None: Prints evaluation metrics and saves relevant plots.
    """
    display_name = "Baseline CNN" if model_name == "baseline" else "ConvNeXt"
    print(f"\n=== Evaluating {display_name} ===")

    # Load the trained model
    model = load_model(model_name, checkpoint_path)
    (
        predictions,
        labels,
        probabilities,
        images,
        latency_ms,
        peak_vram
    ) = run_inference(model, test_loader)

    # Calculate and print evaluation and computational resource metrics
    metrics = calculate_metrics(predictions, labels, probabilities)
    print_results(
        model_name=model_name,
        model=model,
        metrics=metrics,
        latency_ms=latency_ms,
        peak_vram=peak_vram
    )

    # Plot a confusion matrix for each evaluated model
    plot_confusion_matrix(
        model_name=model_name,
        confusion_matrix_values=metrics["confusion_matrix"]
    )

    # Plot additional visualisations for ConvNeXt
    if model_name == "convnext":
        plot_roc_curve(labels=labels, probabilities=probabilities)
        plot_accuracy_vs_confidence(
            labels=labels,
            predictions=predictions,
            probabilities=probabilities
        )
        plot_prediction_examples(
            model_name=model_name,
            labels=labels,
            predictions=predictions,
            probabilities=probabilities,
            images=images,
            correct=True
        )
        plot_prediction_examples(
            model_name=model_name,
            labels=labels,
            predictions=predictions,
            probabilities=probabilities,
            images=images,
            correct=False
        )


def main():
    """
    Prepare the data and evaluate both trained models.

    Returns:
        None: Prints dataset summaries, evaluation summaries and saves plots.
    """
    # Check the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Build training, validation and test DataLoaders
    train_loader, val_loader = build_dataloader(is_train=True, batch_size=BATCH_SIZE)
    test_loader = build_dataloader(is_train=False, batch_size=BATCH_SIZE)

    # Print selected computing device and dataset summaries
    print(f"Using device: {DEVICE}")
    print_dataset_summary(train_loader, val_loader, test_loader)

    # Evaluate both models using best saved checkpoint
    evaluate_model(
        model_name="baseline",
        checkpoint_path=BASELINE_CHECKPOINT,
        test_loader=test_loader
    )
    evaluate_model(
        model_name="convnext",
        checkpoint_path=CONVNEXT_CHECKPOINT,
        test_loader=test_loader
    )


if __name__ == "__main__":
    main()