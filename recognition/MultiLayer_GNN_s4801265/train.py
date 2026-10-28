import torch
from torch import nn

from dataset import load_dataset
from modules import MLP, GCN

import matplotlib.pyplot as plt
import csv
import time

import argparse

import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix
)

# Training settings
SEED = 42
LEARNING_RATE = 0.01
EPOCHS = 1000
DROPOUT = 0.5
HIDDEN_DIM = 32

def model_forward(model, data):
    """
    Perform a forward pass using the selected model.

    Args:
        model: MLP or GCN model.
        data: Graph containing node features and connections.

    Returns:
        Class logits for all nodes.
    """
    if isinstance(model, GCN):
        return model(data.x, data.edge_index)

    return model(data.x)

def train_epoch(model, data, optimizer, criterion):
    """
    Train the selected model for one epoch.

    Args:
        model: MLP or GCN model.
        data: Graph containing node features, labels and masks.
        optimizer: Optimiser used to update model parameters.
        criterion: Loss function used during training.

    Returns:
        Training cross-entropy loss for the epoch.
    """
    model.train()

    # Clear gradients from previous epoch
    optimizer.zero_grad()

    # Generate logits for all nodes
    output = model_forward(model, data)

    # Calculate loss using training labels only
    loss = criterion(
        output[data.train_mask],
        data.y[data.train_mask]
    )

    # Calculate gradients
    loss.backward()

    # Update model parameters
    optimizer.step()

    return loss.item()


def evaluate(model, data, mask, criterion):
    """
    Evaluate model loss and accuracy on a selected data split.

    Args:
        model: Trained MLP or GCN model.
        data: Graph containing node features and labels.
        mask: Boolean mask selecting the nodes to evaluate.
        criterion: Loss function used for evaluation.

    Returns:
        Tuple containing cross-entropy loss and classification accuracy.
    """
    model.eval()

    with torch.no_grad():
        output = model_forward(model, data)

        loss = criterion(
            output[mask],
            data.y[mask]
        )

        predictions = output[mask].argmax(dim=1)

        accuracy = (
            predictions == data.y[mask]
        ).float().mean().item()

    return loss.item(), accuracy


def evaluate_test_set(model, data, model_name):
    """
    Evaluate classification performance on the test set.

    Calculates accuracy, macro-F1, per-class precision, recall,
    F1 score, multiclass AUROC and the confusion matrix.
    Saves test predictions and class probabilities to a CSV file.

    Args:
        model: Trained MLP or GCN model.
        data: Graph containing node features, labels and test mask.
        model_name: Name of the model used for output filenames.

    Returns:
        None
    """
    model.eval()

    with torch.no_grad():
        # Generate predictions for all nodes
        output = model_forward(model, data)

        # Select only test nodes
        test_output = output[data.test_mask]
        test_labels = data.y[data.test_mask]

        # Convert logits into class probabilities
        probabilities = torch.softmax(test_output, dim=1)

        # Find the predicted class for each node
        predictions = probabilities.argmax(dim=1)

    # Move results to CPU for evaluation
    labels = test_labels.cpu().numpy()
    predictions = predictions.cpu().numpy()
    probabilities = probabilities.cpu().numpy()

    # Get original node indices
    node_indices = data.test_mask.nonzero(
        as_tuple=True
    )[0].cpu().numpy()

    # Calculate classification metrics
    accuracy = accuracy_score(labels, predictions)

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            labels,
            predictions,
            labels=[0, 1, 2, 3],
            zero_division=0
        )
    )

    macro_f1 = f1.mean()

    # Calculate multiclass AUROC
    auroc = roc_auc_score(
        labels,
        probabilities,
        labels=[0, 1, 2, 3],
        multi_class="ovr",
        average="macro"
    )

    # Calculate confusion matrix
    matrix = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1, 2, 3]
    )

    # Print results
    print(f"\n=== FINAL {model_name.upper()} TEST RESULTS ===")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print(f"Macro AUROC (OvR): {auroc:.4f}")

    print("\nPer-Class Results:")

    for class_id in range(4):
        print(
            f"Class {class_id}: "
            f"Precision={precision[class_id]:.4f}, "
            f"Recall={recall[class_id]:.4f}, "
            f"F1={f1[class_id]:.4f}, "
            f"Support={support[class_id]}"
        )

    print("\nConfusion Matrix:")
    print(matrix)

    # Save individual test predictions and probabilities
    with open(f"{model_name}_predictions.csv", "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "node_id",
            "true_label",
            "predicted_label",
            "confidence",
            "prob_class_0",
            "prob_class_1",
            "prob_class_2",
            "prob_class_3"
        ])

        for node_id, label, prediction, probs in zip(
            node_indices, labels, predictions, probabilities
        ):
            writer.writerow([
                int(node_id),
                int(label),
                int(prediction),
                float(probs.max()),
                *[float(p) for p in probs]
            ])

    print(f"\nPredictions saved to {model_name}_predictions.csv")


def plot_training_results(training_losses, validation_losses,
                          training_accuracies, validation_accuracies,
                          model_name):
    """
    Plot and save training and validation results.

    Args:
        training_losses: Training loss for each epoch.
        validation_losses: Validation loss for each epoch.
        training_accuracies: Training accuracy for each epoch.
        validation_accuracies: Validation accuracy for each epoch.
        model_name: Name of the model used for plot titles and filenames.

    Returns:
        None
    """
    epochs = range(1, len(training_losses) + 1)

    # Plot training and validation losses
    plt.figure(figsize=(8, 5))

    plt.plot(epochs, training_losses, label="Training Loss")
    plt.plot(epochs, validation_losses, label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title(f"{model_name.upper()} Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(f"{model_name}_loss.png", dpi=200)
    plt.close()

    # Plot training and validation accuracy
    plt.figure(figsize=(8, 5))

    plt.plot(epochs, training_accuracies, label="Training Accuracy")
    plt.plot(epochs, validation_accuracies, label="Validation Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{model_name.upper()} Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(f"{model_name}_accuracy.png", dpi=200)
    plt.close()

    print("\nTraining plots saved successfully.")


def plot_analysis_results(model, data, model_name, threshold):
    """
    Plot classification, confidence and embedding results.

    Args:
        model: Best trained MLP or GCN.
        data: Graph containing features, labels and masks.
        model_name: "mlp" or "gcn".
        threshold: Confidence threshold selected on validation data.

    Returns:
        None
    """
    model.eval()

    with torch.no_grad():
        output = model_forward(model, data)

        test_output = output[data.test_mask]
        labels = data.y[data.test_mask]

        probabilities = torch.softmax(test_output, dim=1)
        confidences, predictions = probabilities.max(dim=1)

        correct = predictions == labels

    # Move values to CPU for plotting
    labels_np = labels.cpu().numpy()
    predictions_np = predictions.cpu().numpy()
    confidences_np = confidences.cpu().numpy()
    correct_np = correct.cpu().numpy()

    class_names = ["Class 0", "Class 1", "Class 2", "Class 3"]

    # Confusion matrix
    matrix = confusion_matrix(
        labels_np, predictions_np, labels=[0, 1, 2, 3]
    )

    plt.figure(figsize=(7, 6))
    plt.imshow(matrix, cmap="Blues")
    plt.colorbar(label="Number of Nodes")

    plt.xticks(range(4), class_names)
    plt.yticks(range(4), class_names)

    for i in range(4):
        for j in range(4):
            plt.text(
                j, i, str(matrix[i, j]),
                ha="center", va="center",
                color="white" if matrix[i, j] > matrix.max() / 2 else "black"
            )

    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.title(f"{model_name.upper()} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"{model_name}_confusion_matrix.png", dpi=200)
    plt.close()

    # Reliability diagram
    bin_confidences = []
    bin_accuracies = []

    for i in range(10):
        lower = i / 10
        upper = (i + 1) / 10

        if i == 9:
            mask = (confidences_np >= lower) & (confidences_np <= upper)
        else:
            mask = (confidences_np >= lower) & (confidences_np < upper)

        if mask.any():
            bin_confidences.append(confidences_np[mask].mean())
            bin_accuracies.append(correct_np[mask].mean())

    plt.figure(figsize=(7, 6))

    plt.plot(
        [0, 1], [0, 1], "k--",
        label="Perfect Calibration"
    )

    plt.plot(
        bin_confidences, bin_accuracies,
        "o-", label=model_name.upper()
    )

    plt.xlabel("Average Confidence")
    plt.ylabel("Actual Accuracy")
    plt.title(f"{model_name.upper()} Reliability Diagram")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(f"{model_name}_reliability.png", dpi=200)
    plt.close()

    # Correct vs incorrect confidence distribution
    plt.figure(figsize=(8, 5))

    bins = np.linspace(0, 1, 21)

    plt.hist(
        confidences_np[correct_np],
        bins=bins,
        alpha=0.6,
        density=True,
        label="Correct Predictions"
    )

    plt.hist(
        confidences_np[~correct_np],
        bins=bins,
        alpha=0.6,
        density=True,
        label="Incorrect Predictions"
    )

    plt.xlabel("Prediction Confidence")
    plt.ylabel("Probability Density")
    plt.title(f"{model_name.upper()} Confidence Distribution")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(f"{model_name}_confidence.png", dpi=200)
    plt.close()

    # Accuracy vs manual-review workload
    review_percentages = []
    accepted_accuracies = []

    for current_threshold in np.linspace(0, 0.99, 100):
        accepted = confidences_np >= current_threshold

        if accepted.sum() == 0:
            continue

        accepted_accuracy = correct_np[accepted].mean() * 100
        review_percentage = (~accepted).mean() * 100

        review_percentages.append(review_percentage)
        accepted_accuracies.append(accepted_accuracy)

    plt.figure(figsize=(8, 5))

    plt.plot(
        review_percentages,
        accepted_accuracies,
        label="Accuracy vs Workload"
    )

    # Mark the threshold selected using validation data
    if threshold is not None:
        accepted = confidences_np >= threshold

        if accepted.any():
            plt.scatter(
                (~accepted).mean() * 100,
                correct_np[accepted].mean() * 100,
                label=f"Selected Threshold: {threshold:.2f}",
                s=80,
                zorder=5
            )

    plt.xlabel("Nodes Sent for Manual Review (%)")
    plt.ylabel("Accepted Prediction Accuracy (%)")
    plt.title(f"{model_name.upper()} Manual Review Trade-off")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(f"{model_name}_manual_review.png", dpi=200)
    plt.close()

    # GCN learned embeddings using t-SNE
    if model_name == "gcn":
        with torch.no_grad():
            embeddings = model.get_embeddings(
                data.x, data.edge_index
            )

        embeddings = embeddings.cpu().numpy()
        all_labels = data.y.cpu().numpy()

        # Sample 4000 nodes for faster t-SNE
        rng = np.random.default_rng(42)
        indices = rng.choice(
            len(all_labels),
            size=min(4000, len(all_labels)),
            replace=False
        )

        embeddings = embeddings[indices]
        sampled_labels = all_labels[indices]

        tsne = TSNE(
            n_components=2,
            perplexity=30,
            random_state=42,
            init="pca",
            learning_rate="auto"
        )

        reduced = tsne.fit_transform(embeddings)

        plt.figure(figsize=(9, 7))

        for class_id in range(4):
            mask = sampled_labels == class_id

            plt.scatter(
                reduced[mask, 0],
                reduced[mask, 1],
                s=8,
                alpha=0.6,
                label=class_names[class_id]
            )

        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        plt.title("GCN Learned Node Embeddings")
        plt.legend()
        plt.grid(True, alpha=0.2)
        plt.tight_layout()

        plt.savefig("gcn_tsne.png", dpi=200)
        plt.close()

    print(f"\n{model_name.upper()} analysis plots saved successfully.")



def calculate_ece(probabilities, labels, num_bins=10):
    """
    Calculate Expected Calibration Error using confidence bins.

    Args:
        probabilities: Class probabilities for each node.
        labels: Actual class labels.
        num_bins: Number of confidence bins.

    Returns:
        Expected Calibration Error.
    """
    confidences, predictions = probabilities.max(dim=1)
    correct = (predictions == labels).float()

    ece = 0.0

    for bin_id in range(num_bins):
        lower = bin_id / num_bins
        upper = (bin_id + 1) / num_bins

        # Include the upper bound in the final bin
        if bin_id == num_bins - 1:
            in_bin = (confidences >= lower) & (confidences <= upper)
        else:
            in_bin = (confidences >= lower) & (confidences < upper)

        if in_bin.any().item():
            bin_accuracy = correct[in_bin].mean().item()
            bin_confidence = confidences[in_bin].mean().item()
            bin_fraction = in_bin.float().mean().item()

            ece += bin_fraction * abs(
                bin_accuracy - bin_confidence
            )

    return ece


def select_confidence_threshold(
    model, data, target_accuracy=0.95
):
    """
    Select a confidence threshold using validation data.

    Args:
        model: Trained MLP or GCN model.
        data: Graph containing validation features and labels.
        target_accuracy: Minimum desired accuracy for accepted nodes.

    Returns:
        Selected threshold, or None if no threshold qualifies.
    """
    model.eval()

    with torch.no_grad():
        output = model_forward(model, data)
        probabilities = torch.softmax(
            output[data.val_mask], dim=1
        )

        labels = data.y[data.val_mask]

        confidences, predictions = probabilities.max(dim=1)
        correct = predictions == labels

    # Candidate thresholds from 0.50 to 0.99
    thresholds = [i / 100 for i in range(50, 100)]

    for threshold in thresholds:
        accepted = confidences >= threshold

        # Avoid calculating accuracy on an empty set
        if accepted.sum().item() == 0:
            continue

        accepted_accuracy = (
            correct[accepted].float().mean().item()
        )

        if accepted_accuracy >= target_accuracy:
            print(
                f"\nSelected Confidence Threshold: {threshold:.2f}"
            )
            print(
                f"Validation Accepted Accuracy: "
                f"{accepted_accuracy:.4f}"
            )

            return threshold

    print("\nNo threshold met the target validation accuracy.")
    return None


def analyse_confidence(model, data, mask, threshold, split_name):
    """
    Analyse confidence reliability and manual-review workload
    for a selected data split.

    Args:
        model: Trained MLP or GCN model.
        data: Graph containing features and labels.
        mask: Boolean mask for the selected split.
        threshold: Confidence threshold selected on validation data.
        split_name: Name of the evaluated split.

    Returns:
        None
    """
    model.eval()

    with torch.no_grad():
        output = model_forward(model, data)

        probabilities = torch.softmax(
            output[mask], dim=1
        )

        labels = data.y[mask]

        confidences, predictions = probabilities.max(dim=1)
        correct = predictions == labels

    # Expected Calibration Error
    ece = calculate_ece(probabilities, labels)

    # Average confidence for correct/incorrect predictions
    correct_confidence = (
        confidences[correct].mean().item()
        if correct.any().item() else float("nan")
    )

    incorrect_confidence = (
        confidences[~correct].mean().item()
        if (~correct).any().item() else float("nan")
    )

    print(f"\n=== {split_name.upper()} CONFIDENCE ANALYSIS ===")
    print(f"ECE: {ece:.4f}")
    print(f"Mean Correct Confidence: {correct_confidence:.4f}")
    print(f"Mean Incorrect Confidence: {incorrect_confidence:.4f}")

    # Evaluate manual-review performance
    if threshold is None:
        print("No qualifying manual-review threshold.")
        return

    accepted = confidences >= threshold

    accepted_count = accepted.sum().item()
    total_count = len(labels)

    accepted_percentage = 100 * accepted_count / total_count
    review_percentage = 100 - accepted_percentage

    print(f"\nConfidence Threshold: {threshold:.2f}")
    print(f"Automatically Accepted: {accepted_percentage:.2f}%")
    print(f"Sent for Manual Review: {review_percentage:.2f}%")

    if accepted_count > 0:
        accepted_accuracy = (
            correct[accepted].float().mean().item()
        )

        print(
            f"Accepted Prediction Accuracy: "
            f"{accepted_accuracy:.4f}"
        )
    else:
        print("Accepted Prediction Accuracy: N/A")


def measure_resources(model, data, device):
    """
    Measure model parameters, inference time and peak GPU memory
    during inference.

    Args:
        model: Trained MLP or GCN model.
        data: Graph containing node features.
        device: CPU or GPU device.

    Returns:
        None
    """
    model.eval()

    # Count total model parameters
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )

    print(f"\n=== {type(model).__name__.upper()} RESOURCE MEASUREMENTS ===")
    print(f"Model Parameters: {parameter_count:,}")

    # Warm up before timing
    with torch.inference_mode():
        for _ in range(5):
            model_forward(model, data)

    # Synchronise GPU operations before timing
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

    # Measure repeated forward passes
    inference_times = []

    with torch.inference_mode():
        for _ in range(20):
            if device.type == "cuda":
                torch.cuda.synchronize()

            start_time = time.perf_counter()

            output = model_forward(model, data)
            test_output = output[data.test_mask]

            # Ensure GPU operations have completed
            if device.type == "cuda":
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - start_time
            inference_times.append(elapsed)

    average_time = sum(inference_times) / len(inference_times)

    print(f"Average Inference Time: {average_time:.6f} seconds")

    if device.type == "cuda":
        peak_memory = (
            torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        )

        print(f"Peak Inference GPU Memory: {peak_memory:.2f} MiB")
    else:
        print("Peak Inference GPU Memory: N/A (CPU execution)")


def analyse_failures(model, data, num_examples=5):
    """
    Identify high confidence incorrect test-node predictions.

    Args:
        model: Trained MLP or GCN model.
        data: Graph containing node features and test labels.
        num_examples: Number of failures to display.

    Returns:
        None
    """
    model.eval()

    with torch.no_grad():
        output = model_forward(model, data)
        probabilities = torch.softmax(output, dim=1)

        confidences, predictions = probabilities.max(dim=1)

    # Find node IDs belonging to the test set
    test_indices = data.test_mask.nonzero(
        as_tuple=True
    )[0]

    # Find incorrectly classified test nodes
    incorrect = (
        predictions[test_indices] != data.y[test_indices]
    )

    incorrect_indices = test_indices[incorrect]

    # Sort incorrect nodes by descending confidence
    sorted_indices = incorrect_indices[
        torch.argsort(
            confidences[incorrect_indices],
            descending=True
        )
    ]

    print(f"\n=== {type(model).__name__.upper()} FAILURE ANALYSIS ===")

    for node_id in sorted_indices[:num_examples]:
        node = node_id.item()

        print(f"\nNode ID: {node}")
        print(f"Actual Class: {data.y[node].item()}")
        print(f"Predicted Class: {predictions[node].item()}")
        print(f"Confidence: {confidences[node].item():.4f}")

        print(
            "Class Probabilities:",
            probabilities[node].cpu().tolist()
        )


def main():
    """
    Run the complete training and evaluation pipeline.

    Loads the dataset, initialises and trains the selected model,
    saves the best checkpoint based on validation loss, and
    performs test evaluation, confidence analysis, resource
    measurements, failure analysis and visualisations.

    Returns:
        None
    """
    # Select which model to train
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["mlp", "gcn"],
        default="mlp"
    )

    args = parser.parse_args()
    model_name = args.model

    # Set random seed for reproducibility
    torch.manual_seed(SEED)

    # Choose GPU if available, otherwise CPU
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    # Load the dataset and transfer it to the device
    data = load_dataset(seed=SEED).to(device)

    # Initialise the selected model
    if model_name == "mlp":
        model = MLP(
            input_dim=128,
            hidden_dim=HIDDEN_DIM,
            output_dim=4,
            dropout=DROPOUT
        ).to(device)

        epochs = EPOCHS

    else:
        model = GCN(
            input_dim=128,
            hidden_dim=HIDDEN_DIM,
            output_dim=4,
            dropout=DROPOUT
        ).to(device)

        epochs = EPOCHS

    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    # Define the optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    print(f"{model_name.upper()} initialised successfully.")

    # Reset peak GPU memory before training
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    best_val_loss = float("inf")
    best_epoch = 0
    best_model_path = f"{model_name}_best.pt"

    training_losses = []
    validation_losses = []

    training_accuracies = []
    validation_accuracies = []

    # Train the selected model
    for epoch in range(1, epochs + 1):

        # Train the selected model for one epoch
        train_epoch(model, data, optimizer, criterion)

        # Evaluate training performance
        train_loss, train_accuracy = evaluate(
            model, data, data.train_mask, criterion
        )

        # Evaluate validation performance
        val_loss, val_accuracy = evaluate(
            model, data, data.val_mask, criterion
        )

        # Record losses and accuracy
        training_losses.append(train_loss)
        validation_losses.append(val_loss)

        training_accuracies.append(train_accuracy)
        validation_accuracies.append(val_accuracy)


        # Save model when validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch

            torch.save(model.state_dict(), best_model_path)

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Accuracy: {val_accuracy:.4f}"
        )

    # Measure peak GPU memory during training
    if device.type == "cuda":
        torch.cuda.synchronize(device)

        peak_training_memory = (
            torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        )

        print(
            f"\nPeak Training GPU Memory: "
            f"{peak_training_memory:.2f} MiB"
        )
    else:
        print("\nPeak Training GPU Memory: N/A (CPU execution)")

    # Load the best model after training
    model.load_state_dict(
        torch.load(
            best_model_path,
            map_location=device,
            weights_only=True
        )
    )


    model.eval()

    print(f"\nBest {model_name.upper()} model loaded.")

    # Evaluate the best saved model on training nodes
    train_loss, train_accuracy = evaluate(
        model, data, data.train_mask, criterion
    )

    # Evaluate the best saved model on validation nodes
    val_loss, val_accuracy = evaluate(
        model, data, data.val_mask, criterion
    )

    print(f"\n=== BEST {model_name.upper()} MODEL RESULTS ===")
    print(f"Best Epoch: {best_epoch}")
    print(f"Training Loss: {train_loss:.4f}")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")


    # Save training and validation plots
    plot_training_results(
        training_losses,
        validation_losses,
        training_accuracies,
        validation_accuracies,
        model_name
    )

    # Select a confidence threshold using validation data
    threshold = select_confidence_threshold(
        model,
        data,
        target_accuracy=0.95
    )

    # Measure resource requirements
    measure_resources(model, data, device)

    # Final test-set classification evaluation
    evaluate_test_set(model, data, model_name)

    # Analyse test-set confidence and manual-review performance
    analyse_confidence(
        model,
        data,
        data.test_mask,
        threshold,
        "Test"
    )

    # Investigate confidently incorrect predictions
    analyse_failures(model, data, num_examples=5)

    # Plot classification, confidence and embedding results
    plot_analysis_results(
        model,
        data,
        model_name,
        threshold
    )

if __name__ == "__main__":
    main()
