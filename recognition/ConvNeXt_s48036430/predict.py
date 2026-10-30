"""Evaluate a trained ConvNeXt model and generate test-result plots."""

from pathlib import Path
import time

import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

from dataset import BATCH_SIZE, CLASS_TO_LABEL, load_dataset, load_train_validation
from modules import ConvNeXtClassifier

MODEL_PATH = Path("./models/convnext_best.pth")
PLOT_PATH = Path("./plots/test_metrics_vs_threshold.png")
IGNORED_INTERVAL_PLOT_PATH = Path("./plots/test_accuracy_by_ignored_interval.png")
EXAMPLE_PLOT_PATH = Path("./plots/example_classifications.png")


def report_model_size_and_latency(model, data_loader, device):
    """Print the model parameter count and average inference latency."""
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    sample_images, _ = next(iter(data_loader))
    sample_image = sample_images[:1].to(device)

    model.eval()
    with torch.inference_mode():
        for _ in range(5):
            model(sample_image)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        start_time = time.perf_counter()
        for _ in range(20):
            model(sample_image)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed_time = time.perf_counter() - start_time

    latency_ms = elapsed_time * 1000 / 20
    print(f"Model parameters: {parameter_count:,}")
    print(f"Inference latency: {latency_ms:.2f} ms/image on {device}")


def collect_ad_probabilities(model, data_loader, device, loss_function=None):
    """Collect AD probabilities, labels, and optional average loss."""
    model.eval()
    probabilities = []
    labels = []
    total_loss = 0.0
    total = 0

    with torch.no_grad():
        for images, batch_labels in data_loader:
            batch_labels = batch_labels.to(device)
            logits = model(images.to(device))
            probabilities.append(
                torch.softmax(logits, dim=1)[:, CLASS_TO_LABEL["AD"]].cpu()
            )
            labels.append(batch_labels.cpu())
            if loss_function is not None:
                batch_size = batch_labels.size(0)
                total_loss += loss_function(logits, batch_labels).item() * batch_size
                total += batch_size

    ad_probabilities = torch.cat(probabilities)
    collected_labels = torch.cat(labels)
    average_loss = total_loss / total if total else None
    return ad_probabilities, collected_labels, average_loss


def find_best_ad_threshold(ad_probabilities, labels):
    """Select the AD probability threshold with the highest accuracy."""
    unique_probabilities = torch.unique(ad_probabilities, sorted=True)
    candidates = [0.0, 0.5, 1.0]
    candidates.extend(
        ((unique_probabilities[:-1] + unique_probabilities[1:]) / 2).tolist()
    )

    best_threshold = 0.5
    best_accuracy = -1.0
    for threshold in candidates:
        predictions = torch.where(
            ad_probabilities >= threshold,
            CLASS_TO_LABEL["AD"],
            CLASS_TO_LABEL["NC"],
        )
        accuracy = (predictions == labels).float().mean().item()
        if (accuracy, -abs(threshold - 0.5)) > (
            best_accuracy,
            -abs(best_threshold - 0.5),
        ):
            best_threshold = threshold
            best_accuracy = accuracy

    return best_threshold, best_accuracy


def calculate_metrics(ad_probabilities, labels, threshold):
    """Calculate classification accuracy and AD recall at a threshold."""
    predictions = torch.where(
        ad_probabilities >= threshold,
        CLASS_TO_LABEL["AD"],
        CLASS_TO_LABEL["NC"],
    )
    accuracy = (predictions == labels).float().mean().item()
    ad_mask = labels == CLASS_TO_LABEL["AD"]
    ad_total = ad_mask.sum().item()
    ad_correct = ((predictions == CLASS_TO_LABEL["AD"]) & ad_mask).sum().item()
    ad_recall = ad_correct / ad_total if ad_total else 0.0
    return accuracy, ad_recall


def calculate_binary_auroc(scores, positive_labels):
    """Calculate AUROC for binary labels, accounting for tied scores."""
    positive_count = positive_labels.sum().item()
    negative_count = (~positive_labels).sum().item()
    if not positive_count or not negative_count:
        return float("nan")

    sorted_scores, sort_indices = torch.sort(scores)
    sorted_labels = positive_labels[sort_indices]
    _, tie_group_indices, tie_group_counts = torch.unique_consecutive(
        sorted_scores, return_inverse=True, return_counts=True
    )
    rank_ends = tie_group_counts.cumsum(0).to(torch.float64)
    rank_starts = rank_ends - tie_group_counts.to(torch.float64) + 1
    average_ranks = (rank_starts + rank_ends) / 2
    positive_rank_sum = average_ranks[tie_group_indices][sorted_labels].sum().item()
    return (positive_rank_sum - positive_count * (positive_count + 1) / 2) / (
        positive_count * negative_count
    )


def report_classification_metrics(ad_probabilities, labels):
    """Print per-class precision, recall, F1, and AUROC metrics."""
    predictions = torch.where(
        ad_probabilities >= 0.5,
        CLASS_TO_LABEL["AD"],
        CLASS_TO_LABEL["NC"],
    )
    class_metrics = []

    print("Test classification metrics (threshold=0.5):")
    print(f"{'Class':<8} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUROC':>10}")
    for class_name in ("AD", "NC"):
        class_label = CLASS_TO_LABEL[class_name]
        true_positive_mask = labels == class_label
        predicted_positive_mask = predictions == class_label
        true_positive = (true_positive_mask & predicted_positive_mask).sum().item()
        false_positive = (~true_positive_mask & predicted_positive_mask).sum().item()
        false_negative = (true_positive_mask & ~predicted_positive_mask).sum().item()

        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        class_scores = ad_probabilities if class_name == "AD" else 1 - ad_probabilities
        auroc = calculate_binary_auroc(class_scores, true_positive_mask)
        class_metrics.append((precision, recall, f1, auroc))
        print(
            f"{class_name:<8} {precision:>10.4f} {recall:>10.4f} "
            f"{f1:>10.4f} {auroc:>10.4f}"
        )

    macro_f1 = sum(metrics[2] for metrics in class_metrics) / len(class_metrics)
    print(f"Macro-F1: {macro_f1:.4f}")


def plot_test_metrics_by_threshold(ad_probabilities, labels, plot_path):
    """Plot test accuracy and AD recall across probability thresholds."""
    thresholds = torch.arange(1, 100, dtype=torch.float32) / 100
    metrics = [
        calculate_metrics(ad_probabilities, labels, threshold.item())
        for threshold in thresholds
    ]
    accuracies, ad_recalls = zip(*metrics)

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(thresholds.tolist(), accuracies, label="Test accuracy")
    axis.plot(thresholds.tolist(), ad_recalls, label="Test AD recall")
    axis.set(
        title="Test Metrics by AD Probability Threshold",
        xlabel="AD probability threshold",
        ylabel="Metric",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axis.set_xticks([index / 10 for index in range(11)])
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def plot_test_accuracy_by_ignored_interval(ad_probabilities, labels, plot_path):
    """Plot accuracy and coverage as central predictions are ignored."""
    interval_widths = torch.linspace(0, 1, 101)
    accuracies = []
    percentages_remaining = []
    predictions = torch.where(
        ad_probabilities >= 0.5,
        CLASS_TO_LABEL["AD"],
        CLASS_TO_LABEL["NC"],
    )

    for width in interval_widths:
        lower_bound = (1 - width.item()) / 2
        upper_bound = (1 + width.item()) / 2
        remaining = (ad_probabilities < lower_bound) | (ad_probabilities > upper_bound)

        remaining_count = remaining.sum().item()
        percentages_remaining.append(remaining_count / len(labels))
        if remaining_count:
            accuracies.append(
                (predictions[remaining] == labels[remaining]).float().mean().item()
            )
        else:
            accuracies.append(float("nan"))

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(interval_widths.tolist(), accuracies, label="Test accuracy")
    axis.plot(
        interval_widths.tolist(),
        percentages_remaining,
        label="Percentage remaining",
    )
    axis.set(
        title="Test Accuracy When Ignoring Central Predictions",
        xlabel="Ignored interval width",
        ylabel="Proportion",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axis.set_xticks([index / 10 for index in range(11)])
    axis.grid(True, alpha=0.3)
    axis.legend()
    plot_path = Path(plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved ignored-interval test plot to {plot_path}")


def plot_example_classifications(model, data_loader, device, plot_path):
    """Plot example images grouped by true class and AD probability."""
    row_labels = (CLASS_TO_LABEL["AD"], CLASS_TO_LABEL["NC"])
    column_titles = (
        "AD probability 0.0 - 0.1",
        "AD probability 0.4 - 0.5",
        "AD probability 0.5 - 0.6",
        "AD probability 0.9 - 1.0",
    )
    examples = [[None for _ in column_titles] for _ in row_labels]

    model.eval()
    with torch.no_grad():
        for images, labels in data_loader:
            probabilities = torch.softmax(model(images.to(device)), dim=1)[
                :, CLASS_TO_LABEL["AD"]
            ].cpu()

            for image, label, probability in zip(images, labels, probabilities):
                if label.item() == CLASS_TO_LABEL["AD"]:
                    row_index = 0
                elif label.item() == CLASS_TO_LABEL["NC"]:
                    row_index = 1
                else:
                    continue

                probability_value = probability.item()
                if probability_value < 0.1:
                    column_index = 0
                elif 0.4 <= probability_value < 0.5:
                    column_index = 1
                elif 0.5 < probability_value <= 0.6:
                    column_index = 2
                elif probability_value > 0.9:
                    column_index = 3
                else:
                    continue

                if examples[row_index][column_index] is None:
                    examples[row_index][column_index] = (
                        image.squeeze(0).cpu(),
                        probability_value,
                    )

            if all(example is not None for row in examples for example in row):
                break

    figure, axes = plt.subplots(2, 4, figsize=(16, 7))
    for row_index, (true_label, axes_row) in enumerate(zip(row_labels, axes)):
        for column_index, axis in enumerate(axes_row):
            example = examples[row_index][column_index]
            axis.set_title(column_titles[column_index])
            axis.set_xticks([])
            axis.set_yticks([])
            if column_index == 0:
                label_name = "AD" if true_label == CLASS_TO_LABEL["AD"] else "NC"
                axis.set_ylabel(f"True {label_name}")

            if example is None:
                axis.text(0.5, 0.5, "No matching example", ha="center", va="center")
                continue

            image, probability = example
            axis.imshow(
                torch.rot90(image, k=1, dims=(0, 1)), cmap="gray", vmin=0, vmax=1
            )
            axis.set_xlabel(f"AD probability: {probability:.3f}")

    figure.tight_layout()
    plot_path = Path(plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved example classifications plot to {plot_path}")


def evaluate_and_plot_test_results(
    model,
    validation_loader,
    test_loader,
    device,
    loss_function,
    plot_path,
    ignored_interval_plot_path,
):
    """Evaluate validation and test data and save test-result plots."""
    validation_probabilities, validation_labels, validation_loss = (
        collect_ad_probabilities(
            model, validation_loader, device, loss_function=loss_function
        )
    )
    threshold, validation_accuracy = find_best_ad_threshold(
        validation_probabilities, validation_labels
    )
    _, validation_ad_recall = calculate_metrics(
        validation_probabilities, validation_labels, threshold
    )
    print(
        f"Validation-tuned AD threshold: {threshold:.4f} | "
        f"validation accuracy: {validation_accuracy:.2%} | "
        f"AD recall: {validation_ad_recall:.2%} | "
        f"loss: {validation_loss:.4f}"
    )

    test_probabilities, test_labels, test_loss = collect_ad_probabilities(
        model, test_loader, device, loss_function=loss_function
    )
    report_classification_metrics(test_probabilities, test_labels)
    for name, test_threshold in (
        ("validation-tuned", threshold),
        ("default threshold 0.5", 0.5),
    ):
        test_accuracy, test_ad_recall = calculate_metrics(
            test_probabilities, test_labels, test_threshold
        )
        print(
            f"Test results ({name}) | "
            f"loss: {test_loss:.4f} | "
            f"accuracy: {test_accuracy:.2%} | "
            f"AD recall: {test_ad_recall:.2%}"
        )

    plot_path = Path(plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plot_test_metrics_by_threshold(test_probabilities, test_labels, plot_path)
    print(f"Saved test threshold metrics plot to {plot_path}")
    plot_test_accuracy_by_ignored_interval(
        test_probabilities,
        test_labels,
        ignored_interval_plot_path,
    )


def main():
    """Load the trained model, evaluate it, and generate result plots."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Trained model checkpoint not found: {MODEL_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model = ConvNeXtClassifier().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    _, validation_loader = load_train_validation(batch_size=BATCH_SIZE)
    test_loader = load_dataset("test", batch_size=BATCH_SIZE)
    loss_function = torch.nn.CrossEntropyLoss()

    print(f"Loaded model checkpoint from {MODEL_PATH} on {device}.")
    report_model_size_and_latency(model, test_loader, device)
    evaluate_and_plot_test_results(
        model,
        validation_loader,
        test_loader,
        device,
        loss_function,
        PLOT_PATH,
        IGNORED_INTERVAL_PLOT_PATH,
    )
    plot_example_classifications(model, test_loader, device, EXAMPLE_PLOT_PATH)


if __name__ == "__main__":
    main()
