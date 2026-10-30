"""Train and evaluate a ConvNeXt brain-slice classifier."""

from collections import Counter
from pathlib import Path
from time import perf_counter

import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torchvision import transforms

from dataset import BATCH_SIZE, CLASS_TO_LABEL, load_dataset, load_train_validation
from modules import ConvNeXtClassifier

EPOCHS = 300
LEARNING_RATE = 1e-4
MIN_LEARNING_RATE = 1e-6
WARMUP_EPOCHS = 20
WEIGHT_DECAY = 0.05
LABEL_SMOOTHING = 0.1
TRAIN_AUGMENTATION = transforms.Compose(
    [
        transforms.ConvertImageDtype(torch.uint8),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ConvertImageDtype(torch.float32),
        transforms.RandomErasing(p=0.25),
    ]
)


def train_model(model, train_loader, optimizer, device):
    """Train the model for one epoch and return loss and accuracy."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = torch.stack([TRAIN_AUGMENTATION(image) for image in images])
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = nn.functional.cross_entropy(
            logits, labels, label_smoothing=LABEL_SMOOTHING
        )
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_size

    return total_loss / total, correct / total


def evaluate_model(model, validation_loader, loss_function, device):
    """Evaluate a model and return loss, accuracy, AD recall, and miss rate."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    ad_correct = 0
    ad_total = 0

    with torch.no_grad():
        for images, labels in validation_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            predictions = logits.argmax(dim=1)
            batch_size = labels.size(0)
            total_loss += loss_function(logits, labels).item() * batch_size
            correct += (predictions == labels).sum().item()
            total += batch_size
            ad_mask = labels == CLASS_TO_LABEL["AD"]
            ad_correct += ((predictions == CLASS_TO_LABEL["AD"]) & ad_mask).sum().item()
            ad_total += ad_mask.sum().item()

    ad_recall = ad_correct / ad_total if ad_total else 0.0
    false_negative_rate = 1.0 - ad_recall if ad_total else 0.0
    return total_loss / total, correct / total, ad_recall, false_negative_rate


def print_class_counts(name, data_loader):
    """Print the class distribution of a dataset subset."""
    subset = data_loader.dataset
    label_counts = Counter(subset.dataset.samples[index][1] for index in subset.indices)
    class_counts = ", ".join(
        f"{class_name}={label_counts[label]}"
        for class_name, label in CLASS_TO_LABEL.items()
    )
    print(f"{name} set: {class_counts}")


def main():
    """Train the classifier, evaluate its best checkpoint, and plot metrics."""
    train_loader, validation_loader = load_train_validation(batch_size=BATCH_SIZE)
    test_loader = load_dataset("test", batch_size=BATCH_SIZE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvNeXtClassifier().to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    def learning_rate_scale(epoch):
        """Return the warmup and cosine-decay scale for an epoch."""
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS
        progress = (epoch - WARMUP_EPOCHS) / (EPOCHS - WARMUP_EPOCHS)
        minimum_scale = MIN_LEARNING_RATE / LEARNING_RATE
        return minimum_scale + (1.0 - minimum_scale) * 0.5 * (
            1.0 + torch.cos(torch.tensor(torch.pi * progress)).item()
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_scale)

    print(
        f"Training on {device}: {len(train_loader.dataset)} training images, "
        f"{len(validation_loader.dataset)} validation images, "
        f"{EPOCHS} epochs, batch size {BATCH_SIZE}."
    )
    print_class_counts("Train", train_loader)
    print_class_counts("Validation", validation_loader)

    epochs = []
    learning_rates = []
    train_losses = []
    validation_losses = []
    train_accuracies = []
    validation_accuracies = []
    ad_recalls = []
    checkpoint_path = Path("./models/convnext_best.pth")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_validation_score = None

    plt.ion()
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    (
        loss_axis,
        accuracy_axis,
        recall_axis,
        learning_rate_axis,
    ) = axes.flat

    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = perf_counter()
        current_learning_rate = optimizer.param_groups[0]["lr"]
        epoch_loss, epoch_accuracy = train_model(model, train_loader, optimizer, device)
        (
            validation_loss,
            validation_accuracy,
            ad_recall,
            false_negative_rate,
        ) = evaluate_model(model, validation_loader, loss_function, device)
        epoch_duration = perf_counter() - epoch_start_time

        # Use validation accuracy to select the test-accuracy candidate; then prefer AD recall.
        validation_score = (validation_accuracy, ad_recall, -validation_loss)
        if best_validation_score is None or validation_score > best_validation_score:
            best_validation_score = validation_score
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "class_to_label": CLASS_TO_LABEL,
                    "validation_loss": validation_loss,
                    "validation_accuracy": validation_accuracy,
                    "validation_ad_recall": ad_recall,
                    "validation_ad_false_negative_rate": false_negative_rate,
                },
                checkpoint_path,
            )
            print(f"Saved best model parameters to {checkpoint_path}")

        epochs.append(epoch)
        learning_rates.append(current_learning_rate)
        train_losses.append(epoch_loss)
        validation_losses.append(validation_loss)
        train_accuracies.append(epoch_accuracy)
        validation_accuracies.append(validation_accuracy)
        ad_recalls.append(ad_recall)

        # Plot training and validation loss.
        loss_axis.clear()
        loss_axis.plot(epochs, train_losses, marker="o", label="Train")
        loss_axis.plot(epochs, validation_losses, marker="o", label="Validation")
        loss_axis.set(title="Loss", xlabel="Epoch", ylabel="Loss")
        loss_axis.legend()
        loss_axis.grid(True, alpha=0.3)

        # Plot training and validation accuracy.
        accuracy_axis.clear()
        accuracy_axis.plot(epochs, train_accuracies, marker="o", label="Train")
        accuracy_axis.plot(
            epochs, validation_accuracies, marker="o", label="Validation"
        )
        accuracy_axis.set(
            title="Accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1)
        )
        accuracy_axis.legend()
        accuracy_axis.grid(True, alpha=0.3)

        # Plot validation recall for the AD class.
        recall_axis.clear()
        recall_axis.plot(epochs, ad_recalls, marker="o", color="tab:green")
        recall_axis.set(
            title="Validation AD Recall",
            xlabel="Epoch",
            ylabel="Recall",
            ylim=(0, 1),
        )
        recall_axis.grid(True, alpha=0.3)

        # Plot the warmup and cosine learning-rate schedule.
        learning_rate_axis.clear()
        learning_rate_axis.plot(epochs, learning_rates, marker="o", color="tab:purple")
        learning_rate_axis.set(
            title="Learning Rate", xlabel="Epoch", ylabel="Learning rate"
        )
        learning_rate_axis.grid(True, alpha=0.3)

        figure.tight_layout()
        figure.canvas.draw_idle()
        plt.pause(0.01)

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"train loss: {epoch_loss:.4f} | "
            f"train accuracy: {epoch_accuracy:.2%} | "
            f"validation loss: {validation_loss:.4f} | "
            f"validation accuracy: {validation_accuracy:.2%} | "
            f"AD recall: {ad_recall:.2%} | "
            f"learning rate: {current_learning_rate:.2e} | "
            f"time: {epoch_duration:.2f}s"
        )
        scheduler.step()

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Testing best model from epoch {checkpoint['epoch']}.")
    test_loss, test_accuracy, test_ad_recall, _ = evaluate_model(
        model, test_loader, loss_function, device
    )
    print(
        f"Test results | "
        f"loss: {test_loss:.4f} | "
        f"accuracy: {test_accuracy:.2%} | "
        f"AD recall: {test_ad_recall:.2%}"
    )

    plot_directory = Path("./plots")
    plot_directory.mkdir(parents=True, exist_ok=True)
    plot_path = plot_directory / "training_metrics.png"
    figure.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Saved training plots to {plot_path}")
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
