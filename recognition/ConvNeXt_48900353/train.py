import argparse
from datetime import datetime
from pathlib import Path
from time import perf_counter
import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from dataset import create_dataloaders
from modules import build_model


DATASET_ROOT = "~/Documents/Datasets/ADNI/AD_NC"
SEED = 42
MODEL_NAME = "convnext"
BATCH_SIZE = 128
EPOCHS = 200
EARLY_STOPPING_PATIENCE = 1000
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
WARMUP_EPOCHS = 5

CLASS_NAMES = ["NC", "AD"]
LABEL_MAP = {"NC": 0, "AD": 1}
DATASET_SPLIT_NAMES = ["train", "test"]

def run_epoch(model, loader, criterion, device, optimizer=None, epoch=None, total_epochs=EPOCHS, scaler=None):
    """Run one training or evaluation pass and aggregate image-level metrics.

    Args:
        model: model to run training on
        loader: loader containing batches
        criterion: loss function
        device: device for images, targets, and autocasting
        optimizer: optimiser for weight updates
        epoch: epoch number for progress output
        total_epochs: total epoch count shown in progress output
        scaler: an enabled scaler also enables float16 autocasting during training or evaluation.

    Returns:
        tuple: mean loss per image and accuracy, training updates model parameters
    """

    training = optimizer is not None
    amp_enabled = scaler is not None and scaler.is_enabled()
    model.train(training)

    total_loss = 0.0
    total_correct = 0
    total_images = 0
    total_steps = len(loader)

    # validation and testing do not need gradient calculations
    with torch.set_grad_enabled(training):
        for step, (images, labels) in enumerate(loader, start=1):
            # Move both tensors to the selected CPU or CUDA device.
            images = images.to(device)
            labels = labels.to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                outputs = model(images) # forward pass
                loss = criterion(outputs, labels) # calculate loss

            # update weights if training model
            if training:
                if scaler is not None:
                    # Scale gradients to reduce float16 underflow before the optimizer step.
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            # weight by batch size so final batch is counted correctly
            total_loss += loss.item() * labels.size(0)
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total_images += labels.size(0)

            if epoch is not None:
                phase = "Training" if training else "Validation"
                print(
                    f"{phase} - Epoch {epoch}/{total_epochs} - "
                    f"Step {step}/{total_steps} "
                    f"({step / total_steps:.0%}) - "
                    f"Loss: {loss.item():.4f}",
                    end="\r",
                    flush=True,
                )

    return total_loss / total_images, total_correct / total_images

def save_training_plot(history, run_dir):
    """Save training and validation loss and accuracy curves.

    Args:
        history: epoch records containing epoch, train_loss, val_loss, train_accuracy, and val_accuracy
        run_dir: existing directory for the output plot
    """
    epochs = [entry["epoch"] for entry in history]
    train_losses = [entry["train_loss"] for entry in history]
    val_losses = [entry["val_loss"] for entry in history]
    train_accuracies = [entry["train_accuracy"] for entry in history]
    val_accuracies = [entry["val_accuracy"] for entry in history]

    figure, (loss_axis, accuracy_axis) = plt.subplots(1, 2, figsize=(12, 5))

    # create loss plot
    loss_axis.plot(epochs, train_losses, label="Training")
    loss_axis.plot(epochs, val_losses, label="Validation")
    loss_axis.set_title("Loss")
    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("Loss")
    loss_axis.legend()
    loss_axis.grid(True, alpha=0.3)

    # create accuracy plot
    accuracy_axis.plot(epochs, train_accuracies, label="Training")
    accuracy_axis.plot(epochs, val_accuracies, label="Validation")
    accuracy_axis.set_title("Accuracy")
    accuracy_axis.set_xlabel("Epoch")
    accuracy_axis.set_ylabel("Accuracy")
    accuracy_axis.legend()
    accuracy_axis.grid(True, alpha=0.3)

    # save plot under run directory
    figure.tight_layout()
    plot_path = run_dir / "training_history.png"
    figure.savefig(plot_path, dpi=150)
    plt.close(figure)
    print(f"Saved training plot to: {plot_path}")

def save_test_report(labels, predictions, ad_probabilities, test_loss, training_seconds, average_epoch_seconds, run_dir,):
    """Calculate and save test metrics and a confusion matrix with AD positive.

    Args:
        labels: true class indices, including both NC and AD
        predictions: predicted class indices aligned with labels
        ad_probabilities: AD probabilities aligned with labels and used to calculate ROC AUC
        test_loss: mean test loss per image
        training_seconds: total training duration in seconds
        average_epoch_seconds: mean duration of a completed epoch in seconds
        run_dir: existing directory for the report and confusion matrix
    """
    # convert the original class labels into binary labels: NC = 0 and AD = 1
    positive_labels = [1 if label == LABEL_MAP["AD"] else 0 for label in labels]
    positive_predictions = [1 if pred == LABEL_MAP["AD"] else 0 for pred in predictions]

    # calculate classification metrics using the binary labels
    accuracy = accuracy_score(positive_labels, positive_predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        positive_labels,
        positive_predictions,
        labels=[0, 1],
        zero_division=0,
    )
    # build a confusion matrix and calculate the ROC AUC from AD probabilities
    confusion = confusion_matrix(positive_labels, positive_predictions, labels=[0, 1])
    roc_auc = roc_auc_score(positive_labels, ad_probabilities)

    # format the test metrics as a readable report
    training_minutes, training_remaining_seconds = divmod(
        int(training_seconds), 60
    )
    report_lines = [
        "\nModel Training Report:\n"
        f"Training time: {training_minutes}m {training_remaining_seconds}s\n",
        f"Average time per epoch: {average_epoch_seconds:.2f}s\n",

        "\nTest Set Evaluation Report:",
        f"Test loss: {test_loss:.4f}",
        f"Accuracy: {accuracy:.4f}",
        f"Precision (AD positive): {precision[1]:.4f}",
        f"Recall (AD positive): {recall[1]:.4f}",
        f"F1 Score (AD positive): {f1[1]:.4f}",
        f"ROC AUC: {roc_auc:.4f}",
        f"Support: {support[0] + support[1]}",
        f"Negative support: {support[0]}",
        f"Positive support: {support[1]}\n",
    ]
    report_text = "\n".join(report_lines)

    # save the report to the current run directory and print it
    report_path = run_dir / "test_report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    print(report_text)
    print(f"Saved test statistics to: {report_path}")

    # create confusion matrix plot
    confusion_figure, confusion_axis = plt.subplots(figsize=(5, 4))
    confusion_image = confusion_axis.imshow(confusion, cmap="Blues")
    confusion_axis.set_title("Confusion Matrix")
    confusion_axis.set_xlabel("Predicted label")
    confusion_axis.set_ylabel("True label")
    confusion_axis.set_xticks([0, 1])
    confusion_axis.set_yticks([0, 1])
    confusion_axis.set_xticklabels(["NC", "AD"])
    confusion_axis.set_yticklabels(["NC", "AD"])

    # display each confusion-matrix count
    for row in range(confusion.shape[0]):
        for col in range(confusion.shape[1]):
            value = confusion[row, col]
            color = "white" if value > confusion.max() / 2 else "black"
            confusion_axis.text(col, row, str(value), ha="center", va="center", color=color)

    # save the confusion matrix plot under the run directory
    confusion_figure.colorbar(confusion_image, ax=confusion_axis)
    confusion_figure.tight_layout()
    confusion_plot_path = run_dir / "confusion_matrix.png"
    confusion_figure.savefig(confusion_plot_path, dpi=150)
    plt.close(confusion_figure)
    print(f"Saved confusion matrix image to: {confusion_plot_path}")

@torch.inference_mode()
def evaluate_test(model, test_loader, criterion, device, training_seconds, run_dir, average_epoch_seconds, scaler=None,):
    """Evaluate held-out images and save a report without updating model weights.

    Args:
        model: trained model
        test_loader: loader containing test set batches
        criterion: loss function returning a scalar batch-mean loss
        device: device for images, targets, and autocasting
        training_seconds: training duration in seconds for the report
        run_dir: existing output directory for test results
        average_epoch_seconds: average epoch duration in seconds
        scaler: enables float16 autocasting if provided
    """
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_predictions = []
    all_ad_probabilities = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            amp_enabled = scaler is not None and scaler.is_enabled()
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                outputs = model(images) # forward pass
                loss = criterion(outputs, labels) # calculate loss

            # softmax probabilities are used for AUC and confidence statistics
            probabilities = torch.softmax(outputs, dim=1)
            total_loss += loss.item() * labels.size(0)
            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(outputs.argmax(dim=1).cpu().tolist())
            all_ad_probabilities.extend(
                probabilities[:, LABEL_MAP["AD"]].cpu().tolist()
            )

    # save test statistics to run directory
    test_loss = total_loss / len(all_labels)
    save_test_report(
        all_labels,
        all_predictions,
        all_ad_probabilities,
        test_loss,
        training_seconds,
        average_epoch_seconds,
        run_dir,
    )

def parse_args():
    """Parse and validate the command-line training configuration.

    Returns:
        argparse: Dataset path, seed, architecture, batch size, epoch counts, optimizer settings, worker count, and warmup configuration

    """
    parser = argparse.ArgumentParser(
        description="Train an ADNI image classification model."
    )
    parser.add_argument("--dataset-root", default=DATASET_ROOT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--model-name",
        choices=["resnet18", "convnext"],
        default=MODEL_NAME,
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=EARLY_STOPPING_PATIENCE,
    )
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--warmup-epochs", type=int, default=WARMUP_EPOCHS)
    args = parser.parse_args()

    # Reject invalid training settings before loading data or creating a model.
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.early_stopping_patience < 1:
        parser.error("--early-stopping-patience must be at least 1")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be greater than 0")
    if args.weight_decay < 0:
        parser.error("--weight-decay cannot be negative")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    if args.warmup_epochs < 0 or args.warmup_epochs >= args.epochs:
        parser.error("--warmup-epochs must be at least 0 and less than --epochs")

    return args


def main():
    """Train a classifier, restore its best validation checkpoint, and evaluate it.
    """
    args = parse_args()

    # allow reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = (
        torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    )
    print(f"Model: {args.model_name} training on: {device_name}")

    # get train, validation, and test data loaders
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset_root=args.dataset_root,
        class_names=CLASS_NAMES,
        dataset_split_names=DATASET_SPLIT_NAMES,
        label_map=LABEL_MAP,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    # Create the model on the selected device; enable mixed precision only on CUDA.
    model = build_model(args.model_name, num_classes=len(LABEL_MAP)).to(device)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Slightly soften target labels to discourage overconfident predictions.
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # warm up linearly, then decay with cosine annealing
    # The scheduler is stepped once per epoch below, so these values are in epochs.
    warmup_steps = args.warmup_epochs
    cosine_steps = args.epochs - args.warmup_epochs
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cosine_steps,
        eta_min=1e-6,
    )
    if warmup_steps:
        scheduler = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(
                    optimizer,
                    start_factor=0.1,
                    end_factor=1.0,
                    total_iters=warmup_steps,
                ),
                cosine_scheduler,
            ],
            milestones=[warmup_steps],
        )
    else:
        scheduler = cosine_scheduler

    history = []
    best_val_accuracy = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    best_model_state = None
    epoch_times = []

    # main training loop
    training_start_time = datetime.now()
    print("\n--- Beginning Training ---")

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = perf_counter()

        # run training epoch
        train_loss, train_accuracy = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            epoch=epoch,
            total_epochs=args.epochs,
            scaler=scaler,
        )

        # run validation
        val_loss, val_accuracy = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            epoch=epoch,
            total_epochs=args.epochs,
            scaler=scaler,
        )

        scheduler.step()
        epoch_seconds = perf_counter() - epoch_start_time
        epoch_times.append(epoch_seconds)

        # add current epoch results to history
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
        })

        # if val accuracy is higher, store this model
        if best_model_state is None or val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            epochs_without_improvement = 0
            # Clone onto CPU so later optimizer updates cannot alter this checkpoint.
            best_model_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Train loss: {train_loss:.4f}, accuracy: {train_accuracy:.2%} | "
            f"Val loss: {val_loss:.4f}, accuracy: {val_accuracy:.2%} | "
            f"Time: {epoch_seconds:.2f}s"
        )

        # if validation stops improving, early stop
        if epochs_without_improvement >= args.early_stopping_patience:
            print(
                f"Early stopping after {args.early_stopping_patience} epochs "
                "without validation accuracy improvement."
            )
            break

    print("\n--- Finished Training ---")
    training_elapsed = datetime.now() - training_start_time
    training_seconds = int(training_elapsed.total_seconds())
    average_epoch_seconds = sum(epoch_times) / len(epoch_times)

    # use the best validation checkpoint for saving and test evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print(
        f"Saving model from epoch {best_epoch} "
        f"with validation accuracy {best_val_accuracy:.2%}."
    )

    # create a directory for this runs model and plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = Path("Models") / args.model_name / f"{args.model_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # save model under the run directory
    model_path = run_dir / f"{args.model_name}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to: {model_path}")

    # save training plot under the run directory
    save_training_plot(history, run_dir)

    minutes, seconds = divmod(training_seconds, 60)
    print(f"Training completed in {minutes}m {seconds}s")

    # evaluate the model on the test set
    evaluate_test(
        model,
        test_loader,
        criterion,
        device,
        training_seconds,
        run_dir,
        average_epoch_seconds,
        scaler=scaler,
    )


if __name__ == "__main__":
    main()