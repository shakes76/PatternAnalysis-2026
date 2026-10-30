"""Checkpoint, experiment metadata, and training plot utilities."""

import json
from pathlib import Path

import matplotlib.pyplot as plt

def save_checkpoint(model, tokenizer, output_dir, method, epoch, validation_loss):
    """
    Save the best model checkpoint and its tokenizer.

    Full fine-tuning saves the complete model.
    LoRA saves the adapter weights and configuration.

    Args:
        model: Fine-tuned FLAN-T5 model.
        tokenizer: FLAN-T5 tokenizer.
        output_dir (str): Root directory for training outputs.
        method (str): Fine-tuning method, either 'full' or 'lora'.
        epoch (int): Current epoch number.
        validation_loss (float): Current validation loss.

    Returns:
        Path: Directory containing the saved checkpoint.
    """

    checkpoint_dir = Path(output_dir) / method / "best_checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save model weights and configuration using Hugging Face format.
    # For PEFT models, this saves the LoRA adapter rather than the full base model.
    model.save_pretrained(checkpoint_dir)

    # Save the tokenizer needed to reproduce inference.
    tokenizer.save_pretrained(checkpoint_dir)

    # Record checkpoint metadata for later analysis.
    metadata = {
        "method": method,
        "epoch": epoch,
        "validation_loss": validation_loss,
    }

    import json

    with open(checkpoint_dir / "training_metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print(f"Best checkpoint saved to: {checkpoint_dir}")

    return checkpoint_dir

def save_training_config(args):
    """
    Save the experiment configuration for reproducibility.

    Args:
        args: Parsed command-line arguments.
    """
    output_dir = Path(args.output_dir) / args.method
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / "training_config.json"

    config = vars(args).copy()
    config["model_name"] = "google/flan-t5-base"

    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(config, file, indent=4)

    print(f"Training configuration saved to: {config_path}")

def save_training_history(history, output_dir, method):
    """
    Save training and validation metrics for each epoch.

    Args:
        history (list): Per-epoch training and validation metrics.
        output_dir (str): Base output directory.
        method (str): Fine-tuning method.
    """
    experiment_dir = Path(output_dir) / method
    experiment_dir.mkdir(parents=True, exist_ok=True)

    history_path = experiment_dir / "training_history.json"

    with open(history_path, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=4)

    print(f"Training history saved to: {history_path}")

def save_training_summary(summary, output_dir, method):
    """
    Save final training metrics and performance statistics.
    """
    experiment_dir = Path(output_dir) / method
    experiment_dir.mkdir(parents=True, exist_ok=True)

    summary_path = experiment_dir / "training_summary.json"

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print(f"Training summary saved to: {summary_path}")

def plot_training_history(history, output_dir, method):
    """
    Plot and save training and validation loss curves.

    Args:
        history (list): Per-epoch training metrics.
        output_dir (str): Base output directory.
        method (str): Fine-tuning method.
    """
    if not history:
        print("No training history available for plotting.")
        return

    experiment_dir = Path(output_dir) / method
    experiment_dir.mkdir(parents=True, exist_ok=True)

    epochs = [item["epoch"] for item in history]
    training_losses = [item["training_loss"] for item in history]
    validation_losses = [item["validation_loss"] for item in history]

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        training_losses,
        marker="o",
        label="Training Loss",
    )

    plt.plot(
        epochs,
        validation_losses,
        marker="s",
        label="Validation Loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"FLAN-T5 {method.upper()} Training History")
    plt.xticks(epochs)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    figure_path = experiment_dir / "loss_curve.png"
    plt.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Loss curve saved to: {figure_path}")

