"""
Contains the source code for training, validating, testing, and saving the model.

Example usage:
    python train.py --data-dir datasets --epochs 30 --batch-size 32

Arguments:
    --model, str: Model architecture to use, 'convnext' (default) or 'baseline'
    --data-dir, str | Path
    --output-dir, str | Path
    --epochs, int
    --batch-size, int
    --learning-rate, float
    --weight-decay, float
    --val-fraction, float (0<=val_fraction<1)
    --checkpoint-every, int
    --num-workers, int
    --seed, int
    --resume-from, str | Path
    --p_drop, float: Maximum drop path rate for ConvNeXt model (0.0 to 1.0). Default is 0.0 (no drop path).
"""
import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Dict, Optional

import torch
from torch import nn
from torch.optim import AdamW

from dataset import create_dataloaders
from modules import BaselineCNN, ConvNeXt


CLASS_NAMES = ["AD", "NC"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _run_epoch(model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None
) -> Dict[str, float]:
    """
    Run a single epoch of training (or validation) and return summary metrics.
    If optimizer is provided, the model will be trained; otherwise, it will be evaluated.

    Args:
        model (nn.Module): model to train
        loader (torch.utils.data.DataLoader): data loader
        criterion (nn.Module): loss function
        device (torch.device): device to use for training ('cuda' or 'cpu')
        optimizer (Optional[torch.optim.Optimizer], optional): optimizer to use for training. Defaults to None.

    Raises:
        ValueError: _description_

    Returns:
        Dict[str, float]: _description_
    """

    training = optimizer is not None
    model.train(training)
    total_loss, total_correct, total_examples = 0.0, 0, 0
    start = time.perf_counter()

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            if labels.min() < 0 or labels.max() >= logits.shape[1]:
                raise ValueError(
                    f"Invalid labels: {labels.unique().tolist()}, "
                    f"model output shape: {logits.shape}"
                )
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_examples += batch_size

    elapsed = time.perf_counter() - start

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
        "seconds": elapsed,
        "images_per_second": total_examples / elapsed if elapsed > 0 else 0.0,
    }


def evaluate_model(model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device
    ) -> dict:
    """
    Evaluate a trained model and return loss, accuracy and confusion matrix.

    Args:
        model (nn.Module): model to evaluate
        loader (torch.utils.data.DataLoader): data loader to use in evaluation
        device (torch.device): device to use for evaluation ('cuda' or 'cpu')

    Returns:
        dict: summary metrics: loss, accuracy, confusion_matrix, class_names, seconds, images_per_second, num_examples
    """
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    confusion = [[0 for _ in CLASS_NAMES] for _ in CLASS_NAMES]
    start = time.perf_counter()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            predictions = logits.argmax(dim=1)
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (predictions == labels).sum().item()
            total += batch_size
            for actual, predicted in zip(labels.tolist(), predictions.tolist()):
                confusion[actual][predicted] += 1
    elapsed = time.perf_counter() - start
    if total == 0:
        raise ValueError("Evaluation DataLoader yielded no examples")
    return {
        "loss": total_loss / total,
        "accuracy": total_correct / total,
        "confusion_matrix": confusion,
        "class_names": CLASS_NAMES,
        "seconds": elapsed,
        "images_per_second": total / elapsed if elapsed > 0 else 0.0,
        "num_examples": total,
    }


def train_model(
    model: nn.Module,
    train_loader,
    val_loader=None,
    test_loader=None,
    *,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    checkpoint_every: int = 20,
    output_dir: str | Path = "models",
    resume_from: str | Path | None = None,
) -> dict:
    """General use training function for CNN and ConvNeXt models.

    Saves the folliwing in output_dir:
        <model_name>_best.pt       best validation-accuracy
        <model_name>_final.pt      final epoch
        <model_name>_checkpoint.pt latest checkpoint
        <model_name>_history.csv   per-epoch metrics
        <model_name>_history.json  same metrics plus final test evaluation
    
    'test_loader' is evaluated only after training.

    Args:
        model (nn.Module): model to train
        train_loader (_type_): training DataLoader
        val_loader (_type_, optional): validation DataLoader. Defaults to None.
        test_loader (_type_, optional): test DataLoader. Defaults to None.
        epochs (int, optional): number of epochs to train for. Defaults to 100.
        learning_rate (float, optional): learning rate for the optimizer. Defaults to 1e-3.
        weight_decay (float, optional): weight decay for the optimizer. Defaults to 1e-4.
        checkpoint_every (int, optional): save a checkpoint every N epochs. Defaults to 20.
        output_dir (str | Path, optional): directory to save model files. Defaults to "models".
        resume_from (str | Path | None, optional): path to a saved checkpoint to resume training from. Defaults to None.

    Returns:
        dict: training history and final evaluation metrics. Has keys:
            model_name, device, epochs_completed, best_epoch, best_validation_accuracy,
            total_training_seconds, test, history
    """
    model_name = model.__class__.__name__
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    device = torch.device(device)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model = model.to(device)
    class_weights = torch.tensor(
        [1.5, 1.0],
        dtype=torch.float32,
        device=device
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history = []
    start_epoch = 0
    best_val_accuracy = -1.0
    best_epoch = 0

    # option to resume training from a saved checkpoint (very useful)
    if resume_from is not None:
        checkpoint = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        history = checkpoint.get("history", [])
        start_epoch = checkpoint.get("epoch", 0)
        best_val_accuracy = checkpoint.get("best_val_accuracy", -1.0)
        best_epoch = checkpoint.get("best_epoch", 0)
        print(f"Resuming at epoch {start_epoch + 1} from {resume_from}")


    ### MAIN TRAINING LOOP ###
    total_start = time.perf_counter()
    for epoch in range(start_epoch, epochs):
        epoch_start = time.perf_counter()

        # Reset GPU memory statistics for this epoch.
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)

        train_metrics = _run_epoch(model, train_loader, criterion, device, optimizer)

        # Capture peak memory immediately after training
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_vram_allocated_mib = (
                torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            )
            peak_vram_reserved_mib = (
                torch.cuda.max_memory_reserved(device) / (1024 ** 2)
            )
        else:
            peak_vram_allocated_mib = ""
            peak_vram_reserved_mib = ""

        train_metrics = _run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = _run_epoch(model, val_loader, criterion, device) if val_loader else None

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        epoch_seconds = time.perf_counter() - epoch_start

        row = {
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_seconds": train_metrics["seconds"],
            "train_images_per_second": train_metrics["images_per_second"],
            "val_loss": val_metrics["loss"] if val_metrics else "",
            "val_accuracy": val_metrics["accuracy"] if val_metrics else "",
            "val_seconds": val_metrics["seconds"] if val_metrics else "",
            "learning_rate": current_lr,
            "epoch_seconds": epoch_seconds,
            "peak_vram_allocated_mib": peak_vram_allocated_mib,
            "peak_vram_reserved_mib": peak_vram_reserved_mib,
        }
        history.append(row)
        val_text = (
            f" | val loss {val_metrics['loss']:.4f} | val acc {val_metrics['accuracy']:.2%}"
            if val_metrics else ""
        )
        print(
            f"Epoch {epoch + 1}/{epochs} | train loss {train_metrics['loss']:.4f} "
            f"| train acc {train_metrics['accuracy']:.2%}{val_text} "
            f"| {epoch_seconds:.1f}s"
        )

        # save model if its improves validation accuracy
        if val_metrics and val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            best_epoch = epoch + 1
            torch.save(model.state_dict(), output_path / f"{model_name}_best.pt")

        # save checkpoint
        if (epoch + 1) % checkpoint_every == 0:
            checkpoint_data = {
                "epoch": epoch + 1,
                "model_name": model_name,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_accuracy": best_val_accuracy,
                "best_epoch": best_epoch,
                "history": history,
            }
            torch.save(checkpoint_data, output_path / f"{model_name}_checkpoint_{epoch + 1}.pt")


        # Update the CSV after each epoch, so metrics survive interruptions.
        csv_path = output_path / f"{model_name}_history.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

        # save compute being wasted on overfitted model
        if train_metrics["accuracy"] == 1.0:
            print("Training accuracy reached 100%, stopping early.")
            break

    torch.save(model.state_dict(), output_path / f"{model_name}_final.pt")

    # Final test evaluation uses the best validation checkpoint when available.
    test_results = None
    if test_loader is not None:
        best_path = output_path / f"{model_name}_best.pt"
        if val_loader is not None and best_path.exists():
            model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
        test_results = evaluate_model(model, test_loader, device)
        print(f"Final test accuracy: {test_results['accuracy']:.3%}")

    summary = {
        "model_name": model_name,
        "device": str(device),
        "epochs_completed": len(history),
        "best_epoch": best_epoch if val_loader else None,
        "best_validation_accuracy": best_val_accuracy if val_loader else None,
        "total_training_seconds": time.perf_counter() - total_start,
        "test": test_results,
        "history": history,
    }
    with (output_path / f"{model_name}_history.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Train the baseline ADNI CNN")
    parser.add_argument("--model", default="convnext", help="Model architecture to use")
    parser.add_argument("--data-dir", default="datasets", help="Directory containing train/ and test/")
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--p_drop", type=float, default=0.0)
    args = parser.parse_args()

    set_seed(args.seed)
    loaders = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    if args.model == "convnext":
        model = ConvNeXt(num_classes=2, p_drop_max=args.p_drop)
    elif args.model == "baseline":
        if args.p_drop != 0.0:
            raise ValueError("Drop path is not supported for the baseline model")
        model = BaselineCNN(num_classes=2)
    else:
        raise ValueError(f"Unknown model architecture: {args.model}")
    train_model(
        model,
        loaders["train"],
        loaders.get("val"),
        loaders.get("test"),
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        checkpoint_every=args.checkpoint_every,
        output_dir=args.output_dir,
        resume_from=args.resume_from
    )


if __name__ == "__main__":
    main()