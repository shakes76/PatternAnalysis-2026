"""
Training entry point for FLAN-T5 radiology report summarisation.

Supports two fine-tuning methods:
1. Full fine-tuning
2. LoRA fine-tuning

"""

import argparse
import random
import math
import time

import numpy as np
import torch
from torch.optim import AdamW
from tqdm.auto import tqdm

from dataset import (
    MODEL_NAME,
    load_radiology_dataset,
    load_tokenizer,
    preprocess_function,
    create_dataloader,
)
from modules import load_full_model, load_lora_model, inspect_model
from training_artifacts import (
    save_checkpoint,
    save_training_config,
    save_training_history,
    save_training_summary,
    plot_training_history,
)

def parse_args():
    """
    Parse command-line arguments for model training.

    Returns:
        argparse.Namespace: Training configuration.
    """

    parser = argparse.ArgumentParser(
        description="Train FLAN-T5 for radiology report summarisation."
    )

    # Select the fine-tuning approach.
    parser.add_argument(
        "--method",
        type=str,
        choices=["full", "lora"],
        required=True,
        help="Fine-tuning method: full or lora.",
    )

    # Training hyperparameters.
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    # LoRA-specific hyperparameters.
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # Directory for saving training outputs.
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
    )

    # Limit optimizer updates for quick training smoke tests.
    # A value of 0 means no limit.
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=0,
        help="Maximum optimizer steps; 0 means no limit.",
    )

    # Maximum gradient norm used for gradient clipping.
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
    )

    # Optional dataset limits for smoke testing.
    # A value of 0 means using the entire split.
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=0,
        help="Maximum training samples; 0 uses the full training split.",
    )

    parser.add_argument(
        "--max_val_samples",
        type=int,
        default=0,
        help="Maximum validation samples; 0 uses the full validation split.",
    )

    parser.add_argument(
        "--verbose", action="store_true",
        help="Run additional DataLoader and forward-pass diagnostics.",
    )

    return parser.parse_args()

def set_seed(seed):
    """
    Set random seeds for reproducibility.

    Args:
        seed (int): Random seed.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device():
    """
    Select the available computation device.

    Returns:
        torch.device: CUDA device if available, otherwise CPU.
    """

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")

def initialise_model(args):
    """
    Initialise the selected fine-tuning model.

    Args:
        args (argparse.Namespace): Training configuration.

    Returns:
        torch.nn.Module: Initialised model.
    """

    if args.method == "full":
        model = load_full_model()

    else:
        model = load_lora_model(
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
        )

    return model

def initialise_dataloaders(args):
    """
    Prepare small training and validation DataLoaders for testing.

    Only a few samples are used at this stage to verify integration
    without tokenizing the entire dataset.

    Args:
        args (argparse.Namespace): Training configuration.

    Returns:
        tuple: Training and validation DataLoaders.
    """

    # Load the official dataset and FLAN-T5 tokenizer.
    dataset = load_radiology_dataset()
    tokenizer = load_tokenizer()

    # Use the full official splits unless sample limits are specified.
    train_subset = dataset["train"]
    val_subset = dataset["validation"]

    if args.max_train_samples > 0:
        train_subset = train_subset.select(
            range(min(args.max_train_samples, len(train_subset)))
        )

    if args.max_val_samples > 0:
        val_subset = val_subset.select(
            range(min(args.max_val_samples, len(val_subset)))
        )

    print("\n=== Dataset Configuration ===")
    print(f"Training samples: {len(train_subset):,}")
    print(f"Validation samples: {len(val_subset):,}")

    # Apply the same preprocessing to both subsets.
    train_tokenized = train_subset.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
        remove_columns=train_subset.column_names,
    )

    val_tokenized = val_subset.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
        remove_columns=val_subset.column_names,
    )

    # Create batches with dynamic padding and label masking.
    train_loader = create_dataloader(
        train_tokenized,
        tokenizer,
        batch_size=args.batch_size,
        shuffle=True,
    )

    val_loader = create_dataloader(
        val_tokenized,
        tokenizer,
        batch_size=args.batch_size,
        shuffle=False,
    )

    return train_loader, val_loader, tokenizer

def inspect_forward_pass(model, dataloader, device):
    """
    Verify that FLAN-T5 can process a batch and calculate loss.

    This smoke test runs a forward pass without computing gradients
    or updating model parameters.

    Args:
        model: Initialised FLAN-T5 model.
        dataloader: DataLoader containing tokenized radiology reports.
        device (torch.device): Device used for computation.
    """

    print("\n=== Forward Pass Inspection ===")

    # Move the model to the selected device.
    model = model.to(device)
    model.eval()

    # Retrieve one batch from the training DataLoader.
    batch = next(iter(dataloader))

    # Move all tensors to the same device as the model.
    batch = {
        key: value.to(device)
        for key, value in batch.items()
    }

    # Run the model without gradient computation.
    with torch.no_grad():
        outputs = model(**batch)

    # Extract the sequence-to-sequence training loss.
    loss = outputs.loss

    print(f"Forward loss: {loss.item():.4f}")

    # Verify that the model produces a valid numerical loss.
    assert torch.isfinite(loss).item(), "Forward loss is not finite"

    print("Forward pass completed successfully.")

def train_one_epoch(
        model, 
        dataloader, 
        optimizer, 
        device, 
        args,
        remaining_steps=None,
        ):
    """
    Train FLAN-T5 for one epoch using a custom PyTorch training loop.

    Supports:
        - Full fine-tuning and LoRA
        - Gradient accumulation
        - Gradient clipping
        - Optional optimizer-step limit

    Args:
        model: FLAN-T5 model.
        dataloader: Training DataLoader.
        optimizer: PyTorch optimizer.
        device (torch.device): Computation device.
        args (argparse.Namespace): Training configuration.
        remaining_steps (int | None): Maximum optimizer updates
            allowed in the current epoch. None means unlimited.

    Returns:
        tuple: Average training loss and number of optimizer steps.
    """

    model.train()
    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    total_batches = 0
    optimizer_steps = 0

    accumulation_steps = args.gradient_accumulation_steps
    num_batches = len(dataloader)

    # Show one progress bar per epoch instead of printing every optimizer step.
    max_batches = num_batches
    if remaining_steps is not None:
        max_batches = min(num_batches, remaining_steps * accumulation_steps)

    progress = tqdm(
        enumerate(dataloader),
        total=max_batches,
        desc="Training",
        unit="batch",
        leave=False,
    )

    for batch_index, batch in progress:

        # Transfer the current batch to the computation device.
        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }

        # Forward pass: calculate sequence-to-sequence loss.
        outputs = model(**batch)
        loss = outputs.loss

        if not torch.isfinite(loss).item():
            raise RuntimeError(
                f"Non-finite training loss at batch {batch_index + 1}"
            )

        # Keep the original loss for logging.
        total_loss += loss.item()
        total_batches += 1

        # Determine the size of the current accumulation group.
        group_start = (
            batch_index // accumulation_steps
        ) * accumulation_steps

        group_size = min(
            accumulation_steps,
            num_batches - group_start,
        )

        scaled_loss = loss / group_size

        # Backpropagation.
        scaled_loss.backward()

        # Scale loss before accumulating gradients.
        is_last_batch = batch_index + 1 == num_batches

        # Update weights after enough mini-batches or at epoch end.
        should_update = (
            (batch_index + 1) % accumulation_steps == 0
            or is_last_batch
        )

        if should_update:

            # Prevent excessively large gradients.
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=args.max_grad_norm,
            )

            # Apply one optimizer update.
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            optimizer_steps += 1

            progress.set_postfix(
                loss=f"{total_loss / total_batches:.4f}",
                steps=optimizer_steps,
            )

            # Stop when the remaining global optimizer-step budget is reached.
            if (
                remaining_steps is not None
                and optimizer_steps >= remaining_steps
            ):
                break

    progress.close()
    average_loss = total_loss / total_batches

    print(f"\nAverage training loss: {average_loss:.4f}")
    print(f"Optimizer steps completed: {optimizer_steps}")

    return average_loss, optimizer_steps

def validate(model, dataloader, device):
    """
    Evaluate the model on the validation dataset.

    This function calculates average validation loss without
    computing gradients or updating model parameters.

    Args:
        model: FLAN-T5 model.
        dataloader: Validation DataLoader.
        device (torch.device): Computation device.

    Returns:
        float: Average validation loss.
    """

    model.eval()

    total_loss = 0.0
    total_batches = 0

    print("\n=== Validation ===")

    with torch.no_grad():

        for batch in dataloader:

            # Move validation tensors to the selected device.
            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            # Calculate validation loss.
            outputs = model(**batch)
            loss = outputs.loss

            if not torch.isfinite(loss).item():
                raise RuntimeError("Non-finite validation loss detected")

            total_loss += loss.item()
            total_batches += 1

    if total_batches == 0:
        raise ValueError("Validation DataLoader is empty")

    average_loss = total_loss / total_batches

    print(f"Average validation loss: {average_loss:.4f}")

    return average_loss





    
def run_training(model, tokenizer, train_loader, val_loader, device, args):
    """
    Initialise the optimizer and execute the training loop.

    Args:
        model: FLAN-T5 model.
        train_loader: Training DataLoader.
        device (torch.device): Computation device.
        args (argparse.Namespace): Training configuration.
    """

    model = model.to(device)

    # Only optimize trainable parameters.
    # For LoRA, frozen pretrained parameters are excluded.
    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = AdamW(
        trainable_parameters,
        lr=args.learning_rate,
    )

    print("\n=== Training Started ===")
    print(f"Method: {args.method}")
    print(f"Device: {device}")

    total_optimizer_steps = 0
    best_validation_loss = float("inf")
    history = []

    # Start measuring training runtime.
    training_start_time = time.perf_counter()

    # Reset peak GPU memory statistics before training.
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(args.epochs):

        # Calculate the remaining optimizer-step budget.
        remaining_steps = None

        if args.max_train_steps > 0:
            remaining_steps = (
                args.max_train_steps - total_optimizer_steps
            )

            if remaining_steps <= 0:
                break

        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        training_loss, completed_steps = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            args=args,
            remaining_steps=remaining_steps,
        )

        total_optimizer_steps += completed_steps

        # Evaluate the model after each training epoch.
        validation_loss = validate(
            model=model,
            dataloader=val_loader,
            device=device,
        )

        epoch_metrics = {
            "epoch": epoch + 1,
            "training_loss": training_loss,
            "validation_loss": validation_loss,
            "optimizer_steps": completed_steps,
            "cumulative_optimizer_steps": total_optimizer_steps,
        }

        history.append(epoch_metrics)

        save_training_history(
            history=history,
            output_dir=args.output_dir,
            method=args.method,
        )

        # Save a checkpoint only when validation loss improves.
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss

            save_checkpoint(
                model=model,
                tokenizer=tokenizer,
                output_dir=args.output_dir,
                method=args.method,
                epoch=epoch + 1,
                validation_loss=validation_loss,
            )

        print(
            f"\nEpoch {epoch + 1} Summary | "
            f"Validation Loss: {validation_loss:.4f}"
        )

        # Stop once the requested smoke-test limit is reached.
        if (
            args.max_train_steps > 0
            and total_optimizer_steps >= args.max_train_steps
        ):
            break

    # Measure total training runtime.
    training_runtime = time.perf_counter() - training_start_time

    # Record peak GPU memory usage.
    peak_gpu_memory_gb = None

    if device.type == "cuda":
        peak_gpu_memory_gb = (
            torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        )

    print("\n=== Training Performance ===")
    print(f"Training runtime: {training_runtime:.2f} seconds")

    if peak_gpu_memory_gb is not None:
        print(f"Peak GPU memory: {peak_gpu_memory_gb:.2f} GB")
    else:
        print("Peak GPU memory: N/A (CPU execution)")

    summary = {
        "method": args.method,
        "device": str(device),
        "epochs_completed": len(history),
        "total_optimizer_steps": total_optimizer_steps,
        "best_validation_loss": best_validation_loss,
        "best_epoch": min(
            history,
            key=lambda item: item["validation_loss"]
        )["epoch"],
        "training_runtime_seconds": training_runtime,
        "peak_gpu_memory_gb": peak_gpu_memory_gb,
    }

    save_training_summary(
        summary=summary,
        output_dir=args.output_dir,
        method=args.method,
    )

    plot_training_history(
        history=history,
        output_dir=args.output_dir,
        method=args.method,
    )

    print("\nTraining completed successfully.")
    print(f"Total optimizer steps: {total_optimizer_steps}")

def main():
    """Initialise the training configuration and selected model."""

    args = parse_args()

    # Validate configuration before loading the model.
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")

    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if args.gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")

    if args.learning_rate is not None and args.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    if args.lora_rank <= 0:
        raise ValueError("lora_rank must be positive")

    if args.lora_alpha <= 0:
        raise ValueError("lora_alpha must be positive")

    if not 0 <= args.lora_dropout < 1:
        raise ValueError("lora_dropout must be in [0, 1)")

    if args.max_train_samples < 0:
        raise ValueError("max_train_samples must be non-negative")

    if args.max_val_samples < 0:
        raise ValueError("max_val_samples must be non-negative")

    # Use method-specific initial learning rates.
    # These are starting configurations, not tuned values.
    if args.learning_rate is None:
        args.learning_rate = 5e-5 if args.method == "full" else 1e-4

    set_seed(args.seed)

    device = get_device()

    print("\n=== Training Configuration ===")
    print(f"Model: {MODEL_NAME}")
    print(f"Method: {args.method}")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"Random seed: {args.seed}")
    print(f"Output directory: {args.output_dir}")

    # Load the selected model without starting training.
    model = initialise_model(args)

    inspect_model(
        model,
        "Full Fine-Tuning" if args.method == "full" else "LoRA Fine-Tuning",
    )

    # Verify that the selected model and data pipeline can be initialised.
    train_loader, val_loader, tokenizer = initialise_dataloaders(args)

    print("\n=== DataLoader Integration ===")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    # Additional diagnostics are optional to avoid unnecessary computation.
    if args.verbose:
        batch = next(iter(train_loader))
        for key, value in batch.items():
            print(f"{key}: {tuple(value.shape)}")
        inspect_forward_pass(model, train_loader, device)

    print("\nModel and DataLoader integration completed successfully.")

    save_training_config(args)

    # Execute the custom PyTorch training loop.
    run_training(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        args=args,
    )



if __name__ == "__main__":
    main()