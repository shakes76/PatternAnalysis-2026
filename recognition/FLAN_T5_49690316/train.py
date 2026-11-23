from pathlib import Path

import argparse
import subprocess
import sys
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForSeq2Seq

from config import MODEL_ID
from dataset import (
    load_audited_data,
    split_train_development,
    tokenize_example,
)
from modules import RadiologySummarizer

import json
import time


def make_loader(split, tokenizer, batch_size, shuffle):
    features = [
        tokenize_example(example, tokenizer)
        for example in split
    ] # Prepares tokenized examples 

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
    ) # Reuses the tested collator for dynamic padding and -100 target padding

    return DataLoader(
        features,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        generator=torch.Generator().manual_seed(42),
    ) # Returns a dataloader with the features and collator


def run_epoch(
    model,
    loader,
    device,
    optimizer=None,
    start_batch=0,
    accumulated_loss=0.0,
    accumulated_tokens=0,
    checkpoint_callback=None,
    save_every=500,
): # Processes remaining batches once, or every batch for a fresh epoch
    # Passing an optimizer enables training, omitting it performs loss evaluation

    is_training = optimizer is not None
    model.train(is_training) # Selects training or eval behaviour

    # Continue the saved running totals when resuming partway through an epoch
    total_loss = accumulated_loss
    total_tokens = accumulated_tokens

    for batch_index, batch in enumerate(loader):
        if batch_index < start_batch:
            continue # Skip completed batches without running the model again

        batch = {
            name: tensor.to(device)
            for name, tensor in batch.items()
        } # Move all tensors to device

        if is_training:
            optimizer.zero_grad(set_to_none=True)
            # Clear initial gradients if training

        with torch.set_grad_enabled(is_training):
            outputs = model(**batch) # Forward pass
            loss = outputs.loss 

            if not torch.isfinite(loss).item():
                raise RuntimeError("Nonfinite loss")
            # Check that loss is finite

            if is_training:
                loss.backward() # backpropagate
                torch.nn.utils.clip_grad_norm_( # Clipping, weight updates
                    model.parameters(),
                    max_norm=1.0,
                )
                optimizer.step() # Optimize

        token_count = (batch["labels"] != -100).sum().item() # Counts actual target tokens excluding padding
        total_loss += loss.item() * token_count 
        total_tokens += token_count # Accumulates token weighted loss

        next_batch = batch_index + 1 # Position after the completed update

        if is_training and (
            next_batch % 100 == 0 or next_batch == len(loader)
        ):
            print(
                f"Batch {next_batch}/{len(loader)}: "
                f"running loss = {total_loss / total_tokens:.4f}",
                flush=True,
            ) # Report progress periodically and at the end of training

        if (
            is_training
            and checkpoint_callback is not None
            and next_batch % save_every == 0
        ):
            checkpoint_callback(next_batch, total_loss, total_tokens)
            # Ask the caller to save the next batch position and running totals

    return total_loss / total_tokens # Returns average loss per non-padding target token


def main():
    parser = argparse.ArgumentParser() 
    # Creates the parser


    parser.add_argument(
        "--mode",
        choices=["full", "lora"],
        default="full",
    ) # Defines a --mode option restricted to full or lora, defaulting to full

    parser.add_argument("--learning-rate", type=float, default=1e-4) # parses a numeric learning rate
    parser.add_argument("--run-name", default=None) # parses an optional checkpoint folder name
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--train-size", type=int, default=128)
    parser.add_argument("--dev-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-every", type=int, default=500) 
    # Controls how many training batches occur between recovery saves
    parser.add_argument("--test-checkpoint", help="Evaluate saved weights instead of training")
    parser.add_argument("--test-split", choices=["development", "final"], default="final")
    parser.add_argument("--test-samples", type=int, default=10000)
    parser.add_argument("--test-output-name", help="New results folder for --test-checkpoint")

    args = parser.parse_args() # Reads command line options, selected value is available as args.mode

    # Reuse the benchmark implementation for a test-only training-driver mode.
    # This avoids duplicating ROUGE or silently evaluating after every training run.
    if args.test_checkpoint:
        if not args.test_output_name or args.test_samples <= 0:
            parser.error("Testing requires --test-output-name and a positive --test-samples")
        subprocess.run([
            sys.executable, str(Path(__file__).resolve().parent / "evaluate.py"),
            "--checkpoint", args.test_checkpoint,
            "--split", args.test_split,
            "--samples", str(args.test_samples),
            "--output-name", args.test_output_name,
        ], check=True)
        return

    if min(args.epochs, args.train_size, args.dev_size, args.batch_size, args.save_every) <= 0:
        parser.error("Epochs, sample counts, and batch size must be positive")
        # Rejects invalid settings before loading the model

    if args.learning_rate <= 0:
        parser.error("Learning rate must be positive")
        # Rejects invalid settings before loading the model


    torch.manual_seed(42) # Set seed for reproduction

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Device:", device, flush=True) # Select device

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = RadiologySummarizer(
        MODEL_ID,
        use_lora=args.mode == "lora",
    ).to(device)
    # Define model and tokenizer

    _, clean_train = load_audited_data() # Filter data 
    training, development = split_train_development(clean_train) # Keep the 5 percent for development

    training_subset = training.shuffle(seed=42).select(
        range(min(args.train_size, len(training)))
    )
    development_subset = development.shuffle(seed=42).select(
        range(min(args.dev_size, len(development)))
    )

    training_loader = make_loader(
        training_subset,
        tokenizer,
        batch_size=args.batch_size,
        shuffle=True,
    ) # Create data loader for training data
    
    development_loader = make_loader(
        development_subset,
        tokenizer,
        batch_size=args.batch_size,
        shuffle=False,
    ) # Create data loader for development

    optimizer = torch.optim.AdamW(
        [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ],
        lr=args.learning_rate,
    )
    # Optimizer includes only trainable parameters. In LoRA mode, they are the adapters

    run_name = args.run_name or f"pilot_{args.mode}"

    checkpoint_dir = (
        Path(__file__).resolve().parent
        / "checkpoints"
        / run_name
    ) # Save checkpoint

    checkpoint_dir.mkdir(parents=True, exist_ok=True) # Creates checkpoint directory
    resume_path = checkpoint_dir / "training_state.pt" 

    run_settings = { # Records the settings that must match when resuming. Omit the epoch line so we can extend a run
        "model_id": MODEL_ID,
        "mode": args.mode,
        "learning_rate": args.learning_rate,
        "train_size": len(training_subset),
        "dev_size": len(development_subset),
        "batch_size": args.batch_size,
        "seed": 42,
    }

    start_epoch = 1
    resume_batch = 0
    resume_loss = 0.0
    resume_tokens = 0

    best_development_loss = float("inf")

    if args.resume:
        state = torch.load( # Reads local training state onto CPU
            resume_path,
            map_location="cpu",
            weights_only=True,
        )

        if state["settings"] != run_settings: # Prevents accidentally continuing a different experiment
            raise ValueError("Resume settings do not match the saved run")

        model.load_state_dict(state["model"]) # Restores the wrapper weights including adapters in LoRA mode
        optimizer.load_state_dict(state["optimizer"]) # Retains AdamW's accumulated update statistics

        start_epoch = state["completed_epoch"] + 1
        resume_batch = state.get("next_batch", 0)
        resume_loss = state.get("accumulated_loss", 0.0)
        resume_tokens = state.get("accumulated_tokens", 0)
        best_development_loss = state["best_development_loss"] # Preserves progress and checkpoint selection

        torch.set_rng_state(state["torch_rng"]) # Preserves random state

        if device.type == "cuda":
            if state["cuda_rng"] is None:
                raise ValueError("Checkpoint was not created on CUDA")
            torch.cuda.set_rng_state_all(state["cuda_rng"])

        if device.type == "mps":
            if state["mps_rng"] is None:
                raise ValueError("Checkpoint was not created on MPS")
            torch.mps.set_rng_state(state["mps_rng"])

        print(f"Resuming at epoch {start_epoch}, batch offset {resume_batch}", flush=True)

    elif resume_path.exists():
        raise FileExistsError(
            "Run already exists; use --resume or choose a new --run-name"
        ) # Continue an existing resumable run if available


    if not args.resume:
        configuration = {
            "arguments": vars(args),
            "effective_settings": run_settings,
            "device": str(device),
            "torch_version": str(torch.__version__),
            "gpu_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda" else None
            ),
        }

        with (checkpoint_dir / "run_config.json").open("w") as file:
            json.dump(configuration, file, indent=2)

    trainable_count = sum( # Confirms that the requested mode actually has the expected trainable count
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("Learning rate:", args.learning_rate, flush=True)
    print("Training mode:", args.mode, flush=True)
    print("Trainable parameters:", f"{trainable_count:,}", flush=True)

    if not args.resume: # Measure initial loss only if its a fresh run
        initial_loss = run_epoch(model, development_loader, device)
        print(f"Initial development loss: {initial_loss:.4f}", flush=True)

    def save_recovery(
        completed_epoch,
        next_batch=0,
        accumulated_loss=0.0,
        accumulated_tokens=0,
    ):
        """Uses the same saving logic for periodic and epoch-boundary checkpoints"""
        recovery_state = {
            "settings": run_settings,
            "completed_epoch": completed_epoch,
            "next_batch": next_batch,
            "accumulated_loss": accumulated_loss,
            "accumulated_tokens": accumulated_tokens,
            "best_development_loss": best_development_loss,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": (
                torch.cuda.get_rng_state_all()
                if device.type == "cuda" else None
            ),
            "mps_rng": (
                torch.mps.get_rng_state()
                if device.type == "mps" else None
            ),
        }

        temporary_path = checkpoint_dir / "training_state.tmp"
        torch.save(recovery_state, temporary_path)
        temporary_path.replace(resume_path)

        print(
            f"Saved recovery: completed epoch {completed_epoch}, "
            f"batch offset {next_batch}",
            flush=True,
        )

    for epoch in range(start_epoch, args.epochs + 1):
        training_loader.generator.manual_seed(42 + epoch)
        # Recreates the epoch's shuffled batch order

        def save_partial(next_batch, total_loss, total_tokens):
            save_recovery(
                completed_epoch=epoch - 1,
                next_batch=next_batch,
                accumulated_loss=total_loss,
                accumulated_tokens=total_tokens,
            )

        # Start timing after any preceding GPU work has finished
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        elif device.type == "mps":
            torch.mps.synchronize()

        training_started = time.perf_counter()

        training_loss = run_epoch(
            model,
            training_loader,
            device,
            optimizer,
            start_batch=resume_batch,
            accumulated_loss=resume_loss,
            accumulated_tokens=resume_tokens,
            checkpoint_callback=save_partial,
            save_every=args.save_every,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()

        training_seconds = time.perf_counter() - training_started
        development_started = time.perf_counter()

        development_loss = run_epoch(
            model, development_loader, device
        )

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()

        development_seconds = time.perf_counter() - development_started

        # Peak PyTorch-allocated CUDA memory across training and development
        peak_gpu_gib = (
            torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            if device.type == "cuda" else None
        )

        print(
            f"Epoch {epoch}: training loss = {training_loss:.4f}, "
            f"development loss = {development_loss:.4f}",
            flush=True,
        )

        # For resumed epochs, measure only batches processed in this session
        processed_batches = len(training_loader) - resume_batch

        epoch_record = {
            "epoch": epoch,
            "training_loss": training_loss,
            "development_loss": development_loss,
            "training_seconds_this_session": training_seconds,
            "development_seconds": development_seconds,
            "training_batches_this_session": processed_batches,
            "training_batches_per_second": (
                processed_batches / training_seconds
            ),
            "peak_cuda_allocated_gib": peak_gpu_gib,
        }

        with (checkpoint_dir / "metrics.jsonl").open("a") as file:
            file.write(json.dumps(epoch_record) + "\n")

        print(
            f"Training: {training_seconds:.1f}s; "
            f"development: {development_seconds:.1f}s; "
            f"peak CUDA memory: {peak_gpu_gib}",
            flush=True,
        )

        if development_loss < best_development_loss:
            best_development_loss = development_loss
            model.model.save_pretrained(
                checkpoint_dir,
                safe_serialization=True,
            )
            tokenizer.save_pretrained(checkpoint_dir)
            print("Saved best pilot checkpoint", flush=True)

        # Finds best output and saves it

        save_recovery(completed_epoch=epoch)

        resume_batch = 0
        resume_loss = 0.0
        resume_tokens = 0

        


if __name__ == "__main__":
    main()
