"""Train, validate, test and save a segmentation model for the HipMRI prostate task.

Example:
    python train.py --model-type residual --model-name residual_unet --data-dir ./data/keras_slices_data --epochs 20
"""

import argparse
import csv
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from tqdm.auto import tqdm

from dataset import get_dataloaders
from modules import BASE, get_model

# Setup constants
NUM_CLASSES = 6
PROSTATE_LABEL = 5  # Other labels can be found in dataset notebook


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train, validate, test and save a segmentation model for prostate\n"
            "segmentation on the HipMRI 2D slices.\n"
            "\n"
            "The model is trained on the training split and checked against the\n"
            "validation split after every epoch. Whenever validation prostate\n"
            "Dice improves, the weights are saved to <checkpoint-dir>/<model-name>.pt.\n"
            "After the final epoch, that best model is evaluated once on the\n"
            "test split."
        ),
        epilog=(
            "examples:\n"
            "  python train.py --model-type baseline --model-name baseline_unet\n"
            "  python train.py --model-type residual --model-name residual_unet \\\n"
            "      --epochs 30 --batch-size 8 --lr 5e-4"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=["baseline", "residual", "improved"],
        help=(
            "Model architecture to train:\n"
            "  baseline  standard U-Net\n"
            "  residual  U-Net with residual blocks\n"
            "  improved  improved U-Net variant"
        ),
    )
    required.add_argument(
        "--model-name",
        type=str,
        required=True,
        help=(
            "Name for the saved model and output files, without extension.\n"
            "e.g. 'residual_unet' saves to <checkpoint-dir>/residual_unet.pt\n"
            "and <output-dir>/residual_unet_summary.csv.\n"
            "Existing files with the same name are overwritten."
        ),
    )

    paths = parser.add_argument_group("paths")
    paths.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./data/keras_slices_data"),
        help=(
            "Directory containing the dataset split folders\n"
            "(keras_slices_train, keras_slices_seg_train, etc.).\n"
            "Relative paths are resolved from the current directory.\n"
            "(default: %(default)s)"
        ),
    )
    paths.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("./checkpoints"),
        help=(
            "Directory where the trained model is saved.\n"
            "Created if it does not exist.\n"
            "(default: %(default)s)"
        ),
    )
    paths.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help=(
            "Directory for run outputs: <model-name>_summary.csv\n"
            "(settings, training profile and test results),\n"
            "<model-name>_history.csv (per-epoch loss and Dice) and\n"
            "<model-name>_output.txt (copy of the command-line output).\n"
            "Created if it does not exist.\n"
            "(default: %(default)s)"
        ),
    )

    hyper = parser.add_argument_group("training hyperparameters")
    hyper.add_argument(
        "--epochs",
        type=int,
        default=20,
        help=(
            "Number of full passes over the training split.\n"
            "(default: %(default)s)"
        ),
    )
    hyper.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help=(
            "Number of slices per batch. Lower this if the GPU\n"
            "runs out of memory.\n"
            "(default: %(default)s)"
        ),
    )
    hyper.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help=(
            "Learning rate for the Adam optimiser.\n"
            "(default: %(default)s)"
        ),
    )
    hyper.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Random seed for Python, NumPy and PyTorch so runs\n"
            "can be reproduced.\n"
            "(default: %(default)s)"
        ),
    )
    return parser.parse_args()


def set_seed(seed):
    # Fix random seeds to make experiments more reproducible.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loaders(data_dir, batch_size, device):
    """Return train, validation and test DataLoaders."""
    return get_dataloaders(data_dir, batch_size, device)


def build_model(model_type, device):
    """Create the model to train: "baseline", "residual" or "improved"."""
    return get_model(model_type).to(device)


def train_one_epoch(model, loader, optimizer, criterion, device, training=True):
    """Run one training epoch and return (loss, prostate dice)."""
    model.train(training)
    loss_sum, pixel_count = 0.0, 0
    intersection, predicted_count, target_count = 0, 0, 0
    
    # Iterate through the dataset in batches, showing a progress bar.
    for images, masks in tqdm(loader, leave=False, mininterval=5):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, masks)
            if not torch.isfinite(loss).item():
                raise RuntimeError("Non-finite loss encountered")
            if training:
                loss.backward()
                optimizer.step()
        valid = masks != -100 # Remove padded pixels
        pixels = valid.sum().item() # Get batches total loss by multiplying average loss with pixels. Convert from tensor to number.
        loss_sum += loss.item() * pixels
        pixel_count += pixels
        prediction = logits.detach().argmax(dim=1)
        pred_prostate = (prediction == PROSTATE_LABEL) & valid # Criterion for predicted prostate label.
        true_prostate = (masks == PROSTATE_LABEL) & valid # Criterion for true prostate label
        intersection += (pred_prostate & true_prostate).sum().item() # Prediction and truth both positive
        predicted_count += pred_prostate.sum().item()
        target_count += true_prostate.sum().item()
    denominator = predicted_count + target_count
    dice = 2 * intersection / denominator if denominator else float("nan")
    return loss_sum / pixel_count, dice


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate on a validation or test loader and return (loss, prostate dice)."""
    return train_one_epoch(model, loader, None, criterion, device, training=False)


def run_settings(args, model):
    """Return the settings needed to reproduce this run."""
    return {
        "Model type": args.model_type,
        "Model name": args.model_name,
        "Parameters": sum(p.numel() for p in model.parameters()),
        "Epochs": args.epochs,
        "Batch size": args.batch_size,
        "Optimizer": "Adam",
        "Learning rate": args.lr,
        "Seed": args.seed,
        "Data directory": str(args.data_dir),
    }


def training_profile(device, history, training_start):
    """Return training time and peak GPU memory, as in the training notebooks."""
    profile = {
        "Device": str(device),
        "Completed epochs": len(history),
        "Recorded epoch time (min)": sum(row["seconds"] for row in history) / 60,
        "Wall time (min)": (time.perf_counter() - training_start) / 60,
    }
    if device.type == "cuda":
        profile.update({
            "GPU": torch.cuda.get_device_name(device),
            "Peak allocated VRAM (GiB)": torch.cuda.max_memory_allocated(device) / 1024**3,
            "Peak reserved VRAM (GiB)": torch.cuda.max_memory_reserved(device) / 1024**3,
        })
    return profile


def print_profile(title, profile):
    print(title)
    width = max(len(name) for name in profile)
    for name, value in profile.items():
        # Round floats for display only; the CSV keeps full precision.
        shown = f"{value:.3f}" if isinstance(value, float) else value
        print(f"  {name:<{width}}  {shown}")


def save_profile(profile, output_dir, model_name):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{model_name}_summary.csv"
    with open(path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])
        writer.writerows(profile.items())
    return path


def save_history(history, output_dir, model_name):
    """Save per-epoch loss, Dice and time, one row per epoch."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{model_name}_history.csv"
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    return path


def save_model(model, checkpoint_dir, args, history, target_size):
    """Save the weights with the settings needed to rebuild the model, as in the notebooks.

    Called whenever validation Dice improves, so epoch, val_prostate_dice and history
    describe the epoch being saved.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"{args.model_name}.pt"
    payload = dict(
        model_state_dict=model.state_dict(),
        model_type=args.model_type,
        num_classes=NUM_CLASSES,
        base=BASE,
        prostate_label=PROSTATE_LABEL,
        target_size=target_size,
        epoch=len(history),
        val_prostate_dice=history[-1]["val_prostate_dice"],
        history=history,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    torch.save(payload, path)
    return path


class Tee:
    """Write to several streams at once, e.g. the terminal and a log file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def main():
    args = parse_args()

    # Copy everything printed to <output-dir>/<model-name>_output.txt.
    # tqdm progress bars go to stderr, so they stay out of the file.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / f"{args.model_name}_output.txt"
    terminal = sys.stdout
    with open(log_path, "w") as log:
        sys.stdout = Tee(terminal, log)
        try:
            print("Command:", " ".join(sys.argv))
            run(args)
            print("Saved command-line output to", log_path)
        except BaseException:
            # Keep the error in the log as well as showing it in the terminal.
            traceback.print_exc(file=log)
            raise
        finally:
            sys.stdout = terminal


def run(args):
    set_seed(args.seed)

    # Normal device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Dataset:", args.data_dir)

    train_loader, val_loader, test_loader = build_loaders(args.data_dir, args.batch_size, device)
    model = build_model(args.model_type, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Ignore artificial padding added by the dataset loader.
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

    # Start timing and track peak PyTorch GPU memory during training and validation.
    training_start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    # Training and validation, keeping the checkpoint with the best validation prostate Dice.
    history = []
    best_dice, best_epoch, model_path = -1.0, None, None
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss, train_dice = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_dice = evaluate(model, val_loader, criterion, device)
        seconds = time.perf_counter() - epoch_start
        history.append(dict(
            epoch=epoch,
            train_loss=train_loss, train_prostate_dice=train_dice,
            val_loss=val_loss, val_prostate_dice=val_dice,
            seconds=seconds,
        ))
        improved = val_dice > best_dice
        if improved:
            best_dice, best_epoch = val_dice, epoch
            model_path = save_model(
                model, args.checkpoint_dir, args, history, train_loader.dataset.target_size
            )
        elapsed = time.perf_counter() - training_start
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train loss {train_loss:.4f} dice {train_dice:.4f} | "
            f"val loss {val_loss:.4f} dice {val_dice:.4f} | "
            f"{seconds:.1f}s | elapsed {elapsed / 60:.1f} min | "
            f"best val {best_dice:.4f}" + (" (saved)" if improved else ""),
            flush=True,
        )

    # Measure the profile before testing so it covers training and validation only.
    profile = training_profile(device, history, training_start)
    print_profile("Training profile", profile)
    print("Saved history to", save_history(history, args.output_dir, args.model_name))

    if model_path is None:
        sys.exit("Validation prostate Dice was never a number, so no model was saved.")
    print(f"Saved best model (epoch {best_epoch}, val prostate Dice {best_dice:.4f}) to {model_path}")

    # Test the best checkpoint rather than the final epoch's weights.
    best = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    del best

    # Testing
    test_loss, test_dice = evaluate(model, test_loader, criterion, device)
    print(f"Test loss {test_loss:.4f} dice {test_dice:.4f} (best epoch {best_epoch})")

    settings = run_settings(args, model)
    print_profile("Run settings", settings)
    results = {
        "Best epoch": best_epoch,
        "Best val prostate Dice": best_dice,
        "Test loss": test_loss,
        "Test prostate Dice": test_dice,
    }
    path = save_profile({**settings, **profile, **results}, args.output_dir, args.model_name)
    print("Saved summary to", path)


if __name__ == "__main__":
    main()
