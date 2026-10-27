"""One-batch CNN training-step smoke test for COMP3710 ADNI.

This checks that the baseline can receive REAL MRI batches, calculate a loss,
backpropagate gradients, and update parameters. It is NOT full training and
it does not evaluate on the test split or save any model weights.

Run on Windows from the project folder, e.g.:
  python test_models.py --data-root C:\\...\\AD_NC --manifest-dir C:\\...\\prepared_seed42
"""

import argparse
from pathlib import Path

import torch
from torch import nn

from dataset import make_dataloaders
from modules import BaselineCNN


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available in this Python environment.")
    device = torch.device(
        "cuda" if args.device == "cuda" or
        (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    torch.manual_seed(42)

    # The loader validates all three manifest files, including patient overlap.
    loaders = make_dataloaders(
        data_root=args.data_root,
        manifest_dir=args.manifest_dir,
        batch_size=args.batch_size,
        image_size=224,
        num_workers=0,  # Safest choice for initial Windows testing.
        seed=42,
        augment_train=False,
    )
    batch = next(iter(loaders["train"]))  # ONLY a training batch.
    images = batch["image"].to(device)
    labels = batch["label"].to(device)

    model = BaselineCNN(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(images)
    expected_shape = (images.size(0), 2)
    assert tuple(logits.shape) == expected_shape, (
        f"Expected logits {expected_shape}; received {tuple(logits.shape)}"
    )
    assert torch.isfinite(logits).all(), "Non-finite logits"
    assert labels.dtype == torch.int64 and labels.ndim == 1, "Invalid labels"
    assert set(labels.cpu().tolist()).issubset({0, 1}), "Unknown diagnosis labels"

    loss = criterion(logits, labels)
    assert torch.isfinite(loss).item(), "Non-finite cross-entropy loss"
    loss.backward()

    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients, "No gradients were calculated"
    assert all(torch.isfinite(grad).all().item() for grad in gradients), (
        "At least one gradient contains NaN or infinity"
    )
    assert any(torch.count_nonzero(grad).item() > 0 for grad in gradients), (
        "All gradients are zero"
    )
    optimizer.step()  # Exactly ONE test update; no checkpoints are saved.

    print("MODEL: BaselineCNN (class 0=NC, class 1=AD)")
    print(f"Device: {device}")
    print(f"Parameters: {parameter_count:,}")
    print(f"Input batch: {tuple(images.shape)}")
    print(f"Target labels: {tuple(labels.shape)}")
    print(f"Output logits: {tuple(logits.shape)}")
    print(f"Cross-entropy loss (untrained model): {loss.item():.4f}")
    print("PASS: forward, loss, backward and one optimizer step succeeded.")
    print("No full training, test-set evaluation, or weight saving was performed.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, AssertionError, RuntimeError) as error:
        raise SystemExit(f"ERROR: {error}") from error
