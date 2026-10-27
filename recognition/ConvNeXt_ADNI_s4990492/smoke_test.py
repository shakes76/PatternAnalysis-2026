"""Check one GPU training step using real training images."""

import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import ADNISliceDataset
from modules import SimpleCNN


def main():
    torch.manual_seed(42)
    torch.set_num_threads(2)

    if not torch.cuda.is_available():
        raise RuntimeError("Run this script in a Slurm GPU job.")

    base = Path(__file__).resolve().parent
    dataset = ADNISliceDataset(base / "splits" / "train.csv")

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(42),
    )

    images, labels = next(iter(loader))
    images = images.to("cuda")
    labels = labels.to("cuda")

    model = SimpleCNN().to("cuda")
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    before = model.classifier.weight.detach().clone()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()

    optimizer.zero_grad(set_to_none=True)
    logits = model(images)
    loss = criterion(logits, labels)

    if not torch.isfinite(loss).item():
        raise RuntimeError("Loss is not finite.")

    loss.backward()

    gradients_ok = all(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all().item()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    if not gradients_ok:
        raise RuntimeError("Missing or non-finite gradients.")

    optimizer.step()
    torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    peak_mib = torch.cuda.max_memory_allocated() / 1024**2

    parameters_ok = all(
        torch.isfinite(parameter).all().item()
        for parameter in model.parameters()
    )
    changed = not torch.equal(
        before, model.classifier.weight.detach()
    )

    if not parameters_ok or not changed:
        raise RuntimeError("Parameter update check failed.")

    print("GPU:", torch.cuda.get_device_name(0))
    print("Batch shape:", tuple(images.shape))
    print("Batch labels:", labels.cpu().tolist())
    print("Loss before update:", loss.item())
    print("Gradients finite:", gradients_ok)
    print("Classifier weights changed:", changed)
    print("One-step time (seconds):", round(elapsed, 4))
    print("Peak allocated GPU memory (MiB):", round(peak_mib, 2))
    print("Real-data training step passed")


if __name__ == "__main__":
    main()