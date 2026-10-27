"""Measure a short baseline training run after warm-up."""

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

    model = SimpleCNN().cuda()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    warmup_steps = 5
    measured_steps = 30
    iterator = iter(loader)

    def train_step():
        images, labels = next(iterator)
        images = images.cuda()
        labels = labels.cuda()

        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), labels)

        if not torch.isfinite(loss).item():
            raise RuntimeError("Non-finite loss")

        loss.backward()
        optimizer.step()
        return loss.item(), images.size(0)

    for _ in range(warmup_steps):
        train_step()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()

    losses = []
    total_images = 0

    for _ in range(measured_steps):
        loss, count = train_step()
        losses.append(loss)
        total_images += count

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    print("Warm-up steps:", warmup_steps)
    print("Measured steps:", measured_steps)
    print("Measured images:", total_images)
    print("Measured time (seconds):", round(elapsed, 3))
    print("Images per second:", round(total_images / elapsed, 2))
    print("Mean loss, first 10 measured steps:",
          round(sum(losses[:10]) / 10, 4))
    print("Mean loss, last 10 measured steps:",
          round(sum(losses[-10:]) / 10, 4))
    print("Peak allocated GPU memory (MiB):",
          round(torch.cuda.max_memory_allocated() / 1024**2, 2))
    print("Rough training-only epoch estimate (seconds):",
          round(len(dataset) * elapsed / total_images, 1))


if __name__ == "__main__":
    main()