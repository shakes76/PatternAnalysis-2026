"""Check one real-data training step with ConvNeXt-Tiny."""

import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import ADNISliceDataset
from modules import ConvNeXtTiny


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
    images = images.cuda()
    labels = labels.cuda()

    model = ConvNeXtTiny().cuda()
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.05,
    )
    criterion = nn.CrossEntropyLoss()

    parameter_count = sum(p.numel() for p in model.parameters())
    before = model.classifier.weight.detach().clone()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()

    optimizer.zero_grad(set_to_none=True)
    logits = model(images)

    assert logits.shape == (images.size(0), 2)
    assert torch.isfinite(logits).all().item()

    loss = criterion(logits, labels)
    assert torch.isfinite(loss).item()
    loss.backward()

    gradients_ok = all(
        p.grad is not None and torch.isfinite(p.grad).all().item()
        for p in model.parameters()
        if p.requires_grad
    )
    if not gradients_ok:
        raise RuntimeError("Missing or non-finite gradients")

    optimizer.step()
    torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    peak_mib = torch.cuda.max_memory_allocated() / 1024**2

    parameters_ok = all(
        torch.isfinite(p).all().item()
        for p in model.parameters()
    )
    changed = not torch.equal(
        before, model.classifier.weight.detach()
    )
    if not parameters_ok or not changed:
        raise RuntimeError("Parameter update check failed")

    print("GPU:", torch.cuda.get_device_name(0))
    print("Input shape:", tuple(images.shape))
    print("Output shape:", tuple(logits.shape))
    print("Parameter count:", parameter_count)
    print("Loss before update:", loss.item())
    print("Parameter gradients finite:", gradients_ok)
    print("Classifier weights changed:", changed)
    print("First-step time (seconds):", round(elapsed, 4))
    print("Peak allocated GPU memory (MiB):", round(peak_mib, 2))
    print("ConvNeXt real-data training step passed")


if __name__ == "__main__":
    main()