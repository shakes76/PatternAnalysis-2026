"""Check a U-Net forward pass, backward pass, and parameter update."""

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import HipMRIDataset
from modules import UNet2D


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Run this check inside a GPU job.")

    torch.manual_seed(42)
    device = torch.device("cuda")
    dataset = HipMRIDataset(split="train")
    loader = DataLoader(
        dataset, batch_size=2, shuffle=False, num_workers=0
    )
    batch = next(iter(loader))
    images = batch["image"].to(device)
    targets = batch["mask"].to(device)

    model = UNet2D().to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    optimizer.zero_grad(set_to_none=True)
    logits = model(images)
    expected_shape = (images.shape[0], 6, 256, 128)
    if tuple(logits.shape) != expected_shape:
        raise RuntimeError(f"Unexpected output shape: {logits.shape}")
    if not torch.isfinite(logits).all():
        raise RuntimeError("Non-finite logits.")

    loss = criterion(logits, targets)
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite loss.")
    loss.backward()

    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            raise RuntimeError(f"Missing gradient: {name}")
        if not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f"Non-finite gradient: {name}")

    before = model.output.weight.detach().clone()
    optimizer.step()
    if torch.equal(before, model.output.weight.detach()):
        raise RuntimeError("Output weights did not change.")
    if any(
        not torch.isfinite(parameter).all()
        for parameter in model.parameters()
    ):
        raise RuntimeError("Non-finite parameters after update.")

    print("GPU:", torch.cuda.get_device_name(0))
    print("Parameters:", sum(p.numel() for p in model.parameters()))
    print("Input shape:", tuple(images.shape))
    print("Output shape:", tuple(logits.shape))
    print("Initial cross-entropy:", loss.item())
    print("Finite gradients and parameter update: PASS")
    print("This is a smoke test, not a trained-model evaluation.")


if __name__ == "__main__":
    main()
