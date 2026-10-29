"""Run a CPU smoke check for the baseline UNet2D."""

import argparse

import torch

from modules import UNet2D


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--backward", action="store_true")
    args = parser.parse_args()

    if min(args.height, args.width) < 16:
        parser.error("--height and --width must each be at least 16")
    if min(args.batch_size, args.base_channels) < 1:
        parser.error("--batch-size and --base-channels must be positive")

    torch.manual_seed(3710)
    torch.set_num_threads(1)

    model = UNet2D(base_channels=args.base_channels).cpu()
    model.eval()
    x = torch.randn(
        args.batch_size, 1, args.height, args.width,
        device="cpu", requires_grad=args.backward,
    )

    with torch.set_grad_enabled(args.backward):
        logits = model(x)

    expected = (args.batch_size, 6, args.height, args.width)
    if tuple(logits.shape) != expected:
        raise RuntimeError(f"Expected {expected}, got {tuple(logits.shape)}")
    if not torch.isfinite(logits).all():
        raise RuntimeError("Model output contains NaN or Inf")

    print("Device:", x.device)
    print("Input shape:", tuple(x.shape))
    print("Output shape:", tuple(logits.shape))
    print("Parameters:", sum(p.numel() for p in model.parameters()))
    print("Forward check: PASS")

    if args.backward:
        logits.square().mean().backward()
        tensors = [("input", x)] + list(model.named_parameters())
        for name, tensor in tensors:
            if tensor.grad is None:
                raise RuntimeError(f"Missing gradient: {name}")
            if not torch.isfinite(tensor.grad).all():
                raise RuntimeError(f"Non-finite gradient: {name}")
        print("Backward check: PASS")

    print("Model smoke check: PASS")


if __name__ == "__main__":
    main()
