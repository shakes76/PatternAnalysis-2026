"""Measure peak GPU memory usage for baseline and Improved U-Net."""

import torch

from modules import UNet, ImprovedUNet


def main():
    """Profile GPU memory during repeated forward passes."""

    # This script must run on a CUDA-enabled GPU compute node.
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required. Run this script on a Rangpur GPU node."
        )

    device = torch.device("cuda")

    models = [
        ("Baseline U-Net", UNet(in_channels=1, num_classes=6)),
        ("Improved U-Net", ImprovedUNet(in_channels=1, num_classes=6)),
    ]

    for name, model in models:
        # Clear cached memory and reset the peak memory counter.
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Move the model to the GPU and prepare a sample MRI input.
        model = model.to(device).eval()
        image = torch.randn(1, 1, 256, 256, device=device)

        # Perform ten forward passes without gradient tracking.
        with torch.inference_mode():
            for _ in range(10):
                output = model(image)

            # Wait for all GPU operations to complete.
            torch.cuda.synchronize()

        # Report peak memory allocated by PyTorch in MiB.
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

        print(f"{name}: {peak_mb:.2f} MiB peak GPU memory")

        # Release model and input tensors before profiling the next model.
        del model, image, output
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

