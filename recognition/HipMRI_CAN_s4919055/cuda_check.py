"""Synthetic CUDA diagnostic, not CAN3D training or a performance benchmark.

Submit cuda_check.slurm from the remote project folder after creating runs/.
This test uses no MRI data and does not change the patient split.
"""

import os
import socket

import torch
from torch import nn


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("Run this diagnostic inside a Slurm GPU allocation.")
    print("Job:", os.environ["SLURM_JOB_ID"], flush=True)
    print("Host:", socket.gethostname(), flush=True)
    print("PyTorch:", torch.__version__, flush=True)
    print("Package CUDA version:", torch.version.cuda, flush=True)
    print("CUDA available:", torch.cuda.is_available(), flush=True)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable in this allocation; inspect the job logs.")

    device = torch.device("cuda:0")
    print("Device:", torch.cuda.get_device_name(device), flush=True)
    torch.manual_seed(42)
    layer = nn.Conv3d(1, 2, kernel_size=3, padding=1).to(device)
    sample = torch.randn(1, 1, 16, 16, 16, device=device)
    torch.cuda.reset_peak_memory_stats(device)
    output = layer(sample)
    if tuple(output.shape) != (1, 2, 16, 16, 16):
        raise RuntimeError("Unexpected 3D convolution output shape.")
    # Exercise backward using a synthetic scalar, not a segmentation loss.
    diagnostic_value = output.square().mean()
    if not torch.isfinite(diagnostic_value).item():
        raise RuntimeError("Nonfinite synthetic output.")
    diagnostic_value.backward()
    for parameter in layer.parameters():
        if parameter.grad is None or not torch.all(torch.isfinite(parameter.grad)).item():
            raise RuntimeError("Missing or nonfinite convolution gradients.")
    torch.cuda.synchronize(device)
    print("Input shape:", tuple(sample.shape), flush=True)
    print("Output shape:", tuple(output.shape), flush=True)
    print("Diagnostic peak allocated MiB:", torch.cuda.max_memory_allocated(device) / 1024**2, flush=True)
    print("CUDA_CHECK_PASSED", flush=True)
    print("This tiny test does not establish CAN3D memory needs or segmentation accuracy.", flush=True)


if __name__ == "__main__":
    main()
