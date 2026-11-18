"""Explicit execution-device selection for experiments and CPU smoke tests."""

import torch


def select_device(name):
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable. Install CUDA-enabled PyTorch and check the GPU driver, or explicitly use --device cpu for smoke tests.")
    return torch.device(name)
