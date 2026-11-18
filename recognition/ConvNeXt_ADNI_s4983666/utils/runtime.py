"""Device selection, random seeds, and execution environment recording.

Reproducibility: https://docs.pytorch.org/docs/2.6/notes/randomness.html
"""

import importlib.metadata
import os
import platform
import random
import sys

import torch

def seed_everything(seed):
    """Request deterministic execution within one software/hardware environment."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def select_device(requested):
    """Use CUDA when available; explicitly requested unavailable CUDA is an error."""
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable in this Python environment.")
    return torch.device(requested)


def sync_device(device):
    """Synchronize CUDA to avoid reporting asynchronous dispatch time as latency."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def environment_info(device):
    """Record actual package/device versions instead of claiming cross-platform equivalence."""
    return {
        "python": sys.version, "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in ("torch", "Pillow", "matplotlib")},
        "device": str(device), "cuda_build": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
