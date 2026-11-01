"""Shared reproducibility, device timing, logging and JSON helpers."""

import json
import logging
import random
from pathlib import Path
from time import perf_counter

import numpy as np
import torch


GIB = 1024**3
DEFAULT_PROMPT_TEMPLATE = (
    "Simplify this radiology report for a layperson:\n{report}"
)


def set_seed(seed, *, deterministic=False):
    """Seeds Python, NumPy and PyTorch and configures determinism.

    By default, deterministic algorithms are disabled and cuDNN benchmarking
    is enabled. Seeded runs are therefore not guaranteed to be bit-for-bit
    reproducible.

    Args:
        seed: Integer random seed.
        deterministic: Enables deterministic algorithms and disables cuDNN
            benchmarking when True. Defaults to False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def select_device(preference="auto"):
    """Selects a device, checking availability only in auto mode.

    Args:
        preference: "auto" chooses CUDA, then MPS, then CPU. An explicit
            device name or torch.device is used without an availability check.

    Returns:
        The selected torch.device.
    """
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preference)


def synchronize(device):
    """Waits for queued CUDA or MPS work; does nothing on CPU.

    Args:
        device: Device name or torch.device to synchronise.
    """
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


class DeviceClock:
    """Measures wall time after waiting for queued accelerator work."""

    def __init__(self, device):
        """Waits for pending work and starts the timer.

        Args:
            device: Device name or torch.device used by the measured work.
        """
        self.device = torch.device(device)
        synchronize(self.device)
        self.restart()

    def elapsed(self):
        """Waits for pending work and returns elapsed seconds as a float."""
        synchronize(self.device)
        return perf_counter() - self._started

    def restart(self):
        """Resets the timer without synchronising the device.

        Call after elapsed(), once the prior work has finished. This lets the
        next window start after intervening logging or other excluded work.
        """
        self._started = perf_counter()


def cuda_memory_gib(device, statistic):
    """Reads a CUDA allocator statistic, returning None on other devices.

    Args:
        device: Device name or torch.device being measured.
        statistic: torch.cuda memory function accepting a device.

    Returns:
        The memory amount in GiB as a float, or None on CPU/MPS.
    """
    device = torch.device(device)
    return statistic(device) / GIB if device.type == "cuda" else None


def measure_stage(device, operation, *args):
    """Times an operation and records CUDA allocator peaks.

    Synchronises before resetting peaks, retains allocated/cached memory, and
    excludes the peak reset and final statistic reads from elapsed time.

    Args:
        device: Device name or torch.device used by the operation.
        operation: Callable to execute and measure.
        *args: Positional arguments forwarded to operation.

    Returns:
        A (result, resources) tuple. resources holds elapsed seconds and peak
        allocated/reserved memory in GiB, or None for memory on CPU/MPS.
    """
    device = torch.device(device)
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    started = perf_counter()
    result = operation(*args)
    synchronize(device)
    seconds = perf_counter() - started

    return result, {
        "seconds": seconds,
        "peak_allocated_gpu_gib": cuda_memory_gib(
            device, torch.cuda.max_memory_allocated
        ),
        "peak_reserved_gpu_gib": cuda_memory_gib(
            device, torch.cuda.max_memory_reserved
        ),
    }


def configure_logger(name="radiology", *, output_dir=None, level=logging.INFO):
    """Replaces and closes existing handlers on the named logger.

    Adds console logging and optional file logging, and disables propagation.

    Args:
        name: Logger name, also used for the optional log filename.
        output_dir: Directory for <name>.log, created if needed; None skips it.
        level: Logging level; defaults to logging.INFO.

    Returns:
        The configured logging.Logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    handlers = [logging.StreamHandler()]
    if output_dir is not None:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        handlers.append(
            logging.FileHandler(directory / f"{name}.log", encoding="utf-8")
        )
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def write_json(data, path):
    """Writes UTF-8 JSON, creating parent directories as needed.

    Args:
        data: JSON-serialisable configuration, metrics or other data.
        path: Destination filename (str or PathLike).

    Returns:
        The destination as a pathlib.Path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    path.write_text(text, encoding="utf-8")
    return path


def read_json(path):
    """Reads UTF-8 JSON configuration, metrics or other data.

    Args:
        path: Source filename (str or PathLike).

    Returns:
        The decoded Python object.
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))
