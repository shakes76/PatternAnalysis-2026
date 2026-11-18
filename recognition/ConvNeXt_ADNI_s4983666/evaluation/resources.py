"""Measure synchronized forward-only inference separately from data loading."""

import math
import statistics
import time

import torch

from utils.runtime import sync_device
from utils.training_controls import ExecutionControls, autocast_context


def cuda_peak_mib(device: torch.device) -> float | None:
    """Return measured allocation, or None when CUDA memory is unavailable."""
    return torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None


@torch.inference_mode()
def profile_inference(model: torch.nn.Module, image_size: tuple[int, int], device: torch.device,
                      controls: ExecutionControls, batch_sizes: tuple[int, ...] = (1, 64), warmup: int = 10,
                      repeats: int = 100, enabled: bool = True,
                      on_cpu: bool = False) -> dict:
    """Benchmark synthetic tensors at the real input shape without scoring labels.

    Inputs are resident on device. CUDA allocation is reset separately per batch
    size. OOM is recorded as a resource limit, not a substitute timing. Reduced
    repeat counts are permitted for smoke checks but explicitly flagged.
    """
    if type(warmup) is not int or warmup < 1 or type(repeats) is not int or repeats < 1:
        raise ValueError("Profile warmup and repeats must be positive integers.")
    if not batch_sizes or any(type(n) is not int or n < 1 for n in batch_sizes):
        raise ValueError("Profile batch sizes must be positive integers.")
    result = {"status": "measured", "device": str(device), "warmup": warmup,
              "repeats": repeats, "input_shape": [getattr(model, "input_channels", 1), *image_size],
              "publication_protocol_complete": warmup >= 10 and repeats >= 100,
              "precision": controls.precision,
              "timing_scope": "device_resident_forward_only_excludes_loading_and_transfer",
              "batches": []}
    if not enabled or (device.type != "cuda" and not on_cpu):
        result["status"] = "skipped_by_configuration" if not enabled else "not_measured_cpu"
        result["publication_protocol_complete"] = False
        return result
    previous_mode = model.training
    model.eval()
    try:
        for batch_size in batch_sizes:
            images = logits = None
            try:
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                images = torch.zeros(batch_size, getattr(model, "input_channels", 1), *image_size, device=device)
                for _ in range(warmup):
                    with autocast_context(controls, device):
                        model(images)
                sync_device(device)
                elapsed = []
                for _ in range(repeats):
                    sync_device(device)
                    started = time.perf_counter()
                    with autocast_context(controls, device):
                        logits = model(images)
                    sync_device(device)
                    elapsed.append((time.perf_counter() - started) * 1000)
                if not torch.isfinite(logits).all().item():
                    raise ValueError("Inference profile produced non-finite outputs.")
                mean = statistics.mean(elapsed)
                if not math.isfinite(mean) or mean <= 0:
                    raise ValueError("Inference profile produced invalid timing.")
                result["batches"].append({"batch_size": batch_size, "status": "measured",
                                          "mean_batch_ms": mean,
                                          "std_batch_ms": statistics.pstdev(elapsed),
                                          "mean_ms_per_slice": mean / batch_size,
                                          "slices_per_second": batch_size * 1000 / mean,
                                          "peak_cuda_allocated_mib": cuda_peak_mib(device),
                                          "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20
                                          if device.type == "cuda" else None})
            except torch.cuda.OutOfMemoryError:
                result["batches"].append({"batch_size": batch_size, "status": "cuda_out_of_memory"})
                result["status"] = "partial_resource_limit"
                result["publication_protocol_complete"] = False
            finally:
                del images, logits
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    finally:
        model.train(previous_mode)
    return result
