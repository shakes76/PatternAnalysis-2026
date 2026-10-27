"""Synchronized stage timing and PyTorch CUDA allocator peaks.

These counters describe this process's PyTorch allocator, including objects
already resident at stage entry. They are not total GPU VRAM or a delta from
the entry allocation. The caller supplies an honest resident-state description.
"""

from copy import deepcopy
import time

import torch


class StageMeasurement:
    """Measure a CPU/CUDA stage without clearing caches or changing model state.

    CUDA is synchronized before resetting peaks and after the body. Memory
    queries and timing do not allocate model tensors, change RNG state, or alter
    training/evaluation mode. A body exception remains the propagated exception
    even if collecting its final CUDA statistics also fails.
    """

    cuda_fields = (
        "start_cuda_allocated_bytes", "start_cuda_reserved_bytes",
        "peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes",
        "end_cuda_allocated_bytes", "end_cuda_reserved_bytes",
    )

    def __init__(self, device, stage, resident_state):
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("StageMeasurement supports CPU and CUDA devices.")
        self.stage = stage
        self.resident_state = deepcopy(resident_state)
        self.result = None
        self._started = None

    @staticmethod
    def _error(error):
        return {"type": type(error).__name__, "message": str(error)}

    def __enter__(self):
        self.result = {
            "stage": self.stage, "device": str(self.device),
            "resident_state": deepcopy(self.resident_state),
            "status": "running", "elapsed_seconds": None,
            **{field: None for field in self.cuda_fields},
        }
        try:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
                torch.cuda.reset_peak_memory_stats(self.device)
                self.result["start_cuda_allocated_bytes"] = torch.cuda.memory_allocated(self.device)
                self.result["start_cuda_reserved_bytes"] = torch.cuda.memory_reserved(self.device)
            self._started = time.perf_counter()
        except BaseException as error:
            self.result["status"] = "failed"
            self.result["error"] = self._error(error)
            raise
        return self

    def __exit__(self, error_type, error, traceback):
        measurement_error = None
        try:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
                self.result["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(self.device)
                self.result["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved(self.device)
                self.result["end_cuda_allocated_bytes"] = torch.cuda.memory_allocated(self.device)
                self.result["end_cuda_reserved_bytes"] = torch.cuda.memory_reserved(self.device)
        except BaseException as collected_error:
            measurement_error = collected_error
        finally:
            self.result["elapsed_seconds"] = time.perf_counter() - self._started
        failure = error if error is not None else measurement_error
        self.result["status"] = "failed" if failure is not None else "passed"
        if failure is not None:
            self.result["error"] = self._error(failure)
        if measurement_error is not None:
            self.result["measurement_error"] = self._error(measurement_error)
            if error is None:
                raise measurement_error
        return False
