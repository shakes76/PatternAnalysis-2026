"""
profiling.py
Author: Derek McKinney (dm-uq), SN: 45506053

Contains utilities used to conduct performance and resource profiling.

CUDA timing and memory measurements follow PyTorch guidance on asynchronous
execution, synchronisation, and peak-memory statistics.

References:
    PyTorch CUDA semantics:
    https://docs.pytorch.org/docs/stable/notes/cuda.html

"""

# IMPORTS

from statistics import mean, median
from time import perf_counter
from typing import TypeAlias

import torch
from torch import Tensor
from torch_geometric.data import Data

from modules import GCNNetwork4Layer, MLPNetwork3Layer

# CONSTANTS

ProfileModel: TypeAlias = MLPNetwork3Layer | GCNNetwork4Layer


# parameters
def count_parameters(
        model: ProfileModel,
) -> int:
    """
    Count the total number of parameters in a model.

    Args:
        model: Model to profile.

    Returns:
        Total number of model parameters.

    """
    # sum the number of elements across all model parameters
    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


# profiling helper
def _forward_pass(
        model: ProfileModel,
        data: Data,
) -> Tensor:
    """
    Perform a single forward pass for profiling.

    The forward call is selected according to the model architecture, with the
    MLP using only the node features only, while the GCN additionally uses
    graph connectivity via the edge index.

    Args:
        model: Loaded prediction model to profile.
        data: Graph data containing node features and connectivity.

    Returns:
        Raw class logits for all graph nodes.

    Raises:
        TypeError: If the supplied model type is unsupported.

    """
    if isinstance(model, MLPNetwork3Layer):
        return model(data.x)

    if isinstance(model, GCNNetwork4Layer):
        return model(
            data.x,
            data.edge_index,
        )

    raise TypeError(
        f"Unsupported profiling model: {type(model).__name__}"
    )


# latency
@torch.inference_mode()  # measure inference
def measure_inference_latency(
        model: ProfileModel,
        data: Data,
        device: torch.device,
        warmup_runs: int = 10,
        timed_runs: int = 100,
) -> dict[str, float]:
    """
    Measure forward-pass inference latency for a model.

    The graph data is assumed to already reside on the inference device.
    Unmeasured warm-up passes are performed before timing, with CUDA
    synchronised around timed passes. Mean and median latency are recorded.

    Args:
        model: Model to profile.
        data: Graph data already located on the inference device.
        device: Device used for inference.
        warmup_runs: Number of unmeasured forward passes before timing.
        timed_runs: Number of forward passes used for latency measurement.

    Returns:
        Mean and median inference latency in milliseconds.

    """
    # use evaluation mode for frozen-model inference
    model.eval()

    # warm up the model and CUDA execution before measurement
    for _ in range(warmup_runs):
        _forward_pass(
            model=model,
            data=data,
        )

    # clear any queued warm-up work before timed runs begin
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    latencies = []

    # measure individual forward-pass latency
    for _ in range(timed_runs):
        # CUDA execution is asynchronous, must begin from a completed state
        # establish clean starting point
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        start_time = perf_counter()  # start

        _forward_pass(
            model=model,
            data=data,
        )

        # ensure timed CUDA work finished before stopping the clock
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        elapsed_ms = (perf_counter() - start_time) * 1000  # stop
        latencies.append(elapsed_ms)

    return {
        "mean_latency_ms": mean(latencies),
        "median_latency_ms": median(latencies),
    }


# VRAM
@torch.inference_mode()
def measure_peak_vram(
        model: ProfileModel,
        data: Data,
        device: torch.device,
) -> float | None:
    """
    Measure peak GPU memory allocated during model inference.

    The graph data is assumed to already reside on the inference device.
    CUDA peak-memory statistics are reset before the measured forward pass and
    synchronised before the peak allocation is recorded.

    Args:
        model: Model to profile.
        data: Graph data already located on the inference device.
        device: Device used for inference.

    Returns:
        Peak allocated GPU memory in MiB, or None for a non-CUDA device.

    """
    # VRAM profiling only available when inference uses CUDA
    if device.type != "cuda":
        return None

    # use evaluation mode for frozen-model inference
    model.eval()

    # finish any outstanding CUDA work before resetting peak-memory tracker
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    # retain model output until the measured forward pass has finished
    _inference_output = _forward_pass(
        model=model,
        data=data,
    )

    # ensure all allocations from the measured CUDA work have occurred
    torch.cuda.synchronize(device)

    # record peak MiB PyTorch tensor memory allocated during inference
    peak_bytes = torch.cuda.max_memory_allocated(device)
    peak_mib = peak_bytes / (1024 ** 2)  # use MiB for 'small' local card

    del _inference_output  # release temp output

    return peak_mib


# full profiling procedure
def profile_model(
        model: ProfileModel,
        data: Data,
        device: torch.device,
        warmup_runs: int = 10,
        timed_runs: int = 100,
) -> dict:
    """
    Perform inference profiling measurement procedures for a model.

    Args:
        model: Model to profile.
        data: Graph data already located on the inference device.
        device: Device used for inference.
        warmup_runs: Number of unmeasured forward passes before timing.
        timed_runs: Number of forward passes used for latency measurement.

    Returns:
        Dictionary containing parameter count, latency, and peak GPU
        memory measurements.

    """
    # latency profiling
    latency = measure_inference_latency(
        model=model,
        data=data,
        device=device,
        warmup_runs=warmup_runs,
        timed_runs=timed_runs,
    )

    return {
        "parameters": count_parameters(model),  # parameter profiling
        "mean_latency_ms": latency["mean_latency_ms"],
        "median_latency_ms": latency["median_latency_ms"],
        "peak_vram_mib": measure_peak_vram(  # VRAM profiling
            model=model,
            data=data,
            device=device,
        ),
    }


def print_profile(
        model_name: str,
        profile: dict,
) -> None:
    """
    Print formatted profiling results for a model.

    Args:
        model_name: Display name of the profiled model.
        profile: Profiling measurements produced for the model.

    Returns:
        None.

    """
    # profiling results table
    print(f"\n{model_name} Profiling Results")
    result_width = 30  # fixed width
    print("-" * result_width)  # title separator

    # print parameter-count and latency measurements
    print(f"{'Parameters:':<20}{profile['parameters']:,}")
    print(
        f"{'Mean latency:':<20}"
        f"{profile['mean_latency_ms']:.2f} ms"
    )
    print(
        f"{'Median latency:':<20}"
        f"{profile['median_latency_ms']:.2f} ms"
    )

    # print peak VRAM when CUDA profiling is available
    peak_vram = profile["peak_vram_mib"]

    if peak_vram is None:
        print(f"{'Peak VRAM:':<20}N/A")  # VRAM profile not available
    else:
        print(f"{'Peak VRAM:':<20}{peak_vram:.2f} MiB")

    print()  # 'border' space
