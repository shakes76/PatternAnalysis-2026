"""Profile a checkpoint on seeded synthetic FP32 inputs, without MRI data."""

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import torch

from modules import build_model
from resource_monitor import StageMeasurement


SEED = 3710
BATCH_SIZES = (1, 4)
SPATIAL_SHAPES = ((256, 128), (256, 144))
WARMUP_ITERATIONS = 10
TIMED_ITERATIONS = 50


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def profile_checkpoint(checkpoint_path, device):
    """Run the fixed proposal profile; loading and input setup are untimed."""
    checkpoint_path = Path(checkpoint_path).resolve()
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    if device.type not in ("cpu", "cuda"):
        raise ValueError("The profile supports only cpu or cuda.")
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    checkpoint_hash = sha256_file(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model_name = checkpoint["model"]
    config = dict(checkpoint["model_config"])
    if config.get("in_channels") != 1 or config.get("num_classes") != 6:
        raise ValueError("This fixed HipMRI profile requires one input channel and six classes.")
    model = build_model(model_name, **config)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    metadata = {
        "checkpoint_epoch": checkpoint.get("epoch"),
        "limited_training_smoke": checkpoint.get("limited_training_smoke"),
    }
    # An optimizer is never constructed, and checkpoint optimizer tensors are
    # released before profiling. Only the model is moved to the selected device.
    del checkpoint
    model = model.float().to(device).eval()
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    results = []
    with torch.inference_mode():
        for batch_size in BATCH_SIZES:
            for height, width in SPATIAL_SHAPES:
                synchronize(device)
                if device.type == "cuda":
                    # Drop free allocations from the previous shape before this
                    # shape's setup/warmup. Warmup cache is retained for timing.
                    torch.cuda.empty_cache()
                values = torch.rand(
                    (batch_size, 1, height, width),
                    generator=generator, dtype=torch.float32,
                ).to(device)
                expected_shape = (batch_size, 6, height, width)
                for iteration in range(WARMUP_ITERATIONS):
                    logits = model(values)
                    if iteration == WARMUP_ITERATIONS - 1:
                        if tuple(logits.shape) != expected_shape:
                            raise ValueError("The model output violates the six-class spatial contract.")
                        if not torch.isfinite(logits).all().item():
                            raise ValueError("The warmup produced non-finite model logits.")
                    del logits
                synchronize(device)
                stage_name = f"inference_batch{batch_size}_{height}x{width}"
                measurement = StageMeasurement(
                    device,
                    stage_name,
                    "One FP32 eval model and its FP32 synthetic input remain resident; "
                    "outputs and forward temporaries exist during each forward; "
                    "no optimizer or training gradients. Warmup allocator cache is retained.",
                )
                times_ms = []
                with measurement:
                    for _ in range(TIMED_ITERATIONS):
                        synchronize(device)
                        started = perf_counter()
                        logits = model(values)
                        synchronize(device)
                        elapsed_ms = (perf_counter() - started) * 1000.0
                        times_ms.append(elapsed_ms)
                        del logits
                median = float(np.median(times_ms))
                results.append({
                    "input_shape": list(values.shape),
                    "output_shape": list(expected_shape),
                    "warmup_iterations": WARMUP_ITERATIONS,
                    "timed_iterations": TIMED_ITERATIONS,
                    "raw_batch_times_ms": times_ms,
                    "median_batch_ms": median,
                    "p95_batch_ms": float(np.percentile(times_ms, 95, method="linear")),
                    "median_ms_per_slice": median / batch_size,
                    "stage_measurement": measurement.result,
                })
                del values
                print(
                    f"{stage_name}: median={median:.3f} ms; "
                    f"p95={results[-1]['p95_batch_ms']:.3f} ms",
                    flush=True,
                )
    if sha256_file(checkpoint_path) != checkpoint_hash:
        raise RuntimeError("The checkpoint changed during profiling; results are not accepted.")
    project = Path(__file__).resolve().parent
    return {
        "measurement_kind": "fixed_synthetic_fp32_inference_profile",
        "performed_by": "Codex automation; not user-performed verification",
        "created_at": datetime.now(timezone(timedelta(hours=10))).isoformat(),
        "status": "passed",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        **metadata,
        "model": model_name,
        "model_config": config,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "gpu_total_memory_bytes": torch.cuda.get_device_properties(device).total_memory if device.type == "cuda" else None,
        "runtime": {
            "python": platform.python_version(),
            "pytorch": str(torch.__version__),
            "numpy": np.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "platform": platform.platform(),
            "cpu_threads": torch.get_num_threads(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
        "settings": {
            "seed": SEED,
            "input_generator": "Dedicated seeded CPU torch.Generator; setup and transfer precede warmup and timing",
            "combination_order": "batch sizes 1 then 4; each with 256x128 then 256x144",
            "precision": "FP32 model/input/output dtypes; no AMP or autocast; backend TF32 settings recorded above",
            "mode": "eval and inference_mode",
            "warmup_iterations_per_combination": WARMUP_ITERATIONS,
            "timed_iterations_per_combination": TIMED_ITERATIONS,
            "percentile_method": "numpy.percentile(method='linear')",
        },
        "scope": {
            "synthetic_input_only": True,
            "mri_pixels_read": False,
            "test_split_read": False,
            "optimizer_resident": False,
            "timing_includes": "Model forward, Python/host launch and post-forward synchronization overhead; CUDA synchronized before and after each iteration",
            "timing_excludes": "Checkpoint loading, model/input construction, transfers, warmup, metrics, output deletion, plotting and report I/O",
            "memory_includes": "Resident model/input, transient outputs and forward temporaries during the timed stage; reserved memory includes allocator cache retained from this combination's setup/warmup",
            "memory_excludes": "Checkpoint restoration and an optimizer; not total GPU VRAM or training memory",
            "cache_policy": "Before each combination, free unused CUDA cache after synchronization, then construct input and warm up; do not clear warmup cache inside the timed stage",
            "cpu_note": "CPU timing is supported; CUDA memory values are unavailable on CPU and must not be interpreted as zero GPU usage.",
        },
        "source_sha256": {
            name: sha256_file(project / name)
            for name in ("modules.py", "resource_monitor.py", "profile_inference.py")
        },
        "source_hash_basis": "SHA256 of current working-tree file bytes; paths and line endings may differ in other environments",
        "results": results,
        "claim_boundary": "Synthetic forward latency and scoped memory only; no MRI accuracy, test evaluation or cross-environment timing equivalence claim.",
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite previous profile evidence: {args.output}")
    report = profile_checkpoint(args.checkpoint, torch.device(args.device))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also rejects a competing writer appearing after the
    # initial existence check; previous evidence is never overwritten.
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Synthetic inference profile saved: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
