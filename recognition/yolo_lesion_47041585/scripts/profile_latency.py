# Ultralytics, OpenCV and torch calls return partly unknown types, so silence only those
# rules here; every other strict check still applies.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false
"""Measure end-to-end inference latency for the latency criterion (batch size 1).

TIMING SCOPE (declared before measuring): each timed interval starts from a decoded image
already in memory and ends with the final detections and the routing decision. It includes
resizing and padding, the copy to the device, the forward pass, NMS, moving the boxes back to
the CPU, conversion to Detections and the routing rule. It excludes reading the file from
disk and decoding the JPEG, which are timed separately and reported as an extra scope, and
it excludes drawing and saving. The device is synchronised immediately before and after
every interval, because GPU work is asynchronous.

It uses the same inference path as `predict` (the same model settings from the run's saved
config and the same multi-label predictor). Images come from the validation split, so no
test data is touched, and it needs no labels.

    uv run python scripts/profile_latency.py --device 0 --output artifacts/latency.json
"""

import argparse
import json
import os
import platform
import statistics
import subprocess
import time
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from ultralytics.engine.results import Results

from yolo_lesion_47041585.helpers.policy import PolicySpec, route_case
from yolo_lesion_47041585.modules import LesionPredictor
from yolo_lesion_47041585.predict import detections_from_result, load_run


def synchronise(device: str | None) -> None:
    """Wait for queued device work, so a host timer sees the real end of each interval."""
    if torch.cuda.is_available() and (device is None or device not in ("cpu", "mps")):
        torch.cuda.synchronize()
    elif device == "mps" or (device is None and torch.backends.mps.is_available()):
        torch.mps.synchronize()


def nvidia_smi(field: str) -> str | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        )  # fmt: skip
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip().splitlines()[0]


def percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q / 100 * (len(ordered) - 1)))]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/A0-s47041585/weights/best.pt"))
    parser.add_argument("--images", type=Path, default=Path("artifacts/isic2018/yolo/images/val"))
    parser.add_argument("--n-images", type=int, default=100, help="timed inferences, at least 100 for a final measurement")
    parser.add_argument("--warmup", type=int, default=20, help="untimed inferences first, at least 20")
    parser.add_argument("--device", help="Ultralytics device, e.g. cpu, mps or 0")
    parser.add_argument("--tau-m", type=float, default=0.19, help="malignant alert threshold of the routing rule")
    parser.add_argument("--tau-b", type=float, default=0.001, help="benign clearance threshold of the routing rule")
    parser.add_argument("--rule", default="lenient", help="routing rule version")
    parser.add_argument("--output", type=Path, default=Path("artifacts/latency.json"))
    args = parser.parse_args(argv)

    model, settings = load_run(args.checkpoint)
    spec = PolicySpec(args.tau_m, args.tau_b, args.rule)
    paths = sorted(args.images.glob("*.jpg"))
    needed = max(args.n_images, args.warmup)
    if len(paths) < 1:
        raise FileNotFoundError(f"No images in {args.images}")
    chosen = [paths[i % len(paths)] for i in range(needed)]  # cycles if fewer unique images

    # read and decode everything first, timing the extra scope (file read and decode) per image
    decode_seconds: list[float] = []
    decoded: list[NDArray[np.uint8]] = []
    for path in chosen:
        start = time.perf_counter()
        image = cv2.imread(str(path))
        decode_seconds.append(time.perf_counter() - start)
        if image is None:
            raise FileNotFoundError(f"could not read {path}")
        decoded.append(cast(NDArray[np.uint8], image))

    def one(image: NDArray[np.uint8]) -> int:
        results = model.predict(
            image, imgsz=settings.imgsz, conf=settings.conf, iou=settings.iou,
            max_det=settings.max_det, device=args.device, verbose=False,
            predictor=LesionPredictor,
        )  # fmt: skip
        detections = detections_from_result(cast(list[Results], results)[0])
        route_case(detections, spec)
        return len(detections)

    for image in decoded[: args.warmup]:
        one(image)
    synchronise(args.device)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    latencies: list[float] = []
    boxes = 0
    wall_start = time.perf_counter()
    for image in decoded[: args.n_images]:
        synchronise(args.device)
        start = time.perf_counter()
        boxes += one(image)
        synchronise(args.device)
        latencies.append(time.perf_counter() - start)
    wall = time.perf_counter() - wall_start

    affinity = getattr(os, "sched_getaffinity", None)  # Linux only
    mean = statistics.fmean(latencies)
    latency_stats = {
        "mean": mean,
        "median": statistics.median(latencies),
        "p95": percentile(latencies, 95),
        "min": min(latencies),
        "max": max(latencies),
    }
    fps = len(latencies) / sum(latencies)
    with_decode = mean + statistics.fmean(decode_seconds)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    report = {
        "scope": "decoded image in memory -> final detections and routing decision, batch 1, device synchronised, disk read and decode excluded",
        "checkpoint": str(args.checkpoint),
        "images_timed": len(latencies),
        "unique_images": min(len(paths), len(latencies)),
        "warmup_images": args.warmup,
        "latency_seconds": latency_stats,
        "fps_batch_1": fps,
        "fps_including_loop_overhead": len(latencies) / wall,
        "mean_boxes_per_image": boxes / len(latencies),
        "extra_scope_file_read_and_decode_seconds_mean": statistics.fmean(decode_seconds),
        "mean_latency_plus_file_read_and_decode": with_decode,
        "settings": {
            "imgsz": settings.imgsz, "conf": settings.conf, "iou": settings.iou, "max_det": settings.max_det,
            "precision": "fp32", "batch_size": 1, "routing_rule": args.rule,
            "tau_m": args.tau_m, "tau_b": args.tau_b,
        },  # fmt: skip
        "parameters": sum(p.numel() for p in cast(torch.nn.Module, model.model).parameters()),
        "hardware": {
            "device": args.device or "auto",
            "gpu": gpu,
            "driver": nvidia_smi("driver_version"),
            "cpu_count": os.cpu_count(),
            "cpus_allocated": len(affinity(0)) if affinity else None,
            "platform": platform.platform(),
        },
        "software": {
            "python": platform.python_version(), "torch": version("torch"), "ultralytics": version("ultralytics"),
            "opencv": cv2.__version__, "cuda_runtime": torch.version.cuda,
        },  # fmt: skip
        "inference_peak_vram_mb": {
            "allocated": torch.cuda.max_memory_allocated() >> 20,
            "reserved": torch.cuda.max_memory_reserved() >> 20,
        } if torch.cuda.is_available() else None,
    }  # fmt: skip
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))

    print(f"{len(latencies)} images, batch 1, fp32, device {args.device or 'auto'} ({gpu or 'no CUDA GPU'})")
    print(f"latency: mean {mean*1000:.1f} ms, median {latency_stats['median']*1000:.1f} ms, "
          f"p95 {latency_stats['p95']*1000:.1f} ms  ->  {fps:.1f} images/s")
    print(f"with file read and decode added: mean {with_decode*1000:.1f} ms")
    print(f"target: mean at most 1000 ms -> {'met' if mean <= 1.0 else 'NOT met'} on this hardware")
    print(f"written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
