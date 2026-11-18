"""Run a fixed model over a loader and score slices and complete scans."""

import math
import time

import torch

from evaluation.metrics import aggregate_scans, binary_metrics
from utils.runtime import sync_device
from utils.training_controls import ExecutionControls, autocast_context


@torch.inference_mode()
def predict_slices(model: torch.nn.Module, loader, device: torch.device,
                   controls: ExecutionControls) -> tuple[list[dict], float]:
    """Return one row per slice with its AD logit and probability, plus forward time.

    Forward time excludes data loading and host-to-device transfer.
    """
    model.eval()
    rows, forward_seconds = [], 0.0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=device.type == "cuda")
        sync_device(device)
        started = time.perf_counter()
        with autocast_context(controls, device):
            logits = model(images).float()
        sync_device(device)
        forward_seconds += time.perf_counter() - started
        if not torch.isfinite(logits).all():
            raise ValueError("Model produced non-finite logits; evaluation stopped.")
        for index, logit in enumerate(logits.cpu().tolist()):
            rows.append({
                "patient_id": batch["patient_id"][index],
                "image_id": batch["image_id"][index],
                "slice_index": int(batch["slice_index"][index]),
                "relative_path": batch["relative_path"][index],
                "label": int(batch["label"][index]),
                "logit": logit,
                "probability": 1 / (1 + math.exp(-logit)) if logit >= 0 else math.exp(logit) / (1 + math.exp(logit)),
            })
    return rows, forward_seconds


def score_slices(slices: list[dict], expected_slices: int) -> tuple[dict, list[dict]]:
    """Slice and scan metrics; a scan's probability is the mean of its slice probabilities."""
    scans = aggregate_scans(slices, expected_slices=expected_slices)
    scores = {unit: binary_metrics([r["label"] for r in rows], [r["probability"] for r in rows])
              for unit, rows in (("slice", slices), ("scan", scans))}
    scores["n_patients"] = len({row["patient_id"] for row in scans})
    return scores, scans


def evaluate(model: torch.nn.Module, loader, device: torch.device, expected_slices: int,
             controls: ExecutionControls) -> tuple[dict, list[dict], list[dict]]:
    """Predict every slice of a role, then return (scores, slice rows, scan rows)."""
    slices, forward_seconds = predict_slices(model, loader, device, controls)
    scores, scans = score_slices(slices, expected_slices)
    scores["forward_seconds"] = forward_seconds
    scores["mean_forward_ms_per_slice"] = forward_seconds * 1000 / len(slices)
    return scores, slices, scans
