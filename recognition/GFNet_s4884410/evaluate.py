"""
Final evaluation of the trained models: run once, after all training and model choice.

For each model in RUNS, loads its best checkpoint and writes to OUT_DIR/<model>/:
    val_slice.csv, test_slice.csv       one row per slice: pid, slice_pos, label,
                                        logit_ad_minus_nc, prob_ad
    val_patient.csv, test_patient.csv   one row per patient: label, n_slices, prob_ad,
                                        slice_prob_std
    metrics.json                        loss and slice/patient metrics for val and test
    profile.json                        parameters, MACs, peak inference memory, latency

Validation predictions are for choosing thresholds (e.g. the triage rule); test
predictions are for reporting only.

Usage:
    python evaluate.py
"""

import json
import time
from pathlib import Path

import numpy as np
import torch

from dataset import get_datasets, get_loaders
from modules import BaselineCNN, GFNet, count_parameters
from train import DATA_ROOT
from utils import aggregate_patients, classification_metrics, predict, save_rows_csv

# Paths
PROJECT_DIR = Path(__file__).resolve().parent
RUNS = {  # final checkpoint folder for each model
    "cnn": PROJECT_DIR / "runs" / "cnn_20261026_161619",
    "gfnet": PROJECT_DIR / "runs" / "gfnet_20261027_144653",
}
OUT_DIR = PROJECT_DIR / "results" / "eval"

# Settings
BATCH_SIZE = 64
NUM_WORKERS = 1
LATENCY_WARMUP = 20  # untimed runs before timing
LATENCY_RUNS = 100  # timed runs; the median is reported
SLICES_PER_PATIENT = 20


def load_model(run_dir, device):
    """Rebuild a model from run_dir/best.pt, using the settings saved with it.

    Returns:
        (model in eval mode on device, checkpoint dict)
    """
    ckpt = torch.load(Path(run_dir) / "best.pt", map_location="cpu", weights_only=False)
    if ckpt["model"] == "cnn":
        model = BaselineCNN()
    elif ckpt["model"] == "gfnet":
        model = GFNet(**ckpt["config"]["model_args"])
    else:
        raise ValueError(f"Unknown model in checkpoint: {ckpt['model']}")
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt


def evaluate_split(model, loader, device, out_dir, split):
    """Save slice and patient predictions for one split, and return its metrics."""
    loss, rows = predict(model, loader, device)
    patients = aggregate_patients(rows)
    save_rows_csv(rows, out_dir / f"{split}_slice.csv")
    save_rows_csv(patients, out_dir / f"{split}_patient.csv")
    return {
        "loss": loss,
        "n_slices": len(rows),
        "n_patients": len(patients),
        "slice": classification_metrics(
            [r["label"] for r in rows], [r["prob_ad"] for r in rows]
        ),
        "patient": classification_metrics(
            [r["label"] for r in patients], [r["prob_ad"] for r in patients]
        ),
    }


@torch.no_grad()
def latency_ms(model, batch, device, amp):
    """Median forward-pass time in ms for one batch, after LATENCY_WARMUP untimed runs.

    The GPU is synchronised before and after each run so the timing covers the
    full computation.
    """
    use_amp = amp and device.type == "cuda"
    times = []
    for i in range(LATENCY_WARMUP + LATENCY_RUNS):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=use_amp
        ):
            model(batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        if i >= LATENCY_WARMUP:
            times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


@torch.no_grad()
def peak_memory_gb(model, batch, device):
    """Peak GPU memory (GB) allocated by PyTorch for one AMP forward pass, including weights.

    Returns NaN on CPU.
    """
    if device.type != "cuda":
        return float("nan")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        model(batch)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1e9


def count_flops(model):
    """Multiply-accumulates (MACs) for one 224 x 224 slice, counted with thop.

    FFT operations are not counted, so GFNet's figure is a slight undercount.
    Returns None if thop is not installed.
    """
    try:
        from thop import profile
    except ImportError:
        return None
    import copy

    m = copy.deepcopy(model).cpu().float().eval()
    macs, _ = profile(m, inputs=(torch.zeros(1, 1, 224, 224),), verbose=False)
    return float(macs)


def profile_model(model, device):
    """Parameters, MACs, peak inference memory and latency, measured on synthetic input."""
    model.eval()
    one = torch.zeros(1, 1, 224, 224, device=device)
    patient = torch.zeros(SLICES_PER_PATIENT, 1, 224, 224, device=device)
    batch = torch.zeros(BATCH_SIZE, 1, 224, 224, device=device)
    return {
        "parameters": count_parameters(model),
        "macs_per_slice": count_flops(model),
        "peak_inference_vram_gb_batch64": peak_memory_gb(model, batch, device),
        "peak_inference_vram_gb_patient": peak_memory_gb(model, patient, device),
        "latency_ms_per_slice_amp": latency_ms(model, one, device, amp=True),
        "latency_ms_per_patient_amp": latency_ms(model, patient, device, amp=True),
        "latency_ms_per_slice_fp32": latency_ms(model, one, device, amp=False),
        "latency_ms_per_patient_fp32": latency_ms(model, patient, device, amp=False),
        "latency_note": f"median of {LATENCY_RUNS} runs after {LATENCY_WARMUP} warm-up, "
        "synthetic input, data loading excluded",
    }


def main():
    """Evaluate every model in RUNS on validation and test, and profile it."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "none (CPU)"
    print(f"Device: {gpu}")

    datasets, _ = get_datasets(DATA_ROOT, num_workers=NUM_WORKERS)  # no augmentation
    loaders = get_loaders(datasets, BATCH_SIZE, NUM_WORKERS)

    for name, run_dir in RUNS.items():
        out_dir = OUT_DIR / name
        out_dir.mkdir(parents=True, exist_ok=True)
        model, ckpt = load_model(run_dir, device)
        print(
            f"\n== {name}: {Path(run_dir).name}, epoch {ckpt['epoch']}, "
            f"{count_parameters(model):,} parameters =="
        )

        t0 = time.perf_counter()
        metrics = {"run": Path(run_dir).name, "best_epoch": ckpt["epoch"], "gpu": gpu}
        for split in ("val", "test"):
            metrics[split] = evaluate_split(
                model, loaders[split], device, out_dir, split
            )
            s, p = metrics[split]["slice"], metrics[split]["patient"]
            print(
                f"{split:4} loss {metrics[split]['loss']:.4f} | slice acc {s['acc']:.3f} "
                f"auroc {s['auroc']:.3f} | patient acc {p['acc']:.3f} auroc {p['auroc']:.3f} "
                f"sens {p['recall_ad']:.3f} spec {p['recall_nc']:.3f}"
            )
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

        prof = {"gpu": gpu, **profile_model(model, device)}
        (out_dir / "profile.json").write_text(json.dumps(prof, indent=2))
        print(
            f"latency {prof['latency_ms_per_slice_amp']:.2f} ms/slice, "
            f"{prof['latency_ms_per_patient_amp']:.2f} ms/patient (AMP) | "
            f"peak inference {prof['peak_inference_vram_gb_batch64']:.2f} GB (batch 64) | "
            f"{time.perf_counter() - t0:.0f}s"
        )

    print(f"\nOutputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
