"""Train comparable MLP/GCN models with separate selection and calibration data.

Default development runs never evaluate test labels. Use --evaluate-test only
after locking the experiment procedure. --smoke performs at most five epochs.
"""

import argparse
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import torch
from torch.nn import functional as F

from dataset import DEFAULT_PATH, DATA_SHA256, PROJECT_DIR, acquire, make_pyg_graph, sha256
from evaluation import (
    choose_threshold, classification_metrics, fit_temperature, plot_history,
    selective_metrics, split_validation_roles,
)
from modules import make_model, normalized_adjacency


@torch.no_grad()
def forward_latency(model, x, adjacency, repetitions=10):
    model.eval()
    for _ in range(3):
        model(x, adjacency)
    if x.is_cuda:
        torch.cuda.synchronize()
    durations = []
    for _ in range(repetitions):
        start = time.perf_counter()
        model(x, adjacency)
        if x.is_cuda:
            torch.cuda.synchronize()
        durations.append(1000 * (time.perf_counter() - start))
    return {"scope": "one forward pass for the entire graph; excludes data loading",
            "repetitions": repetitions, "median_ms": float(np.median(durations)),
            "min_ms": float(min(durations)), "max_ms": float(max(durations))}


def run_one(name, seed, graph, args, selection_mask, calibration_mask):
    torch.manual_seed(seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    x = graph.x.to(device)
    # Only training and validation targets are moved into the development run.
    train_ids = torch.where(graph.train_mask)[0].to(device)
    selection_ids = torch.where(selection_mask)[0].to(device)
    calibration_ids = torch.where(calibration_mask)[0].to(device)
    train_y = graph.y[graph.train_mask].to(device)
    selection_y = graph.y[selection_mask].to(device)
    calibration_y = graph.y[calibration_mask].to(device)
    adjacency = normalized_adjacency(graph.edge_index.to(device), len(x)) if name == "gcn" else None
    model_config = {"inputs": x.shape[1], "hidden": args.hidden,
                    "classes": 4, "dropout": args.dropout}
    model = make_model(name, **model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_f1, best_epoch, best_state = -1., 0, None
    history = []
    start = time.perf_counter()
    max_epochs = min(args.epochs, 5) if args.smoke else args.epochs
    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits, _ = model(x, adjacency)
        loss = F.cross_entropy(logits[train_ids], train_y)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss")
        loss.backward()
        if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError("Nonfinite training gradients")
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_logits, _ = model(x, adjacency)
            selected = classification_metrics(validation_logits[selection_ids], selection_y)
        history.append({"epoch": epoch, "train_loss": float(loss.detach()),
                        "selection_loss": selected["negative_log_likelihood"],
                        "selection_macro_f1": selected["macro_f1"]})
        if selected["macro_f1"] > best_f1:
            best_f1, best_epoch = selected["macro_f1"], epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0 or epoch == max_epochs:
            print(f"{name} seed={seed} epoch={epoch} train_loss={float(loss.detach()):.4f} "
                  f"selection_macro_f1={selected['macro_f1']:.4f}", flush=True)
        if epoch - best_epoch >= args.patience:
            break
    if device.type == "cuda":
        torch.cuda.synchronize()
    training_seconds = time.perf_counter() - start
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits, _ = model(x, adjacency)
    temperature = fit_temperature(logits[calibration_ids], calibration_y)
    threshold = choose_threshold(logits[calibration_ids], calibration_y,
                                 temperature, args.target_accuracy)
    phase = "smoke" if args.smoke else ("final" if args.evaluate_test else "development")
    destination = args.output / phase / f"{name}_seed{seed}"
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model": name, "model_config": model_config, "model_state": best_state,
        "dataset_sha256": DATA_SHA256, "split_seed": args.split_seed,
        "validation_role_seed": args.split_seed + 1, "initialization_seed": seed,
        "temperature": temperature, "confidence_threshold": threshold,
        "phase": phase, "best_epoch": best_epoch,
        "feature_mean": graph.feature_mean, "feature_scale": graph.feature_scale,
    }
    torch.save(checkpoint, destination / "model.pt")
    versions = {package: importlib.metadata.version(package) for package in
                ("torch", "torch-geometric", "numpy", "scikit-learn", "matplotlib")}
    summary = {
        "phase": phase, "model": name, "initialization_seed": seed,
        "split_seed": args.split_seed, "dataset_sha256": DATA_SHA256,
        "epochs_run": len(history), "best_epoch": best_epoch,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "training_seconds": training_seconds, "device": str(device),
        "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else (platform.processor() or platform.machine()),
        "versions": versions, "python": platform.python_version(),
        "cuda_runtime": torch.version.cuda, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "selection_nodes": int(selection_mask.sum()),
        "calibration_nodes": int(calibration_mask.sum()),
        "best_selection_macro_f1": best_f1,
        "calibration_fit_raw": classification_metrics(logits[calibration_ids], calibration_y),
        "calibration_fit_scaled": classification_metrics(logits[calibration_ids], calibration_y, temperature),
        "temperature": temperature, "target_calibration_accepted_accuracy": args.target_accuracy,
        "calibration_fitted_policy": selective_metrics(logits[calibration_ids], calibration_y, temperature, threshold),
        "test_evaluated": bool(args.evaluate_test),
        "full_graph_forward_latency": forward_latency(model, x, adjacency),
        "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else None,
        "peak_gpu_reserved_mib": torch.cuda.max_memory_reserved(device) / (1024 ** 2) if device.type == "cuda" else None,
        "caveat": "Calibration statistics describe fitting data. Smoke runs do not establish final quality.",
    }
    if args.evaluate_test:
        test_ids = torch.where(graph.test_mask)[0].to(device)
        test_y = graph.y[graph.test_mask].to(device)
        summary["test_raw"] = classification_metrics(logits[test_ids], test_y)
        summary["test_scaled"] = classification_metrics(logits[test_ids], test_y, temperature)
        summary["test_operating_policy"] = selective_metrics(logits[test_ids], test_y, temperature, threshold)
    (destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (destination / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    plot_history(history, destination / "training.png")
    print(f"Saved {phase} run to {destination}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=["mlp", "gcn"], default=["mlp", "gcn"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--data", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--target-accuracy", type=float, default=0.9)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "runs")
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1 or args.hidden < 1 or args.cpu_threads < 1:
        parser.error("Epochs, patience, hidden width and CPU threads must be positive")
    if not 0 <= args.dropout < 1 or args.lr <= 0 or args.weight_decay < 0:
        parser.error("Invalid optimizer/dropout settings")
    if not 0 < args.target_accuracy <= 1:
        parser.error("Target accuracy must be in (0, 1]")
    if args.smoke and args.evaluate_test:
        parser.error("Smoke checks must leave the test set untouched")
    torch.set_num_threads(args.cpu_threads)
    if args.download:
        acquire(args.data)
    if sha256(args.data) != DATA_SHA256:
        raise ValueError("Dataset differs from the audited course archive")
    graph = make_pyg_graph(args.data, args.split_seed)
    selection_mask, calibration_mask = split_validation_roles(
        graph.y, graph.validation_mask, args.split_seed + 1,
    )
    for seed in args.seeds:
        for name in args.models:
            run_one(name, seed, graph, args, selection_mask, calibration_mask)


if __name__ == "__main__":
    main()
