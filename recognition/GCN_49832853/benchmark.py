
import os
from pathlib import Path
import time

import numpy as np
import psutil
import torch
import os
import psutil

from dataset import load_facebook, split_nodes
from modules import GCN, MLP

BASE_DIR = Path(__file__).resolve().parent


def calculate_ece(probabilities, labels, n_bins=10):
    confidence, predictions = probabilities.max(dim=1)
    correct = predictions.eq(labels).float()

    boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        lower = boundaries[i]
        upper = boundaries[i + 1]

        if i == n_bins - 1:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)

        if mask.any():
            bin_accuracy = correct[mask].mean()
            bin_confidence = confidence[mask].mean()
            bin_weight = mask.float().mean()

            ece += (
                bin_weight
                * torch.abs(bin_accuracy - bin_confidence)
            ).item()

    return ece


@torch.inference_mode()
def profile_model(name, model, data):
    checkpoint = BASE_DIR / "checkpoints" / f"{name.lower()}_best.pt"

    model.load_state_dict(
        torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=True
        )
    )
    model.eval()

    # Probability calibration on validation nodes
    logits = model(data.x, data.edge_index)
    probabilities = torch.softmax(
        logits[data.val_mask], dim=1
    )
    labels = data.y[data.val_mask]

    ece = calculate_ece(probabilities, labels)

    confidence, predictions = probabilities.max(dim=1)
    correct = predictions == labels

    # Warm up before CPU timing
    for _ in range(5):
        model(data.x, data.edge_index)

    # Benchmark one full-graph forward pass
    times = []
    for _ in range(30):
        start = time.perf_counter()
        model(data.x, data.edge_index)
        times.append(
            (time.perf_counter() - start) * 1000
        )

    params = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )

    print(f"\n===== {name} =====")
    print(f"Parameters: {params}")
    print(f"ECE (10 bins): {ece:.4f}")
    print(f"Mean confidence: {confidence.mean().item():.4f}")

    if (~correct).any():
        print(
            "Mean confidence on errors:",
            f"{confidence[~correct].mean().item():.4f}"
        )

    print(
        f"Median CPU inference: {np.median(times):.2f} ms"
    )
    print(
        f"Mean CPU inference: {np.mean(times):.2f} ms"
    )

    process = psutil.Process(os.getpid())
    memory = process.memory_info()

    print(
        f"Process peak RAM: "
        f"{memory.peak_wset / (1024 ** 2):.2f} MB"
    )


def main():
    data = split_nodes(load_facebook(), seed=42)

    for name, model_class in [("MLP", MLP), ("GCN", GCN)]:
        model = model_class(
            num_features=data.num_node_features,
            num_classes=int(data.y.max().item()) + 1
        )

        profile_model(name, model, data)


if __name__ == "__main__":
    main()
