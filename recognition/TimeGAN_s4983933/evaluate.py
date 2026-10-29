"""
COMP3710 TimeGAN - Initial Evaluation

Generate synthetic order-book sequences and compare their normalized
feature distributions with real data. Supports V1/V2 preprocessing and
original/V3.2 model architectures.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from dataset import prepare_dataset
from dataset_v2 import prepare_dataset_v2
from modules import TimeGAN
from modules_v3_2 import TimeGANV32


def load_model(checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model_version = checkpoint.get("model_version", "original")
    if model_version == "v3.2":
        model_class = TimeGANV32
    elif model_version == "original":
        model_class = TimeGAN
    else:
        raise ValueError(f"Unknown model version: {model_version}")

    model = model_class(
        feature_dim=checkpoint["feature_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        noise_dim=checkpoint["noise_dim"],
        num_layers=checkpoint["num_layers"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


@torch.no_grad()
def generate_samples(model, count, seq_len, device, batch_size=128):
    batches = []

    for start in range(0, count, batch_size):
        current_size = min(batch_size, count - start)
        noise = torch.rand(
            current_size,
            seq_len,
            model.noise_dim,
            device=device,
        )
        generated = model.generate(noise)
        batches.append(generated.cpu().numpy())

    return np.concatenate(batches, axis=0)


def histogram_kl(real, generated, bins=50, eps=1e-8):
    """Estimate feature-wise KL divergence using histograms."""
    real_flat = real.reshape(-1, real.shape[-1])
    fake_flat = generated.reshape(-1, generated.shape[-1])
    values = []

    for feature in range(real_flat.shape[1]):
        a = real_flat[:, feature]
        b = fake_flat[:, feature]
        lower = min(a.min(), b.min())
        upper = max(a.max(), b.max())

        if upper <= lower:
            values.append(0.0)
            continue

        edges = np.linspace(lower, upper, bins + 1)
        p, _ = np.histogram(a, bins=edges)
        q, _ = np.histogram(b, bins=edges)
        p = p.astype(np.float64) + eps
        q = q.astype(np.float64) + eps
        p /= p.sum()
        q /= q.sum()
        values.append(np.sum(p * np.log(p / q)))

    return np.array(values)


def evaluate(real, generated):
    if real.shape != generated.shape:
        raise ValueError(
            f"Shape mismatch: {real.shape} vs {generated.shape}"
        )

    real_mean = real.mean(axis=(0, 1))
    fake_mean = generated.mean(axis=(0, 1))
    real_std = real.std(axis=(0, 1))
    fake_std = generated.std(axis=(0, 1))

    mean_mae = np.mean(np.abs(real_mean - fake_mean))
    std_mae = np.mean(np.abs(real_std - fake_std))
    kl_values = histogram_kl(real, generated)

    return {
        "mean_mae": float(mean_mae),
        "std_mae": float(std_mae),
        "kl_mean": float(kl_values.mean()),
        "kl_median": float(np.median(kl_values)),
        "kl_max": float(kl_values.max()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    device = torch.device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    dataset_version = checkpoint.get("dataset_version", "v1")
    model_version = checkpoint.get("model_version", "original")

    if dataset_version == "v2":
        dataset = prepare_dataset_v2(args.data_dir)
    elif dataset_version == "v1":
        dataset = prepare_dataset(args.data_dir)
    else:
        raise ValueError(f"Unknown dataset version: {dataset_version}")

    real = dataset["test"]
    if real.shape[-1] != checkpoint["feature_dim"]:
        raise ValueError(
            f"Dataset feature count {real.shape[-1]} does not match "
            f"model feature count {checkpoint['feature_dim']}"
        )

    generated = generate_samples(
        model,
        count=len(real),
        seq_len=real.shape[1],
        device=device,
        batch_size=args.batch_size,
    )
    results = evaluate(real, generated)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Include architecture to avoid overwriting V3.1 with V3.2 samples.
    output_file = output_dir / f"generated_{dataset_version}_{model_version.replace('.', '_')}.npy"
    np.save(output_file, generated)

    print("Checkpoint stage:", checkpoint["stage"])
    print("Dataset version:", dataset_version)
    print("Model version:", model_version)
    print("Real shape:", real.shape)
    print("Generated shape:", generated.shape)

    print("\n=== Initial Evaluation Results ===")
    for key, value in results.items():
        print(f"{key}: {value:.6f}")
    print("\nGenerated sequences saved to:", output_file)


if __name__ == "__main__":
    main()
