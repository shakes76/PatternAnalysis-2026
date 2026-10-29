
"""
COMP3710 Project 7 - TimeGAN Training
Stage 1: Embedder and Recovery pretraining
Stage 2: Supervisor pretraining
"""

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dataset import prepare_dataset
from dataset_v2 import prepare_dataset_v2
from dataset_v3 import prepare_dataset_v3
from dataset_v4 import prepare_dataset_v4
from modules import TimeGAN
from modules_v3_2 import TimeGANV32


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_loader(data, batch_size, shuffle=False):
    tensor = torch.from_numpy(data).float()
    return DataLoader(
        TensorDataset(tensor),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def reconstruction_loss(model, batch, criterion):
    reconstructed = model.reconstruct(batch)
    return criterion(reconstructed, batch)


@torch.no_grad()
def validate_embedding(model, loader, device, criterion):
    model.eval()
    total = 0.0
    count = 0

    for (batch,) in loader:
        batch = batch.to(device)
        loss = reconstruction_loss(model, batch, criterion)
        total += loss.item() * len(batch)
        count += len(batch)

    return total / count


def train_embedding(model, train_loader, val_loader,
                    device, epochs=10, lr=0.001):

    optimizer = torch.optim.Adam(
        list(model.embedder.parameters())
        + list(model.recovery.parameters()),
        lr=lr,
    )
    criterion = nn.MSELoss()

    history = []

    print("\n=== Stage 1: Embedding Pretraining ===",
          flush=True)

    for epoch in range(epochs):
        model.train()
        total = 0.0

        for (batch,) in train_loader:
            batch = batch.to(device)

            loss = reconstruction_loss(
                model, batch, criterion
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += loss.item() * len(batch)

        train_loss = total / len(train_loader.dataset)
        val_loss = validate_embedding(
            model, val_loader, device, criterion
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train MSE: {train_loss:.6f} | "
            f"Val MSE: {val_loss:.6f}",
            flush=True,
        )

    return history, optimizer


def supervised_loss(model, batch, criterion):
    # Freeze Embedder outputs for Supervisor pretraining.
    with torch.no_grad():
        h = model.embedder(batch)

    h_pred = model.supervisor(h)

    return criterion(
        h_pred[:, :-1, :],
        h[:, 1:, :],
    )


@torch.no_grad()
def validate_supervisor(model, loader, device, criterion):
    model.eval()
    total = 0.0
    count = 0

    for (batch,) in loader:
        batch = batch.to(device)
        loss = supervised_loss(model, batch, criterion)
        total += loss.item() * len(batch)
        count += len(batch)

    return total / count


def train_supervisor(model, train_loader, val_loader,
                     device, epochs=10, lr=0.001):

    optimizer = torch.optim.Adam(
        model.supervisor.parameters(),
        lr=lr,
    )
    criterion = nn.MSELoss()

    history = []

    print("\n=== Stage 2: Supervisor Pretraining ===",
          flush=True)

    for epoch in range(epochs):
        model.train()
        total = 0.0

        for (batch,) in train_loader:
            batch = batch.to(device)

            loss = supervised_loss(
                model, batch, criterion
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += loss.item() * len(batch)

        train_loss = total / len(train_loader.dataset)
        val_loss = validate_supervisor(
            model, val_loader, device, criterion
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train MSE: {train_loss:.6f} | "
            f"Val MSE: {val_loss:.6f}",
            flush=True,
        )

    return history, optimizer


def moment_loss(real, fake):
    """Match feature-wise means and standard deviations."""
    real_mean = real.mean(dim=(0, 1))
    fake_mean = fake.mean(dim=(0, 1))

    real_std = real.std(dim=(0, 1), unbiased=False)
    fake_std = fake.std(dim=(0, 1), unbiased=False)

    mean_error = torch.mean(torch.abs(real_mean - fake_mean))
    std_error = torch.mean(torch.abs(real_std - fake_std))

    return mean_error + std_error


def joint_losses(model, real, noise):
    """
    Compute the main TimeGAN joint-training losses.

    This function calculates losses but does not update weights.
    """
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()

    # Real-data latent representations
    h = model.embedder(real)
    x_tilde = model.recovery(h)
    h_supervised = model.supervisor(h)

    # Synthetic latent representations
    e_hat = model.generator(noise)
    h_hat = model.supervisor(e_hat)
    x_hat = model.recovery(h_hat)

    # Discriminator predictions (logits)
    y_real = model.discriminator(h)
    y_fake = model.discriminator(h_hat)
    y_fake_e = model.discriminator(e_hat)

    # 1. Reconstruction loss
    loss_recon = mse(real, x_tilde)

    # 2. Supervised temporal loss
    loss_sup = mse(
        h_supervised[:, :-1, :],
        h[:, 1:, :]
    )

    # 3. Discriminator loss
    loss_d_real = bce(y_real, torch.ones_like(y_real))
    loss_d_fake = bce(y_fake, torch.zeros_like(y_fake))
    loss_d_fake_e = bce(
        y_fake_e, torch.zeros_like(y_fake_e)
    )

    loss_d = loss_d_real + loss_d_fake + loss_d_fake_e

    # 4. Generator adversarial loss
    loss_g_adv = (
        bce(y_fake, torch.ones_like(y_fake))
        + bce(y_fake_e, torch.ones_like(y_fake_e))
    )

    # 5. Moment matching
    loss_moment = moment_loss(real, x_hat)

    return {
        "reconstruction": loss_recon,
        "supervised": loss_sup,
        "discriminator": loss_d,
        "generator_adversarial": loss_g_adv,
        "moment": loss_moment,
    }



def set_requires_grad(network, enabled):
    """Control which network parameters accumulate gradients."""
    for parameter in network.parameters():
        parameter.requires_grad_(enabled)


def joint_train_step(model, real, optimizers, generator_updates=1):
    """One joint batch: G/S repeated, then E/R once and D once."""
    if generator_updates < 1:
        raise ValueError("generator_updates must be >= 1")
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()
    opt_g, opt_er, opt_d = optimizers
    batch_size, seq_len, _ = real.shape

    def sample_noise():
        return torch.rand(
            batch_size, seq_len, model.noise_dim,
            device=real.device,
        )

    # Generator/Supervisor steps: fresh forward and backward on every update.
    set_requires_grad(model.discriminator, False)
    set_requires_grad(model.embedder, False)
    set_requires_grad(model.recovery, False)
    set_requires_grad(model.generator, True)
    set_requires_grad(model.supervisor, True)

    with torch.no_grad():
        h_real = model.embedder(real)

    g_total = adv_total = sup_total = moment_total = 0.0
    for _ in range(generator_updates):
        z = sample_noise()
        e_fake = model.generator(z)
        h_fake = model.supervisor(e_fake)
        x_fake = model.recovery(h_fake)
        h_pred = model.supervisor(h_real)

        logits_h = model.discriminator(h_fake)
        logits_e = model.discriminator(e_fake)
        adv = (
            bce(logits_h, torch.ones_like(logits_h))
            + bce(logits_e, torch.ones_like(logits_e))
        )
        sup = mse(h_pred[:, :-1, :], h_real[:, 1:, :])
        moments = moment_loss(real, x_fake)
        loss_g = adv + 100 * torch.sqrt(sup + 1e-8) + 100 * moments

        opt_g.zero_grad(set_to_none=True)
        loss_g.backward()
        opt_g.step()
        opt_g.zero_grad(set_to_none=True)

        g_total += loss_g.item()
        adv_total += adv.item()
        sup_total += sup.item()
        moment_total += moments.item()

    # Embedder/Recovery step: Supervisor is fixed but transmits gradients.
    set_requires_grad(model.generator, False)
    set_requires_grad(model.supervisor, False)
    set_requires_grad(model.embedder, True)
    set_requires_grad(model.recovery, True)

    h_real = model.embedder(real)
    x_recon = model.recovery(h_real)
    h_pred = model.supervisor(h_real)
    recon = mse(real, x_recon)
    sup_er = mse(h_pred[:, :-1, :], h_real[:, 1:, :])
    loss_er = 10 * torch.sqrt(recon + 1e-8) + 0.1 * sup_er

    opt_er.zero_grad(set_to_none=True)
    loss_er.backward()
    opt_er.step()
    opt_er.zero_grad(set_to_none=True)

    # Discriminator step: no gradient should reach the other networks.
    set_requires_grad(model.embedder, False)
    set_requires_grad(model.recovery, False)
    set_requires_grad(model.discriminator, True)
    with torch.no_grad():
        h_real = model.embedder(real)
        e_fake = model.generator(sample_noise())
        h_fake = model.supervisor(e_fake)

    real_logits = model.discriminator(h_real)
    fake_logits = model.discriminator(h_fake)
    fake_e_logits = model.discriminator(e_fake)
    loss_d = (
        bce(real_logits, torch.ones_like(real_logits))
        + bce(fake_logits, torch.zeros_like(fake_logits))
        + bce(fake_e_logits, torch.zeros_like(fake_e_logits))
    )
    opt_d.zero_grad(set_to_none=True)
    loss_d.backward()
    opt_d.step()
    opt_d.zero_grad(set_to_none=True)

    for network in (
        model.generator, model.supervisor, model.embedder,
        model.recovery, model.discriminator,
    ):
        set_requires_grad(network, True)

    return {
        "generator": g_total / generator_updates,
        "generator_adversarial": adv_total / generator_updates,
        "supervised": sup_total / generator_updates,
        "moment": moment_total / generator_updates,
        "embedder": loss_er.item(),
        "discriminator": loss_d.item(),
    }


@torch.no_grad()
def measure_diversity(model, device, seq_len, noise_dim, samples=64):
    """Compare independent generated samples using a fixed local RNG seed."""
    was_training = model.training
    model.eval()
    try:
        # fork_rng avoids affecting the global RNG stream used by training.
        with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
            torch.manual_seed(12345)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(12345)
            z1 = torch.rand(samples, seq_len, noise_dim, device=device)
            z2 = torch.rand(samples, seq_len, noise_dim, device=device)
            x1 = model.generate(z1)
            x2 = model.generate(z2)
        return {
            "output_difference": float((x1 - x2).abs().mean().item()),
            "sample_std": float(x1.std(dim=0, unbiased=False).mean().item()),
            "temporal_std": float(x1.std(dim=1, unbiased=False).mean().item()),
        }
    finally:
        model.train(was_training)


def train_joint(model, train_loader, device, epochs=5, lr=0.001,
                generator_updates=1):
    """Run joint adversarial training over multiple epochs."""

    opt_g = torch.optim.Adam(
        list(model.generator.parameters())
        + list(model.supervisor.parameters()),
        lr=lr,
    )

    opt_er = torch.optim.Adam(
        list(model.embedder.parameters())
        + list(model.recovery.parameters()),
        lr=lr,
    )

    opt_d = torch.optim.Adam(
        model.discriminator.parameters(),
        lr=lr,
    )

    history = []
    print("Generator updates per batch:", generator_updates)
    print("Stage 3 learning rate:", lr)

    print("\n=== Stage 3: Joint Adversarial Training ===",
          flush=True)

    for epoch in range(epochs):
        model.train()

        total_g = 0.0
        total_er = 0.0
        total_d = 0.0
        total_adv = 0.0
        total_sup = 0.0
        total_moment = 0.0
        count = 0

        for (batch,) in train_loader:
            batch = batch.to(device)

            losses = joint_train_step(
                model, batch, (opt_g, opt_er, opt_d),
                generator_updates=generator_updates,
            )

            n = len(batch)
            total_g += losses["generator"] * n
            total_er += losses["embedder"] * n
            total_d += losses["discriminator"] * n
            total_adv += losses["generator_adversarial"] * n
            total_sup += losses["supervised"] * n
            total_moment += losses["moment"] * n
            count += n

        epoch_losses = {
            "epoch": epoch + 1,
            "generator": total_g / count,
            "embedder": total_er / count,
            "discriminator": total_d / count,
            "generator_adversarial": total_adv / count,
            "supervised": total_sup / count,
            "moment": total_moment / count,
        }
        diversity = measure_diversity(
            model, device,
            seq_len=train_loader.dataset.tensors[0].shape[1],
            noise_dim=model.noise_dim,
        )
        epoch_losses.update(diversity)

        history.append(epoch_losses)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"G: {epoch_losses['generator']:.4f} | "
            f"ER: {epoch_losses['embedder']:.4f} | "
            f"D: {epoch_losses['discriminator']:.4f} | "
            f"Adv: {epoch_losses['generator_adversarial']:.4f} | "
            f"Moment: {epoch_losses['moment']:.4f} | "
            f"Sample Std: {epoch_losses['sample_std']:.8f} | "
            f"Noise Sensitivity: {epoch_losses['output_difference']:.10f}",
            flush=True,
        )

        if not all(
            np.isfinite(epoch_losses[k])
            for k in epoch_losses if k != "epoch"
        ):
            raise RuntimeError(
                "Non-finite loss detected during joint training."
            )

    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument(
        "--dataset-version", choices=["v1", "v2", "v3", "v4"], default="v3"
    )
    parser.add_argument(
        "--model-version", choices=["original", "v3.2"], default="original"
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--supervisor-epochs", type=int, default=10)
    parser.add_argument("--joint-epochs", type=int, default=5)
    parser.add_argument("--joint-lr", type=float, default=0.0001)
    parser.add_argument("--generator-updates", type=int, default=1,
                        help="Generator/Supervisor steps per joint batch")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    if args.generator_updates < 1:
        parser.error("--generator-updates must be at least 1")
    if args.joint_lr <= 0:
        parser.error("--joint-lr must be positive")

    set_seed(42)
    start = time.time()
    device = torch.device(args.device)

    if args.dataset_version == "v4":
        result = prepare_dataset_v4(args.data_dir)
    elif args.dataset_version == "v3":
        result = prepare_dataset_v3(args.data_dir)
    elif args.dataset_version == "v2":
        result = prepare_dataset_v2(args.data_dir)
    else:
        result = prepare_dataset(args.data_dir)

    print("Dataset version:", args.dataset_version)

    # Keep the fitted return transformation parameters with checkpoints.
    # Evaluation must use the same train-only values, never refit on test data.
    normalizer_metadata = {}
    if args.dataset_version == "v4":
        n = result["normalizer"]
        normalizer_metadata = {
            "return_scale": float(n.return_scale),
            "return_mean": float(n.return_mean),
            "return_std": float(n.return_std),
        }

    train_loader = create_loader(
        result["train"], args.batch_size, shuffle=True
    )
    val_loader = create_loader(result["val"], args.batch_size)

    feature_dim = result["train"].shape[2]
    model_class = TimeGANV32 if args.model_version == "v3.2" else TimeGAN
    model = model_class(
        feature_dim=feature_dim,
        hidden_dim=24,
        noise_dim=feature_dim,
    ).to(device)
    print("Model version:", args.model_version)

    # Keep checkpoints separate for each representation and architecture.
    default_dir = (
        f"checkpoints_{args.dataset_version}_v3_2"
        if args.model_version == "v3.2"
        else f"checkpoints_{args.dataset_version}"
    )
    output_dir = Path(args.output_dir or default_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Device:", device)
    print("Train sequences:", len(result["train"]))
    print("Validation sequences:", len(result["val"]))

    embedding_history, embedding_optimizer = train_embedding(
        model, train_loader, val_loader, device, epochs=args.epochs
    )

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": embedding_optimizer.state_dict(),
        "history": embedding_history,
        "feature_dim": feature_dim,
        "hidden_dim": 24,
        "noise_dim": feature_dim,
        "num_layers": 2,
        "stage": "embedding",
        "dataset_version": args.dataset_version,
        "model_version": args.model_version,
        **normalizer_metadata,
    }, output_dir / "embedding.pt")

    supervisor_history, supervisor_optimizer = train_supervisor(
        model, train_loader, val_loader,
        device, epochs=args.supervisor_epochs
    )

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": supervisor_optimizer.state_dict(),
        "embedding_history": embedding_history,
        "supervisor_history": supervisor_history,
        "feature_dim": feature_dim,
        "hidden_dim": 24,
        "noise_dim": feature_dim,
        "num_layers": 2,
        "stage": "supervisor",
        "dataset_version": args.dataset_version,
        "model_version": args.model_version,
        **normalizer_metadata,
    }, output_dir / "supervisor.pt")

    joint_history = train_joint(
        model, train_loader, device, epochs=args.joint_epochs,
        generator_updates=args.generator_updates,
        lr=args.joint_lr,
    )

    torch.save({
        "model_state_dict": model.state_dict(),
        "joint_history": joint_history,
        "joint_lr": args.joint_lr,
        "generator_updates": args.generator_updates,
        "training_variant": (
            "v4.1-asinh-zscore-identity-generator"
            if args.dataset_version == "v4" and args.model_version == "v3.2"
            else "v4-41feature-identity-generator"
            if args.dataset_version == "v3" and args.model_version == "v3.2"
            else ("v3.2-identity-generator" if args.model_version == "v3.2"
                  else "original-generator")
        ),
        "feature_dim": feature_dim,
        "hidden_dim": 24,
        "noise_dim": feature_dim,
        "num_layers": 2,
        "stage": "joint",
        "dataset_version": args.dataset_version,
        "model_version": args.model_version,
        **normalizer_metadata,
    }, output_dir / "timegan_joint.pt")

    print("\nSaved checkpoints:")
    print(output_dir / "embedding.pt")
    print(output_dir / "supervisor.pt")
    print(output_dir / "timegan_joint.pt")
    print("Runtime (seconds):", round(time.time() - start, 2))
    print("Stages 1, 2 and 3 completed!")


if __name__ == "__main__":
    main()
