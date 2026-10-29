
"""COMP3710 TimeGAN V4.3: resume weighted Stage 1, train stages 2 and 3."""

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
from modules_v4_2 import TimeGANV42
from stage1_v4_3_compare import return_aware_loss, reconstruct_metrics


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


def joint_train_step(model, real, optimizers, generator_updates=1,
                     zero_normalized=0.0, return_weight=1.0, move_weight=5.0):
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
    recon = return_aware_loss(
        x_recon, real, zero_normalized, return_weight, move_weight
    )
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
                generator_updates=1, zero_normalized=0.0,
                return_weight=1.0, move_weight=5.0):
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
                zero_normalized=zero_normalized,
                return_weight=return_weight, move_weight=move_weight,
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


def audit_reconstruction(model, data, normalizer, device, label):
    """Report validation-only reconstruction performance after each stage."""
    metrics = reconstruct_metrics(model, data, normalizer, device)
    print(f"\n=== {label}: Validation Reconstruction ===", flush=True)
    for key, value in metrics.items():
        print(f"  {key}: {value:.10g}", flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="V4.3: continue from weighted-return Stage 1 checkpoint"
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--stage1-checkpoint", default="checkpoints_v4_3_stage1/weighted_returns.pt")
    parser.add_argument("--supervisor-epochs", type=int, default=2)
    parser.add_argument("--joint-epochs", type=int, default=2)
    parser.add_argument("--joint-lr", type=float, default=0.0001)
    parser.add_argument("--generator-updates", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--return-weight", type=float, default=1.0)
    parser.add_argument("--move-weight", type=float, default=5.0)
    parser.add_argument("--output-dir", default="checkpoints_v4_3_full")
    args = parser.parse_args()
    if (args.supervisor_epochs < 1 or args.joint_epochs < 1
            or args.generator_updates < 1 or args.batch_size < 1
            or args.joint_lr <= 0 or args.return_weight <= 0
            or args.move_weight < 1):
        parser.error("Invalid training parameters; epochs/updates/batch/lr must be positive")

    set_seed(args.seed)
    start = time.time()
    device = torch.device(args.device)
    result = prepare_dataset_v4(args.data_dir)
    normalizer = result["normalizer"]
    zero_normalized = -float(normalizer.return_mean) / float(normalizer.return_std)

    train_loader = create_loader(result["train"], args.batch_size, shuffle=True)
    val_loader = create_loader(result["val"], args.batch_size)
    ckpt_path = Path(args.stage1_checkpoint)
    if not ckpt_path.is_file():
        parser.error(f"Stage 1 checkpoint not found: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if checkpoint.get("variant") != "weighted_returns":
        parser.error("Stage 1 checkpoint must have variant='weighted_returns'")
    if checkpoint.get("model_version") != "v4.2" or checkpoint.get("dataset_version") != "v4":
        parser.error("Stage 1 checkpoint must use model v4.2 and dataset v4")
    if checkpoint.get("seed") != args.seed:
        parser.error("Requested seed does not match the Stage 1 checkpoint seed")
    if not np.isclose(checkpoint.get("return_scale", np.nan), normalizer.return_scale):
        parser.error("Return scale disagrees with current training dataset")
    for key in ("return_mean", "return_std"):
        if not np.isclose(checkpoint.get(key, np.nan), getattr(normalizer, key), atol=1e-9):
            parser.error(f"{key} disagrees with current dataset")
    for key in ("return_weight", "move_weight"):
        # Older Stage-1 checkpoints may not store these; the original experiment
        # used CLI defaults 1.0 and 5.0 respectively.
        expected = getattr(args, key)
        if key in checkpoint and not np.isclose(checkpoint[key], expected):
            parser.error(f"{key} differs from saved Stage 1 hyperparameter")

    model = TimeGANV42(feature_dim=41, hidden_dim=24, noise_dim=41, num_layers=2).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "feature_dim": 41, "hidden_dim": 24, "noise_dim": 41, "num_layers": 2,
        "dataset_version": "v4", "model_version": "v4.2",
        "training_variant": "v4.3-weighted-reconstruction",
        "return_scale": float(normalizer.return_scale),
        "return_mean": float(normalizer.return_mean),
        "return_std": float(normalizer.return_std),
        "return_weight": args.return_weight, "move_weight": args.move_weight,
        "zero_normalized": zero_normalized,
        "stage1_checkpoint": str(ckpt_path), "seed": args.seed,
    }

    print("Dataset version: v4 | Model version: v4.2", flush=True)
    print("Loaded weighted Stage 1 checkpoint:", ckpt_path, flush=True)
    print("Train/validation sequences:", len(result["train"]), len(result["val"]), flush=True)
    print("Stage 3 return loss weights:", args.return_weight, args.move_weight, flush=True)
    audit_reconstruction(model, result["val"], normalizer, device, "Stage 1 (loaded)")

    supervisor_history, supervisor_optimizer = train_supervisor(
        model, train_loader, val_loader, device, epochs=args.supervisor_epochs
    )
    audit_reconstruction(model, result["val"], normalizer, device, "Stage 2")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": supervisor_optimizer.state_dict(),
        "supervisor_history": supervisor_history,
        "stage": "supervisor", **metadata,
    }, output_dir / "supervisor.pt")

    joint_history = train_joint(
        model, train_loader, device, epochs=args.joint_epochs,
        generator_updates=args.generator_updates, lr=args.joint_lr,
        zero_normalized=zero_normalized,
        return_weight=args.return_weight, move_weight=args.move_weight,
    )
    val_metrics = audit_reconstruction(
        model, result["val"], normalizer, device, "Stage 3"
    )
    torch.save({
        "model_state_dict": model.state_dict(),
        "joint_history": joint_history,
        "supervisor_history": supervisor_history,
        "joint_lr": args.joint_lr,
        "generator_updates": args.generator_updates,
        "validation_reconstruction": val_metrics,
        "stage": "joint", **metadata,
    }, output_dir / "timegan_joint.pt")
    print("\nSaved checkpoints:", flush=True)
    print(output_dir / "supervisor.pt", flush=True)
    print(output_dir / "timegan_joint.pt", flush=True)
    print("Runtime (seconds):", round(time.time() - start, 2), flush=True)
    print("V4.3 Stage 2 and Stage 3 completed!", flush=True)


if __name__ == "__main__":
    main()
