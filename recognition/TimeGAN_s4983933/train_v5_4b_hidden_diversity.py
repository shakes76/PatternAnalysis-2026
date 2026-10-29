import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dataset_v4 import prepare_dataset_v4
from modules_v4_2 import TimeGANV42
from lob_head_v5_2 import ConstraintAwareLOBHeadV52
from train_v4_3 import moment_loss


def pair_difference(x):
    n = (x.shape[0] // 2) * 2
    return (x[:n:2] - x[1:n:2]).abs().mean()


def set_grad(module, enabled):
    for p in module.parameters():
        p.requires_grad_(enabled)


@torch.no_grad()
def evaluate(model, head, test, count=512):
    model.eval()
    head.eval()

    x = test[:count]
    n, t, _ = x.shape

    # Fixed seed to make evaluations comparable.
    with torch.random.fork_rng():
        torch.manual_seed(12345)
        z = torch.rand(n, t, model.noise_dim)

    e_fake = model.generator(z)
    h_fake = model.supervisor(e_fake)
    fake = head(h_fake)
    fake_raw = head.forward_raw(h_fake)

    real_pair = pair_difference(x[..., :40]).item()
    fake_pair = pair_difference(fake).item()

    prices = fake_raw[..., 0::2]
    asks = prices[..., 0::2]
    bids = prices[..., 1::2]
    spread = asks[..., 0] - bids[..., 0]

    return {
        "real_normalized_pair_diff": float(real_pair),
        "fake_normalized_pair_diff": float(fake_pair),
        "fake_across_sample_std": float(
            fake.std(dim=0, unbiased=False).mean()
        ),
        "fake_temporal_difference": float(
            (fake[:, 1:] - fake[:, :-1]).abs().mean()
        ),
        "fake_spread_mean": float(spread.mean()),
        "fake_spread_std": float(
            spread.std(unbiased=False)
        ),
        "generator_hidden_pair_diff": float(
            pair_difference(e_fake)
        ),
        "supervisor_hidden_pair_diff": float(
            pair_difference(h_fake)
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--diversity-weight", type=float, default=0.5)
    parser.add_argument("--target-fraction", type=float, default=0.5)
    parser.add_argument("--generator-updates", type=int, default=2)
    parser.add_argument(
        "--output-dir",
        default="outputs_v5_4b_hidden_diversity",
    )
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(4)

    device = torch.device("cpu")

    data = prepare_dataset_v4("data")
    normalizer = data["normalizer"]

    train = torch.tensor(
        np.asarray(data["train"]),
        dtype=torch.float32,
    )
    val = torch.tensor(
        np.asarray(data["val"]),
        dtype=torch.float32,
    )

    loader = DataLoader(
        TensorDataset(train),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    ckpt = torch.load(
        "checkpoints_v4_3_full/timegan_joint.pt",
        map_location="cpu",
        weights_only=True,
    )

    model = TimeGANV42(
        feature_dim=ckpt["feature_dim"],
        hidden_dim=ckpt["hidden_dim"],
        noise_dim=ckpt["noise_dim"],
        num_layers=ckpt["num_layers"],
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])

    head_ckpt = torch.load(
        "outputs_v5_3_lob_head/best_lob_head.pt",
        map_location="cpu",
        weights_only=True,
    )

    head = ConstraintAwareLOBHeadV52(
        hidden_dim=ckpt["hidden_dim"],
        minimum=normalizer.book_normalizer.minimum,
        scale=normalizer.book_normalizer.scale,
        min_spread=0.01,
        min_gap=0.01,
    ).to(device)

    head.load_state_dict(head_ckpt["head_state_dict"])

    # Keep reconstruction representation fixed.
    set_grad(model.embedder, False)
    set_grad(model.recovery, False)
    set_grad(head, False)

    opt_g = torch.optim.Adam(
        list(model.generator.parameters())
        + list(model.supervisor.parameters()),
        lr=args.lr,
    )

    opt_d = torch.optim.Adam(
        model.discriminator.parameters(),
        lr=args.lr,
    )

    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    folder = Path(args.output_dir)
    folder.mkdir(parents=True, exist_ok=True)

    baseline = evaluate(model, head, val)

    print("=== V5.4 Diversity-aware TimeGAN ===")
    print("Train windows:", len(train))
    print("Validation windows:", len(val))
    print("Generator updates:", args.generator_updates)
    print("Diversity weight:", args.diversity_weight)
    print("Target fraction:", args.target_fraction)
    print("\n=== Before training ===")
    print(json.dumps(baseline, indent=2), flush=True)

    history = []

    for epoch in range(args.epochs):
        model.generator.train()
        model.supervisor.train()
        model.discriminator.train()

        total = {
            "generator_loss": 0.0,
            "diversity_loss": 0.0,
            "real_pair": 0.0,
            "fake_pair": 0.0,
            "discriminator_loss": 0.0,
        }
        steps = 0

        for (real,) in loader:
            real = real.to(device)
            b, t, _ = real.shape

            with torch.no_grad():
                h_real = model.embedder(real)
                target_lob = real[..., :40]

            # Generator and Supervisor optimization.
            set_grad(model.discriminator, False)
            set_grad(model.generator, True)
            set_grad(model.supervisor, True)

            for _ in range(args.generator_updates):
                z = torch.rand(
                    b, t, ckpt["noise_dim"],
                    device=device,
                )

                e_fake = model.generator(z)
                h_fake = model.supervisor(e_fake)

                x_fake = model.recovery(h_fake)
                lob_fake = head(h_fake)
                h_pred = model.supervisor(h_real)

                logits_h = model.discriminator(h_fake)
                logits_e = model.discriminator(e_fake)

                adv = (
                    bce(
                        logits_h,
                        torch.ones_like(logits_h),
                    )
                    + bce(
                        logits_e,
                        torch.ones_like(logits_e),
                    )
                )

                sup = mse(
                    h_pred[:, :-1, :],
                    h_real[:, 1:, :],
                )

                moments = moment_loss(real, x_fake)

                # V5.4b: match diversity in latent space.
                real_pair = pair_difference(
                    h_real
                ).detach()

                fake_pair = pair_difference(h_fake)

                target = (
                    args.target_fraction * real_pair
                ).detach()

                diversity_loss = torch.relu(
                    (target - fake_pair)
                    / (target + 1e-6)
                )

                loss_g = (
                    adv
                    + 100 * torch.sqrt(sup + 1e-8)
                    + 100 * moments
                    + args.diversity_weight
                    * diversity_loss
                )

                opt_g.zero_grad(set_to_none=True)
                loss_g.backward()
                opt_g.step()

                total["generator_loss"] += loss_g.item()
                total["diversity_loss"] += (
                    diversity_loss.item()
                )
                total["real_pair"] += real_pair.item()
                total["fake_pair"] += (
                    fake_pair.detach().item()
                )

            # Discriminator optimization.
            set_grad(model.generator, False)
            set_grad(model.supervisor, False)
            set_grad(model.discriminator, True)

            with torch.no_grad():
                z = torch.rand(
                    b, t, ckpt["noise_dim"],
                    device=device,
                )
                e_fake = model.generator(z)
                h_fake = model.supervisor(e_fake)

            logits_real = model.discriminator(h_real)
            logits_fake = model.discriminator(h_fake)
            logits_fake_e = model.discriminator(e_fake)

            loss_d = (
                bce(
                    logits_real,
                    torch.ones_like(logits_real),
                )
                + bce(
                    logits_fake,
                    torch.zeros_like(logits_fake),
                )
                + bce(
                    logits_fake_e,
                    torch.zeros_like(logits_fake_e),
                )
            )

            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            opt_d.step()

            total["discriminator_loss"] += loss_d.item()
            steps += 1

        for key in [
            "generator_loss",
            "diversity_loss",
            "real_pair",
            "fake_pair",
        ]:
            total[key] /= (
                steps * args.generator_updates
            )

        total["discriminator_loss"] /= steps

        validation = evaluate(model, head, val)

        record = {
            "epoch": epoch + 1,
            "training": total,
            "validation": validation,
        }
        history.append(record)

        print(
            f"\nEpoch {epoch + 1}/{args.epochs}"
        )
        print(
            f"G Loss={total['generator_loss']:.6f}"
            f" | D Loss={total['discriminator_loss']:.6f}"
            f" | Diversity Loss={total['diversity_loss']:.6f}"
        )
        print(
            "Validation fake LOB pair difference:",
            validation["fake_normalized_pair_diff"],
        )
        print(
            "Validation fake Spread std:",
            validation["fake_spread_std"],
        )

        if not all(np.isfinite(v) for v in total.values()):
            raise RuntimeError("Non-finite training metric")

    result = {
        "experiment": "v5.4b_hidden_diversity_finetuning",
        "source_checkpoint": (
            "checkpoints_v4_3_full/timegan_joint.pt"
        ),
        "epochs": args.epochs,
        "lr": args.lr,
        "diversity_weight": args.diversity_weight,
        "target_fraction": args.target_fraction,
        "baseline_validation": baseline,
        "history": history,
    }

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "source_checkpoint": (
                "checkpoints_v4_3_full/timegan_joint.pt"
            ),
            "feature_dim": ckpt["feature_dim"],
            "hidden_dim": ckpt["hidden_dim"],
            "noise_dim": ckpt["noise_dim"],
            "num_layers": ckpt["num_layers"],
            "experiment": "v5.4b-hidden-diversity",
        },
        folder / "timegan_finetuned.pt",
    )

    with open(folder / "results.json", "w") as f:
        json.dump(result, f, indent=2)

    print("\n=== Final comparison ===")
    print(
        "Before pair difference:",
        baseline["fake_normalized_pair_diff"],
    )
    print(
        "After pair difference:",
        history[-1]["validation"][
            "fake_normalized_pair_diff"
        ],
    )
    print(
        "Before spread std:",
        baseline["fake_spread_std"],
    )
    print(
        "After spread std:",
        history[-1]["validation"][
            "fake_spread_std"
        ],
    )

    print("\nSaved:", folder / "timegan_finetuned.pt")
    print("Saved:", folder / "results.json")
    print("V5.4 experiment complete!")


if __name__ == "__main__":
    main()
