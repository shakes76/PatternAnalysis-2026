import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from dataset_v4 import prepare_dataset_v4
from modules_v4_2 import TimeGANV42
from lob_head_v5_8 import MLPConstraintAwareLOBHead

SEED = 42
EPOCHS = 5
BATCH = 64
STEPS_PER_EPOCH = 100
LR = 3e-4
OUT = Path("outputs_v5_12_depth_aware")

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(4)
OUT.mkdir(parents=True, exist_ok=True)


class StochasticVolumeHead(nn.Module):
    def __init__(self, hidden_dim=24):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(hidden_dim + 20, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 20),
        )

        # Begin close to the previous decoder.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, hidden, base_log_volume, noise):
        correction = self.net(
            torch.cat([hidden, noise], dim=-1)
        )

        return F.softplus(
            base_log_volume + correction
        )


def correlated_noise(n, t, generator=None):
    eps = torch.randn(
        n, t, 20, generator=generator
    )

    result = torch.zeros_like(eps)
    result[:, 0] = eps[:, 0]

    for step in range(1, t):
        result[:, step] = (
            0.8 * result[:, step - 1]
            + 0.6 * eps[:, step]
        )

    return result


def depth_features(x, normalizer):
    # Convert the 40 normalized LOB features
    # into 20 log1p-volume features.
    book = normalizer.book_normalizer

    minimum = torch.as_tensor(
        book.minimum, dtype=torch.float32
    )
    scale = torch.as_tensor(
        book.scale, dtype=torch.float32
    )

    raw = x[..., :40] * scale + minimum

    # Ask1, Bid1, Ask2, Bid2, ...
    return raw[..., 1::2]


def moment_loss(pred, target):
    # Shape: N x T x 20
    p = pred.reshape(-1, 20)
    r = target.reshape(-1, 20)

    mean_loss = F.mse_loss(
        p.mean(0), r.mean(0)
    )

    std_loss = F.mse_loss(
        p.std(0, unbiased=False),
        r.std(0, unbiased=False)
    )

    # Distribution tails, computed per LOB level.
    quantile_loss = 0.0
    for q in [0.5, 0.9, 0.99]:
        quantile_loss = quantile_loss + F.mse_loss(
            torch.quantile(p, q, dim=0),
            torch.quantile(r, q, dim=0)
        )

    pred_change = torch.abs(
        pred[:, 1:] - pred[:, :-1]
    ).mean(dim=(0, 1))

    real_change = torch.abs(
        target[:, 1:] - target[:, :-1]
    ).mean(dim=(0, 1))

    temporal_loss = F.mse_loss(
        pred_change, real_change
    )

    return (
        mean_loss
        + std_loss
        + quantile_loss
        + 2.0 * temporal_loss
    )


def metrics(pred, real):
    pred_np = np.expm1(pred.detach().numpy())
    real_np = np.expm1(real.detach().numpy())

    def stats(x):
        return {
            "mean": float(x.mean()),
            "std": float(x.std()),
            "p99": float(np.quantile(x, 0.99)),
            "across_sample_std": float(
                x.std(axis=0).mean()
            ),
            "temporal_change": float(
                np.abs(np.diff(x, axis=1)).mean()
            ),
        }

    return {
        "generated": stats(pred_np),
        "real": stats(real_np),
    }


def load_generator():
    ck = torch.load(
        "outputs_v5_10_latent_matching/timegan_finetuned.pt",
        map_location="cpu",
        weights_only=True,
    )

    model = TimeGANV42(
        feature_dim=ck["feature_dim"],
        hidden_dim=ck["hidden_dim"],
        noise_dim=ck["noise_dim"],
        num_layers=ck["num_layers"],
    )
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    return model, ck


def main():
    data = prepare_dataset_v4("data")
    normalizer = data["normalizer"]

    model, ck = load_generator()

    book = normalizer.book_normalizer

    hc = torch.load(
        "outputs_v5_9_variance_aware/variance_aware_best.pt",
        map_location="cpu",
        weights_only=True,
    )

    decoder = MLPConstraintAwareLOBHead(
        hidden_dim=hc["hidden_dim"],
        minimum=book.minimum,
        scale=book.scale,
        min_spread=0.01,
        min_gap=0.01,
        intermediate=hc["intermediate"],
    )

    decoder.load_state_dict(hc["head_state_dict"])
    decoder.eval()

    for parameter in decoder.parameters():
        parameter.requires_grad_(False)

    train = torch.tensor(
        np.asarray(data["train"]), dtype=torch.float32
    )
    val = torch.tensor(
        np.asarray(data["val"]), dtype=torch.float32
    )

    train_log = depth_features(train, normalizer)
    val_log = depth_features(val, normalizer)

    head = StochasticVolumeHead(
        hidden_dim=ck["hidden_dim"]
    )

    optimizer = torch.optim.Adam(
        head.parameters(), lr=LR
    )

    # Fixed noise and generator states for fair
    # epoch-by-epoch validation.
    val_rng = torch.Generator().manual_seed(12345)

    with torch.no_grad():
        z_val = torch.rand(
            512, 24, ck["noise_dim"],
            generator=val_rng
        )
        h_val = model.supervisor(
            model.generator(z_val)
        )
        base_val = decoder.forward_raw(
            h_val
        )[..., 1::2]

        noise_val = correlated_noise(
            512, 24, val_rng
        )

    reference_val = val_log[:512]

    def validate():
        head.eval()

        with torch.no_grad():
            pred = head(
                h_val, base_val, noise_val
            )

            score = float(
                moment_loss(
                    pred, reference_val
                ).item()
            )

        return score, metrics(
            pred, reference_val
        )

    best_score, initial_metrics = validate()
    best_state = copy.deepcopy(
        head.state_dict()
    )
    best_epoch = 0

    print("=== V5.12 Depth-aware Training ===")
    print(
        "Initial validation loss:",
        best_score,
        flush=True
    )
    print(json.dumps(
        initial_metrics, indent=2
    ), flush=True)

    for epoch in range(1, EPOCHS + 1):
        head.train()
        total = 0.0

        for _ in range(STEPS_PER_EPOCH):
            real_indices = torch.randint(
                0, len(train_log), (BATCH,)
            )
            target = train_log[real_indices]

            with torch.no_grad():
                z = torch.rand(
                    BATCH, 24, ck["noise_dim"]
                )
                hidden = model.supervisor(
                    model.generator(z)
                )
                base = decoder.forward_raw(
                    hidden
                )[..., 1::2]

            stochastic = correlated_noise(
                BATCH, 24
            )

            pred = head(
                hidden, base, stochastic
            )

            loss = moment_loss(pred, target)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total += float(loss.item())

        val_score, val_metrics = validate()

        if val_score < best_score:
            best_score = val_score
            best_state = copy.deepcopy(
                head.state_dict()
            )
            best_epoch = epoch

        print(
            f"Epoch {epoch}/{EPOCHS}"
            f" | Train={total/STEPS_PER_EPOCH:.6f}"
            f" | Val={val_score:.6f}"
            f" | Generated mean="
            f"{val_metrics['generated']['mean']:.2f}"
            f" | Generated std="
            f"{val_metrics['generated']['std']:.2f}"
            f" | Generated P99="
            f"{val_metrics['generated']['p99']:.2f}"
            f" | Temporal="
            f"{val_metrics['generated']['temporal_change']:.2f}",
            flush=True
        )

    head.load_state_dict(best_state)
    final_score, final_metrics = validate()

    torch.save({
        "head_state_dict": head.state_dict(),
        "hidden_dim": ck["hidden_dim"],
        "best_epoch": best_epoch,
        "model_description":
            "stochastic volume-only correction head"
    }, OUT / "best_volume_head.pt")

    results = {
        "best_epoch": best_epoch,
        "best_validation_loss": final_score,
        "initial_validation": initial_metrics,
        "best_validation": final_metrics,
        "training_source": "train",
        "model_selection_source": "validation",
        "test_used_for_training": False,
    }

    with open(OUT / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== V5.12 BEST VALIDATION ===")
    print(json.dumps(results, indent=2))
    print("\nSaved:", OUT / "best_volume_head.pt")
    print("V5.12 training complete!")


if __name__ == "__main__":
    main()
