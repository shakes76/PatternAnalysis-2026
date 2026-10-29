import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dataset_v4 import prepare_dataset_v4
from modules_v4_2 import TimeGANV42
from lob_head_v5_8 import MLPConstraintAwareLOBHead
from train_v5_12_depth_aware import (
    StochasticVolumeHead,
    correlated_noise,
)

SEED = 371013
EPOCHS = 6
STEPS_PER_EPOCH = 80
BATCH = 64
LR = 1e-4
EMA_ALPHA = 0.2

CORR_WEIGHT = 4.0
TEMP_WEIGHT = 2.0
DIST_WEIGHT = 1.0

N_VAL = 512
OUT = Path("outputs_v5_13_correlation_aware")
OUT.mkdir(parents=True, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(4)


def ema_volumes(x, alpha=EMA_ALPHA):
    parts = [x[:, 0]]
    previous = x[:, 0]
    for t in range(1, x.shape[1]):
        previous = (
            alpha * x[:, t]
            + (1.0 - alpha) * previous
        )
        parts.append(previous)
    return torch.stack(parts, dim=1)


def corr_matrix(x):
    # x: N x T x 20, raw share volumes.
    v = x.reshape(-1, 20)
    v = v - v.mean(dim=0, keepdim=True)
    v = v / (
        v.square().mean(dim=0, keepdim=True)
        .sqrt().clamp_min(1e-4)
    )
    return (v.T @ v) / v.shape[0]


def lag_corr(x, lag):
    a = x[:, :-lag]
    b = x[:, lag:]

    am = a.mean(dim=(0, 1), keepdim=True)
    bm = b.mean(dim=(0, 1), keepdim=True)

    a = a - am
    b = b - bm

    numerator = (a * b).mean(dim=(0, 1))
    denominator = (
        a.square().mean(dim=(0, 1)).sqrt()
        * b.square().mean(dim=(0, 1)).sqrt()
    ).clamp_min(1e-6)

    return numerator / denominator


def distribution_loss(fake, real):
    # Stable log1p-volume distribution matching.
    f = torch.log1p(fake).reshape(-1, 20)
    r = torch.log1p(real).reshape(-1, 20)

    mean_loss = F.mse_loss(
        f.mean(0), r.mean(0)
    )
    std_loss = F.mse_loss(
        f.std(0, unbiased=False),
        r.std(0, unbiased=False)
    )

    quantile_loss = 0.0
    for q in (0.5, 0.9, 0.99):
        quantile_loss = quantile_loss + F.mse_loss(
            torch.quantile(f, q, dim=0),
            torch.quantile(r, q, dim=0)
        )

    return mean_loss + std_loss + quantile_loss


def correlation_loss(fake, target_matrix):
    # Use log volumes for stable optimisation,
    # matching the actual TRAIN log-volume matrix.
    log_fake = torch.log1p(fake)
    matrix = corr_matrix(log_fake)

    mask = ~torch.eye(
        20, dtype=torch.bool, device=fake.device
    )

    return F.mse_loss(
        matrix[mask], target_matrix[mask]
    )


def temporal_loss(fake, target_lag1, target_lag5,
                  target_change):
    log_fake = torch.log1p(fake)

    lag1 = lag_corr(log_fake, 1)
    lag5 = lag_corr(log_fake, 5)

    # Match raw-share average event-to-event change,
    # normalised to avoid an enormous loss scale.
    change = (
        fake[:, 1:] - fake[:, :-1]
    ).abs().mean()

    loss = (
        F.mse_loss(lag1, target_lag1)
        + F.mse_loss(lag5, target_lag5)
        + (
            (change - target_change)
            / target_change.clamp_min(1.0)
        ).square()
    )

    return loss


def load_models(data):
    ck = torch.load(
        "outputs_v5_10_latent_matching/timegan_finetuned.pt",
        map_location="cpu",
        weights_only=True
    )

    gan = TimeGANV42(
        feature_dim=ck["feature_dim"],
        hidden_dim=ck["hidden_dim"],
        noise_dim=ck["noise_dim"],
        num_layers=ck["num_layers"]
    )
    gan.load_state_dict(ck["model_state_dict"])
    gan.eval()

    hc = torch.load(
        "outputs_v5_9_variance_aware/variance_aware_best.pt",
        map_location="cpu",
        weights_only=True
    )

    bn = data["normalizer"].book_normalizer

    decoder = MLPConstraintAwareLOBHead(
        hidden_dim=hc["hidden_dim"],
        minimum=bn.minimum,
        scale=bn.scale,
        intermediate=hc["intermediate"],
        min_spread=0.01,
        min_gap=0.01
    )
    decoder.load_state_dict(hc["head_state_dict"])
    decoder.eval()

    vc = torch.load(
        "outputs_v5_12_depth_aware/best_volume_head.pt",
        map_location="cpu",
        weights_only=True
    )

    head = StochasticVolumeHead(
        hidden_dim=vc["hidden_dim"]
    )
    head.load_state_dict(vc["head_state_dict"])

    for module in (gan, decoder):
        for p in module.parameters():
            p.requires_grad_(False)

    return gan, decoder, head, ck


def get_real_volumes(data, split):
    x = data["normalizer"].inverse_transform(
        np.asarray(data[split], dtype=np.float64)
    )
    volume = np.expm1(x[..., 1:40:2])
    return torch.tensor(
        np.maximum(volume, 0), dtype=torch.float32
    )


def generate_volumes(
    gan, decoder, head, ck, count, rng
):
    with torch.no_grad():
        z = torch.rand(
            count, 24, ck["noise_dim"],
            generator=rng
        )
        hidden = gan.supervisor(
            gan.generator(z)
        )
        base = decoder.forward_raw(
            hidden
        )[..., 1::2]

    noise = correlated_noise(
        count, 24, generator=rng
    )

    predicted_log = head(hidden, base, noise)

    # Softplus output is already a positive log1p
    # volume, so expm1 converts to raw shares.
    raw_volume = torch.expm1(
        predicted_log.clamp(max=12)
    )

    return ema_volumes(raw_volume)


def describe(x):
    a = x.detach().cpu().numpy()
    adjacent = []
    for side in (0, 1):
        for level in range(9):
            j = 2 * level + side
            k = j + 2
            adjacent.append(
                np.corrcoef(
                    a[..., j].ravel(),
                    a[..., k].ravel()
                )[0, 1]
            )

    return {
        "mean": float(a.mean()),
        "std": float(a.std()),
        "p99": float(np.quantile(a, 0.99)),
        "temporal_change": float(
            np.abs(np.diff(a, axis=1)).mean()
        ),
        "lag1_mean": float(lag_corr(x, 1).mean()),
        "lag5_mean": float(lag_corr(x, 5).mean()),
        "adjacent_corr_mean": float(
            np.mean(adjacent)
        ),
        "adjacent_corr_abs_mean": float(
            np.mean(np.abs(adjacent))
        ),
    }


def main():
    data = prepare_dataset_v4("data")

    real_train = get_real_volumes(data, "train")
    real_val = get_real_volumes(data, "val")[:N_VAL]

    # Targets are calculated exclusively from TRAIN.
    train_log = torch.log1p(real_train)
    target_matrix = corr_matrix(train_log).detach()
    target_lag1 = lag_corr(train_log, 1).detach()
    target_lag5 = lag_corr(train_log, 5).detach()
    target_change = (
        real_train[:, 1:] - real_train[:, :-1]
    ).abs().mean().detach()

    # Validation uses actual validation statistics,
    # but these values never enter training gradients.
    val_log = torch.log1p(real_val)
    val_matrix = corr_matrix(val_log).detach()
    val_lag1 = lag_corr(val_log, 1).detach()
    val_lag5 = lag_corr(val_log, 5).detach()
    val_change = (
        real_val[:, 1:] - real_val[:, :-1]
    ).abs().mean().detach()

    gan, decoder, head, ck = load_models(data)

    optimizer = torch.optim.Adam(
        head.parameters(), lr=LR
    )

    # Fixed latent/noise samples across validation
    # epochs for reproducible checkpoint selection.
    val_rng = torch.Generator().manual_seed(202613)

    with torch.no_grad():
        z = torch.rand(
            N_VAL, 24, ck["noise_dim"],
            generator=val_rng
        )
        val_hidden = gan.supervisor(
            gan.generator(z)
        )
        val_base = decoder.forward_raw(
            val_hidden
        )[..., 1::2]
        val_noise = correlated_noise(
            N_VAL, 24, generator=val_rng
        )

    def validation():
        head.eval()
        with torch.no_grad():
            log_volume = head(
                val_hidden, val_base, val_noise
            )
            raw = torch.expm1(
                log_volume.clamp(max=12)
            )
            prediction = ema_volumes(raw)

            dist = distribution_loss(
                prediction, real_val
            )
            corr = correlation_loss(
                prediction, val_matrix
            )
            temp = temporal_loss(
                prediction, val_lag1,
                val_lag5, val_change
            )

            score = (
                DIST_WEIGHT * dist
                + CORR_WEIGHT * corr
                + TEMP_WEIGHT * temp
            )

        return (
            float(score),
            {
                "distribution_loss": float(dist),
                "correlation_loss": float(corr),
                "temporal_loss": float(temp),
                "generated": describe(prediction),
                "real": describe(real_val),
            }
        )

    initial_score, initial = validation()
    best_score = initial_score
    best_state = copy.deepcopy(head.state_dict())
    best_epoch = 0

    print("=== V5.13 CORRELATION-AWARE TRAINING ===")
    print(
        "Initial validation:",
        json.dumps(initial, indent=2),
        flush=True
    )

    for epoch in range(1, EPOCHS + 1):
        head.train()
        epoch_loss = 0.0

        for _ in range(STEPS_PER_EPOCH):
            rng = torch.Generator().manual_seed(
                int(torch.randint(1, 2**30, (1,)).item())
            )

            count = BATCH
            indices = torch.randint(
                0, len(real_train), (count,)
            )

            target = real_train[indices]
            prediction = generate_volumes(
                gan, decoder, head, ck, count, rng
            )

            dist = distribution_loss(
                prediction, target
            )
            corr = correlation_loss(
                prediction, target_matrix
            )
            temp = temporal_loss(
                prediction, target_lag1,
                target_lag5, target_change
            )

            loss = (
                DIST_WEIGHT * dist
                + CORR_WEIGHT * corr
                + TEMP_WEIGHT * temp
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                head.parameters(), max_norm=5.0
            )
            optimizer.step()

            epoch_loss += float(loss.item())

        val_score, val_result = validation()

        if val_score < best_score:
            best_score = val_score
            best_state = copy.deepcopy(
                head.state_dict()
            )
            best_epoch = epoch

        g = val_result["generated"]

        print(
            f"Epoch {epoch}/{EPOCHS}"
            f" | Train={epoch_loss/STEPS_PER_EPOCH:.4f}"
            f" | Val={val_score:.4f}"
            f" | CorrLoss="
            f"{val_result['correlation_loss']:.4f}"
            f" | AdjCorr={g['adjacent_corr_mean']:.4f}"
            f" | Lag1={g['lag1_mean']:.4f}"
            f" | Lag5={g['lag5_mean']:.4f}"
            f" | Change={g['temporal_change']:.2f}"
            f" | Std={g['std']:.2f}",
            flush=True
        )

    head.load_state_dict(best_state)
    best_val_score, best_val = validation()

    torch.save({
        "head_state_dict": head.state_dict(),
        "hidden_dim": ck["hidden_dim"],
        "best_epoch": best_epoch,
        "ema_alpha": EMA_ALPHA,
        "source_checkpoint": (
            "outputs_v5_12_depth_aware/"
            "best_volume_head.pt"
        ),
    }, OUT / "best_volume_head.pt")

    results = {
        "best_epoch": best_epoch,
        "best_validation_score": best_val_score,
        "initial_validation": initial,
        "best_validation": best_val,
        "configuration": {
            "epochs": EPOCHS,
            "steps_per_epoch": STEPS_PER_EPOCH,
            "batch_size": BATCH,
            "lr": LR,
            "ema_alpha": EMA_ALPHA,
            "correlation_weight": CORR_WEIGHT,
            "temporal_weight": TEMP_WEIGHT,
            "distribution_weight": DIST_WEIGHT,
        },
        "train_only_gradient_targets": True,
        "selection_split": "validation",
        "test_used": False,
    }

    with open(OUT / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== V5.13 BEST VALIDATION ===")
    print(json.dumps(results, indent=2))
    print("\nSaved:", OUT / "best_volume_head.pt")
    print("V5.13 training complete!")


if __name__ == "__main__":
    main()
