
import copy
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from dataset import prepare_dataset_v4
from modules import TimeGANV42, MultiFactorVolumeHead

SEED = 371014
LR = 5e-4

def correlation(x):
    x = x.reshape(-1, 20)
    x = x - x.mean(0, keepdim=True)
    x = x / x.std(
        0, unbiased=False, keepdim=True
    ).clamp_min(1e-5)
    return x.T @ x / len(x)

def lag_corr(x, lag):
    a = x[:, :-lag]
    b = x[:, lag:]

    a = a - a.mean((0, 1), keepdim=True)
    b = b - b.mean((0, 1), keepdim=True)

    return (
        (a * b).mean((0, 1))
        / (
            a.square().mean((0, 1)).sqrt()
            * b.square().mean((0, 1)).sqrt()
        ).clamp_min(1e-6)
    )

def training_targets(x):
    log = torch.log1p(x)
    return {
        "log_mean": log.mean((0, 1)),
        "log_std": log.reshape(-1, 20).std(
            0, unbiased=False
        ),
        "quantiles": [
            torch.quantile(
                log.reshape(-1, 20), q, dim=0
            )
            for q in (0.5, 0.9, 0.99)
        ],
        "corr": correlation(log).detach(),
        "lag1": lag_corr(log, 1).detach(),
        "lag5": lag_corr(log, 5).detach(),
        "change": (
            x[:, 1:] - x[:, :-1]
        ).abs().mean().detach(),
    }

def objective(log_fake, target):
    log_fake = log_fake.clamp(0, 11)
    flat = log_fake.reshape(-1, 20)
    raw = torch.expm1(log_fake)

    dist = F.mse_loss(
        flat.mean(0), target["log_mean"]
    ) + F.mse_loss(
        flat.std(0, unbiased=False),
        target["log_std"]
    )

    for q, expected in zip(
        (0.5, 0.9, 0.99),
        target["quantiles"]
    ):
        dist = dist + F.mse_loss(
            torch.quantile(flat, q, dim=0),
            expected
        )

    matrix = correlation(log_fake)
    mask = ~torch.eye(
        20, dtype=torch.bool
    )
    corr = F.mse_loss(
        matrix[mask],
        target["corr"][mask]
    )

    temporal = (
        F.mse_loss(
            lag_corr(log_fake, 1),
            target["lag1"]
        )
        + F.mse_loss(
            lag_corr(log_fake, 5),
            target["lag5"]
        )
    )

    change = (
        raw[:, 1:] - raw[:, :-1]
    ).abs().mean()
    temporal = temporal + (
        (
            change - target["change"]
        ) / target["change"].clamp_min(1)
    ).square()

    total = (
        dist + 2.0 * corr + temporal
    )
    return total

def rank_statistics(log_volume):
    a = log_volume.detach().numpy()
    a = a.reshape(-1, 20)
    a = (a - a.mean(0)) / np.maximum(
        a.std(0), 1e-8
    )

    eigen = np.linalg.eigvalsh(
        np.cov(a, rowvar=False)
    )[::-1]
    eigen = np.maximum(eigen, 1e-12)
    p = eigen / eigen.sum()

    return {
        "pc1": float(p[0]),
        "effective_rank": float(
            np.exp(-(p * np.log(p)).sum())
        ),
    }

def stats(log_volume):
    raw = torch.expm1(
        log_volume.detach()
    ).numpy()

    c = correlation(
        log_volume.detach()
    ).numpy()

    adjacent = [
        c[j, j+2]
        for j in range(18)
    ]

    return {
        "mean": float(raw.mean()),
        "std": float(raw.std()),
        "p99": float(
            np.quantile(raw, .99)
        ),
        "temporal_change": float(
            np.abs(np.diff(raw, axis=1)).mean()
        ),
        "lag1": float(
            lag_corr(log_volume, 1).mean()
        ),
        "lag5": float(
            lag_corr(log_volume, 5).mean()
        ),
        "adjacent_corr": float(
            np.mean(adjacent)
        ),
        **rank_statistics(log_volume)
    }

def train_v514(
    smoke_test=False,
    data_dir="data",
    output_dir=None,
    checkpoint=None,
):
    EPOCHS = 1 if smoke_test else 8
    STEPS = 1 if smoke_test else 80
    BATCH = 8 if smoke_test else 64
    N_VAL = 32 if smoke_test else 512
    OUT = Path(output_dir or (
        "outputs_smoke_v514" if smoke_test
        else "outputs_v5_14_multifactor"
    ))
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(4)

    data = prepare_dataset_v4(data_dir)
    norm = data["normalizer"]

    def real_volume(split):
        raw = norm.inverse_transform(
            np.asarray(
                data[split], dtype=np.float64
            )
        )
        vol = np.maximum(
            np.expm1(
                raw[..., 1:40:2]
            ), 0
        )
        return torch.tensor(
            vol, dtype=torch.float32
        )

    train = real_volume("train")
    val = real_volume("val")[:N_VAL]

    train_target = training_targets(train)
    val_target = training_targets(val)

    ck = torch.load(
        checkpoint or (
            "outputs_v5_10_latent_matching/"
            "timegan_finetuned.pt"
        ),
        map_location="cpu",
        weights_only=True
    )

    gan = TimeGANV42(
        feature_dim=ck["feature_dim"],
        hidden_dim=ck["hidden_dim"],
        noise_dim=ck["noise_dim"],
        num_layers=ck["num_layers"]
    )
    gan.load_state_dict(
        ck["model_state_dict"]
    )
    gan.eval()

    for p in gan.parameters():
        p.requires_grad_(False)

    head = MultiFactorVolumeHead(
        hidden_dim=ck["hidden_dim"],
        mean_log=train_target["log_mean"],
        std_log=train_target["log_std"]
    )

    optimizer = torch.optim.Adam(
        head.parameters(), lr=LR
    )

    rng = torch.Generator().manual_seed(
        SEED + 100
    )

    # Fixed validation latent samples.
    with torch.no_grad():
        z_val = torch.rand(
            N_VAL, 24, ck["noise_dim"],
            generator=rng
        )
        h_val = gan.supervisor(
            gan.generator(z_val)
        )
        eps_val = torch.randn(
            N_VAL, 24, 20,
            generator=rng
        )

    def validate():
        head.eval()
        with torch.no_grad():
            log_vol = head(h_val, eps_val)
            loss = objective(
                log_vol, val_target
            )
        return float(loss), stats(log_vol)

    best_loss, baseline = validate()
    best_state = copy.deepcopy(
        head.state_dict()
    )
    best_epoch = 0

    print(
        "=== V5.14 MULTIFACTOR TRAINING ===",
        flush=True
    )
    print(
        "Initial validation:",
        json.dumps(baseline, indent=2),
        flush=True
    )
    print(
        "Real validation:",
        json.dumps(
            stats(torch.log1p(val)),
            indent=2
        ),
        flush=True
    )

    history = []

    for epoch in range(1, EPOCHS + 1):
        head.train()
        total = 0.0

        for _ in range(STEPS):
            with torch.no_grad():
                z = torch.rand(
                    BATCH, 24,
                    ck["noise_dim"]
                )
                h = gan.supervisor(
                    gan.generator(z)
                )

            eps = torch.randn(
                BATCH, 24, 20
            )

            log_vol = head(h, eps)
            loss = objective(
                log_vol, train_target
            )

            optimizer.zero_grad(
                set_to_none=True
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                head.parameters(), 5.0
            )
            optimizer.step()

            total += float(loss.item())

        val_loss, metrics = validate()

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(
                head.state_dict()
            )

        history.append({
            "epoch": epoch,
            "train_loss": total / STEPS,
            "val_loss": val_loss,
            **metrics
        })

        print(
            f"Epoch {epoch}/{EPOCHS}"
            f" | Train={total/STEPS:.4f}"
            f" | Val={val_loss:.4f}"
            f" | Rank="
            f"{metrics['effective_rank']:.3f}"
            f" | PC1={metrics['pc1']:.3f}"
            f" | AdjCorr="
            f"{metrics['adjacent_corr']:.3f}"
            f" | Lag1={metrics['lag1']:.3f}"
            f" | Lag5={metrics['lag5']:.3f}"
            f" | Change="
            f"{metrics['temporal_change']:.2f}"
            f" | P99={metrics['p99']:.1f}",
            flush=True
        )

    head.load_state_dict(best_state)
    final_loss, final_metrics = validate()

    torch.save({
        "head_state_dict": head.state_dict(),
        "hidden_dim": ck["hidden_dim"],
        "best_epoch": best_epoch,
        "architecture": "20 independent AR factors "
                        "+ limited TimeGAN conditioning",
    }, OUT / "best_multifactor_head.pt")

    results = {
        "best_epoch": best_epoch,
        "best_validation_loss": final_loss,
        "initial_validation": baseline,
        "best_validation": final_metrics,
        "history": history,
        "train_only_optimisation": True,
        "test_used": False,
    }

    with open(OUT / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(
        "\n=== V5.14 BEST VALIDATION ==="
    )
    print(
        json.dumps(
            {k: v for k, v in results.items()
             if k != "history"},
            indent=2
        )
    )
    print(
        "\nV5.14 training complete!"
    )
