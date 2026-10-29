import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from dataset_v4 import prepare_dataset_v4
from modules_v4_2 import TimeGANV42
from lob_head_v5_8 import MLPConstraintAwareLOBHead
from lob_head_v5_2 import check_raw_constraints

SEED = 42
EPOCHS = 3
LR = 1e-4
BATCH_SIZE = 64
TEMPORAL_WEIGHT = 1.0
VARIANCE_WEIGHT = 0.1

torch.set_num_threads(4)
torch.manual_seed(SEED)
np.random.seed(SEED)


def load_models(data):
    ck = torch.load(
        "checkpoints_v4_3_full/timegan_joint.pt",
        map_location="cpu",
        weights_only=True,
    )

    hc = torch.load(
        "outputs_v5_8_mlp_constrained/best_head.pt",
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

    for p in model.parameters():
        p.requires_grad_(False)

    book = data["normalizer"].book_normalizer

    head = MLPConstraintAwareLOBHead(
        hidden_dim=ck["hidden_dim"],
        minimum=book.minimum,
        scale=book.scale,
        min_spread=0.01,
        min_gap=0.01,
        intermediate=hc["intermediate"],
    )
    head.load_state_dict(hc["head_state_dict"])

    return model, head


def pair_diff(x):
    n = (len(x) // 2) * 2
    return float(
        np.abs(x[:n:2] - x[1:n:2]).mean()
    )


def variance_loss(pred, target):
    # Feature-wise std across batch and time.
    pred_std = pred.reshape(-1, 40).std(
        dim=0, unbiased=False
    )
    real_std = target.reshape(-1, 40).std(
        dim=0, unbiased=False
    ).detach()

    # Protect against nearly constant features.
    denominator = real_std.clamp_min(0.02)

    return (
        ((pred_std - real_std) / denominator) ** 2
    ).mean()


def temporal_loss(pred, target):
    return F.mse_loss(
        pred[:, 1:] - pred[:, :-1],
        target[:, 1:] - target[:, :-1],
    )


@torch.no_grad()
def evaluate(model, head, windows):
    model.eval()
    head.eval()

    predictions = []
    for batch in windows.split(128):
        hidden = model.embedder(batch)
        predictions.append(head(hidden).cpu())

    pred = torch.cat(predictions)
    real = windows[..., :40]

    pred_np = pred.numpy()
    real_np = real.numpy()

    return {
        "mse": float(F.mse_loss(pred, real)),
        "temporal_mse": float(
            temporal_loss(pred, real)
        ),
        "variance_loss": float(
            variance_loss(pred, real)
        ),
        "pair_difference": pair_diff(pred_np),
        "real_pair_difference": pair_diff(real_np),
        "temporal_change": float(
            np.abs(np.diff(pred_np, axis=1)).mean()
        ),
        "real_temporal_change": float(
            np.abs(np.diff(real_np, axis=1)).mean()
        ),
        "across_sample_std": float(
            pred_np.std(axis=0).mean()
        ),
        "real_across_sample_std": float(
            real_np.std(axis=0).mean()
        ),
    }


def train_arm(name, model, starting_state,
              template, train, val, variance_weight):
    head = copy.deepcopy(template)
    head.load_state_dict(starting_state)

    optimizer = torch.optim.Adam(
        head.parameters(), lr=LR
    )

    # Identical mini-batch ordering in both arms.
    rng = torch.Generator()
    rng.manual_seed(SEED)

    loader = DataLoader(
        TensorDataset(train),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=rng,
    )

    initial = evaluate(model, head, val)
    best_score = initial["mse"] + (
        TEMPORAL_WEIGHT * initial["temporal_mse"]
    ) + (
        VARIANCE_WEIGHT * initial["variance_loss"]
    )

    best_state = copy.deepcopy(head.state_dict())
    best_epoch = 0
    history = []

    print(f"\n=== {name} ===", flush=True)
    print("Initial validation:",
          json.dumps(initial, indent=2), flush=True)

    for epoch in range(1, EPOCHS + 1):
        head.train()
        total_loss = 0.0

        for (x,) in loader:
            with torch.no_grad():
                hidden = model.embedder(x)

            target = x[..., :40]
            pred = head(hidden)

            mse = F.mse_loss(pred, target)
            temp = temporal_loss(pred, target)
            var = variance_loss(pred, target)

            loss = (
                mse
                + TEMPORAL_WEIGHT * temp
                + variance_weight * var
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(x)

        val_metrics = evaluate(model, head, val)

        # Same selection criterion across both arms.
        score = (
            val_metrics["mse"]
            + TEMPORAL_WEIGHT
            * val_metrics["temporal_mse"]
            + VARIANCE_WEIGHT
            * val_metrics["variance_loss"]
        )

        if score < best_score:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(
                head.state_dict()
            )

        history.append({
            "epoch": epoch,
            "train_loss": total_loss / len(train),
            "validation": val_metrics,
        })

        print(
            f"Epoch {epoch}/{EPOCHS}"
            f" | Train={total_loss/len(train):.6f}"
            f" | Val MSE={val_metrics['mse']:.6f}"
            f" | Val VarLoss="
            f"{val_metrics['variance_loss']:.6f}"
            f" | Pair="
            f"{val_metrics['pair_difference']:.6f}"
            f" | Change="
            f"{val_metrics['temporal_change']:.6f}",
            flush=True,
        )

    head.load_state_dict(best_state)

    return head, {
        "name": name,
        "variance_weight": variance_weight,
        "best_epoch": best_epoch,
        "initial_validation": initial,
        "history": history,
        "best_validation": evaluate(model, head, val),
    }


def main():
    data = prepare_dataset_v4("data")

    train = torch.tensor(
        np.asarray(data["train"]), dtype=torch.float32
    )
    val = torch.tensor(
        np.asarray(data["val"]), dtype=torch.float32
    )
    test = torch.tensor(
        np.asarray(data["test"]), dtype=torch.float32
    )

    model, template = load_models(data)
    initial_state = copy.deepcopy(
        template.state_dict()
    )

    output = Path("outputs_v5_9_variance_aware")
    output.mkdir(parents=True, exist_ok=True)

    results = {
        "experiment": "v5.9_variance_aware_ablation",
        "epochs": EPOCHS,
        "lr": LR,
        "batch_size": BATCH_SIZE,
        "temporal_weight": TEMPORAL_WEIGHT,
        "variance_weight": VARIANCE_WEIGHT,
    }

    for name, weight in [
        ("control", 0.0),
        ("variance_aware", VARIANCE_WEIGHT),
    ]:
        head, record = train_arm(
            name, model, initial_state,
            template, train, val, weight
        )

        record["test"] = evaluate(model, head, test)

        with torch.no_grad():
            checks = []
            for batch in test.split(128):
                h = model.embedder(batch)
                raw = head.forward_raw(h)
                checks.append(check_raw_constraints(raw))

        record["constraints"] = {
            k: all(c[k] for c in checks)
            for k in checks[0]
        }

        torch.save(
            {
                "head_state_dict": head.state_dict(),
                "hidden_dim": 24,
                "intermediate": 128,
                "best_epoch": record["best_epoch"],
                "variance_weight": weight,
            },
            output / f"{name}_best.pt",
        )

        results[name] = record

    with open(output / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== V5.9 FINAL TEST COMPARISON ===")

    for name in ["control", "variance_aware"]:
        r = results[name]
        print(f"\n{name} | Best epoch: {r['best_epoch']}")
        print(json.dumps(r["test"], indent=2))
        print("Constraints:", r["constraints"])

    print("\nSaved:", output / "results.json")
    print("V5.9 experiment complete!")


if __name__ == "__main__":
    main()
