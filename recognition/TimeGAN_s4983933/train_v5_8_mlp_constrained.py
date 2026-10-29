import argparse
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
from train_v5_7_decoder_ablation import (
    evaluate_decoder,
)


def initialise_from_train(head, data):
    """Initialise the 39 output parameters from TRAIN only."""
    normalizer = data["normalizer"]
    raw = normalizer.inverse_transform(
        np.asarray(data["train"])
    )[..., :40]

    prices = raw[..., 0::2]
    sizes = raw[..., 1::2]
    asks = prices[..., 0::2]
    bids = prices[..., 1::2]

    spread = asks[..., 0] - bids[..., 0]
    ask_gaps = np.diff(asks, axis=-1)
    bid_gaps = -np.diff(bids, axis=-1)

    targets = np.concatenate([
        [spread.mean()],
        ask_gaps.mean(axis=(0, 1)),
        bid_gaps.mean(axis=(0, 1)),
        sizes.mean(axis=(0, 1)),
    ])

    minimums = np.concatenate([
        [head.min_spread],
        np.full(18, head.min_gap),
        np.zeros(20),
    ])

    residual = targets - minimums
    if np.any(residual <= 0):
        raise ValueError(
            "Invalid mean initialization targets"
        )

    bias = np.log(np.expm1(residual))

    last = head.projection[-1]
    with torch.no_grad():
        # Small nonzero weights preserve gradient flow
        # to earlier MLP layers.
        torch.nn.init.normal_(
            last.weight, mean=0.0, std=0.001
        )
        last.bias.copy_(
            torch.tensor(bias, dtype=torch.float32)
        )

    print(
        "Mean initialization: spread=",
        float(targets[0]),
        "ask_gap=",
        float(targets[1:10].mean()),
        "bid_gap=",
        float(targets[10:19].mean()),
        flush=True,
    )


def pair_difference(x):
    n = len(x) // 2 * 2
    return float(
        np.abs(x[:n:2] - x[1:n:2]).mean()
    )


@torch.no_grad()
def detail_metrics(embedder, head, windows):
    embedder.eval()
    head.eval()

    results = []
    for (x,) in DataLoader(
        TensorDataset(windows),
        batch_size=128,
        shuffle=False,
    ):
        h = embedder(x)
        results.append(head(h).cpu())

    pred = torch.cat(results).numpy()
    real = windows[..., :40].numpy()

    return {
        "mse": float(np.mean((pred - real)**2)),
        "temporal_mse": float(np.mean(
            (np.diff(pred, axis=1)
             - np.diff(real, axis=1))**2
        )),
        "pair_difference": pair_difference(pred),
        "real_pair_difference": pair_difference(real),
        "temporal_change": float(
            np.abs(np.diff(pred, axis=1)).mean()
        ),
        "real_temporal_change": float(
            np.abs(np.diff(real, axis=1)).mean()
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--batch-size", type=int, default=64
    )
    parser.add_argument(
        "--lr", type=float, default=0.001
    )
    parser.add_argument(
        "--temporal-weight", type=float, default=1.0
    )
    parser.add_argument(
        "--output-dir",
        default="outputs_v5_8_mlp_constrained",
    )
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(4)

    data = prepare_dataset_v4("data")

    train = torch.tensor(
        np.asarray(data["train"]),
        dtype=torch.float32,
    )
    val = torch.tensor(
        np.asarray(data["val"]),
        dtype=torch.float32,
    )
    test = torch.tensor(
        np.asarray(data["test"]),
        dtype=torch.float32,
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
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    book = data["normalizer"].book_normalizer

    head = MLPConstraintAwareLOBHead(
        hidden_dim=ckpt["hidden_dim"],
        minimum=book.minimum,
        scale=book.scale,
        min_spread=0.01,
        min_gap=0.01,
        intermediate=128,
    )

    initialise_from_train(head, data)

    optimizer = torch.optim.Adam(
        head.parameters(), lr=args.lr
    )

    loader = DataLoader(
        TensorDataset(train),
        batch_size=args.batch_size,
        shuffle=True,
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    baseline = detail_metrics(
        model.embedder, head, val
    )
    best_score = (
        baseline["mse"]
        + args.temporal_weight
        * baseline["temporal_mse"]
    )
    best_epoch = 0
    best_state = copy.deepcopy(head.state_dict())
    history = []

    print(
        "=== V5.8 MLP Constraint-aware Reconstruction ===",
        flush=True,
    )
    print(
        "Train / Val / Test:",
        len(train), len(val), len(test),
    )
    print(
        "Initial validation:",
        json.dumps(baseline, indent=2),
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        head.train()
        running = 0.0

        for (x,) in loader:
            with torch.no_grad():
                hidden = model.embedder(x)

            prediction = head(hidden)
            target = x[..., :40]

            loss_mse = F.mse_loss(
                prediction, target
            )
            loss_temporal = F.mse_loss(
                prediction[:, 1:] - prediction[:, :-1],
                target[:, 1:] - target[:, :-1],
            )

            loss = (
                loss_mse
                + args.temporal_weight * loss_temporal
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(x)

        metrics = detail_metrics(
            model.embedder, head, val
        )
        score = (
            metrics["mse"]
            + args.temporal_weight
            * metrics["temporal_mse"]
        )

        if score < best_score:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(
                head.state_dict()
            )

        history.append({
            "epoch": epoch,
            "train_loss": running / len(train),
            "validation": metrics,
        })

        print(
            f"Epoch {epoch}/{args.epochs}"
            f" | Train={running/len(train):.6f}"
            f" | Val MSE={metrics['mse']:.6f}"
            f" | Val Temp MSE={metrics['temporal_mse']:.6f}"
            f" | Val Pair={metrics['pair_difference']:.6f}"
            f" | Val Change={metrics['temporal_change']:.6f}",
            flush=True,
        )

    head.load_state_dict(best_state)

    val_result = detail_metrics(
        model.embedder, head, val
    )
    test_result = detail_metrics(
        model.embedder, head, test
    )

    with torch.no_grad():
        constraints = []
        for batch in test.split(128):
            hidden = model.embedder(batch)
            raw = head.forward_raw(hidden)
            constraints.append(
                check_raw_constraints(raw)
            )

    all_constraints = {
        key: all(result[key] for result in constraints)
        for key in constraints[0]
    }

    result = {
        "experiment": "v5.8_mlp_constrained",
        "best_epoch": best_epoch,
        "initial_validation": baseline,
        "history": history,
        "best_validation": val_result,
        "test": test_result,
        "constraints": all_constraints,
    }

    torch.save(
        {
            "head_state_dict": best_state,
            "hidden_dim": ckpt["hidden_dim"],
            "intermediate": 128,
            "best_epoch": best_epoch,
            "source_embedder":
                "checkpoints_v4_3_full/timegan_joint.pt",
        },
        out / "best_head.pt",
    )

    with open(out / "results.json", "w") as f:
        json.dump(result, f, indent=2)

    print("\n=== Best Validation ===")
    print(json.dumps(val_result, indent=2))
    print("\n=== Test ===")
    print(json.dumps(test_result, indent=2))
    print("\n=== Constraints ===")
    print(json.dumps(all_constraints, indent=2))
    print("\nBest epoch:", best_epoch)
    print("Saved:", out / "best_head.pt")
    print("Saved:", out / "results.json")
    print("V5.8 training complete!")


if __name__ == "__main__":
    main()
