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
from lob_head_v5_2 import (
    ConstraintAwareLOBHeadV52,
    check_raw_constraints,
)


def pair_diff(x):
    n = (len(x) // 2) * 2
    return float(
        np.abs(x[:n:2] - x[1:n:2]).mean()
    )


def temporal_mse(pred, target):
    pred_delta = pred[:, 1:] - pred[:, :-1]
    true_delta = target[:, 1:] - target[:, :-1]
    return F.mse_loss(pred_delta, true_delta)


def build_model(data):
    ckpt = torch.load(
        "checkpoints_v4_3_full/timegan_joint.pt",
        map_location="cpu",
        weights_only=True,
    )
    hc = torch.load(
        "outputs_v5_3_lob_head/best_lob_head.pt",
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

    for p in model.parameters():
        p.requires_grad_(False)

    book = data["normalizer"].book_normalizer

    head = ConstraintAwareLOBHeadV52(
        hidden_dim=ckpt["hidden_dim"],
        minimum=book.minimum,
        scale=book.scale,
        min_spread=0.01,
        min_gap=0.01,
    )
    head.load_state_dict(hc["head_state_dict"])

    return model, head


@torch.no_grad()
def evaluate(model, head, windows, temporal_weight):
    model.eval()
    head.eval()

    total_mse = 0.0
    total_temporal_mse = 0.0
    count = 0
    preds = []
    targets = []

    loader = DataLoader(
        TensorDataset(windows),
        batch_size=128,
        shuffle=False,
    )

    for (x,) in loader:
        h = model.embedder(x)
        pred = head(h)
        target = x[..., :40]

        n = len(x)
        total_mse += F.mse_loss(
            pred, target
        ).item() * n

        total_temporal_mse += temporal_mse(
            pred, target
        ).item() * n

        count += n
        preds.append(pred.cpu())
        targets.append(target.cpu())

    pred = torch.cat(preds).numpy()
    target = torch.cat(targets).numpy()

    mse = total_mse / count
    tmse = total_temporal_mse / count

    temporal_pred = float(
        np.abs(np.diff(pred, axis=1)).mean()
    )
    temporal_real = float(
        np.abs(np.diff(target, axis=1)).mean()
    )

    return {
        "mse": mse,
        "temporal_mse": tmse,
        "combined_loss": (
            mse + temporal_weight * tmse
        ),
        "real_pair_diff": pair_diff(target),
        "pred_pair_diff": pair_diff(pred),
        "real_temporal_change": temporal_real,
        "pred_temporal_change": temporal_pred,
        "temporal_change_ratio": (
            temporal_pred / max(temporal_real, 1e-12)
        ),
        "real_across_sample_std": float(
            target.std(axis=0).mean()
        ),
        "pred_across_sample_std": float(
            pred.std(axis=0).mean()
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--temporal-weight", type=float, default=1.0
    )
    parser.add_argument(
        "--output-dir",
        default="outputs_v5_6_dynamic_lob",
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

    model, head = build_model(data)

    train_loader = DataLoader(
        TensorDataset(train),
        batch_size=args.batch_size,
        shuffle=True,
    )

    optimizer = torch.optim.Adam(
        head.parameters(),
        lr=args.lr,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline = evaluate(
        model, head, val, args.temporal_weight
    )

    best_val = baseline["combined_loss"]
    best_epoch = 0
    best_state = copy.deepcopy(head.state_dict())

    history = []

    print("=== V5.6 Dynamic-aware LOB Reconstruction ===")
    print("Train/Val/Test:", len(train), len(val), len(test))
    print("Only LOB Head is trainable.")
    print("Temporal weight:", args.temporal_weight)
    print("\n=== Initial V5.3 Validation ===")
    print(json.dumps(baseline, indent=2), flush=True)

    for epoch in range(1, args.epochs + 1):
        head.train()
        total = 0.0
        total_mse = 0.0
        total_temporal = 0.0

        for (x,) in train_loader:
            with torch.no_grad():
                h = model.embedder(x)

            pred = head(h)
            target = x[..., :40]

            loss_mse = F.mse_loss(pred, target)
            loss_temporal = temporal_mse(
                pred, target
            )

            loss = (
                loss_mse
                + args.temporal_weight * loss_temporal
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            n = len(x)
            total += loss.item() * n
            total_mse += loss_mse.item() * n
            total_temporal += loss_temporal.item() * n

        train_metrics = {
            "combined_loss": total / len(train),
            "mse": total_mse / len(train),
            "temporal_mse": total_temporal / len(train),
        }

        validation = evaluate(
            model, head, val, args.temporal_weight
        )

        history.append({
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation,
        })

        if validation["combined_loss"] < best_val:
            best_val = validation["combined_loss"]
            best_epoch = epoch
            best_state = copy.deepcopy(
                head.state_dict()
            )

        print(
            f"\nEpoch {epoch}/{args.epochs}"
            f" | Train Loss={train_metrics['combined_loss']:.6f}"
            f" | Val MSE={validation['mse']:.6f}"
            f" | Val Temporal MSE={validation['temporal_mse']:.6f}"
            f" | Val Pair={validation['pred_pair_diff']:.8f}"
            f" | Val Temporal Change="
            f"{validation['pred_temporal_change']:.8f}",
            flush=True,
        )

    head.load_state_dict(best_state)

    best_validation = evaluate(
        model, head, val, args.temporal_weight
    )
    test_result = evaluate(
        model, head, test, args.temporal_weight
    )

    with torch.no_grad():
        h = model.embedder(test[:64])
        raw = head.forward_raw(h)
        constraints = check_raw_constraints(raw)

    if not all(constraints.values()):
        raise RuntimeError(
            "LOB constraint validation failed"
        )

    torch.save(
        {
            "head_state_dict": best_state,
            "best_epoch": best_epoch,
            "hidden_dim": 24,
            "source_head": (
                "outputs_v5_3_lob_head/best_lob_head.pt"
            ),
            "temporal_weight": args.temporal_weight,
        },
        output_dir / "best_dynamic_lob_head.pt",
    )

    result = {
        "experiment": "v5.6_dynamic_lob",
        "best_epoch": best_epoch,
        "initial_validation": baseline,
        "history": history,
        "best_validation": best_validation,
        "test": test_result,
        "constraints": constraints,
    }

    with open(
        output_dir / "results.json", "w"
    ) as f:
        json.dump(result, f, indent=2)

    print("\n=== Best Validation ===")
    print(json.dumps(best_validation, indent=2))
    print("\n=== Test ===")
    print(json.dumps(test_result, indent=2))
    print("\n=== Constraints ===")
    print(json.dumps(constraints, indent=2))
    print("\nBest epoch:", best_epoch)
    print("Saved:", output_dir / "best_dynamic_lob_head.pt")
    print("Saved:", output_dir / "results.json")
    print("V5.6 experiment complete!")


if __name__ == "__main__":
    main()
