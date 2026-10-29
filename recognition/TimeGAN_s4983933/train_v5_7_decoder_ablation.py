import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from dataset_v4 import prepare_dataset_v4
from modules_v4_2 import TimeGANV42
from lob_head_v5_2 import ConstraintAwareLOBHeadV52


class MLPDecoder(nn.Module):
    """Unconstrained, time-step-wise latent-to-LOB decoder."""
    def __init__(self, hidden_dim=24, intermediate=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, intermediate),
            nn.ReLU(),
            nn.Linear(intermediate, intermediate),
            nn.ReLU(),
            nn.Linear(intermediate, 40),
        )

    def forward(self, hidden):
        return self.network(hidden)


def pair_difference(x):
    n = (len(x) // 2) * 2
    return float(np.abs(x[:n:2] - x[1:n:2]).mean())


def calculate_metrics(pred, target):
    mse = float(np.mean((pred - target) ** 2))
    pred_delta = np.diff(pred, axis=1)
    real_delta = np.diff(target, axis=1)

    return {
        "mse": mse,
        "temporal_mse": float(
            np.mean((pred_delta - real_delta) ** 2)
        ),
        "pair_difference": pair_difference(pred),
        "across_sample_std": float(
            pred.std(axis=0).mean()
        ),
        "temporal_change": float(
            np.abs(pred_delta).mean()
        ),
        "temporal_change_ratio": float(
            np.abs(pred_delta).mean()
            / max(np.abs(real_delta).mean(), 1e-12)
        ),
    }


@torch.no_grad()
def evaluate_decoder(embedder, decoder, windows):
    embedder.eval()
    decoder.eval()

    predictions = []
    targets = []

    loader = DataLoader(
        TensorDataset(windows),
        batch_size=128,
        shuffle=False,
    )

    for (x,) in loader:
        hidden = embedder(x)
        pred = decoder(hidden)
        predictions.append(pred.cpu().numpy())
        targets.append(x[..., :40].cpu().numpy())

    return calculate_metrics(
        np.concatenate(predictions),
        np.concatenate(targets),
    )


def load_components(data):
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

    for p in model.parameters():
        p.requires_grad_(False)

    head_ckpt = torch.load(
        "outputs_v5_6_dynamic_lob/best_dynamic_lob_head.pt",
        map_location="cpu",
        weights_only=True,
    )

    book = data["normalizer"].book_normalizer

    constrained = ConstraintAwareLOBHeadV52(
        hidden_dim=ckpt["hidden_dim"],
        minimum=book.minimum,
        scale=book.scale,
        min_spread=0.01,
        min_gap=0.01,
    )
    constrained.load_state_dict(
        head_ckpt["head_state_dict"]
    )
    constrained.eval()

    for p in constrained.parameters():
        p.requires_grad_(False)

    return model, constrained, ckpt


class OriginalRecovery40(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.recovery = model.recovery

    def forward(self, hidden):
        return self.recovery(hidden)[..., :40]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument(
        "--temporal-weight", type=float, default=1.0
    )
    parser.add_argument(
        "--output-dir",
        default="outputs_v5_7_decoder_ablation",
    )
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(4)

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

    model, constrained, ckpt = load_components(data)

    decoder = MLPDecoder(
        hidden_dim=ckpt["hidden_dim"]
    )

    optimizer = torch.optim.Adam(
        decoder.parameters(), lr=args.lr
    )

    loader = DataLoader(
        TensorDataset(train),
        batch_size=args.batch_size,
        shuffle=True,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== V5.7 Decoder Ablation ===")
    print("Train / Validation / Test:",
          len(train), len(val), len(test))
    print("Frozen Embedder: V4.3")
    print("Trainable Decoder: MLP")
    print("Temporal weight:", args.temporal_weight)

    original_recovery = OriginalRecovery40(model)

    reference = {
        "v4_3_recovery": evaluate_decoder(
            model.embedder, original_recovery, val
        ),
        "v5_6_constrained": evaluate_decoder(
            model.embedder, constrained, val
        ),
    }

    print("\n=== Reference Validation ===")
    print(json.dumps(reference, indent=2), flush=True)

    best_score = float("inf")
    best_epoch = None
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        decoder.train()

        total_loss = 0.0
        samples = 0

        for (x,) in loader:
            with torch.no_grad():
                hidden = model.embedder(x)

            target = x[..., :40]
            pred = decoder(hidden)

            reconstruction = F.mse_loss(pred, target)

            pred_delta = pred[:, 1:] - pred[:, :-1]
            target_delta = target[:, 1:] - target[:, :-1]

            dynamic = F.mse_loss(
                pred_delta, target_delta
            )

            loss = (
                reconstruction
                + args.temporal_weight * dynamic
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(x)
            samples += len(x)

        val_metrics = evaluate_decoder(
            model.embedder, decoder, val
        )

        score = (
            val_metrics["mse"]
            + args.temporal_weight
            * val_metrics["temporal_mse"]
        )

        if score < best_score:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(
                decoder.state_dict()
            )

        history.append({
            "epoch": epoch,
            "train_loss": total_loss / samples,
            "validation": val_metrics,
        })

        print(
            f"Epoch {epoch}/{args.epochs}"
            f" | Train={total_loss / samples:.6f}"
            f" | Val MSE={val_metrics['mse']:.6f}"
            f" | Val Temp MSE="
            f"{val_metrics['temporal_mse']:.6f}"
            f" | Val Pair="
            f"{val_metrics['pair_difference']:.6f}"
            f" | Val Temp Change="
            f"{val_metrics['temporal_change']:.6f}",
            flush=True,
        )

    decoder.load_state_dict(best_state)

    # Test set only after validation-based selection.
    test_metrics = {
        "v4_3_recovery": evaluate_decoder(
            model.embedder, original_recovery, test
        ),
        "v5_6_constrained": evaluate_decoder(
            model.embedder, constrained, test
        ),
        "v5_7_mlp": evaluate_decoder(
            model.embedder, decoder, test
        ),
    }

    train_mean = train[..., :40].mean(
        dim=(0, 1)
    ).numpy()

    real_test = test[..., :40].numpy()
    baseline = np.broadcast_to(
        train_mean, real_test.shape
    )

    test_metrics["train_mean_baseline"] = (
        calculate_metrics(baseline, real_test)
    )

    real_delta = np.diff(real_test, axis=1)

    test_metrics["real_reference"] = {
        "pair_difference": pair_difference(
            real_test
        ),
        "across_sample_std": float(
            real_test.std(axis=0).mean()
        ),
        "temporal_change": float(
            np.abs(real_delta).mean()
        ),
    }

    result = {
        "experiment": "v5.7_decoder_ablation",
        "best_epoch": best_epoch,
        "epochs": args.epochs,
        "lr": args.lr,
        "temporal_weight": args.temporal_weight,
        "reference_validation": reference,
        "training_history": history,
        "test": test_metrics,
    }

    torch.save(
        {
            "model_state_dict": best_state,
            "hidden_dim": ckpt["hidden_dim"],
            "intermediate": 128,
            "best_epoch": best_epoch,
            "source_embedder": (
                "checkpoints_v4_3_full/timegan_joint.pt"
            ),
        },
        output_dir / "best_mlp_decoder.pt",
    )

    with open(
        output_dir / "results.json", "w"
    ) as f:
        json.dump(result, f, indent=2)

    print("\n=== Test Comparison ===")
    print(json.dumps(test_metrics, indent=2))
    print("\nBest epoch:", best_epoch)
    print("Saved:", output_dir / "best_mlp_decoder.pt")
    print("Saved:", output_dir / "results.json")
    print("V5.7 experiment complete!")


if __name__ == "__main__":
    main()
