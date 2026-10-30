"""Train an MLP or GCN using training targets and selection targets only.

No test, calibration or policy targets are requested. Outputs include the best
checkpoint, full history, basic resource measurements and learning curves.
"""

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import random
import subprocess
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import psutil
from sklearn.metrics import f1_score
import torch
from torch import nn

from dataset import DEFAULT_ROOT, NodeClassificationDataset, load_facebook, sha256_file
from modules import GCN, MLP


@dataclass(frozen=True)
class TrainConfig:
    """Small, predeclared baseline configuration; split seed is independent."""

    seed: int = 42
    split_seed: int = 42
    hidden_features: int = 64
    dropout: float = 0.5
    learning_rate: float = 0.01
    weight_decay: float = 0.0005
    max_epochs: int = 300
    patience: int = 30
    min_delta: float = 0.0001
    device: str = "auto"
    cpu_threads: int = 1
    model_type: str = "MLP"


def configure_reproducibility(config: TrainConfig) -> torch.device:
    """Use deterministic operations; fail rather than silently accept nondeterminism."""
    if min(config.max_epochs, config.patience, config.cpu_threads) < 1:
        raise ValueError("Epoch limit, patience and CPU threads must be positive")
    if config.learning_rate <= 0 or config.weight_decay < 0 or config.min_delta < 0:
        raise ValueError("Invalid learning rate, weight decay or minimum improvement")
    if config.model_type not in ("MLP", "GCN"):
        raise ValueError("Model type must be MLP or GCN")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.set_num_threads(config.cpu_threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    chosen = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(chosen if config.device == "auto" else config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def model_logits(model, features, edge_index=None, mask=None):
    """Mask outputs, never features, for fixed-graph message passing."""
    logits = model(features) if edge_index is None else model(features, edge_index)
    return logits if mask is None else logits[mask]


@torch.no_grad()
def evaluate(model: nn.Module, features: torch.Tensor, targets: torch.Tensor,
             edge_index=None, mask=None) -> dict:
    """Basic loss/accuracy/macro-F1, with dropout disabled for validation."""
    model.eval()
    logits = model_logits(model, features, edge_index, mask)
    loss = nn.functional.cross_entropy(logits, targets)
    if not torch.isfinite(loss):
        raise RuntimeError("Evaluation loss is not finite")
    predictions = logits.argmax(dim=1)
    return {
        "loss": float(loss.item()),
        "accuracy": float((predictions == targets).float().mean().item()),
        "macro_f1": float(f1_score(
            targets.cpu().numpy(), predictions.cpu().numpy(),
            labels=list(range(logits.size(1))), average="macro", zero_division=0,
        )),
    }


def plot_history(history: list[dict], path: Path) -> None:
    """Plot basic training curves required by the assessment."""
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [r["optimization_loss"] for r in history], label="Training (dropout on)")
    axes[0].plot(epochs, [r["selection_loss"] for r in history], label="Selection (dropout off)")
    axes[0].set(ylabel="Cross-entropy", xlabel="Epoch")
    axes[1].plot(epochs, [r["selection_accuracy"] for r in history], label="Selection accuracy")
    axes[1].plot(epochs, [r["selection_macro_f1"] for r in history], label="Selection macro-F1")
    axes[1].set(ylabel="Score", xlabel="Epoch", ylim=(0, 1))
    for axis in axes:
        axis.legend()
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def source_record() -> dict:
    """Record the parent Git revision plus hashes of the actual executable code."""
    folder = Path(__file__).resolve().parent
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=folder, capture_output=True, text=True,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=folder, capture_output=True, text=True,
    )
    return {
        "git_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "git_worktree_dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
        "code_sha256": {
            name: sha256_file(folder / name)
            for name in ("dataset.py", "modules.py", "train.py")
        },
    }


def train_model(
    dataset: NodeClassificationDataset, config: TrainConfig, output_dir: Path,
) -> dict:
    """Fit on train nodes, select by selection macro-F1, and restore the best model.

    Checkpoints contain weights and primitive metadata, not datasets. No resume
    mechanism or final-test evaluation is implemented at this stage.
    """
    if dataset.audit["split_seed"] != config.split_seed:
        raise ValueError("Configuration and dataset split seeds do not match")
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Output directory is not empty; choose a fresh run directory")
    device = configure_reproducibility(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)

    # GCN sees all unlabeled graph context; only permitted labels are fetched.
    edge_index = train_mask = selection_mask = None
    if config.model_type == "GCN":
        train_x = selection_x = dataset.graph.x.to(device)
        edge_index = dataset.graph.edge_index.to(device)
        train_mask = dataset.masks["train"].to(device)
        selection_mask = dataset.masks["selection"].to(device)
    else:
        train_x = dataset.graph.x[dataset.masks["train"]].to(device)
        selection_x = dataset.graph.x[dataset.masks["selection"]].to(device)
    train_y = dataset.targets("train").to(device)
    selection_y = dataset.targets("selection").to(device)
    model_config = {
        "in_features": dataset.graph.num_node_features,
        "hidden_features": config.hidden_features,
        "num_classes": dataset.audit["classes"],
        "dropout": config.dropout,
    }
    model = {"MLP": MLP, "GCN": GCN}[config.model_type](**model_config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
    )
    process = psutil.Process()
    peak_sampled_rss = process.memory_info().rss
    metadata = {
        "model_type": config.model_type, "model_config": model_config,
        "training_config": asdict(config), "data_provenance": dataset.provenance,
        "split_mask_sha256": {
            name: report["mask_sha256"] for name, report in dataset.audit["splits"].items()
        },
        "source": source_record(),
        "versions": {"torch": str(torch.__version__), "numpy": np.__version__},
    }
    checkpoint_path = output_dir / "best.pt"
    history = []
    best_score = -1.0
    stale_epochs = 0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        # Cross-entropy receives raw logits and ONLY the training target vector.
        loss = nn.functional.cross_entropy(
            model_logits(model, train_x, edge_index, train_mask), train_y,
        )
        if not torch.isfinite(loss):
            raise RuntimeError("Training loss is not finite")
        loss.backward()
        for parameter in model.parameters():
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise RuntimeError("Missing or nonfinite training gradients")
        optimizer.step()
        selection = evaluate(model, selection_x, selection_y, edge_index, selection_mask)
        row = {
            "epoch": epoch, "optimization_loss": float(loss.item()),
            **{f"selection_{name}": value for name, value in selection.items()},
        }
        history.append(row)
        if selection["macro_f1"] > best_score + config.min_delta:
            best_score = selection["macro_f1"]
            stale_epochs = 0
            torch.save({
                **metadata, "epoch": epoch, "selection_metrics": selection,
                "model_state_dict": {
                    name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
                },
            }, checkpoint_path)
        else:
            stale_epochs += 1
        peak_sampled_rss = max(peak_sampled_rss, process.memory_info().rss)
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:3d}: train CE={row['optimization_loss']:.5f}, "
                f"selection CE={selection['loss']:.5f}, "
                f"accuracy={selection['accuracy']:.4f}, F1={selection['macro_f1']:.4f}",
                flush=True,
            )
        if stale_epochs >= config.patience:
            break
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - started
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    summary = {
        **metadata,
        "best_epoch": checkpoint["epoch"], "epochs_completed": len(history),
        "stopped_early": len(history) < config.max_epochs,
        "last_optimization_loss": history[-1]["optimization_loss"],
        "restored_training_metrics": evaluate(model, train_x, train_y, edge_index, train_mask),
        "restored_selection_metrics": evaluate(model, selection_x, selection_y, edge_index, selection_mask),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "training_seconds": training_seconds,
        "peak_sampled_process_rss_mib": peak_sampled_rss / (1024 ** 2),
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else None,
        "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved(device) / (1024 ** 2) if device.type == "cuda" else None,
        "timing_scope": "training, selection validation and checkpoint writes; excludes data load, setup and plots",
        "memory_scope": "CUDA allocator peak through restored-checkpoint evaluation; process RSS sampled at setup and per epoch",
        "checkpoint": "best.pt", "reserved_splits_evaluated": [],
    }
    (output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    plot_history(history, output_dir / "learning_curves.png")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def train_mlp(dataset, config, output_dir):
    """Keep the Stage 3 entry point restricted to the feature-only baseline."""
    if config.model_type != "MLP":
        raise ValueError("train_mlp requires model_type=MLP; use train_model for GCN")
    return train_model(dataset, config, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("mlp", "gcn"), default="mlp")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda:0"))
    args = parser.parse_args()
    config = TrainConfig(
        seed=args.seed, split_seed=args.split_seed, max_epochs=args.max_epochs,
        patience=args.patience, device=args.device, model_type=args.model.upper(),
    )
    output_dir = args.output_dir or Path(__file__).resolve().parent / "outputs" / f"{args.model}_seed{args.seed}"
    train_model(load_facebook(args.data_root, args.split_seed), config, output_dir)


if __name__ == "__main__":
    main()
