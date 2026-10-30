"""Selection-only evaluation of frozen MLP/GCN checkpoints (Stage 5).

No calibration, policy or test targets are accessed. Metrics use sklearn;
inference and hidden representations use the original PyTorch/PyG models.

Metric API: https://scikit-learn.org/1.7/modules/generated/sklearn.metrics.roc_auc_score.html.
t-SNE method: van der Maaten & Hinton (2008),
https://www.jmlr.org/papers/v9/vandermaaten08a.html (sklearn implementation).
"""

import argparse
import csv
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.manifold import TSNE
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)
import torch
import torch_geometric

from dataset import CLASS_NAMES, DEFAULT_ROOT, load_facebook, sha256_file
from modules import GCN, MLP
from train import TrainConfig, configure_reproducibility, source_record

PROJECT = Path(__file__).resolve().parent


def classification_metrics(targets, probabilities, class_names=CLASS_NAMES) -> dict:
    """Class IDs and probability columns must follow class_names order.

    Macro metrics weight classes equally. AUROC uses each class versus all
    others, then the unweighted mean. Reject missing classes rather than report
    an undefined AUROC as a fabricated number.
    """
    targets = np.asarray(targets)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    classes = np.arange(len(class_names))
    if targets.ndim != 1 or targets.dtype.kind not in "iu":
        raise ValueError("Targets must be a one-dimensional integer vector")
    if probabilities.shape != (len(targets), len(classes)) or len(targets) == 0:
        raise ValueError("Probability dimensions do not match targets/classes")
    if (not np.isfinite(probabilities).all() or (probabilities < 0).any()
            or (probabilities > 1).any()
            or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-6, rtol=0)):
        raise ValueError("Expected finite, nonnegative probabilities summing to one")
    if not np.array_equal(np.unique(targets), classes):
        raise ValueError("Every class must occur for multiclass AUROC")
    predicted = probabilities.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predicted, labels=classes, zero_division=0,
    )
    auc = roc_auc_score(targets, probabilities, labels=classes,
                        multi_class="ovr", average=None)
    return {
        "nodes": len(targets), "accuracy": float(accuracy_score(targets, predicted)),
        "macro_precision": float(precision.mean()), "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()), "macro_ovr_auroc": float(auc.mean()),
        "per_class": [
            {"class_id": int(i), "class_name": name, "support": int(support[i]),
             "precision": float(precision[i]), "recall": float(recall[i]),
             "f1": float(f1[i]), "ovr_auroc": float(auc[i])}
            for i, name in enumerate(class_names)
        ],
        "confusion_matrix": confusion_matrix(targets, predicted, labels=classes).tolist(),
    }


def load_model(checkpoint_path, dataset, device, expected_type):
    """Check data/split identity before loading a frozen checkpoint safely."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["model_type"] != expected_type:
        raise ValueError("Unexpected checkpoint model type")
    if checkpoint["data_provenance"]["sha256"] != dataset.provenance["sha256"]:
        raise ValueError("Checkpoint dataset hash does not match")
    masks = {name: report["mask_sha256"] for name, report in dataset.audit["splits"].items()}
    if checkpoint["split_mask_sha256"] != masks:
        raise ValueError("Checkpoint split hashes do not match")
    cls = {"MLP": MLP, "GCN": GCN}[expected_type]
    model = cls(**checkpoint["model_config"]).to(device).eval()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model, checkpoint


@torch.inference_mode()
def benchmark_forward(forward, device, warmup=20, repeats=100):
    """Synchronised wall time of full-graph model calls, excluding transfers.

    First forward includes GCN normalization cache creation. Steady-state calls
    retain that fixed-graph cache. CUDA synchronization includes completed work,
    not merely asynchronous launch time. No softmax, metrics or plotting timed.
    """
    if warmup < 1 or repeats < 2:
        raise ValueError("Need positive warmup and at least two timed repeats")

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def timed_call():
        sync()
        started = time.perf_counter()
        forward()
        sync()
        return (time.perf_counter() - started) * 1000

    first_ms = timed_call()
    for _ in range(warmup):
        forward()
    sync()
    samples = [timed_call() for _ in range(repeats)]
    return {
        "first_forward_ms": first_ms, "warmup_calls": warmup, "timed_calls": repeats,
        "mean_ms": float(np.mean(samples)), "median_ms": float(np.median(samples)),
        "p95_ms": float(np.percentile(samples, 95)), "min_ms": float(np.min(samples)),
    }, samples


@torch.inference_mode()
def evaluate_checkpoint(dataset, checkpoint_path, expected_type, device,
                        warmup=20, repeats=100):
    """Evaluate only selection labels; retain node-level outputs on CPU only."""
    gc.collect()
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    model, checkpoint = load_model(checkpoint_path, dataset, device, expected_type)
    features = dataset.graph.x.to(device)
    edges = dataset.graph.edge_index.to(device) if expected_type == "GCN" else None

    def forward():
        return model(features) if edges is None else model(features, edges)

    timing, samples = benchmark_forward(forward, device, warmup, repeats)
    # Same full-node forward workload for both models; subset only for metrics.
    mask = dataset.masks["selection"].to(device)
    logits = forward()[mask]
    probabilities = logits.softmax(dim=1).cpu().numpy()
    targets = dataset.targets("selection").numpy()
    metrics = classification_metrics(targets, probabilities)
    loss = float(torch.nn.functional.cross_entropy(
        logits, torch.from_numpy(targets).to(device)).item())
    # Recomputed metrics must agree with the originally selected checkpoint.
    recorded = checkpoint["selection_metrics"]
    for key, actual in (("loss", loss), ("accuracy", metrics["accuracy"]),
                        ("macro_f1", metrics["macro_f1"])):
        if not np.isclose(actual, recorded[key], atol=1e-6, rtol=1e-5):
            raise ValueError(f"Restored {key} disagrees with checkpoint")
    resources = {
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "inference_nodes": dataset.graph.num_nodes,
        "peak_cuda_allocated_mib": (torch.cuda.max_memory_allocated(device) / 1024**2
                                     if device.type == "cuda" else None),
        "peak_cuda_reserved_mib": (torch.cuda.max_memory_reserved(device) / 1024**2
                                    if device.type == "cuda" else None),
        **timing,
    }
    # Hidden features are after conv1/ReLU with dropout disabled. Do not change
    # the architecture or checkpoint to expose them; labels never enter this step.
    hidden = (torch.relu(model.conv1(features, edges))[mask].cpu().numpy()
              if expected_type == "GCN" else None)
    result = {
        "model": expected_type, "checkpoint_sha256": sha256_file(Path(checkpoint_path)),
        "checkpoint_epoch": checkpoint["epoch"], "cross_entropy": loss,
        "metrics": metrics, "resources": resources,
    }
    return result, probabilities, hidden, samples


def save_plot(figure, stem):
    """PNG for README plus vector SVG for export."""
    figure.savefig(str(stem) + ".png", dpi=180, bbox_inches="tight")
    figure.savefig(str(stem) + ".svg", bbox_inches="tight")
    plt.close(figure)


def plot_confusion(metrics, name, stem, partition="selection"):
    counts = np.asarray(metrics["confusion_matrix"])
    normalized = counts / counts.sum(axis=1, keepdims=True)
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    for row, column in np.ndindex(counts.shape):
        axis.text(column, row, f"{counts[row, column]}\n{normalized[row, column]:.1%}",
                  ha="center", va="center", color="white" if normalized[row, column] > .5 else "black")
    axis.set(xticks=range(4), yticks=range(4), xticklabels=CLASS_NAMES,
             yticklabels=CLASS_NAMES, xlabel="Predicted class", ylabel="True class",
             title=f"{name}: {partition} nodes (n={metrics['nodes']:,})")
    figure.colorbar(image, ax=axis, label="Fraction within true class")
    figure.tight_layout()
    save_plot(figure, stem)


def project_embeddings(hidden):
    """Single predeclared t-SNE on all selection embeddings, with no label input."""
    settings = dict(n_components=2, perplexity=30.0, init="pca", learning_rate="auto",
                    max_iter=1000, random_state=42, method="barnes_hut", n_jobs=1)
    coordinates = TSNE(**settings).fit_transform(hidden)
    if not np.isfinite(coordinates).all():
        raise RuntimeError("Nonfinite t-SNE coordinates")
    return coordinates, settings


def plot_embeddings(coordinates, targets, stem):
    figure, axis = plt.subplots(figsize=(7, 6))
    colors = ("#0072B2", "#E69F00", "#009E73", "#CC79A7")
    for i, name in enumerate(CLASS_NAMES):
        selected = targets == i
        axis.scatter(*coordinates[selected].T, s=9, alpha=.65, color=colors[i],
                     label=f"{name} (n={selected.sum()})", edgecolors="none")
    axis.set(title="GCN hidden embeddings: selection nodes, t-SNE seed 42",
             xlabel="t-SNE dimension 1", ylabel="t-SNE dimension 2")
    axis.legend(markerscale=2)
    figure.tight_layout()
    save_plot(figure, stem)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_evaluation(dataset, mlp_checkpoint, gcn_checkpoint, output_dir,
                   device="auto", warmup=20, repeats=100):
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Choose a fresh output directory")
    device = configure_reproducibility(TrainConfig(device=device))
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    per_class = []
    comparison = []
    raw = {}
    hidden = None
    for name, checkpoint in (("MLP", mlp_checkpoint), ("GCN", gcn_checkpoint)):
        result, probabilities, representation, samples = evaluate_checkpoint(
            dataset, checkpoint, name, device, warmup, repeats,
        )
        results.append(result)
        raw[name + "_probabilities"] = probabilities
        raw[name + "_latency_ms"] = np.asarray(samples)
        plot_confusion(result["metrics"], name, output_dir / f"{name.lower()}_confusion_selection")
        per_class.extend({"model": name, **row} for row in result["metrics"]["per_class"])
        comparison.append({"model": name, "best_epoch": result["checkpoint_epoch"],
                           **{k: result["metrics"][k] for k in
                              ("accuracy", "macro_precision", "macro_recall", "macro_f1", "macro_ovr_auroc")},
                           **result["resources"]})
        if representation is not None:
            hidden = representation
        print(f"{name}: accuracy={result['metrics']['accuracy']:.6f}, "
              f"macro-F1={result['metrics']['macro_f1']:.6f}, "
              f"macro-OvR AUROC={result['metrics']['macro_ovr_auroc']:.6f}", flush=True)
    coordinates, settings = project_embeddings(hidden)
    targets = dataset.targets("selection").numpy()
    plot_embeddings(coordinates, targets, output_dir / "gcn_tsne_selection")
    raw.update(gcn_hidden=hidden, tsne_coordinates=coordinates,
               selection_node_ids=np.flatnonzero(dataset.masks["selection"].numpy()))
    np.savez_compressed(output_dir / "node_outputs.npz", **raw)
    report = {
        "partition": "selection", "reserved_partitions_evaluated": [],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "versions": {"torch": str(torch.__version__), "pyg": torch_geometric.__version__,
                     "sklearn": sklearn.__version__, "numpy": np.__version__},
        "data_sha256": dataset.provenance["sha256"],
        "selection_mask_sha256": dataset.audit["splits"]["selection"]["mask_sha256"],
        "source": {**source_record(), "evaluate_py_sha256": sha256_file(Path(__file__))},
        "latency_scope": "full 22470-node forward, inputs resident, eval/inference_mode; no transfers, softmax, metrics or plots; synchronized per call",
        "memory_scope": "fresh per-model CUDA allocator peak from model/input setup through selection metrics; excludes subsequent embedding extraction; not total device usage",
        "tsne": {"settings": settings, "nodes": len(targets), "input_dimensions": hidden.shape[1],
                 "representation": "ReLU(conv1) in eval mode", "labels_used_for": "plot colours only"},
        "models": results,
    }
    (output_dir / "selection_metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_csv(output_dir / "comparison_selection.csv", comparison)
    write_csv(output_dir / "per_class_selection.csv", per_class)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--mlp-checkpoint", type=Path, default=PROJECT / "outputs/mlp_seed42/best.pt")
    parser.add_argument("--gcn-checkpoint", type=Path, default=PROJECT / "outputs/gcn_seed42/best.pt")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "outputs/evaluation_selection_seed42")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda:0"), default="auto")
    args = parser.parse_args()
    run_evaluation(load_facebook(args.data_root), args.mlp_checkpoint,
                   args.gcn_checkpoint, args.output_dir, args.device)


if __name__ == "__main__":
    main()
