"""Prediction, evaluation, and result exports for Facebook page classification.

Restore a trained MLP or GCN, evaluate a selected split, and export the results.
Use the same facebook.npz archive as training; the checkpoint supplies the
model configuration and split seed needed to recreate the training split.

References:
    Model architecture and operator attribution are documented in modules.py.
    Course-linked dataset: https://github.com/gayanku/facebook_page-page_dataset
    Original node labels used to verify the archive's category ordering:
    https://github.com/gayanku/facebook_page-page_dataset/blob/main/facebook_large/musae_facebook_target.csv

AI assistance:
    OpenAI Codex assisted in generating, revising and debugging this code.
    The README section "Artificial Intelligence Usage Disclosure" documents
    the scope of assistance and the data audits, smoke runs, complete training
    runs and saved-output checks used to verify the implementation.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch_geometric.data import Data

from dataset import load_facebook_page_page
from train import create_model, forward_model, load_checkpoint, resolve_device


# The modified Gayanku archive encodes page_type categories alphabetically.
# Verified against the original musae_facebook_target.csv records.
PAGE_CLASSES = {
    "0": "Company",
    "1": "Government",
    "2": "Politician",
    "3": "TV show",
}


def load_trained_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    """Restore a model and its metadata from a training checkpoint."""

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint was not found: {checkpoint_path}"
        )

    checkpoint = load_checkpoint(checkpoint_path, device)

    required = {"model_config", "seed", "class_values"}
    missing = required.difference(checkpoint)

    if missing:
        raise ValueError(
            f"Checkpoint is missing metadata: {sorted(missing)}"
        )

    model = create_model(
        **checkpoint["model_config"]
    ).to(device)

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    return model, checkpoint


def predict_scores(
    model: nn.Module,
    data: Data,
    model_name: str,
) -> torch.Tensor:
    """Return class scores using model.eval() and torch.no_grad()."""

    model.eval()

    with torch.no_grad():
        if model_name == "mlp":
            outputs = model(data.x)
        elif model_name == "gcn":
            outputs = model(
                data.x,
                data.edge_index,
            )
        else:
            raise ValueError(
                f"Unknown model name: {model_name!r}"
            )

    if not bool(torch.isfinite(outputs).all()):
        raise ValueError(
            "Model predictions contain NaN or infinite values"
        )

    return outputs


def measure_inference(
    model: nn.Module,
    data: Data,
    model_name: str,
    warmup_runs: int = 5,
    repeats: int = 30,
) -> dict[str, Any]:
    """Time full-graph forward passes with inputs already on the device.

    Loading, device transfer, metrics, confidence analysis, and plotting are
    excluded. Per-node time is an amortized figure, not isolated-node latency.
    """

    if warmup_runs < 0:
        raise ValueError("timing-warmup cannot be negative")
    if repeats < 1:
        raise ValueError("timing-runs must be at least 1")

    device = data.x.device
    use_cuda = device.type == "cuda"
    model.eval()
    baseline_memory = None
    peak_memory = None
    extra_memory = None
    timings_ms = []

    print(f"> Timing full-graph inference: {warmup_runs} warmups, {repeats} runs")
    with torch.no_grad():
        for _ in range(warmup_runs):
            outputs = forward_model(model, data, model_name)
            if use_cuda:
                torch.cuda.synchronize(device)
            del outputs

        if use_cuda:
            torch.cuda.synchronize(device)
            baseline_memory = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)

        for _ in range(repeats):
            if use_cuda:
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            outputs = forward_model(model, data, model_name)
            if use_cuda:
                torch.cuda.synchronize(device)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            timings_ms.append(elapsed_ms)
            del outputs

        if use_cuda:
            peak_memory = torch.cuda.max_memory_allocated(device)
            extra_memory = max(0, peak_memory - baseline_memory)

    mean_ms = float(np.mean(timings_ms))
    memory_scale = 1024.0 ** 2
    device_name = (
        torch.cuda.get_device_name(device)
        if use_cuda else platform.processor() or platform.machine()
    )
    return {
        "scope": "full_graph_forward_only",
        "device": str(device),
        "device_name": device_name,
        "torch_version": str(torch.__version__),
        "torch_cpu_threads": torch.get_num_threads(),
        "num_nodes": int(data.num_nodes),
        "num_directed_edge_entries": int(data.edge_index.shape[1]),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "warmup_runs": warmup_runs,
        "timed_runs": repeats,
        "mean_ms": mean_ms,
        "median_ms": float(np.median(timings_ms)),
        "std_ms": float(np.std(timings_ms)),
        "min_ms": float(min(timings_ms)),
        "max_ms": float(max(timings_ms)),
        "timings_ms": timings_ms,
        "amortized_mean_microseconds_per_node": mean_ms * 1000.0 / data.num_nodes,
        "cuda_baseline_allocated_mib": (
            baseline_memory / memory_scale if use_cuda else None
        ),
        "cuda_peak_allocated_mib": (
            peak_memory / memory_scale if use_cuda else None
        ),
        "cuda_peak_increment_mib": (
            extra_memory / memory_scale if use_cuda else None
        ),
        "memory_note": (
            "PyTorch peak allocated tensor memory during timed inference, "
            "including model and input tensors; excludes CUDA context and "
            "other processes. This is not training peak memory."
            if use_cuda else "GPU memory is not applicable to this CPU run."
        ),
    }


def get_node_embeddings(
    model: nn.Module,
    data: Data,
    model_name: str,
) -> torch.Tensor:
    """Return first-layer hidden features after ReLU, before dropout."""

    model.eval()
    with torch.no_grad():
        if model_name == "mlp":
            embeddings = model.linear1(data.x)
        elif model_name == "gcn":
            # Use the full graph so every node keeps its original neighbours.
            embeddings = model.conv1(data.x, data.edge_index)
        else:
            raise ValueError(f"Unknown model name: {model_name!r}")

        embeddings = model.relu(embeddings)

    if not bool(torch.isfinite(embeddings).all()):
        raise ValueError("Hidden features contain NaN or infinite values")

    return embeddings.detach().cpu()


def save_embedding_plot(
    output_dir: Path,
    embeddings: torch.Tensor,
    outputs: torch.Tensor,
    data: Data,
    mask: torch.Tensor,
    model_name: str,
    seed: int,
    split_name: str,
    max_nodes: int = 2000,
) -> None:
    """Project learned hidden features to 2D, then colour by true category.

    Sampling and t-SNE use no labels. Separate model projections have arbitrary
    axes and should be interpreted as qualitative views of local structure.
    """

    if max_nodes < 3:
        raise ValueError("embedding-max-nodes must be at least 3")

    node_indices = mask.detach().cpu().nonzero(as_tuple=False).flatten().numpy()
    available_nodes = len(node_indices)
    if available_nodes < 3:
        print("Skipped t-SNE: at least three selected pages are needed")
        return

    # A label-independent sample keeps the same pages across both models.
    if available_nodes > max_nodes:
        rng = np.random.default_rng(seed)
        node_indices = np.sort(
            rng.choice(node_indices, size=max_nodes, replace=False)
        )

    features = embeddings.detach().cpu()[node_indices].numpy()
    perplexity = min(30.0, float(len(node_indices) - 1))
    initialization = "pca" if features.shape[1] >= 2 else "random"

    from sklearn.manifold import TSNE

    print(f"> Creating t-SNE plot for {len(node_indices)} {split_name} pages")
    projection = TSNE(
        n_components=2,
        perplexity=perplexity,
        init=initialization,
        learning_rate="auto",
        max_iter=1000,
        random_state=seed,
        n_jobs=1,
    )
    coordinates = projection.fit_transform(features)
    if not bool(np.isfinite(coordinates).all()):
        raise ValueError("t-SNE produced non-finite coordinates")

    # Ground-truth labels are read only after calculating point positions.
    labels = data.y.detach().cpu()[node_indices].numpy()
    predictions = outputs.detach().cpu()[node_indices].argmax(dim=1).numpy()
    class_names = [
        PAGE_CLASSES.get(str(value), str(value))
        for value in data.class_values
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "embedding_tsne.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["node_index", "page_name", "true_class", "predicted_class", "tsne_1", "tsne_2"]
        )
        for row, node_index in enumerate(node_indices):
            writer.writerow(
                [
                    int(node_index),
                    data.page_name[int(node_index)],
                    class_names[int(labels[row])],
                    class_names[int(predictions[row])],
                    float(coordinates[row, 0]),
                    float(coordinates[row, 1]),
                ]
            )

    settings = {
        "model": model_name,
        "seed": seed,
        "split": split_name,
        "representation": "first hidden layer after ReLU, before dropout",
        "hidden_dimensions": int(features.shape[1]),
        "available_nodes": available_nodes,
        "plotted_nodes": len(node_indices),
        "maximum_nodes": max_nodes,
        "sampling": "uniform without replacement, seeded, independent of labels",
        "perplexity": perplexity,
        "initialization": initialization,
        "learning_rate": "auto",
        "max_iter": 1000,
        "n_jobs": 1,
        "interpretation": (
            "Qualitative local structure only; axes and global distances "
            "are not directly comparable between separate projections"
        ),
    }
    with (output_dir / "embedding_tsne_settings.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(settings, handle, indent=2, allow_nan=False)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(10, 6))
    for class_index, class_name in enumerate(class_names):
        selected = labels == class_index
        axis.scatter(
            coordinates[selected, 0],
            coordinates[selected, 1],
            color=f"C{class_index}",
            s=10,
            alpha=0.65,
            linewidths=0,
            label=f"{class_name} (n={int(selected.sum())})",
        )

    axis.set(
        title=(
            f"{model_name.upper()} hidden representations: t-SNE\n"
            f"{len(node_indices)}/{available_nodes} {split_name} pages | seed {seed}"
        ),
        xlabel="t-SNE dimension 1",
        ylabel="t-SNE dimension 2",
    )
    axis.legend(
        title="True page category", markerscale=2,
        loc="upper left", bbox_to_anchor=(1.02, 1.0),
    )
    figure.text(
        0.5, 0.015,
        "Colours show true categories. t-SNE axes are arbitrary.",
        ha="center", fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.035, 1, 1))
    figure.savefig(output_dir / "embedding_tsne.png", dpi=160)
    plt.close(figure)


def calculate_metrics(
    outputs: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, Any]:
    """Calculate accuracy, macro-F1, AUROC, and a confusion matrix."""

    if outputs.ndim != 2 or labels.ndim != 1:
        raise ValueError(
            "Expected outputs=[nodes, classes] and labels=[nodes]"
        )

    if mask.dtype != torch.bool or mask.shape != labels.shape:
        raise ValueError(
            "The split mask must be Boolean with one value per node"
        )

    if outputs.shape[0] != labels.numel() or not bool(mask.any()):
        raise ValueError(
            "Predictions must match labels and the split cannot be empty"
        )

    selected_outputs = outputs[mask].detach().cpu()
    y_true = labels[mask].detach().cpu().numpy()
    probabilities = selected_outputs.softmax(dim=1).numpy()
    y_pred = probabilities.argmax(axis=1)
    class_ids = list(range(outputs.shape[1]))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=class_ids,
        zero_division=0,
    )

    # Evaluate each class against all remaining classes.
    per_class_auroc: list[float | None] = []

    for class_index in class_ids:
        binary_labels = (y_true == class_index).astype(np.int64)

        if np.unique(binary_labels).size < 2:
            per_class_auroc.append(None)
        else:
            class_auroc = roc_auc_score(
                binary_labels,
                probabilities[:, class_index],
            )
            per_class_auroc.append(float(class_auroc))

    # A complete macro AUROC needs positive and negative examples per class.
    macro_auroc = None

    if all(score is not None for score in per_class_auroc):
        macro_auroc = float(np.mean(per_class_auroc))

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=class_ids,
    )

    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized_matrix = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=np.float64),
        where=row_totals != 0,
    )

    return {
        "count": int(y_true.size),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1.mean()),
        "macro_auroc_ovr": macro_auroc,
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "per_class_support": support.tolist(),
        "per_class_auroc_ovr": per_class_auroc,
        "confusion_matrix": matrix.tolist(),
        "normalized_confusion_matrix": normalized_matrix.tolist(),
    }


def evaluate_confidence(
    outputs: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    threshold: float,
) -> dict[str, Any]:
    """Measure predictions accepted by a maximum-softmax threshold.

    The score measures model confidence; it is not a calibrated probability
    that the prediction is correct. Rejected pages are flagged for review.
    """

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("The confidence threshold must be between 0 and 1")
    if mask.dtype != torch.bool or mask.shape != labels.shape:
        raise ValueError("Expected a Boolean mask with one value per node")
    if not bool(mask.any()):
        raise ValueError("The confidence evaluation split cannot be empty")

    probabilities = outputs[mask].detach().cpu().softmax(dim=1)
    confidences, predictions = probabilities.max(dim=1)
    correct = predictions == labels[mask].detach().cpu()
    accepted = confidences >= threshold

    total = int(correct.numel())
    accepted_count = int(accepted.sum().item())
    accepted_correct = int(correct[accepted].sum().item())
    accepted_errors = accepted_count - accepted_correct

    # Accuracy is undefined when no predictions are accepted.
    accepted_accuracy = None
    accepted_error_rate = None
    if accepted_count > 0:
        accepted_accuracy = accepted_correct / accepted_count
        accepted_error_rate = accepted_errors / accepted_count

    return {
        "threshold": float(threshold),
        "total_count": total,
        "accepted_count": accepted_count,
        "rejected_count": total - accepted_count,
        "coverage": accepted_count / total,
        "accepted_correct": accepted_correct,
        "accepted_errors": accepted_errors,
        "accepted_accuracy": accepted_accuracy,
        "accepted_error_rate": accepted_error_rate,
    }


def select_confidence_threshold(
    outputs: torch.Tensor,
    labels: torch.Tensor,
    validation_mask: torch.Tensor,
    min_coverage: float = 0.5,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Choose a threshold using validation labels only.

    Maximise accepted accuracy while meeting the minimum validation coverage.
    Break ties by greater coverage, then by a lower threshold. The minimum
    coverage is an experiment setting, not an assignment requirement.
    """

    if not 0.0 < min_coverage <= 1.0:
        raise ValueError("Minimum validation coverage must be above 0 and at most 1")

    curve = []
    for step in range(101):
        result = evaluate_confidence(
            outputs,
            labels,
            validation_mask,
            threshold=step / 100.0,
        )
        curve.append(result)

    eligible = [
        result for result in curve
        if result["coverage"] >= min_coverage
    ]
    best = max(
        eligible,
        key=lambda result: (
            result["accepted_accuracy"],
            result["coverage"],
            -result["threshold"],
        ),
    )

    rule = {
        "confidence_measure": "maximum_softmax_score",
        "selection_split": "val",
        "minimum_validation_coverage": float(min_coverage),
        "selection_rule": (
            "Highest accepted validation accuracy; ties prefer greater "
            "coverage, then a lower threshold"
        ),
        "threshold": best["threshold"],
        "validation_results": best,
    }
    return rule, curve


def save_confidence_analysis(
    output_dir: Path,
    rule: dict[str, Any],
    curve: list[dict[str, Any]],
) -> None:
    """Save the validation threshold search separately from test results."""

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "confidence_rule.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(rule, handle, indent=2, allow_nan=False)

    with (output_dir / "validation_confidence_curve.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve[0]))
        writer.writeheader()
        writer.writerows(curve)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = [
        result for result in curve
        if result["accepted_accuracy"] is not None
    ]
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(
        [result["coverage"] for result in points],
        [result["accepted_accuracy"] for result in points],
        marker=".",
        label="Validation thresholds",
    )
    selected = rule["validation_results"]
    axis.scatter(
        selected["coverage"],
        selected["accepted_accuracy"],
        color="red",
        zorder=3,
        label=f"Selected threshold: {rule['threshold']:.2f}",
    )
    axis.set(
        xlabel="Fraction of validation pages accepted (coverage)",
        ylabel="Accuracy among accepted validation pages",
        title="Validation accuracy and coverage",
        xlim=(0, 1.02),
        ylim=(0, 1.02),
    )
    axis.legend()
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_dir / "validation_confidence_curve.png", dpi=160)
    plt.close(figure)


def save_confusion_matrix(
    output_dir: Path,
    metrics: dict[str, Any],
    class_names: list[str],
) -> None:
    """Save matrix counts, normalized values, and a labelled plot."""

    matrix = np.asarray(metrics["confusion_matrix"])
    normalized_matrix = np.asarray(
        metrics["normalized_confusion_matrix"]
    )
    expected_shape = (len(class_names), len(class_names))

    if matrix.shape != expected_shape or normalized_matrix.shape != expected_shape:
        raise ValueError(
            "Confusion matrices must be square and match the class names"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    for filename, values in (
        ("confusion_matrix.csv", matrix),
        ("confusion_matrix_normalized.csv", normalized_matrix),
    ):
        with (output_dir / filename).open(
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["true_class / predicted_class", *class_names]
            )

            for class_name, row in zip(class_names, values):
                writer.writerow([class_name, *row.tolist()])

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 6))
    plot = axis.imshow(
        normalized_matrix,
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
    )
    figure.colorbar(
        plot,
        ax=axis,
        label="Proportion within true class",
    )

    model_name = str(metrics.get("model", "classifier")).upper()
    split_name = metrics.get("split", "selected split")

    axis.set(
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted category",
        ylabel="True category",
        title=f"{model_name} confusion matrix ({split_name})",
    )
    plt.setp(
        axis.get_xticklabels(),
        rotation=30,
        ha="right",
    )

    for row in range(len(class_names)):
        for column in range(len(class_names)):
            value = normalized_matrix[row, column]
            axis.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value > 0.5 else "black",
            )

    figure.tight_layout()
    figure.savefig(
        output_dir / "confusion_matrix.png",
        dpi=160,
    )
    plt.close(figure)


def save_prediction_outputs(
    output_dir: Path,
    data: Data,
    outputs: torch.Tensor,
    mask: torch.Tensor,
    metrics: dict[str, Any],
) -> None:
    """Save metrics, predictions, failure candidates, and confusion matrices."""

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    class_values = [
        str(value)
        for value in data.class_values
    ]
    class_names = [
        PAGE_CLASSES.get(value, value)
        for value in class_values
    ]

    report = {
        **metrics,
        "class_values": class_values,
        "class_names": class_names,
    }

    with (output_dir / "metrics.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            report,
            handle,
            indent=2,
            allow_nan=False,
        )

    node_indices = (
        mask.nonzero(as_tuple=False)
        .flatten()
        .cpu()
        .tolist()
    )
    probabilities = (
        outputs[mask]
        .detach()
        .cpu()
        .softmax(dim=1)
    )
    confidences, predictions = probabilities.max(dim=1)
    labels = data.y[mask].detach().cpu()

    # Each mutual connection is stored once per direction in this dataset.
    degrees = torch.bincount(
        data.edge_index[0].detach().cpu(),
        minlength=data.num_nodes,
    )

    rows = []

    for row_index, node_index in enumerate(node_indices):
        actual = int(labels[row_index])
        predicted = int(predictions[row_index])

        rows.append(
            {
                "node_index": node_index,
                "page_name": data.page_name[node_index],
                "true_label": actual,
                "true_class": class_names[actual],
                "predicted_label": predicted,
                "predicted_class": class_names[predicted],
                "confidence": float(confidences[row_index]),
                "decision": (
                    "accept"
                    if float(confidences[row_index]) >= metrics["confidence_rule"]["threshold"]
                    else "review"
                ),
                "correct": predicted == actual,
                "node_degree": int(degrees[node_index]),
            }
        )

    failures = [
        row for row in rows
        if not row["correct"]
    ]

    # Candidate cases only: their causes still need written analysis.
    failure_examples = sorted(
        failures,
        key=lambda row: row["confidence"],
        reverse=True,
    )[:5]

    fieldnames = (
        "node_index",
        "page_name",
        "true_label",
        "true_class",
        "predicted_label",
        "predicted_class",
        "confidence",
        "decision",
        "correct",
        "node_degree",
    )

    for filename, selected_rows in (
        ("predictions.csv", rows),
        ("incorrect_predictions.csv", failures),
        ("failure_examples.csv", failure_examples),
    ):
        with (output_dir / filename).open(
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
            )
            writer.writeheader()
            writer.writerows(selected_rows)

    save_confusion_matrix(
        output_dir,
        report,
        class_names,
    )


def run_prediction(args: argparse.Namespace) -> None:
    """Load the model, recreate its seeded split, and evaluate predictions."""

    device = resolve_device(args.device)
    model, checkpoint = load_trained_model(
        args.checkpoint,
        device,
    )
    config = checkpoint["model_config"]
    model_name = config["model_name"]
    seed = int(checkpoint["seed"])
    # Older checkpoints used the training seed for their data split too.
    split_seed = int(checkpoint.get("split_seed", seed))

    print(f"> Device: {device}")
    print(
        f"> Loading {model_name.upper()} checkpoint (seed {seed})"
    )
    print(f"> Data split seed: {split_seed}")

    data = load_facebook_page_page(
        args.data_path,
        seed=split_seed,
    )

    if int(data.x.shape[1]) != config["in_channels"]:
        raise ValueError(
            "Dataset feature count differs from the trained model"
        )

    if int(data.num_classes) != config["out_channels"]:
        raise ValueError(
            "Dataset class count differs from the trained model"
        )

    if list(data.class_values) != list(checkpoint["class_values"]):
        raise ValueError(
            "Dataset class order differs from the training checkpoint"
        )

    data = data.to(device)

    if args.split == "all":
        mask = torch.ones_like(data.y, dtype=torch.bool)
    else:
        mask = getattr(data, f"{args.split}_mask")

    inference_profile = measure_inference(
        model,
        data,
        model_name,
        warmup_runs=args.timing_warmup,
        repeats=args.timing_runs,
    )
    outputs = predict_scores(model, data, model_name)
    metrics = calculate_metrics(outputs, data.y, mask)
    metrics.update(
        {
            "model": model_name,
            "seed": seed,
            "split_seed": split_seed,
            "split_ratios": list(data.split_ratios),
            "split": args.split,
            "best_epoch": checkpoint.get("best_epoch"),
            "checkpoint": str(args.checkpoint.resolve()),
            "inference_profile": inference_profile,
        }
    )

    # Choose the rule on validation pages, then apply it unchanged below.
    rule, confidence_curve = select_confidence_threshold(
        outputs,
        data.y,
        data.val_mask,
        min_coverage=args.min_validation_coverage,
    )
    metrics["confidence_rule"] = rule
    metrics["selective_prediction"] = evaluate_confidence(
        outputs,
        data.y,
        mask,
        threshold=rule["threshold"],
    )

    run_dir = (
        args.output_dir
        / f"{model_name}_seed{seed}_{args.split}"
    )
    save_prediction_outputs(
        run_dir,
        data,
        outputs,
        mask,
        metrics,
    )

    with (run_dir / "inference_profile.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(inference_profile, handle, indent=2, allow_nan=False)

    save_confidence_analysis(run_dir, rule, confidence_curve)

    if not args.skip_embedding_plot:
        embeddings = get_node_embeddings(model, data, model_name)
        save_embedding_plot(
            run_dir,
            embeddings,
            outputs,
            data,
            mask,
            model_name,
            seed,
            args.split,
            max_nodes=args.embedding_max_nodes,
        )

    auc = metrics["macro_auroc_ovr"]
    auc_text = "undefined" if auc is None else f"{auc:.4f}"

    print(f"Evaluated pages: {metrics['count']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro-F1: {metrics['macro_f1']:.4f}")
    print(f"Macro AUROC (one-vs-rest): {auc_text}")
    print(
        f"Full-graph inference: {inference_profile['mean_ms']:.3f} ms mean, "
        f"{inference_profile['median_ms']:.3f} ms median "
        f"({inference_profile['timed_runs']} timed runs)"
    )
    if inference_profile["cuda_peak_allocated_mib"] is None:
        print("Peak GPU memory: not applicable (CPU run)")
    else:
        print(
            "Peak PyTorch GPU memory during inference: "
            f"{inference_profile['cuda_peak_allocated_mib']:.2f} MiB"
        )
    selective = metrics["selective_prediction"]
    accepted_accuracy = selective["accepted_accuracy"]
    accepted_text = (
        "undefined" if accepted_accuracy is None
        else f"{accepted_accuracy:.4f}"
    )
    print(f"Confidence threshold (selected on validation): {rule['threshold']:.2f}")
    print(
        f"Accepted pages: {selective['accepted_count']}/{selective['total_count']} "
        f"(coverage {selective['coverage']:.2%})"
    )
    print(f"Accuracy among accepted pages: {accepted_text}")
    print(f"Pages flagged for review: {selective['rejected_count']}")
    print(f"Saved prediction outputs to: {run_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    """Read prediction settings from the command line."""

    parser = argparse.ArgumentParser(
        description="Evaluate a trained Facebook page classifier"
    )
    parser.add_argument(
        "--data-path",
        "--data",
        dest="data_path",
        type=Path,
        required=True,
        help="Path to facebook.npz",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the trained model checkpoint",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "test", "all"),
        default="test",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device selection: auto, cpu, cuda, or cuda:N",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/predictions"),
    )

    parser.add_argument(
        "--min-validation-coverage",
        type=float,
        default=0.5,
        help=(
            "Experiment setting: minimum fraction of validation predictions "
            "accepted when choosing the confidence threshold (default: 0.5)"
        ),
    )

    parser.add_argument(
        "--embedding-max-nodes",
        type=int,
        default=2000,
        help="Maximum pages in the seeded t-SNE sample (default: 2000)",
    )
    parser.add_argument(
        "--skip-embedding-plot",
        action="store_true",
        help="Skip t-SNE when only prediction metrics are needed",
    )

    parser.add_argument(
        "--timing-warmup",
        type=int,
        default=5,
        help="Untimed warmup forward passes (default: 5)",
    )
    parser.add_argument(
        "--timing-runs",
        type=int,
        default=30,
        help="Measured full-graph forward passes (default: 30)",
    )

    return parser.parse_args()


def main() -> None:
    """Run the prediction program."""

    args = parse_args()
    run_prediction(args)


if __name__ == "__main__":
    main()
