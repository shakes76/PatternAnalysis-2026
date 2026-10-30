from pathlib import Path
import json
import time
import resource
from sklearn.manifold import TSNE
import torch
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    roc_auc_score,
)
import argparse

from dataset import load_facebook
from modules import GCN, MLP


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
CLASS_NAMES = ["company", "government", "politician", "tvshow"]

def load_model(name, graph, device):
    model_class = GCN if name == "gcn" else MLP

    model = model_class(
        input_dim=graph.num_node_features,
        hidden_dim=64,
        num_classes=4
    ).to(device)

    checkpoint = OUTPUT_DIR / f"{name}_best.pt"
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True)
    )
    model.eval()
    return model


def predict(model, graph, name):
    with torch.no_grad():
        if name == "gcn":
            logits = model(graph.x, graph.edge_index)
        else:
            logits = model(graph.x)

        probabilities = torch.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1)

    return predictions, probabilities

def evaluate_classification(predictions, probabilities, graph, name):
    """Evaluate predictions on held-out test nodes."""

    mask = graph.test_mask

    y_true = graph.y[mask].cpu().numpy()
    y_pred = predictions[mask].cpu().numpy()
    y_prob = probabilities[mask].cpu().numpy()

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    auroc = roc_auc_score(
        y_true,
        y_prob,
        multi_class="ovr",
        average="macro",
        labels=np.arange(4)
    )

    print(f"\n{name.upper()} Classification Results")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print(f"Macro-AUROC: {auroc:.4f}")

    print(classification_report(
        y_true,
        y_pred,
        labels=range(4),
        target_names=CLASS_NAMES,
        zero_division=0
    ))

    matrix = confusion_matrix(y_true, y_pred, labels=range(4))

    fig, ax = plt.subplots(figsize=(7, 6))
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=CLASS_NAMES
    )
    display.plot(ax=ax, cmap="Blues", xticks_rotation=30)
    ax.set_title(f"{name.upper()} Test Confusion Matrix")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{name}_confusion.png", dpi=150)
    plt.close(fig)

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "auroc": float(auroc),
    }


def reliability_analysis(probabilities, labels, name):
    """Measure confidence calibration on held-out test nodes."""

    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = (predictions == labels)

    # Divide prediction confidence into 10 equal-width bins.
    bins = np.linspace(0.0, 1.0, 11)
    bin_ids = np.clip(np.digitize(confidence, bins) - 1, 0, 9)

    mean_confidence = []
    empirical_accuracy = []
    ece = 0.0

    for i in range(10):
        selected = bin_ids == i
        count = int(selected.sum())

        if count == 0:
            continue

        avg_confidence = float(confidence[selected].mean())
        avg_accuracy = float(correct[selected].mean())

        mean_confidence.append(avg_confidence)
        empirical_accuracy.append(avg_accuracy)

        # Weight each bin's calibration gap by its sample count.
        ece += (count / len(confidence)) * abs(
            avg_accuracy - avg_confidence
        )

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.plot(
        [0, 1], [0, 1],
        "--", color="gray", label="Perfect calibration"
    )
    ax.plot(
        mean_confidence,
        empirical_accuracy,
        marker="o",
        label=name.upper()
    )

    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Observed accuracy")
    ax.set_title(f"{name.upper()} Reliability (ECE={ece:.4f})")
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / f"{name}_reliability.png",
        dpi=150
    )
    plt.close(fig)

    return float(ece)


def rejection_analysis(probabilities, labels, name, threshold=None):
    """Calculate accuracy and coverage for confidence thresholds."""

    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predictions == labels

    rows = []

    for t in np.linspace(0.0, 1.0, 201):
        accepted = confidence >= t

        coverage = float(accepted.mean())
        accepted_accuracy = (
            float(correct[accepted].mean())
            if accepted.any() else None
        )

        rows.append({
            "threshold": float(t),
            "coverage": coverage,
            "accuracy": accepted_accuracy,
        })

    if threshold is None:
        # Choose the highest-coverage validation threshold
        # that achieves at least 97% accepted accuracy.
        eligible = [
            row for row in rows
            if row["accuracy"] is not None
            and row["accuracy"] >= 0.97
            and row["coverage"] > 0
        ]

        if not eligible:
            raise ValueError(
                f"{name}: No validation threshold achieves 97% accuracy"
            )

        threshold = max(eligible, key=lambda row: row["coverage"])[
            "threshold"
        ]

    accepted = confidence >= threshold

    result = {
        "threshold": float(threshold),
        "coverage": float(accepted.mean()),
        "accepted_accuracy": (
            float(correct[accepted].mean())
            if accepted.any() else None
        ),
        "referred_nodes": int((~accepted).sum()),
        "total_nodes": len(labels),
    }

    return result, rows


def plot_rejection(rows, name):
    """Plot the trade-off between coverage and accepted accuracy."""

    valid = [r for r in rows if r["accuracy"] is not None]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        [r["coverage"] for r in valid],
        [r["accuracy"] for r in valid],
        marker=".",
        markersize=3
    )

    ax.set_xlabel("Coverage (fraction automatically classified)")
    ax.set_ylabel("Accuracy among accepted predictions")
    ax.set_title(f"{name.upper()} Coverage–Accuracy Trade-off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{name}_rejection.png", dpi=150)
    plt.close(fig)



def analyse_failures(predictions, probabilities, graph, name):
    """Inspect five representative misclassified test nodes."""

    test_indices = torch.where(graph.test_mask)[0].cpu().numpy()
    labels = graph.y[graph.test_mask].cpu().numpy()
    predicted = predictions[graph.test_mask].cpu().numpy()
    confidence = probabilities[graph.test_mask].max(dim=1).values.cpu().numpy()

    errors = np.where(predicted != labels)[0]
    ranked = errors[np.argsort(-confidence[errors])]

    # Include confident errors and lower-confidence errors.
    selected = list(ranked[:3]) + list(ranked[-2:])
    selected = list(dict.fromkeys(selected))

    edges = graph.edge_index.cpu().numpy()
    all_labels = graph.y.cpu().numpy()

    cases = []

    for pos in selected:
        node = int(test_indices[pos])

        neighbours = np.concatenate([
            edges[1, edges[0] == node],
            edges[0, edges[1] == node]
        ])
        neighbours = np.unique(neighbours)

        counts = np.bincount(
            all_labels[neighbours], minlength=4
        ) if len(neighbours) else np.zeros(4, dtype=int)

        case = {
            "node_index": node,
            "true_class": CLASS_NAMES[int(labels[pos])],
            "predicted_class": CLASS_NAMES[int(predicted[pos])],
            "confidence": round(float(confidence[pos]), 4),
            "degree": int(len(neighbours)),
            "neighbour_class_counts": {
                class_name: int(count)
                for class_name, count in zip(CLASS_NAMES, counts)
            }
        }
        cases.append(case)

    with open(OUTPUT_DIR / f"{name}_failures.json", "w") as f:
        json.dump(cases, f, indent=2)

    print(f"\n{name.upper()} Misclassification Examples")
    for case in cases:
        print(
            f"Node {case['node_index']}: "
            f"{case['true_class']} -> {case['predicted_class']} | "
            f"Confidence={case['confidence']:.4f} | "
            f"Neighbours={case['degree']}"
        )

    return cases


def plot_embeddings(model, graph):
    """Visualise the GCN hidden representations using t-SNE."""

    model.eval()

    with torch.no_grad():
        embeddings = model.conv1(
            graph.x, graph.edge_index
        )
        embeddings = model.relu(embeddings).cpu().numpy()

    rng = np.random.default_rng(42)
    sample = rng.choice(
        len(embeddings),
        size=min(2000, len(embeddings)),
        replace=False
    )

    projected = TSNE(
        n_components=2,
        perplexity=30,
        random_state=42,
        init="pca",
        learning_rate="auto"
    ).fit_transform(embeddings[sample])

    labels = graph.y.cpu().numpy()[sample]

    fig, ax = plt.subplots(figsize=(8, 6))

    for class_id, class_name in enumerate(CLASS_NAMES):
        selected = labels == class_id
        ax.scatter(
            projected[selected, 0],
            projected[selected, 1],
            s=9,
            alpha=0.7,
            label=class_name
        )

    ax.set_title("GCN Hidden Representations (t-SNE)")
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.legend()

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "gcn_tsne.png", dpi=150)
    plt.close(fig)


def profile_inference(model, graph, name, device):
    """Measure full-graph inference latency and memory."""

    def forward():
        if name == "gcn":
            return model(graph.x, graph.edge_index)
        return model(graph.x)

    def synchronise():
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

    model.eval()

    def allocated_mb():
        if device.type == "mps":
            return torch.mps.current_allocated_memory() / 1024**2
        if device.type == "cuda":
            return torch.cuda.memory_allocated(device) / 1024**2
        return None

    synchronise()
    baseline_mb = allocated_mb()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode():
        for _ in range(3):
            output = forward()
            synchronise()
            del output

        timings = []
        observed_allocations = []

        for _ in range(10):
            synchronise()
            start = time.perf_counter()

            output = forward()
            synchronise()

            timings.append(
                (time.perf_counter() - start) * 1000
            )

            # Observe memory while output is still alive.
            memory = allocated_mb()
            if memory is not None:
                observed_allocations.append(memory)

            del output

    synchronise()

    result = {
        "device": str(device),
        "parameters": sum(
            p.numel() for p in model.parameters()
        ),
        "mean_inference_ms": float(np.mean(timings)),
    }

    if device.type == "cuda":
        result["cuda_peak_allocated_mb"] = (
            torch.cuda.max_memory_allocated(device) / 1024**2
        )

    elif device.type == "mps":
        maximum = max(observed_allocations)

        result["mps_baseline_allocated_mb"] = baseline_mb
        result["mps_max_observed_allocated_mb"] = maximum
        result["mps_observed_increase_mb"] = (
            maximum - baseline_mb
        )
        result["mps_driver_allocated_mb"] = (
            torch.mps.driver_allocated_memory() / 1024**2
        )

    print(f"\n{name.upper()} Resource Profile")
    print(json.dumps(result, indent=2))

    return result





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        choices=["cpu", "mps", "cuda"],
        default="cpu",
        help="Device used for inference and evaluation",
    )
    parser.add_argument(
        "--model",
        choices=["mlp", "gcn", "both"],
        default="both",
        help="Select which model to evaluate",
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    if device.type == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is not available")

    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available")

    print(f"Evaluation device: {device}")
    graph = load_facebook(seed=42).to(device)
    

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prediction_cache = {}
    results = {}

    models = ("mlp", "gcn") if args.model == "both" else (args.model,)

    for name in models:
        model = load_model(name, graph, device)
        predictions, probabilities = predict(model, graph, name)

        prediction_cache[name] = {
            "predictions": predictions.detach().cpu(),
            "probabilities": probabilities.detach().cpu(),
        }

        # Evaluate classification performance on test nodes.
        metrics = evaluate_classification(
            predictions, probabilities, graph, name
        )

        test_probabilities = probabilities[
            graph.test_mask
        ].cpu().numpy()

        test_labels = graph.y[
            graph.test_mask
        ].cpu().numpy()

        # Measure confidence calibration.
        ece = reliability_analysis(
            test_probabilities, test_labels, name
        )
        print(f"{name.upper()} Expected Calibration Error: {ece:.4f}")

        # Select confidence threshold using validation nodes.
        val_probabilities = probabilities[
            graph.val_mask
        ].cpu().numpy()

        val_labels = graph.y[
            graph.val_mask
        ].cpu().numpy()

        val_result, _ = rejection_analysis(
            val_probabilities, val_labels, name
        )

        selected_threshold = val_result["threshold"]

        # Evaluate the selected threshold on held-out test nodes.
        test_result, test_rows = rejection_analysis(
            test_probabilities,
            test_labels,
            name,
            threshold=selected_threshold
        )

        plot_rejection(test_rows, name)

        print(
            f"{name.upper()} threshold={selected_threshold:.3f} | "
            f"coverage={test_result['coverage']:.1%} | "
            f"accepted accuracy={test_result['accepted_accuracy']:.1%} | "
            f"referred={test_result['referred_nodes']}"
        )

        # Analyse representative classification errors.
        failures = analyse_failures(
            predictions, probabilities, graph, name
        )

        # Visualise learned GCN representations.
        if name == "gcn":
            plot_embeddings(model, graph)

        # Measure inference speed and memory usage.
        profiling = profile_inference(
            model, graph, name, device
        )

        # Store results for the final report.
        results[name] = {
            **metrics,
            "ece": ece,
            "rejection": test_result,
            "failure_cases": failures,
            "profiling": profiling,
        }


    # Compare both models on the same representative GCN failures.
    if "mlp" in prediction_cache and "gcn" in prediction_cache:
        comparison = []

        for case in results["gcn"]["failure_cases"]:
            node = case["node_index"]

            row = {
                **case,
                "models": {},
            }

            for model_name in ("mlp", "gcn"):
                cached = prediction_cache[model_name]
                predicted_id = int(cached["predictions"][node])
                confidence = float(
                    cached["probabilities"][node].max()
                )

                row["models"][model_name] = {
                    "predicted_class": CLASS_NAMES[predicted_id],
                    "confidence": round(confidence, 4),
                    "correct": (
                        predicted_id == int(graph.y[node].item())
                    ),
                }

            comparison.append(row)

        with open(
            OUTPUT_DIR / "model_failure_comparison.json", "w"
        ) as f:
            json.dump(comparison, f, indent=2)

        print("\nGCN versus MLP Failure Comparison")

        for row in comparison:
            mlp = row["models"]["mlp"]
            gcn = row["models"]["gcn"]

            print(
                f"Node {row['node_index']}: "
                f"True={row['true_class']} | "
                f"MLP={mlp['predicted_class']} "
                f"({mlp['confidence']:.4f}) | "
                f"GCN={gcn['predicted_class']} "
                f"({gcn['confidence']:.4f})"
            )


    with open(OUTPUT_DIR / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nEvaluation complete. Results saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()