"""Load a saved checkpoint, run inference, and write evidence for manual analysis."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from dataset import CLASS_NAMES, DEFAULT_PATH, PROJECT_DIR, make_pyg_graph, sha256
from evaluation import classification_metrics, reliability, selective_metrics
from modules import make_model, normalized_adjacency


def plot_diagnostics(logits, labels, temperature, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    truth = labels.cpu().numpy()
    for scale, title in ((1., "Raw"), (temperature, "Temperature scaled")):
        probabilities = (logits.cpu() / scale).softmax(dim=1).numpy()
        _, bins = reliability(probabilities, truth)
        populated = [item for item in bins if item["count"]]
        axes[0].plot([item["confidence"] for item in populated],
                     [item["accuracy"] for item in populated], marker="o", label=title)
        confidence = probabilities.max(axis=1)
        correct = probabilities.argmax(axis=1) == truth
        order = np.argsort(-confidence, kind="stable")
        accepted = np.arange(1, len(truth)+1)
        risk = np.cumsum(~correct[order]) / accepted
        axes[1].plot(accepted / len(truth), risk, label=title)
    axes[0].plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
    axes[0].set(xlabel="Mean predicted confidence", ylabel="Empirical accuracy", xlim=(0, 1), ylim=(0, 1))
    axes[1].set(xlabel="Accepted fraction (coverage)", ylabel="Accepted-case error rate", xlim=(0, 1))
    for axis in axes:
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def plot_embedding(embedding, labels, phase, destination, max_points=2000):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    random = np.random.default_rng(42)
    truth = labels.cpu().numpy()
    selected = np.concatenate([
        random.permutation(np.flatnonzero(truth == label))[:max_points//len(CLASS_NAMES)]
        for label in range(len(CLASS_NAMES))
    ])
    coordinates = TSNE(n_components=2, perplexity=min(30., (len(selected)-1)/3),
                       init="pca", learning_rate="auto", random_state=42).fit_transform(
                           embedding.cpu().numpy()[selected])
    figure, axis = plt.subplots(figsize=(6.3, 4.6))
    for label, name in enumerate(CLASS_NAMES):
        group = truth[selected] == label
        axis.scatter(coordinates[group, 0], coordinates[group, 1], s=5, alpha=0.65, label=name)
    axis.set(title=f"{phase}: t-SNE of learned embeddings", xlabel="t-SNE 1", ylabel="t-SNE 2")
    axis.legend(markerscale=2, fontsize=8)
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def error_cases(graph, logits, evaluation_ids, temperature, count=5):
    """Return diverse inspection candidates; do not infer unverified root causes."""
    probabilities = (logits.cpu() / temperature).softmax(dim=1).numpy()
    raw_probabilities = logits.cpu().softmax(dim=1).numpy()
    prediction = probabilities.argmax(axis=1)
    truth = graph.y.numpy()
    ids = evaluation_ids.cpu().numpy()
    wrong = ids[prediction[ids] != truth[ids]]
    if not len(wrong):
        return []
    confidence = probabilities.max(axis=1)
    margin = np.sort(probabilities, axis=1)[:, -1] - np.sort(probabilities, axis=1)[:, -2]
    edges = graph.edge_index.numpy()
    degree = np.bincount(edges[0], minlength=len(truth))
    ordered = np.concatenate((
        wrong[np.argsort(-confidence[wrong])[:2]],
        wrong[np.argsort(degree[wrong])[:1]],
        wrong[np.argsort(margin[wrong])[:1]],
        wrong[np.argsort(-degree[wrong])[:1]],
        wrong[np.argsort(-confidence[wrong])],
    ))
    chosen = list(dict.fromkeys(int(node) for node in ordered))[:count]
    results = []
    for node in chosen:
        neighbours = edges[1, edges[0] == node]
        training_neighbours = neighbours[graph.train_mask.numpy()[neighbours]]
        results.append({
            "node_id": node, "true_class": CLASS_NAMES[int(truth[node])],
            "predicted_class": CLASS_NAMES[int(prediction[node])],
            "raw_confidence": float(raw_probabilities[node].max()),
            "scaled_confidence": float(confidence[node]), "top_two_margin": float(margin[node]),
            "nonself_degree": int(degree[node]),
            "training_neighbour_class_counts": {
                name: int((truth[training_neighbours] == label).sum())
                for label, name in enumerate(CLASS_NAMES)
            },
            "neighbour_prediction_agreement": float((prediction[neighbours] == prediction[node]).mean()) if len(neighbours) else None,
            "interpretation": "Pending manual inspection; these statistics do not prove a root cause.",
        })
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--embedding", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if args.split == "test" and checkpoint["phase"] != "final":
        parser.error("Use a final, locked-procedure checkpoint for test-set evaluation")
    if sha256(args.data) != checkpoint["dataset_sha256"]:
        raise ValueError("Checkpoint and dataset checksums do not match")
    graph = make_pyg_graph(args.data, checkpoint["split_seed"])
    torch.testing.assert_close(graph.feature_mean, checkpoint["feature_mean"])
    torch.testing.assert_close(graph.feature_scale, checkpoint["feature_scale"])
    device = torch.device(args.device)
    model = make_model(checkpoint["model"], **checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    adjacency = normalized_adjacency(graph.edge_index.to(device), graph.num_nodes) if checkpoint["model"] == "gcn" else None
    with torch.no_grad():
        logits, embedding = model(graph.x.to(device), adjacency)
    logits, embedding = logits.cpu(), embedding.cpu()
    mask = getattr(graph, f"{args.split}_mask")
    ids = torch.where(mask)[0]
    truth = graph.y[mask]
    temperature = checkpoint["temperature"]
    threshold = checkpoint["confidence_threshold"]
    output = args.output or args.checkpoint.parent / f"predictions_{args.split}"
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "checkpoint_phase": checkpoint["phase"], "evaluation_split": args.split,
        "raw": classification_metrics(logits[mask], truth),
        "scaled": classification_metrics(logits[mask], truth, temperature),
        "policy": selective_metrics(logits[mask], truth, temperature, threshold),
        "error_cases": error_cases(graph, logits, ids, temperature),
        "note": (
            "Validation predictions include selection/calibration nodes and are not an independent final estimate."
            if args.split == "validation" else
            "Held-out test predictions from the locked procedure; do not tune on these results."
        ),
    }
    (output / "evaluation.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    probabilities = (logits[mask] / temperature).softmax(dim=1)
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["node_id", "true_label", "predicted_label", "confidence", "accepted"])
        for node, label, probability in zip(ids.tolist(), truth.tolist(), probabilities):
            writer.writerow([node, label, int(probability.argmax()), float(probability.max()),
                             bool(probability.max() >= threshold)])
    plot_diagnostics(logits[mask], truth, temperature, output / "confidence.png")
    if args.embedding:
        plot_embedding(embedding[mask], truth, checkpoint["phase"], output / "embedding.png")
    print(f"Loaded checkpoint and saved {len(ids)} {args.split} predictions to {output}")
    print(json.dumps({"raw_accuracy": summary["raw"]["accuracy"],
                      "macro_f1": summary["raw"]["macro_f1"],
                      "test_evaluated": args.split == "test"}, indent=2))


if __name__ == "__main__":
    main()
