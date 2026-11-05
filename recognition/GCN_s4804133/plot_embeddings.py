"""Create a class-coloured TSNE plot of a trained gcn's hidden embeddings"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE

from dataset import CLASS_NAMES, load_graph
from modules import GCN
from predict import build_model


def main() -> None:
    """Load a checkpoint, project sampled embeddings, and save the plot"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="outputs/gcn_seed42_five-seed.pt",
        help="trained GCN checkpoint path",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=1000,
        help="number of nodes to plot from each class",
    )
    parser.add_argument(
        "--output",
        default="figures/gcn_tsne_embeddings.png",
        help="output image path",
    )
    args = parser.parse_args()
    if args.samples_per_class < 1:
        parser.error("--samples-per-class must be at least 1")

    # use the gpu when available, otherwise run on the cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_graph().to(device)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"GCN checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = build_model(checkpoint, data.x.size(1), len(CLASS_NAMES), device)
    if not isinstance(model, GCN):
        raise TypeError("The selected checkpoint did not create a GCN model")

    # run all but the final layer to get each node's learned hidden representation
    with torch.inference_mode():
        hidden = data.x
        for layer in model.layers[:-1]:
            hidden = F.relu(layer(hidden, data.edge_index))
            hidden = F.dropout(hidden, p=model.dropout, training=False)

    embeddings = hidden.cpu().numpy()
    labels = data.y.cpu().numpy()
    # use a fixed random seed so the same nodes are sampled each run
    rng = np.random.default_rng(42)
    sample_indices = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        class_indices = np.flatnonzero(labels == class_id)
        if class_indices.size < args.samples_per_class:
            raise ValueError(
                f"Class {class_name!r} has {class_indices.size} nodes, fewer than "
                f"--samples-per-class={args.samples_per_class}"
            )
        sample_indices.extend(
            rng.choice(class_indices, size=args.samples_per_class, replace=False)
        )

    # balance the visualisation by selecting the same number from every class
    sample_indices = np.asarray(sample_indices)
    rng.shuffle(sample_indices)
    # reduce the high dimensional vectors to two dimensions for plotting
    coordinates = TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        learning_rate="auto",
        max_iter=1000,
        random_state=42,
    ).fit_transform(embeddings[sample_indices])

    sample_labels = labels[sample_indices]
    fig, ax = plt.subplots(figsize=(8, 6))
    # plot each class separately so its points share a legend colour
    for class_id, class_name in enumerate(CLASS_NAMES):
        selected = sample_labels == class_id
        ax.scatter(
            coordinates[selected, 0],
            coordinates[selected, 1],
            s=8,
            alpha=0.65,
            label=class_name,
        )

    ax.set(
        title="GCN hidden-layer embeddings (t-SNE)",
        xlabel="t-SNE 1",
        ylabel="t-SNE 2",
    )
    ax.legend(markerscale=2)
    fig.tight_layout()
    output_path = Path(args.output)
    # create the output folder if it does not already exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved t-SNE plot to {output_path}")
    print(
        f"Plotted {len(sample_indices)} nodes "
        f"({args.samples_per_class} sampled from each class)."
    )


if __name__ == "__main__":
    main()
