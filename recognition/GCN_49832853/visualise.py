
from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from dataset import load_facebook, split_nodes
from modules import GCN


BASE_DIR = Path(__file__).resolve().parent
CLASS_NAMES = [
    "company",
    "government",
    "politician",
    "tvshow"
]


@torch.no_grad()
def extract_embeddings(model, data):
    model.eval()

    # Extract 64-dimensional hidden representations
    hidden = model.conv1(data.x, data.edge_index)
    hidden = F.relu(hidden)

    return hidden


def main():
    # Use the original validation split
    data = split_nodes(load_facebook(), seed=42)

    model = GCN(
        num_features=data.num_node_features,
        num_classes=4
    )

    checkpoint = BASE_DIR / "checkpoints" / "gcn_best.pt"

    model.load_state_dict(
        torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=True
        )
    )
    model.eval()

    # Extract graph embeddings
    embeddings = extract_embeddings(model, data)

    # Visualise validation nodes only
    mask = data.val_mask

    X = embeddings[mask].cpu().numpy()
    y = data.y[mask].cpu().numpy()

    # Identify incorrect predictions for annotation
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        predictions = logits[mask].argmax(dim=1)

    wrong = predictions.cpu().numpy() != y

    print("Embeddings shape:", X.shape)
    print("Validation errors:", wrong.sum())

    # Reduce 64 dimensions to 2
    print("Running t-SNE...")

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        max_iter=1000,
        random_state=42
    )

    X_2d = tsne.fit_transform(X)

    # Plot with ground-truth category colours
    plt.figure(figsize=(9, 7))

    for class_id, class_name in enumerate(CLASS_NAMES):
        selected = y == class_id

        plt.scatter(
            X_2d[selected, 0],
            X_2d[selected, 1],
            s=12,
            alpha=0.65,
            label=class_name
        )

    # Mark incorrectly classified nodes
    plt.scatter(
        X_2d[wrong, 0],
        X_2d[wrong, 1],
        marker="x",
        c="black",
        s=20,
        linewidths=0.6,
        label="Misclassified"
    )

    plt.title("GCN Node Embeddings (Validation Set)")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend()
    plt.tight_layout()

    figures_dir = BASE_DIR / "figures"
    figures_dir.mkdir(exist_ok=True)

    output = figures_dir / "gcn_tsne_validation.png"
    plt.savefig(output, dpi=200)
    plt.close()

    print("Saved:", output)


if __name__ == "__main__":
    main()
