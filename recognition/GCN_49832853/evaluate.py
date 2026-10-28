
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
)

from dataset import load_facebook, split_nodes
from modules import GCN, MLP


BASE_DIR = Path(__file__).resolve().parent
CLASS_NAMES = [
    "company",
    "government",
    "politician",
    "tvshow",
]

SEED = 42


@torch.no_grad()
def evaluate_model(name, model, data, device):
    # Load the best checkpoint from training
    checkpoint = BASE_DIR / "checkpoints" / f"{name.lower()}_best.pt"

    model.load_state_dict(
        torch.load(
            checkpoint,
            map_location=device,
            weights_only=True
        )
    )

    model = model.to(device)
    model.eval()

    # Use the same complete graph for inference
    logits = model(data.x, data.edge_index)

    # Evaluate only on validation nodes
    mask = data.val_mask

    probabilities = torch.softmax(
        logits[mask], dim=1
    ).cpu().numpy()

    y_true = data.y[mask].cpu().numpy()
    y_pred = probabilities.argmax(axis=1)

    # Classification metrics
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro"
    )

    auroc = roc_auc_score(
        y_true,
        probabilities,
        labels=np.arange(len(CLASS_NAMES)),
        multi_class="ovr",
        average="macro"
    )

    print(f"\n===== {name} Evaluation =====")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print(f"Macro-AUROC: {auroc:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=np.arange(len(CLASS_NAMES)),
            target_names=CLASS_NAMES,
            zero_division=0
        )
    )

    # Confusion matrix
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(len(CLASS_NAMES))
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASS_NAMES
    )

    disp.plot(values_format="d", xticks_rotation=30)
    plt.title(f"{name} Validation Confusion Matrix")
    plt.tight_layout()

    figures_dir = BASE_DIR / "figures"
    figures_dir.mkdir(exist_ok=True)

    plt.savefig(
        figures_dir / f"{name.lower()}_confusion_matrix.png",
        dpi=150
    )
    plt.close()

    # Identify high-confidence incorrect predictions
    node_ids = torch.where(mask)[0].cpu().numpy()

    confidence = probabilities.max(axis=1)
    wrong_indices = np.flatnonzero(y_pred != y_true)

    ranked_errors = wrong_indices[
        np.argsort(-confidence[wrong_indices])
    ]

    print("\nTop 5 High-Confidence Errors:")

    for idx in ranked_errors[:5]:
        print(
            f"Node {node_ids[idx]} | "
            f"True: {CLASS_NAMES[y_true[idx]]} | "
            f"Predicted: {CLASS_NAMES[y_pred[idx]]} | "
            f"Confidence: {confidence[idx]:.4f}"
        )


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    data = split_nodes(
        load_facebook(), seed=SEED
    ).to(device)

    num_features = data.num_node_features
    num_classes = int(data.y.max().item()) + 1

    models = {
        "MLP": MLP(num_features, num_classes),
        "GCN": GCN(num_features, num_classes)
    }

    for name, model in models.items():
        evaluate_model(name, model, data, device)


if __name__ == "__main__":
    main()
