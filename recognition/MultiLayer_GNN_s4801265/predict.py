import argparse
from pathlib import Path

import torch

from dataset import load_dataset
from modules import MLP, GCN


# Settings must match those used during training.
SEED = 42
HIDDEN_DIM = 32
DROPOUT = 0.5
NUM_CLASSES = 4


def load_trained_model(model_name, device, checkpoint_path=None):
    """Load the trained MLP or GCN from its saved state dictionary.

    Args:
        model_name: Either "mlp" or "gcn".
        device: CPU or CUDA device used for inference.
        checkpoint_path: Optional path to a saved state dictionary.

    Returns:
        Model in evaluation mode.

    Raises:
        FileNotFoundError: If the checkpoint does not exist.
    """
    if model_name == "mlp":
        model = MLP(
            input_dim=128,
            hidden_dim=HIDDEN_DIM,
            output_dim=NUM_CLASSES,
            dropout=DROPOUT
        )
    else:
        model = GCN(
            input_dim=128,
            hidden_dim=HIDDEN_DIM,
            output_dim=NUM_CLASSES,
            dropout=DROPOUT
        )

    path = Path(checkpoint_path or f"{model_name}_best.pt")
    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}. "
            f"Train the model first using: python train.py --model {model_name}"
        )

    # train.py saves model.state_dict(), not a checkpoint dictionary.
    state_dict = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()  # Disable dropout for deterministic inference.

    print(f"Loaded {model_name.upper()} weights from {path}")
    return model


def predict_nodes(model, data, model_name):
    """Predict every node's class and class probabilities.

    Args:
        model: Trained MLP or GCN.
        data: Preprocessed graph with node features and edges.
        model_name: Either "mlp" or "gcn".

    Returns:
        Tensor of class probabilities of shape [num_nodes, 4].
    """
    model.eval()

    with torch.no_grad():
        if model_name == "gcn":
            logits = model(data.x, data.edge_index)
        else:
            logits = model(data.x)

        probabilities = torch.softmax(logits, dim=1)

    return probabilities


def print_prediction(node_id, probabilities, data, threshold=None):
    """Display the predicted category and confidence for one node.

    Args:
        node_id: Original node index in the full graph.
        probabilities: Predicted class probabilities for every node.
        data: Graph containing labels and train/val/test masks.
        threshold: Optional confidence cutoff for manual review.
    """
    node_probs = probabilities[node_id].cpu()
    confidence, predicted_class = node_probs.max(dim=0)
    predicted_class = int(predicted_class.item())
    confidence = float(confidence.item())

    if bool(data.train_mask[node_id].item()):
        split = "Training"
    elif bool(data.val_mask[node_id].item()):
        split = "Validation"
    else:
        split = "Test"

    # Labels are shown for evaluation only, never used in prediction.
    true_class = int(data.y[node_id].item())

    print(f"\n=== NODE {node_id} ===")
    print(f"Dataset Split: {split}")
    print(f"Predicted Class: {predicted_class}")
    print(f"Prediction Confidence: {confidence:.4f}")
    print("Class Probabilities:")

    for class_id, probability in enumerate(node_probs.tolist()):
        print(f"  Class {class_id}: {probability:.4f}")

    print(f"True Class: {true_class}")
    print(f"Correct Prediction: {predicted_class == true_class}")

    if threshold is not None:
        decision = (
            "Automatically Accepted"
            if confidence >= threshold
            else "Sent for Manual Review"
        )
        print(f"Confidence Threshold: {threshold:.2f}")
        print(f"Decision: {decision}")


def main():
    """Load a trained model and display predictions for graph nodes."""
    parser = argparse.ArgumentParser(
        description="Run inference using a trained Facebook Page-Page MLP or GCN."
    )
    parser.add_argument(
        "--model", choices=["mlp", "gcn"], default="gcn",
        help="Model to use for inference (default: gcn)."
    )
    parser.add_argument(
        "--node", type=int, default=None,
        help="Original graph node index to predict."
    )
    parser.add_argument(
        "--num_samples", type=int, default=5,
        help="Number of test nodes to display when --node is omitted."
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Optional validation-selected confidence cutoff from 0 to 1."
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Optional path to the model's saved .pt weights."
    )
    args = parser.parse_args()

    if args.threshold is not None and not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1.")
    if args.num_samples < 1:
        parser.error("--num_samples must be at least 1.")

    # Match the same fixed dataset split and preprocessing as train.py.
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    data = load_dataset(seed=SEED).to(device)

    model = load_trained_model(
        args.model, device, checkpoint_path=args.checkpoint
    )
    probabilities = predict_nodes(model, data, args.model)

    if args.node is not None:
        if not 0 <= args.node < data.num_nodes:
            parser.error(
                f"--node must be between 0 and {data.num_nodes - 1}."
            )
        node_indices = [args.node]
    else:
        # By default, show held-out test nodes rather than training nodes.
        node_indices = (
            data.test_mask.nonzero(as_tuple=True)[0]
            .cpu().tolist()[:args.num_samples]
        )

    for node_id in node_indices:
        print_prediction(node_id, probabilities, data, args.threshold)

    print(f"\nInference complete. Displayed {len(node_indices)} node(s).")


if __name__ == "__main__":
    main()
