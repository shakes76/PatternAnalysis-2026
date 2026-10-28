"""Predict the category of one Facebook node with a saved GCN or MLP.

Place beside dataset.py, modules.py and checkpoints/.
Example:
    python predict.py --node 9077 --model gcn
    python predict.py --node 9077 --model mlp --show-label
"""

import argparse
from pathlib import Path

import torch

from dataset import load_facebook
from modules import GCN, MLP

BASE_DIR = Path(__file__).resolve().parent
CLASS_NAMES = ["company", "government", "politician", "tvshow"]


@torch.inference_mode()
def predict_node(node_id, model_name="gcn", threshold=0.90, show_label=False):
    data = load_facebook()
    if not 0 <= node_id < data.num_nodes:
        raise ValueError(f"Node index must be from 0 to {data.num_nodes - 1}; got {node_id}.")

    model_cls = {"gcn": GCN, "mlp": MLP}[model_name]
    model = model_cls(data.num_node_features, int(data.y.max().item()) + 1)
    checkpoint = BASE_DIR / "checkpoints" / f"{model_name}_best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()

    # Full graph context is needed for message passing in GCN.
    logits = model(data.x, data.edge_index)
    probs = torch.softmax(logits[node_id], dim=0)
    predicted_id = int(torch.argmax(probs).item())
    confidence = float(probs[predicted_id].item())

    print(f"Model: {model_name.upper()}")
    print(f"Node index (row in dataset): {node_id}")
    print(f"Predicted class: {CLASS_NAMES[predicted_id]}")
    print(f"Maximum softmax score: {confidence:.4%}")
    print("Class scores:")
    for index in torch.argsort(probs, descending=True).tolist():
        print(f"  {CLASS_NAMES[index]:<12}: {probs[index].item():.4%}")

    if confidence < threshold:
        print(f"Review policy: FLAG FOR HUMAN REVIEW (threshold {threshold:.2f})")
    else:
        print(f"Review policy: AUTO-ACCEPT (threshold {threshold:.2f})")
    print("Note: a high softmax score does not guarantee correctness.")

    if show_label:
        print(f"Ground-truth label (offline evaluation only): {CLASS_NAMES[data.y[node_id].item()]}")


def main():
    parser = argparse.ArgumentParser(description="Predict one Facebook page node category")
    parser.add_argument("--node", type=int, required=True,
                        help="Zero-based row index in facebook.npz")
    parser.add_argument("--model", choices=["gcn", "mlp"], default="gcn")
    parser.add_argument("--threshold", type=float, default=0.90,
                        help="Threshold fixed using validation data")
    parser.add_argument("--show-label", action="store_true",
                        help="Display known dataset label for offline debugging only")
    args = parser.parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be in [0, 1].")
    predict_node(args.node, args.model, args.threshold, args.show_label)


if __name__ == "__main__":
    main()
