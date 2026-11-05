"""Loads a trained model checkpoint, rebuilds the matching GCN or MLP, predicts a class for every node in the Facebook page graph, reports test-set metrics, and saves the predictions and probabilities"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


# import helpers and modules
from dataset import CLASS_NAMES, load_graph, make_masks
from modules import GCN, MLP


def select_device() -> torch.device:
    """Prefer CUDA, then Apple MPS, and otherwise use CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    print("Warning: CUDA and MPS not found. Using CPU")
    return torch.device("cpu")


def build_model(
    checkpoint: dict,
    in_dim: int,
    out_dim: int,
    device: torch.device,
) -> torch.nn.Module:
    """Recreate the model architecture and load the checkpoint weights."""
    saved_args = checkpoint.get("args")
    if not isinstance(saved_args, dict):
        raise ValueError("checkpoint is missing the training arguments")

    model_name = saved_args.get("model")
    hidden_dim = saved_args.get("hidden")
    dropout = saved_args.get("dropout")
    if not isinstance(hidden_dim, int) or dropout is None:
        raise ValueError("checkpoint is missing hidden size or dropout settings")

    if model_name == "gcn":
        num_layers = saved_args.get("num_layers") # read num of layers if GCN
        if not isinstance(num_layers, int):
            raise ValueError("GCN checkpoint is missing num_layers")
        model = GCN(in_dim, hidden_dim, out_dim, num_layers=num_layers, dropout=dropout)
    elif model_name == "mlp":
        model = MLP(in_dim, hidden_dim, out_dim, dropout=dropout)
    else:
        raise ValueError(f"unsupported model type in checkpoint: {model_name!r}")

    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("checkpoint is missing the model state_dict")
    model.load_state_dict(state_dict) # load saved model weights
    return model.to(device).eval() # move model to device and set to evaluation mode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="path to a .pt file created by train.py")
    parser.add_argument(
        "--output",
        help="output .npz path (default: outputs/predictions_<checkpoint-name>.npz)",
    ) # optional output path for saving predictions
    args = parser.parse_args()

    device = select_device()
    print(f"Using device: {device}")

    # Keep the split seed fixed; the checkpoint seed controls training randomness only.
    data = make_masks(load_graph(), seed=42).to(device) # evaluation uses the same node split regardless of seed used to train checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = build_model(checkpoint, data.x.size(1), len(CLASS_NAMES), device)

    with torch.inference_mode(): # disable gradient computation for inference
        logits = model(data.x, data.edge_index)
        probabilities = F.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1)

    # Print test metrics for a quick check; saved probabilities also support later analysis.
    test_mask = data.test_mask
    test_loss = F.cross_entropy(logits[test_mask], data.y[test_mask]).item()
    test_accuracy = (predictions[test_mask] == data.y[test_mask]).float().mean().item()
    print(f"Model: {checkpoint['args']['model']} | training seed: {checkpoint['args']['seed']}")
    print(f"Test loss: {test_loss:.4f} | test accuracy: {test_accuracy:.4f}")
    print(f"Test nodes: {int(test_mask.sum().item())}")

    output_path = Path(args.output) if args.output else Path("outputs") / f"predictions_{Path(args.checkpoint).stem}.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        node_index=np.arange(data.num_nodes),
        predictions=predictions.cpu().numpy(),
        probabilities=probabilities.cpu().numpy(),
        train_mask=data.train_mask.cpu().numpy(),
        val_mask=data.val_mask.cpu().numpy(),
        test_mask=test_mask.cpu().numpy(),
    )
    print(f"Saved predictions and probabilities for {data.num_nodes} nodes to {output_path}")


if __name__ == "__main__":
    main()
