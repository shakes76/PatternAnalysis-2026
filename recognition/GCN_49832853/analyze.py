
from pathlib import Path

import torch
from dataset import load_facebook, split_nodes
from modules import GCN, MLP

ROOT = Path(__file__).resolve().parent
NAMES = ["company", "government", "politician", "tvshow"]


@torch.no_grad()
def analyse_model(name, model, data):
    path = ROOT / "checkpoints" / f"{name.lower()}_best.pt"
    model.load_state_dict(
        torch.load(path, map_location="cpu", weights_only=True)
    )
    model.eval()

    logits = model(data.x, data.edge_index)
    probs = torch.softmax(logits, dim=1)
    confidence, predictions = probs.max(dim=1)

    mask = data.val_mask
    correct = predictions == data.y
    errors = mask & ~correct
    total_errors = errors.sum().item()

    print(f"\n===== {name} Confidence Analysis =====")
    print("Threshold | Coverage | Accepted Acc | Errors Flagged")

    for threshold in [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]:
        accepted = mask & (confidence >= threshold)
        count = accepted.sum().item()

        coverage = count / mask.sum().item()
        accuracy = (
            correct[accepted].float().mean().item()
            if count else float("nan")
        )
        flagged = (
            (errors & ~accepted).sum().item() / total_errors
            if total_errors else 0.0
        )

        print(
            f"{threshold:8.2f} | "
            f"{coverage:7.2%} | "
            f"{accuracy:11.2%} | "
            f"{flagged:13.2%}"
        )

    # Inspect five high-confidence mistakes.
    wrong_nodes = torch.where(errors)[0]
    order = torch.argsort(confidence[wrong_nodes], descending=True)
    selected = wrong_nodes[order[:5]]

    src, dst = data.edge_index

    print("\nHigh-confidence failure cases:")

    for node in selected.tolist():
        neighbors = torch.unique(dst[src == node])
        neighbors = neighbors[neighbors != node]

        # Only use labels of training neighbors for diagnosis.
        known = neighbors[data.train_mask[neighbors]]
        counts = torch.bincount(data.y[known], minlength=4)

        print(
            f"\nNode {node}: "
            f"true={NAMES[data.y[node].item()]}, "
            f"pred={NAMES[predictions[node].item()]}, "
            f"confidence={confidence[node].item():.6f}"
        )
        print("Degree:", len(neighbors))
        print("Labeled training neighbors:")
        for label, count in zip(NAMES, counts.tolist()):
            print(f"  {label}: {count}")


def main():
    data = split_nodes(load_facebook(), seed=42)

    for name, model_class in [("MLP", MLP), ("GCN", GCN)]:
        model = model_class(128, 4)
        analyse_model(name, model, data)


if __name__ == "__main__":
    main()
