"""Evaluate the GCN and save predictions."""

import argparse
import csv
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, roc_auc_score

from dataset import load_graph
from modules import GCN, normalise_edges



# Default checkpoint.
ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "training_results" / "gcn_20261030_002758" / "model.pt"
CLASSES = ["company", "government", "politician", "tvshow"]

WARMUP = 20
REPEATS = 100



def main():
    """Evaluate a checkpoint and save predictions."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--threshold", type=float, default=0.85)
    args = parser.parse_args()
    path = ROOT / args.model
    folder = path.parent / "prediction"



    # Restore model and split.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    params = ckpt["params"]
    data = load_graph(seed=ckpt["seed"]).to(device)
    model = GCN(data.num_node_features, params["hidden_dim"],
                params["depth"], params["dropout"]).to(device)
    model.load_state_dict(ckpt["weights"])
    model.eval()
    folder.mkdir(parents=True, exist_ok=True)



    # Time forward passes after warm-up.
    with torch.no_grad():
        for _ in range(WARMUP):
            model(data.x, data.edge_index)

        if device == "cuda":
            torch.cuda.synchronize()
        start = perf_counter()

        for _ in range(REPEATS):
            out = model(data.x, data.edge_index)

        if device == "cuda":
            torch.cuda.synchronize()
        ms = (perf_counter() - start) * 1000 / REPEATS



    # Convert scores to probabilities.
    prob = out.cpu().softmax(dim=1)
    pred = prob.argmax(dim=1)
    y = data.y.cpu()
    test = data.test_mask.cpu()



    # Save test metrics.
    report = classification_report(
        y[test], pred[test], labels=[0, 1, 2, 3],
        target_names=CLASSES, digits=4, zero_division=0,
    )
    auc = roc_auc_score(y[test].numpy(), prob[test].numpy(),
                         labels=[0, 1, 2, 3], multi_class="ovr", average="macro")

    with (folder / "metrics.txt").open("w", encoding="utf-8") as f:
        f.write(report)
        f.write(f"\nMacro AUROC: {auc:.4f}\n")



    # Save validation and test predictions.
    columns = ["node", "split", "true", "pred"]
    columns += [f"p_{name}" for name in CLASSES]

    with (folder / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)

        for split, mask in [("val", data.val_mask), ("test", data.test_mask)]:
            nodes = mask.cpu().nonzero(as_tuple=True)[0].tolist()
            for node in nodes:
                writer.writerow([
                    node, split, CLASSES[int(y[node])], CLASSES[int(pred[node])],
                    *prob[node].tolist(),
                ])



    # Flag low-confidence predictions.
    with (folder / "low_confidence.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["node", "split", "true", "pred", "confidence"])

        for split, mask in [("val", data.val_mask), ("test", data.test_mask)]:
            nodes = mask.cpu().nonzero(as_tuple=True)[0].tolist()
            for node in nodes:
                confidence = float(prob[node].max())
                if confidence < args.threshold:
                    writer.writerow([
                        node, split, CLASSES[int(y[node])], CLASSES[int(pred[node])],
                        confidence,
                    ])



    # Extract last-hidden-layer features.
    with torch.no_grad():
        source, target, coef = normalise_edges(data.edge_index, data.num_nodes, data.x.dtype)
        x = data.x
        for layer in model.layers[:-1]:
            x = F.relu(layer(x, source, target, coef))

    torch.save({
        "x": x.cpu(),
        "y": y,
        "classes": CLASSES,
        "val_mask": data.val_mask.cpu(),
        "test_mask": test,
    }, folder / "embeddings.pt")



    # Save inference time.
    with (folder / "predict_info.txt").open("w", encoding="utf-8") as f:
        f.write(f"Model: {path.parent.name}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Inference (ms/graph): {ms:.4f}\n")
        f.write(f"Threshold: {args.threshold}\n")


    print(report)
    print(f"Macro AUROC: {auc:.4f}")
    print(f"Saved: {folder}")


    # Summarise test predictions at the chosen threshold.
    automatic = prob[test].max(dim=1).values >= args.threshold
    count = int(automatic.sum())
    total = int(test.sum())
    accuracy = (pred[test][automatic] == y[test][automatic]).float().mean().item()
    review = total - count

    print(f"\nTest: {total} nodes, threshold = {args.threshold:.2f}")
    print(f"Automatic: {count} ({count / total:.2%})")
    print(f"Auto accuracy: {accuracy:.2%}")
    print(f"Review: {review} ({review / total:.2%})")



if __name__ == "__main__":
    main()
