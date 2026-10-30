"""Label-free inference on the real fixed Facebook graph with a frozen policy.

Example: python predict.py --model gcn --node-ids 8505 5712 9141 9229 1626
Test-partition inference is explicit; Stage 7 demonstrations use selection only.
This driver does not calculate accuracy, refit calibration or select thresholds.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from confidence import acceptance_mask, collect_logits, probabilities
from dataset import CLASS_NAMES, DEFAULT_ROOT, load_facebook, sha256_file
from evaluate import PROJECT, save_plot, write_csv
from train import TrainConfig, configure_reproducibility

EXAMPLE_SELECTION_NODES = (8505, 5712, 9141, 9229, 1626)
DEFAULT_POLICY = PROJECT / "outputs/confidence_seed42/referral_policy.json"


def validate_files(checkpoint_path, policy_path, model_name):
    """Fail before loading/downloading data when required artifacts are absent."""
    if not Path(checkpoint_path).is_file():
        raise FileNotFoundError(
            f"Weights not found: {checkpoint_path}. Recreate with train.py "
            f"--model {model_name.lower()} (see README for the full command)."
        )
    if not Path(policy_path).is_file():
        raise FileNotFoundError(
            f"Frozen policy not found: {policy_path}. For a fresh reproduction, "
            "run confidence.py after training BOTH models. For final evaluation, "
            "restore the approved policy; do not refit it."
        )


def read_policy(policy_path, checkpoint_path, dataset, model_name):
    """Bind calibration/referral settings to exact weights, data and node splits."""
    policy = json.loads(Path(policy_path).read_text(encoding="utf-8"))
    try:
        if (policy["schema_version"] != 1
                or policy["temperature_fitting_partition"] != "calibration"
                or policy["threshold_selection_partition"] != "policy"):
            raise ValueError("Unsupported policy schema or fitting roles")
        if policy["data_sha256"] != dataset.provenance["sha256"]:
            raise ValueError("Policy dataset hash does not match")
        masks = {name: row["mask_sha256"] for name, row in dataset.audit["splits"].items()}
        if policy["split_mask_sha256"] != masks:
            raise ValueError("Policy split hashes do not match")
        settings = policy["models"][model_name]
        if settings["checkpoint_sha256"] != sha256_file(Path(checkpoint_path)):
            raise ValueError("Policy checkpoint hash does not match; use its paired frozen checkpoint")
        temperature, threshold = settings["temperature"], settings["threshold"]
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("Invalid policy temperature")
        if threshold is not None and (not np.isfinite(threshold) or not 0 <= threshold <= 1):
            raise ValueError("Invalid policy threshold")
    except (KeyError, TypeError) as error:
        raise ValueError(f"Malformed policy: {error}") from error
    return policy, settings


def select_nodes(dataset, partition, node_ids=None):
    """Partition membership uses masks only, never ground-truth labels."""
    if partition not in ("selection", "test"):
        raise ValueError("Inference partition must be selection or test")
    if node_ids is None:
        node_ids = (EXAMPLE_SELECTION_NODES if partition == "selection"
                    else np.flatnonzero(dataset.masks[partition].numpy())[:5].tolist())
    nodes = np.asarray(node_ids)
    if nodes.ndim != 1 or len(nodes) == 0 or nodes.dtype.kind not in "iu":
        raise ValueError("Provide a nonempty list of integer node IDs")
    if len(np.unique(nodes)) != len(nodes):
        raise ValueError("Node IDs must be unique")
    if (nodes < 0).any() or (nodes >= dataset.graph.num_nodes).any():
        raise ValueError("Node ID outside the graph")
    if not dataset.masks[partition].numpy()[nodes].all():
        raise ValueError(f"All requested node IDs must belong to the {partition} partition")
    return nodes


def prediction_rows(node_ids, logits, temperature, threshold):
    raw, calibrated = probabilities(logits), probabilities(logits, temperature)
    accepted = acceptance_mask(calibrated, threshold)
    predicted = calibrated.argmax(axis=1)
    return [
        {"node_id": int(node), "predicted_class_id": int(predicted[i]),
         "predicted_class": CLASS_NAMES[predicted[i]],
         "raw_confidence": float(raw[i].max()),
         "calibrated_confidence": float(calibrated[i].max()),
         "decision": "accept" if accepted[i] else "refer",
         **{f"probability_{name}": float(calibrated[i, j]) for j, name in enumerate(CLASS_NAMES)}}
        for i, node in enumerate(node_ids)
    ]


def plot_predictions(rows, model_name, threshold, stem):
    """Display at most 20 rows; the CSV always contains all requested nodes."""
    shown = rows[:20]
    figure, axis = plt.subplots(figsize=(9, max(4, .6 * len(shown))))
    positions = np.arange(len(shown))
    colors = ("#0072B2", "#E69F00", "#009E73", "#CC79A7")
    for j, (name, color) in enumerate(zip(CLASS_NAMES, colors)):
        axis.barh(positions + (j - 1.5) * .18,
                  [row[f"probability_{name}"] for row in shown],
                  height=.17, label=name, color=color)
    if threshold is not None:
        axis.axvline(threshold, linestyle="--", color="black", label=f"Referral threshold {threshold:.4f}")
    axis.set(yticks=positions,
             yticklabels=[f"{r['node_id']}: {r['predicted_class']} ({r['decision']})" for r in shown],
             xlabel="Temperature-scaled class probability", xlim=(0, 1.02),
             title=f"{model_name}: label-free predictions on the fixed Facebook graph")
    axis.invert_yaxis()
    axis.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    figure.tight_layout()
    save_plot(figure, stem)


def run_prediction(dataset, checkpoint_path, policy_path, output_dir,
                   model_name="GCN", node_ids=None, partition="selection", device="auto"):
    if model_name not in ("MLP", "GCN"):
        raise ValueError("Model must be MLP or GCN")
    validate_files(checkpoint_path, policy_path, model_name)
    policy, settings = read_policy(policy_path, checkpoint_path, dataset, model_name)
    nodes = select_nodes(dataset, partition, node_ids)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Choose a fresh output directory to preserve previous predictions")
    chosen_device = configure_reproducibility(TrainConfig(device=device))
    # GCN requires full graph context even when only a few outputs are requested.
    logits = collect_logits(dataset, checkpoint_path, model_name, chosen_device)[nodes]
    rows = prediction_rows(nodes, logits, settings["temperature"], settings["threshold"])
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "predictions.csv", rows)
    summary = {
        "model": model_name, "partition": partition, "nodes": len(rows),
        "device": str(chosen_device), "labels_accessed": False,
        "checkpoint_sha256": settings["checkpoint_sha256"],
        "policy_sha256": sha256_file(Path(policy_path)), "data_sha256": policy["data_sha256"],
        "temperature": settings["temperature"], "threshold": settings["threshold"],
        "accepted": sum(row["decision"] == "accept" for row in rows),
        "referred": sum(row["decision"] == "refer" for row in rows),
    }
    (output_dir / "prediction_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    plot_predictions(rows, model_name, settings["threshold"], output_dir / "predictions")
    for row in rows:
        print(f"node {row['node_id']}: {row['predicted_class']}, "
              f"confidence={row['calibrated_confidence']:.6f}, {row['decision']}")
    print(f"Wrote {len(rows)} predictions to {output_dir}")
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("mlp", "gcn"), default="gcn")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--node-ids", nargs="+", type=int)
    parser.add_argument("--partition", choices=("selection", "test"), default="selection")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda:0"), default="auto")
    args = parser.parse_args()
    checkpoint = args.checkpoint or PROJECT / f"outputs/{args.model}_seed42/best.pt"
    output = args.output_dir or PROJECT / f"outputs/predict_{args.model}_{args.partition}"
    try:
        validate_files(checkpoint, args.policy, args.model.upper())
        run_prediction(load_facebook(args.data_root), checkpoint, args.policy, output,
                       args.model.upper(), args.node_ids, args.partition, args.device)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"predict.py: error: {error}\n")


if __name__ == "__main__":
    main()
