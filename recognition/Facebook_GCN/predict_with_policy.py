"""Apply a checkpoint-bound, frozen policy without consulting evaluation labels.

The CSV deliberately excludes reference labels and does not calculate accuracy.
Saved study results describe post-hoc original-test reuse, not new validation.
"""

from pathlib import Path
import argparse
import csv
import json
import math
import sys

sys.dont_write_bytecode = True
from policy97_experiment import accepted_mask, verify_policy_binding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--nodes", choices=["test", "all"], default="test")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Output exists; choose a new path to preserve the previous export.")
    sys.path.insert(0, str(args.source_dir.resolve()))
    import torch
    from dataset import CLASS_NAMES, DATA_SHA256, DEFAULT_PATH, make_pyg_graph, sha256
    from modules import make_model, normalized_adjacency

    torch.set_num_threads(4)
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["phase"] != "final" or checkpoint["model"] != "gcn":
        raise ValueError("This policy requires an original final GCN checkpoint")
    data = args.data or DEFAULT_PATH
    if sha256(data) != DATA_SHA256:
        raise ValueError("Expected the pinned course dataset")
    verify_policy_binding(policy, args.checkpoint, DATA_SHA256, checkpoint["initialization_seed"])
    if policy["model_config"] != checkpoint["model_config"] or policy["split_seed"] != checkpoint["split_seed"]:
        raise ValueError("Policy and checkpoint configuration mismatch")
    if not math.isclose(policy["temperature"], checkpoint["temperature"], rel_tol=0, abs_tol=0):
        raise ValueError("Policy must use the original saved temperature")
    if not math.isfinite(policy["temperature"]) or policy["temperature"] <= 0:
        raise ValueError("A finite positive temperature is required")
    if policy["validation_role_seed"] != checkpoint["validation_role_seed"]:
        raise ValueError("Policy and checkpoint calibration roles differ")
    graph = make_pyg_graph(data, checkpoint["split_seed"])
    torch.testing.assert_close(graph.feature_mean, checkpoint["feature_mean"])
    torch.testing.assert_close(graph.feature_scale, checkpoint["feature_scale"])
    device = torch.device(args.device)
    model = make_model(checkpoint["model"], **checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    adjacency = normalized_adjacency(graph.edge_index.to(device), graph.num_nodes)
    with torch.no_grad():
        logits, _ = model(graph.x.to(device), adjacency)
    ids = torch.where(graph.test_mask)[0] if args.nodes == "test" else torch.arange(graph.num_nodes)
    probabilities = (logits.cpu()[ids] / policy["temperature"]).softmax(1)
    decisions = accepted_mask(probabilities, policy["new_threshold"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["node_id", "predicted_label", "predicted_class", "confidence", "accepted", "route", "policy_id"])
        for node, pred, confidence, accepted in zip(ids.tolist(), probabilities.argmax(1).tolist(), probabilities.max(1).values.tolist(), decisions.tolist()):
            writer.writerow([node, pred, CLASS_NAMES[pred], confidence, accepted,
                             "automatic_candidate" if accepted else "human_review", policy["policy_id"]])
    print(json.dumps({"predictions": len(ids), "accepted": int(decisions.sum()), "human_review": int((~decisions).sum()),
                      "reference_labels_used_for_decisions": False, "threshold": policy["new_threshold"],
                      "accuracy_computed": False, "output": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
