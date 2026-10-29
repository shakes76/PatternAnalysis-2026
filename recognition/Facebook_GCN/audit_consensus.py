"""Post-evaluation autopsy of five missed-politician cases; never a routing input.

Yuhan Liao - 49839991. Training-neighbour labels and nearest-feature labels
are diagnostic evidence only. Observational agreement does not prove a cause
or justify changing reference annotations without content-level review.
"""
from pathlib import Path
import argparse

import numpy as np
import torch
from torch.nn import functional as F

from consensus_policy import load_members, prepare_inputs, read
from dataset import CLASS_NAMES, canonical_edges, load_arrays, sha256
from modules import make_model, normalized_adjacency
from refinement_study import save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--baseline-study", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Choose a new autopsy output file.")
    torch.set_num_threads(4)
    x, _, roles = prepare_inputs(args.study_dir / "members", args.data)
    gcns, _ = load_members(args.study_dir / "members")
    gcn, gcn_temperature = gcns[0]
    _, labels, raw_edges = load_arrays(args.data)
    cases = read(args.study_dir / "CONSENSUS_RESULTS.json")["case_tracking"]
    folder = args.baseline_study / "mlp_seed42"
    policy = read(folder / "policy.json")
    frozen = read(args.baseline_study / "ALL_POLICIES_FROZEN.json")
    if sha256(folder / "policy.json") != frozen["mlp_seed42"] or sha256(folder / "model.pt") != policy["checkpoint_sha256"]:
        raise ValueError("Baseline policy/checkpoint mismatch")
    if sha256(args.baseline_study / "split_ids.npz") != sha256(args.study_dir / "members/split_ids.npz"):
        raise ValueError("Autopsy baseline must share the same split")
    checkpoint = torch.load(folder / "model.pt", map_location="cpu", weights_only=True)
    if checkpoint["model"] != "mlp" or checkpoint["initialization_seed"] != 42 or checkpoint["temperature"] != policy["temperature"]:
        raise ValueError("Autopsy baseline identity mismatch")
    model = make_model("mlp", **checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    with torch.no_grad():
        p = (model(x)[0] / policy["temperature"]).softmax(1)
    edges = canonical_edges(raw_edges, len(x))
    training = np.zeros(len(x), dtype=bool)
    training[roles["train"]] = True
    train_ids = torch.from_numpy(roles["train"])
    normalized = F.normalize(x, dim=1)
    for case in cases:
        node = case["node_id"]
        neighbours = np.unique(edges[1, (edges[0] == node) & (edges[1] != node)])
        train_neighbours = neighbours[training[neighbours]]
        scores = normalized[train_ids] @ normalized[node]
        closest = train_ids[scores.topk(3).indices].numpy()
        label, confidence = int(p[node].argmax()), float(p[node].max())
        isolated_edges = edges[:, (edges[0] != node) & (edges[1] != node)]
        isolated_graph = normalized_adjacency(torch.from_numpy(isolated_edges), len(x))
        with torch.no_grad():
            isolated_p = (gcn(x, isolated_graph)[0][node] / gcn_temperature).softmax(0)
        case.update({"true_class": CLASS_NAMES[int(labels[node])], "mlp_prediction": CLASS_NAMES[label],
                     "mlp_confidence": confidence, "nonself_degree": int(len(neighbours)),
                     "training_neighbour_counts": {name: int((labels[train_neighbours] == index).sum()) for index, name in enumerate(CLASS_NAMES)},
                     "nearest_three_training_feature_classes": [CLASS_NAMES[int(labels[item])] for item in closest],
                     "isolated_seed42_prediction": CLASS_NAMES[int(isolated_p.argmax())],
                     "isolated_seed42_confidence": float(isolated_p.max()),
                     "isolated_seed42_politician_probability": float(isolated_p[3])})
    save(args.output, {"selection": "First five node IDs among seed42 baseline accepted missed true politicians; diagnostic, not representative sampling",
         "cases": cases, "baseline_checkpoint_sha256": policy["checkpoint_sha256"],
         "audit_source_sha256": sha256(Path(__file__)),
         "intervention": "Remove each selected node's incident nonself edges, renormalise graph, keep unweighted seed42 weights fixed; no labels enter forward",
         "limitations": ["Selected-error graph sensitivity, not retraining ablation or proof of the data-generation cause", "No original page-content/annotation audit",
                         "Nearest features and training neighbours are associations, not causal explanations",
                         "No autopsy labels or case IDs are used by prediction routing"]})
    print("Saved five-case observational autopsy without changing models or routes.")


if __name__ == "__main__":
    main()
