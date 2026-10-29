"""Reload a five-role-study model and export label-free triage decisions."""
from pathlib import Path
import argparse
import csv
import json
import math

import numpy as np
import torch

from dataset import CLASS_NAMES, DATA_SHA256, DEFAULT_PATH, canonical_edges, load_arrays, sha256, standardize_features
from modules import make_model, normalized_adjacency
from policy97_experiment import accepted_mask
from validation_study import ROLES, SPLIT_SEED, class_decisions


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--run", choices=[f"{name}_seed{seed}" for name in ["mlp", "gcn"] for seed in [42, 43, 44]], default="gcn_seed42")
    parser.add_argument("--data", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--policy", choices=["global97", "class98"], default="class98")
    parser.add_argument("--subset", choices=["all", "final"], default="final")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Output already exists; preserve it and choose a new filename.")
    torch.set_num_threads(4)
    folder = args.study_dir / args.run
    frozen = read(args.study_dir / "ALL_POLICIES_FROZEN.json")
    if sha256(folder / "policy.json") != frozen[args.run]:
        raise ValueError("Policy file differs from its pre-evaluation frozen hash")
    policy = read(folder / "policy.json")
    if sha256(folder / "model.pt") != policy["checkpoint_sha256"]:
        raise ValueError("Checkpoint differs from the policy's fitted model")
    manifest = read(args.study_dir / "SPLIT_MANIFEST.json")
    if sha256(args.data) != DATA_SHA256 or manifest["dataset_sha256"] != DATA_SHA256:
        raise ValueError("Pinned dataset required")
    if sha256(args.study_dir / "split_ids.npz") != manifest["split_sha256"]:
        raise ValueError("Split/scaling artifact has changed")
    checkpoint = torch.load(folder / "model.pt", map_location="cpu", weights_only=True)
    expected_run = f"{checkpoint['model']}_seed{checkpoint['initialization_seed']}"
    if expected_run != args.run or checkpoint["split_seed"] != SPLIT_SEED or checkpoint["dataset_sha256"] != DATA_SHA256:
        raise ValueError("Checkpoint study identity mismatch")
    if checkpoint["phase"] != "within_graph_five_role_study" or checkpoint["model"] != policy["model"] or checkpoint["initialization_seed"] != policy["seed"]:
        raise ValueError("Policy/checkpoint mismatch")
    temperature = policy["temperature"]
    if not math.isfinite(temperature) or temperature <= 0 or temperature != checkpoint["temperature"]:
        raise ValueError("Finite positive fitted temperature required")
    if policy["class98_thresholds"] != checkpoint["class_thresholds"] or policy["global97_threshold"] != checkpoint["global97_threshold"]:
        raise ValueError("Policy thresholds differ from the frozen checkpoint")
    raw, _, edges = load_arrays(args.data)
    with np.load(args.study_dir / "split_ids.npz", allow_pickle=False) as saved:
        roles = {name: saved[name].copy() for name in ROLES}
        stored_mean, stored_scale = saved["feature_mean"].copy(), saved["feature_scale"].copy()
    if not np.array_equal(np.sort(np.concatenate(list(roles.values()))), np.arange(len(raw))):
        raise ValueError("Study roles overlap or fail to cover the graph")
    x, mean, scale = standardize_features(raw, roles["train"])
    if not np.array_equal(mean, stored_mean) or not np.array_equal(scale, stored_scale):
        raise ValueError("Training-only scaling differs from recorded values")
    model = make_model(checkpoint["model"], **checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    adjacency = normalized_adjacency(torch.from_numpy(canonical_edges(edges, len(raw))), len(raw)) if checkpoint["model"] == "gcn" else None
    ids = roles["final"] if args.subset == "final" else np.arange(len(raw))
    with torch.no_grad():
        logits, _ = model(torch.from_numpy(x), adjacency)
        probabilities = (logits[torch.from_numpy(ids)] / temperature).softmax(1)
    accepted = class_decisions(probabilities, policy["class98_thresholds"]) if args.policy == "class98" else accepted_mask(probabilities, policy["global97_threshold"])
    prediction, confidence = probabilities.argmax(1), probabilities.max(1).values
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["node_id", "predicted_label", "predicted_class", "confidence", "accepted", "route", "reason", "policy"])
        for node, label, certainty, decision in zip(ids.tolist(), prediction.tolist(), confidence.tolist(), accepted.tolist()):
            cutoff = policy["class98_thresholds"][label] if args.policy == "class98" else policy["global97_threshold"]
            reason = "eligible_research_candidate" if decision else ("class_policy_infeasible_on_fitting_data" if cutoff > 1 else "below_frozen_threshold")
            writer.writerow([node, label, CLASS_NAMES[label], certainty, decision,
                             "automatic_candidate" if decision else "human_review", reason, args.policy])
    print(json.dumps({"predictions": len(ids), "accepted": int(accepted.sum()),
                      "human_review": int((~accepted).sum()), "reference_labels_exported_or_used_for_decisions": False,
                      "independent_external_validation": False, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
