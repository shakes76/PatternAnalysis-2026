"""Apply a frozen exact-threshold policy with optional random quality-review sampling.

Yuhan Liao - 49839991. Decisions never consult reference labels.
"""
from pathlib import Path
import argparse
import csv
import json
import math

import numpy as np
import torch

from dataset import CLASS_NAMES, DATA_SHA256, DEFAULT_PATH, canonical_edges, load_arrays, sha256, standardize_features
from exact_policy import exact_decisions, random_audit_mask
from modules import make_model, normalized_adjacency
from validation_study import ROLES, SPLIT_SEED


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--run", choices=[f"{variant}_seed{seed}" for variant in ["unweighted_exact", "weighted_exact"] for seed in [42, 43, 44]], default="unweighted_exact_seed42")
    parser.add_argument("--data", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--subset", choices=["all", "final"], default="final")
    parser.add_argument("--audit-fraction", type=float, default=0.)
    parser.add_argument("--audit-seed", type=int, default=20261026)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Choose a new output filename; previous exports are preserved.")
    if not math.isfinite(args.audit_fraction) or not 0 <= args.audit_fraction <= 1:
        parser.error("Audit fraction must be in [0, 1]")
    torch.set_num_threads(4)
    folder = args.study_dir / args.run
    policy = read(folder / "policy.json")
    frozen = read(args.study_dir / "ALL_POLICIES_FROZEN.json")
    if sha256(folder / "policy.json") != frozen[args.run] or sha256(folder / "model.pt") != policy["checkpoint_sha256"]:
        raise ValueError("Frozen policy/checkpoint binding mismatch")
    if sha256(args.study_dir / "FROZEN_PROTOCOL.json") != policy["protocol_sha256"]:
        raise ValueError("Protocol binding mismatch")
    if sha256(args.data) != DATA_SHA256 or policy["dataset_sha256"] != DATA_SHA256:
        raise ValueError("Pinned dataset required")
    if sha256(args.study_dir / "split_ids.npz") != policy["split_sha256"]:
        raise ValueError("Split/scaling binding mismatch")
    checkpoint = torch.load(folder / "model.pt", map_location="cpu", weights_only=True)
    if args.run != f"{policy['variant']}_seed{policy['seed']}" or checkpoint["initialization_seed"] != policy["seed"]:
        raise ValueError("Model/policy identity mismatch")
    if checkpoint["split_seed"] != SPLIT_SEED or checkpoint["model"] != "gcn" or checkpoint["dataset_sha256"] != DATA_SHA256:
        raise ValueError("Checkpoint configuration mismatch")
    expected_phase = "posthoc_weighted_refinement" if policy["variant"] == "weighted_exact" else "within_graph_five_role_study"
    if checkpoint["phase"] != expected_phase:
        raise ValueError("Checkpoint phase mismatch")
    temperature = policy["temperature"]
    if not math.isfinite(temperature) or temperature <= 0 or temperature != checkpoint["temperature"]:
        raise ValueError("Temperature binding mismatch")
    if policy["variant"] == "weighted_exact":
        protocol = read(args.study_dir / "FROZEN_PROTOCOL.json")
        torch.testing.assert_close(checkpoint["class_weights"], torch.tensor(protocol["class_weights"]))
    raw, _, edges = load_arrays(args.data)
    with np.load(args.study_dir / "split_ids.npz", allow_pickle=False) as saved:
        roles = {name: saved[name].copy() for name in ROLES}
        mean, scale = saved["feature_mean"].copy(), saved["feature_scale"].copy()
    if not np.array_equal(np.sort(np.concatenate(list(roles.values()))), np.arange(len(raw))):
        raise ValueError("Roles must be disjoint and exhaustive")
    x, actual_mean, actual_scale = standardize_features(raw, roles["train"])
    if not np.array_equal(actual_mean, mean) or not np.array_equal(actual_scale, scale):
        raise ValueError("Recorded training-only scaling mismatch")
    model = make_model("gcn", **checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    adjacency = normalized_adjacency(torch.from_numpy(canonical_edges(edges, len(raw))), len(raw))
    ids = roles["final"] if args.subset == "final" else np.arange(len(raw))
    with torch.no_grad():
        logits, _ = model(torch.from_numpy(x), adjacency)
        p = (logits[torch.from_numpy(ids)] / temperature).softmax(1)
    accepted = exact_decisions(p, policy["thresholds"])
    audited = random_audit_mask(accepted, args.audit_fraction, args.audit_seed)
    prediction, confidence = p.argmax(1), p.max(1).values
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["node_id", "predicted_label", "predicted_class", "confidence", "threshold",
                         "policy_accepted", "random_audit_selected", "route", "reason", "policy_id"])
        for node, label, certainty, eligible, audit in zip(ids.tolist(), prediction.tolist(), confidence.tolist(), accepted.tolist(), audited.tolist()):
            cutoff = policy["thresholds"][label]
            route = "human_review_threshold" if not eligible else ("human_review_random_audit" if audit else "automatic_candidate")
            reason = "no_feasible_class_policy" if cutoff > 1 else ("below_exact_threshold" if not eligible else ("random_quality_sample" if audit else "eligible_research_candidate"))
            writer.writerow([node, label, CLASS_NAMES[label], certainty, cutoff, eligible, audit, route, reason, policy["id"]])
    print(json.dumps({"predictions": len(ids), "threshold_eligible": int(accepted.sum()),
                      "threshold_review": int((~accepted).sum()), "additional_random_audit": int(audited.sum()),
                      "automatic_after_audit": int((accepted & ~audited).sum()), "audit_fraction_of_eligible": args.audit_fraction,
                      "reference_labels_exported_or_used_for_decisions": False,
                      "human_correction_accuracy_assumed": False, "external_independent_validation": False,
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
