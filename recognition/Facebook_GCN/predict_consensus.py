"""Reload frozen consensus policies and export routes without reference labels.

Yuhan Liao - 49839991. This is a post-hoc research candidate, not a risk guarantee.
"""
from pathlib import Path
import argparse
import csv
import json
import math

import numpy as np
import torch

from consensus_policy import (MEMBERS, calibrated_average, load_members,
                              member_probabilities, prepare_inputs, read, route)
from dataset import CLASS_NAMES, DATA_SHA256, DEFAULT_PATH, sha256
from exact_policy import random_audit_mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--policy", choices=["mean_only", "unanimous"], default="unanimous")
    parser.add_argument("--subset", choices=["all", "final"], default="final")
    parser.add_argument("--audit-fraction", type=float, default=0.)
    parser.add_argument("--audit-seed", type=int, default=20261026)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Choose a new output filename; existing exports are preserved.")
    if not math.isfinite(args.audit_fraction) or not 0 <= args.audit_fraction <= 1:
        parser.error("Audit fraction must be in [0, 1]")
    torch.set_num_threads(4)
    folder = args.study_dir
    policy = read(folder / f"{args.policy}_policy.json")
    frozen = read(folder / "ALL_POLICIES_FROZEN.json")
    if sha256(folder / f"{args.policy}_policy.json") != frozen[args.policy]:
        raise ValueError("Frozen ensemble policy mismatch")
    if policy["name"] != args.policy or policy["require_unanimity"] != (args.policy == "unanimous"):
        raise ValueError("Policy identity mismatch")
    if sha256(folder / "FROZEN_PROTOCOL.json") != policy["protocol_sha256"]:
        raise ValueError("Ensemble protocol mismatch")
    base = folder / "members"
    if sha256(base / "ALL_POLICIES_FROZEN.json") != policy["members_manifest_sha256"]:
        raise ValueError("Member manifest mismatch")
    if sha256(base / "split_ids.npz") != policy["split_sha256"] or policy["dataset_sha256"] != DATA_SHA256:
        raise ValueError("Ensemble data/split mismatch")
    x, adjacency, roles = prepare_inputs(base, args.data)
    members, identities = load_members(base)
    if identities != policy["member_identities"]:
        raise ValueError("Ensemble member identity mismatch")
    p, agreement, votes = calibrated_average(member_probabilities(members, x, adjacency), policy["temperature"])
    ids = roles["final"] if args.subset == "final" else np.arange(len(x))
    ids_tensor = torch.from_numpy(ids)
    p, agreement, votes = p[ids_tensor], agreement[ids_tensor], votes[:, ids_tensor]
    accepted = route(p, agreement, policy["thresholds"], policy["require_unanimity"])
    audited = random_audit_mask(accepted, args.audit_fraction, args.audit_seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["node_id", "predicted_label", "predicted_class", "confidence", "threshold", "six_votes", "unanimous",
                         "policy_accepted", "random_audit_selected", "route", "reason", "policy_id"])
        for index, node in enumerate(ids.tolist()):
            label = int(p[index].argmax())
            eligible, audit = bool(accepted[index]), bool(audited[index])
            disagreement = policy["require_unanimity"] and not bool(agreement[index])
            destination = "human_review_disagreement" if disagreement else ("human_review_threshold" if not eligible else ("human_review_random_audit" if audit else "automatic_candidate"))
            reason = "model_disagreement" if disagreement else ("below_frozen_class_threshold" if not eligible else ("random_quality_sample" if audit else "eligible_research_candidate"))
            writer.writerow([node, label, CLASS_NAMES[label], float(p[index].max()), policy["thresholds"][label],
                             ";".join(str(v) for v in votes[:, index].tolist()), bool(agreement[index]), eligible,
                             audit, destination, reason, policy["id"]])
    print(json.dumps({"policy": args.policy, "cases": len(ids), "policy_accepted": int(accepted.sum()),
                      "policy_review": int((~accepted).sum()), "disagreement_cases": int((~agreement).sum()),
                      "additional_random_review": int(audited.sum()), "no_ground_truth_used_in_routes": True,
                      "human_correction_measured": False, "external_validation": False, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
