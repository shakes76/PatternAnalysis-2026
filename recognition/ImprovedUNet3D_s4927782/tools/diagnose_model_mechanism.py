"""Summarize frozen v1 model mechanisms without fitting or changing predictions.

FP32, unaugmented inputs only. TEST access requires an explicit acknowledgement
and is restricted to the eight previously examined M015 visits. Backward is an
optional diagnostic on three prespecified cases, in eval mode, with no optimizer.
The fourteen-case allowlist adds two exploratory TRAIN controls chosen after
initial descriptive contrast statistics; these are not prespecified controls.
Only a fresh directory containing a JSON summary is written; arrays are not saved.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys


PROJECT = Path(__file__).resolve().parents[1]
FROZEN_SHA = {
    "protocol": "9e2585934579890d7194e5460338f2accc12c15463ea529ac969fd94d95bee91",
    "manifest": "c47fe880d353629348786b852e3f699291af1a4d5628b2980bf42a74635d0424",
    "sources": "a3cf4ed4eb21c9d79df9a50b53308df797a30cb112655f4ee85d809ad7916e54",
    "identity": "eacb84bf20626a6ec9d7f6d36489a81e2d4e4af9fbe4ade1ce8f508a2022e08a",
}
CHECKPOINT_SHA = {
    "unet2d": "cfdcfa08147c295962e7d19f2ad16eb20a08590f9a28e00e95a281417aec3c10",
    "improved3d": "9ea46e13a95617ec4ad80c21ba459087fb62941fb04406c7ffeade22ad014b62",
}
ALLOWED_CASES = {
    "B006/Week0": "train", "K019/Week1": "train",
    "B040/Week0": "validation", "K018/Week0": "validation",
    "R016/Week7": "train", "T014/Week3": "train",
    **{f"M015/Week{i}": "test" for i in range(8)},
}
EXPLORATORY_TRAIN_CASES = frozenset(("R016/Week7", "T014/Week3"))
BACKWARD_CASES = frozenset(("B006/Week0", "B040/Week0", "M015/Week2"))
QUANTILES = (0, .01, .05, .25, .5, .75, .95, .99, 1)


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def select_cases(manifest, requested, backward, ack):
    if manifest.get("partial") or manifest.get("development_only"):
        raise ValueError("Development/smoke manifests are forbidden.")
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("Explicit, unique --case keys are required.")
    if len(set(backward)) != len(backward) or not set(backward) <= BACKWARD_CASES:
        raise ValueError("Backward is restricted to the three prespecified cases.")
    if not set(backward) <= set(requested):
        raise ValueError("Every --backward-case must also be an explicit --case.")
    records = {f"{r['patient']}/{r['visit']}": r for r in manifest["records"]}
    if len(records) != len(manifest["records"]):
        raise ValueError("Duplicate manifest case keys.")
    selected = []
    for key in requested:
        if key not in ALLOWED_CASES or key not in records:
            raise ValueError(f"Case is outside the fixed fourteen-case diagnostic set: {key}.")
        record = records[key]
        if record["split"] != ALLOWED_CASES[key]:
            raise ValueError("Case split differs from the frozen diagnostic allowlist.")
        if record["split"] == "test" and not ack:
            raise ValueError("M015 TEST access requires --diagnostic-test-ack.")
        selected.append(record)
    return selected


def quantiles(values):
    import numpy as np
    values = np.asarray(values, dtype=np.float64)
    return {str(q): float(np.quantile(values, q)) if values.size else None for q in QUANTILES}


def class5_summary(rows, num_classes=6):
    """Summarize only GT-class5 logit rows, never save the underlying rows.

    FP32 logits are summarized with stable float64 softmax/reductions. Rank uses
    strictly greater logits (ties share rank); argmax ties select the lowest ID.
    """
    import numpy as np
    rows = np.asarray(rows, dtype=np.float64).reshape(-1, num_classes)
    finite = np.isfinite(rows).all(axis=1)
    valid = rows[finite]
    n = valid.shape[0]
    if n:
        exponent = np.exp(valid - valid.max(axis=1, keepdims=True))
        probabilities = exponent / exponent.sum(axis=1, keepdims=True)
        ranks = 1 + (valid > valid[:, 5, None]).sum(axis=1)
        competitor = np.max(np.delete(valid, 5, axis=1), axis=1)
        winners = np.argmax(valid, axis=1)
        counts = np.bincount(winners, minlength=num_classes).tolist()
    else:
        probabilities = np.empty((0, num_classes))
        ranks, competitor, winners = np.empty(0), np.empty(0), np.empty(0)
        counts = [0] * num_classes
    return {
        "gt_class5_voxels": int(rows.shape[0]), "finite_gt_logit_rows": int(n),
        "nonfinite_gt_logit_rows": int((~finite).sum()),
        "statistics_scope": "finite GT-class5 rows only; FP32 logits, float64 summaries",
        "mean_softmax_probability_by_class": [float(v) for v in probabilities.mean(axis=0)] if n else [None] * num_classes,
        "class5_rank_counts": {str(i): int((ranks == i).sum()) for i in range(1, num_classes + 1)},
        "rank_definition": "1 + number of strictly larger logits; ties share rank",
        "class5_margin_over_max_competitor_quantiles": quantiles(valid[:, 5] - competitor if n else []),
        "class5_tied_with_max_competitor_voxels": int((valid[:, 5] == competitor).sum()) if n else 0,
        "gt_class5_predicted_as_class_counts": counts,
        "gt_class5_errors_predicted_as_class_counts": [count if i != 5 else 0 for i, count in enumerate(counts)],
    }


def tensor_summary(tensor):
    import torch
    values = tensor.detach()
    finite = torch.isfinite(values)
    selected = values[finite].double()
    return {
        "elements": values.numel(), "nonfinite_elements": int((~finite).sum().item()),
        "mean_finite": float(selected.mean().item()) if selected.numel() else None,
        "variance_finite_population": float(selected.var(unbiased=False).item()) if selected.numel() else None,
    }


class GroupNormCapture:
    """Keep numerical summaries only; no feature tensors or graphs are retained."""

    def __init__(self, model):
        import torch
        self.rows, self.handles = [], []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.GroupNorm):
                def hook(layer, inputs, output, name=name):
                    self.rows.append({"layer": name, "groups": layer.num_groups,
                                      "input": tensor_summary(inputs[0]), "output": tensor_summary(output)})
                self.handles.append(module.register_forward_hook(hook))

    def close(self):
        for handle in self.handles:
            handle.remove()


def nearest_target(target, size):
    """Independent floor-index nearest rule used by PyTorch's 'nearest' mode."""
    import torch
    resized = target
    for axis, length in enumerate(size, start=1):
        old = resized.shape[axis]
        indices = torch.div(torch.arange(length, device=target.device) * old, length, rounding_mode="floor")
        resized = resized.index_select(axis, indices)
    return resized


def independent_head_loss(logits, target, class_weights, smooth=1e-5):
    """Explicit weighted NLL and per-class soft-Dice, without core loss helpers."""
    import torch
    target = nearest_target(target, logits.shape[2:])
    values = logits.float()
    log_probabilities = values - torch.logsumexp(values, dim=1, keepdim=True)
    nll = -log_probabilities.gather(1, target.unsqueeze(1)).squeeze(1)
    voxel_weights = class_weights.float()[target]
    ce = (nll * voxel_weights).sum() / voxel_weights.sum()
    probabilities = log_probabilities.exp()
    dice_values, counts = [], []
    for index in range(values.shape[1]):
        actual = target == index
        count = actual.sum()
        counts.append(int(count.item()))
        probability = probabilities[:, index]
        intersection = probability[actual].sum()
        dice_values.append((2 * intersection + smooth) / (probability.sum() + count + smooth))
    dice_loss = 1 - torch.stack(dice_values).mean()
    return ce, dice_loss, counts, target


def state_sha256(model):
    import torch
    result = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        header = json.dumps([name, str(value.dtype), list(value.shape)], separators=(",", ":")).encode()
        raw = value.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes()
        result.update(header + b"\0" + raw)
    return result.hexdigest()


def gradient_summary(model, heads):
    import torch
    rows = []
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        row = {"parameter": name, "component": name.split(".")[0], "elements": parameter.numel(),
               "gradient_present": gradient is not None}
        if gradient is not None:
            finite = torch.isfinite(gradient)
            row.update(nonfinite_elements=int((~finite).sum().item()),
                       zero_elements=int((gradient == 0).sum().item()),
                       all_zero=bool((gradient == 0).all().item()),
                       l2_norm=float(torch.linalg.vector_norm(gradient.double()).item()) if bool(finite.all().item()) else None)
        rows.append(row)
    logits = [{"head": name, "gradient_present": head.grad is not None,
               **(tensor_summary(head.grad) if head.grad is not None else {}),
               "all_zero": bool((head.grad == 0).all().item()) if head.grad is not None else None}
              for name, head in heads]
    return {"parameters": rows, "head_logits": logits,
            "all_parameter_gradients_present": all(r["gradient_present"] for r in rows),
            "all_parameter_gradients_finite": all(r["gradient_present"] and r["nonfinite_elements"] == 0 for r in rows),
            "any_all_zero_parameter_gradient": any(r.get("all_zero", False) for r in rows)}


def diagnose_3d(model, image, target, device, class_weights, backward, core):
    import numpy as np
    import torch
    from torch.nn import functional as F
    x = torch.from_numpy(np.ascontiguousarray(image[None, None])).to(device)
    y = torch.from_numpy(np.ascontiguousarray(target[None])).long().to(device)
    before = state_sha256(model)
    model.zero_grad(set_to_none=True)
    capture = GroupNormCapture(model)
    result = {"precision": "FP32; autocast disabled", "model_mode": "eval", "backward_requested": backward}
    try:
        with torch.set_grad_enabled(backward), torch.autocast(device_type=device.type, enabled=False):
            output = model(x)
            heads = [("main", output["logits"])] + list(zip(("half", "quarter"), output["aux_logits"]))
            if len(heads) != 3:
                raise ValueError("Frozen 3D model must expose main/half/quarter heads.")
            if backward:
                for _, head in heads:
                    head.retain_grad()
            all_finite = all(bool(torch.isfinite(head).all().item()) for _, head in heads)
            result["all_head_logits_finite"] = all_finite
            result["head_logits_finiteness"] = [
                {"head": name, "shape": list(head.shape), "elements": head.numel(),
                 "nonfinite_elements": int((~torch.isfinite(head)).sum().item())}
                for name, head in heads
            ]
            result["heads"] = []
            independent_total = 0.0
            if all_finite:
                for (name, head), weight in zip(heads, (1.0, .5, .25)):
                    with torch.no_grad():
                        ce, dice, counts, resized = independent_head_loss(head.detach(), y, class_weights)
                        actual_ce = core._categorical_head_loss(head.detach(), y, 1., 0., 1e-5, class_weights)
                        actual_dice = core._categorical_head_loss(head.detach(), y, 0., 1., 1e-5, class_weights)
                        reference_target = F.interpolate(y.unsqueeze(1).float(), size=head.shape[2:], mode="nearest").squeeze(1).long()
                        gt_rows = head.detach().movedim(1, -1)[resized == 5].cpu().numpy()
                        ce_error, dice_error = abs(float(ce.item()) - float(actual_ce.item())), abs(float(dice.item()) - float(actual_dice.item()))
                        result["heads"].append({"head": name, "shape": list(head.shape), "weight_before_normalization": weight,
                                                "nearest_target_class_counts": counts,
                                                "nearest_target_matches_torch": bool(torch.equal(resized, reference_target)),
                                                "independent_ce": float(ce.item()), "actual_ce": float(actual_ce.item()),
                                                "independent_dice_loss": float(dice.item()), "actual_dice_loss": float(actual_dice.item()),
                                                "ce_absolute_difference": ce_error, "dice_absolute_difference": dice_error,
                                                "formula_matches_at_1e_minus5": ce_error <= 1e-5 and dice_error <= 1e-5,
                                                "gt_class5": class5_summary(gt_rows)})
                        independent_total += weight * (float(ce.item()) + float(dice.item())) / 1.75
                actual = core.segmentation_loss(output, y, class_weights=class_weights)
                actual_finite = bool(torch.isfinite(actual).item())
                result["loss"] = {"actual": float(actual.item()) if actual_finite else None,
                                  "independent_weighted_total": independent_total,
                                  "head_weight_sum": 1.75, "actual_finite": actual_finite,
                                  "formula_matches_at_1e_minus5": actual_finite and abs(float(actual.item()) - independent_total) <= 1e-5}
                if backward and actual_finite:
                    actual.backward()
                    result["backward"] = {"status": "completed", **gradient_summary(model, heads)}
                else:
                    result["backward"] = {"status": "not_requested" if not backward else "skipped_nonfinite_loss"}
            else:
                result["backward"] = {"status": "skipped_nonfinite_logits" if backward else "not_requested"}
    finally:
        capture.close()
        result["group_norm_layers"] = capture.rows
        result["all_group_norm_values_finite"] = all(
            row[side]["nonfinite_elements"] == 0 for row in capture.rows for side in ("input", "output")
        )
        model.zero_grad(set_to_none=True)
        after = state_sha256(model)
        result["state_dict_before_sha256"], result["state_dict_after_sha256"] = before, after
        result["parameters_and_buffers_unchanged"] = before == after
    return result


def diagnose_2d(model, image, target, device):
    import numpy as np
    import torch
    rows, nonfinite, elements = [], 0, 0
    before = state_sha256(model)
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
        for start in range(0, image.shape[0], 32):
            x = torch.from_numpy(np.ascontiguousarray(image[start:start + 32, None])).to(device)
            logits = model(x)["logits"]
            elements += logits.numel()
            nonfinite += int((~torch.isfinite(logits)).sum().item())
            selected = torch.from_numpy(np.ascontiguousarray(target[start:start + 32] == 5)).to(device)
            rows.append(logits.movedim(1, -1)[selected].cpu().numpy())
    after = state_sha256(model)
    return {"precision": "FP32; autocast disabled", "model_mode": "eval", "slice_batch_size": 32,
            "backward": {"status": "not_requested_2d_forward_only"}, "logit_elements": elements,
            "nonfinite_logit_elements": nonfinite, "all_head_logits_finite": nonfinite == 0,
            "gt_class5": class5_summary(np.concatenate(rows)), "group_norm_layers": [],
            "state_dict_before_sha256": before, "state_dict_after_sha256": after,
            "parameters_and_buffers_unchanged": before == after}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, default=PROJECT / "artifacts/full_seed42/unet2d/best.pt")
    parser.add_argument("--model-checkpoint", type=Path, default=PROJECT / "artifacts/full_seed42/improved3d/best.pt")
    parser.add_argument("--protocol", type=Path, default=PROJECT / "test_protocol.json")
    parser.add_argument("--training-sources", type=Path, default=PROJECT / "training_sources.json")
    parser.add_argument("--cache-identity", type=Path)
    parser.add_argument("--case", action="append", required=True)
    parser.add_argument("--backward-case", action="append", default=[], help="3D only; B006/Week0, B040/Week0 or M015/Week2. Also requires --case.")
    parser.add_argument("--diagnostic-test-ack", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args(argv)


def verify_documents(paths):
    documents = {}
    for name, path in paths.items():
        if sha256(path) != FROZEN_SHA[name]:
            raise ValueError(f"Frozen v1 {name} SHA mismatch.")
        documents[name] = json.loads(path.read_text())
    protocol, manifest, sources, identity = (documents[n] for n in ("protocol", "manifest", "sources", "identity"))
    if protocol["training_seed"] != 42 or protocol["training_epochs_per_model"] != 30:
        raise ValueError("Only the frozen v1 seed42 protocol is permitted.")
    if any(d["manifest_sha256"] != FROZEN_SHA["manifest"] for d in (protocol, sources, identity)):
        raise ValueError("Frozen evidence manifest identity mismatch.")
    if identity["record_count"] != 211 or set(identity["records"]) != {r["cache_path"] for r in manifest["records"]}:
        raise ValueError("Cache identity must cover all 211 manifest records exactly.")
    for relative, digest in sources["source_file_sha256"].items():
        if sha256(PROJECT / relative) != digest:
            raise ValueError(f"Frozen training source changed: {relative}.")
    for name, digest in CHECKPOINT_SHA.items():
        if protocol["models"][name]["checkpoint_sha256"] != digest:
            raise ValueError("Protocol checkpoint is not the frozen v1 model.")
    return documents


def main(argv=None):
    args = parse_args(argv)
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise ValueError("Output directory must be new; refusing to overwrite evidence.")
    paths = {"manifest": args.manifest.expanduser().resolve(), "protocol": args.protocol.expanduser().resolve(),
             "sources": args.training_sources.expanduser().resolve(),
             "identity": (args.cache_identity or args.manifest.parent / "cache_identity.json").expanduser().resolve()}
    docs = verify_documents(paths)
    selected = select_cases(docs["manifest"], args.case, args.backward_case, args.diagnostic_test_ack)
    sys.path.insert(0, str(PROJECT))
    import numpy as np
    import torch
    import modules as core
    from tools.diagnose_frozen_inference import load_cache
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable.")
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    models, configs, evidence = {}, {}, {}
    for name, checkpoint_path in (("unet2d", args.baseline_checkpoint), ("improved3d", args.model_checkpoint)):
        checkpoint_path = checkpoint_path.expanduser().resolve()
        if output.is_relative_to(checkpoint_path.parent) or output.is_relative_to(paths["manifest"].parent):
            raise ValueError("Output must not be inside frozen checkpoint/cache directories.")
        if sha256(checkpoint_path) != CHECKPOINT_SHA[name]:
            raise ValueError("Only the fixed v1 seed42 checkpoint SHA is permitted; seed43 is forbidden.")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        config = checkpoint["config"]
        if (config["seed"] != 42 or config["model"] != name or config["manifest_sha256"] != FROZEN_SHA["manifest"]
                or config["num_classes"] != 6 or config["label_ids"] != [0, 1, 2, 3, 4, 5]
                or config["target_shape_dhw"] != docs["manifest"]["target_shape_dhw"]
                or config.get("development_only") or config.get("max_batches") is not None
                or checkpoint["epoch"] != docs["protocol"]["models"][name]["selected_epoch"]):
            raise ValueError("Checkpoint configuration/epoch differs from frozen v1.")
        factory = core.UNet2D if name == "unet2d" else core.ImprovedUNet3D
        model = factory(6, config["base_channels"]).to(device).eval()
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        models[name], configs[name] = model, config
        evidence[name] = {"checkpoint": str(checkpoint_path), "checkpoint_sha256": CHECKPOINT_SHA[name],
                          "selected_epoch": checkpoint["epoch"], "state_dict_before_sha256": state_sha256(model), "config": config}
        del checkpoint
    if configs["unet2d"]["resolved_class_weights"] != configs["improved3d"]["resolved_class_weights"]:
        raise ValueError("Frozen models must use the same training-derived class weights.")
    weights = torch.tensor(configs["improved3d"]["resolved_class_weights"], dtype=torch.float32, device=device)
    report = {"schema_version": 1, "diagnostic_only": True, "no_new_final_test_scores": True,
              "no_parameter_updates": True, "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "explicit_case_keys": args.case, "explicit_3d_backward_case_keys": args.backward_case,
              "case_selection": {
                  "original_case_keys": [key for key in args.case if key not in EXPLORATORY_TRAIN_CASES],
                  "exploratory_added_train_case_keys": [key for key in args.case if key in EXPLORATORY_TRAIN_CASES],
                  "exploratory_selection_basis": "R016/Week7 and T014/Week3 were appended after initial descriptive statistics as the two lowest-contrast cases among the 143 TRAIN caches; post-hoc exploratory TRAIN controls, not prespecified or independent validation.",
                  "backward_allowlist_unchanged": sorted(BACKWARD_CASES)},
              "diagnostic_test_ack": args.diagnostic_test_ack,
              "evidence": {k: {"path": str(v), "sha256": sha256(v)} for k, v in paths.items()},
              "tool_sha256": sha256(__file__), "cache_loader_sha256": sha256(PROJECT / "tools/diagnose_frozen_inference.py"),
              "models": evidence, "runtime": {"python": platform.python_version(), "numpy": np.__version__,
              "torch": str(torch.__version__), "cuda": torch.version.cuda, "device": str(device),
              "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
              "precision": "FP32; autocast/TF32 disabled; deterministic algorithms warn_only"},
              "limits": ["Retrospective mechanism diagnostics, not new final TEST evaluation or model selection.",
                         "No augmentation, postprocessing, translated inputs, optimizer, or saved arrays.",
                         "Backward checks current frozen weights only; they do not reconstruct historical training gradients or AMP effects.",
                         "GroupNorm/score associations do not establish a cause; annotation truth is not validated here.",
                         "2D is forward-only in 32-slice chunks; 3D loss/backward uses one entire real volume in eval mode."],
              "cases": []}
    output.mkdir(parents=True, exist_ok=False)
    for record in selected:
        key = f"{record['patient']}/{record['visit']}"
        row = {"key": key, "split": record["split"], "models": {}}
        try:
            image, target, spacing, cache = load_cache(paths["manifest"], record, docs["manifest"], docs["identity"])
            row.update(cache=cache, shape_dhw=list(image.shape), spacing_dhw_mm=spacing.tolist())
            row["models"]["unet2d"] = diagnose_2d(models["unet2d"], image, target, device)
            row["models"]["improved3d"] = diagnose_3d(models["improved3d"], image, target, device, weights, key in args.backward_case, core)
            items = list(row["models"].values())
            good = all(i["parameters_and_buffers_unchanged"] and i["all_head_logits_finite"] for i in items)
            three = row["models"]["improved3d"]
            good = good and three["all_group_norm_values_finite"]
            good = good and three.get("loss", {}).get("formula_matches_at_1e_minus5", False)
            good = good and all(h["nearest_target_matches_torch"] and h["formula_matches_at_1e_minus5"] for h in three["heads"])
            if key in args.backward_case:
                good = good and three["backward"].get("all_parameter_gradients_present", False) and three["backward"].get("all_parameter_gradients_finite", False)
            row["status"] = "checks_passed" if good else "check_failure"
        except Exception as error:
            row.update(status="blocked", error_type=type(error).__name__, error=str(error))
        report["cases"].append(row)
    for name, model in models.items():
        evidence[name]["state_dict_after_sha256"] = state_sha256(model)
        evidence[name]["parameters_and_buffers_unchanged"] = evidence[name]["state_dict_before_sha256"] == evidence[name]["state_dict_after_sha256"]
        evidence[name]["checkpoint_file_unchanged"] = sha256(evidence[name]["checkpoint"]) == CHECKPOINT_SHA[name]
    unchanged = all(e["parameters_and_buffers_unchanged"] and e["checkpoint_file_unchanged"] for e in evidence.values())
    report["status"] = "checks_passed" if unchanged and all(r["status"] == "checks_passed" for r in report["cases"]) else "incomplete_or_check_failure"
    destination = output / "mechanism.json"
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Frozen mechanism diagnostic {report['status']}: {destination}")
    return 0 if report["status"] == "checks_passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
