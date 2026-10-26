"""Build a bounded public summary from six explicit private diagnostic JSONs.

No MRI, mask, cache, checkpoint, prediction, network or cluster access. Unknown
input fields are discarded: the public schema is reconstructed from numeric
whitelists. All fourteen diagnostic cases are required. This is retrospective
evidence and never replaces the frozen v1 TEST result.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import tempfile


PROJECT = Path(__file__).resolve().parents[1]
FROZEN = {
    "manifest": "c47fe880d353629348786b852e3f699291af1a4d5628b2980bf42a74635d0424",
    "protocol": "9e2585934579890d7194e5460338f2accc12c15463ea529ac969fd94d95bee91",
    "sources": "a3cf4ed4eb21c9d79df9a50b53308df797a30cb112655f4ee85d809ad7916e54",
    "identity": "eacb84bf20626a6ec9d7f6d36489a81e2d4e4af9fbe4ade1ce8f508a2022e08a",
}
WEIGHTS = {"unet2d": "cfdcfa08147c295962e7d19f2ad16eb20a08590f9a28e00e95a281417aec3c10",
           "improved3d": "9ea46e13a95617ec4ad80c21ba459087fb62941fb04406c7ffeade22ad014b62"}
CASES = {"B006/Week0": "train", "K019/Week1": "train", "B040/Week0": "validation", "K018/Week0": "validation",
         "R016/Week7": "train", "T014/Week3": "train", **{f"M015/Week{i}": "test" for i in range(8)}}
EXPLORATORY = ("R016/Week7", "T014/Week3")
BACKWARD = {"B006/Week0", "B040/Week0", "M015/Week2"}
ROLES = ("source", "mechanism", "cache_descriptors", "archive_identity", "official_masks", "support_intervention")
STAGES = ("native_scaled_before_normalization", "canonical_normalized_unclipped", "canonical_normalized_clipped",
          "cached_linear_before_support_forcing", "cached_after_support_forcing")
SOURCE_CONTRASTS = ("body_iqr", "bladder_minus_body_median", "bladder_minus_body_median_div_body_iqr",
                    "prostate_minus_body_median", "prostate_minus_body_median_div_body_iqr")
CACHE_CONTRASTS = ("bladder_median_minus_body_median", "bladder_body_iqr_standardized_median_gap",
                   "prostate_median_minus_body_median", "prostate_body_iqr_standardized_median_gap", "bladder_median_minus_prostate_median")
MARGIN_KEYS = ("0.05", "0.5", "0.95")
MAX_BYTES = 100_000


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), "Invalid evidence SHA-256.")
    return value


def number(value, nullable=True):
    if nullable and value is None:
        return None
    require(type(value) in (int, float) and math.isfinite(value), "Expected a finite numeric summary field.")
    return value


def count(value):
    require(type(value) is int and value >= 0, "Expected a nonnegative integer count.")
    return value


def nums(row, keys):
    return {key: number(row[key]) for key in keys}


def counts(row, keys):
    return {key: count(row[key]) for key in keys}


def case_key(row):
    return f"{row['patient']}/{row['visit']}"


def index_fourteen(rows, field):
    indexed = {row[field]: row for row in rows}
    require(len(rows) == len(indexed) == 14 and set(indexed) == set(CASES), "Exactly the fourteen diagnostic cases are required.")
    require(all(row["split"] == CASES[key] for key, row in indexed.items()), "Diagnostic case split changed.")
    return indexed


def roi(stats):
    q = stats["quantiles"]
    return {"n": count(stats["n"]), "mean": number(stats["mean"]), "std": number(stats["std"]),
            "median": number(q["q50"]), "iqr": number(q["q75"] - q["q25"]) if q["q75"] is not None and q["q25"] is not None else None,
            "zero_fraction": number(stats["zero_fraction"]), "clip_boundary_fraction": number(stats["at_or_beyond_clip_fraction"])}


def scores(row):
    probabilities = row["mean_softmax_probability_by_class"]
    require(len(probabilities) == 6, "Six categorical probabilities required.")
    result = {"n_gt": count(row["gt_class5_voxels"]), "nonfinite_rows": count(row["nonfinite_gt_logit_rows"]),
              "mean_probabilities": [number(v) for v in probabilities],
              "rank_counts": counts(row["class5_rank_counts"], tuple(str(i) for i in range(1, 7))),
              "margin_quantiles": nums(row["class5_margin_over_max_competitor_quantiles"], MARGIN_KEYS),
              "gt_predicted_as_counts": [count(v) for v in row["gt_class5_predicted_as_class_counts"]]}
    require(len(result["gt_predicted_as_counts"]) == 6, "Six prediction counts required.")
    require(sum(result["rank_counts"].values()) + result["nonfinite_rows"] == result["n_gt"], "Rank support is inconsistent.")
    require(sum(result["gt_predicted_as_counts"]) + result["nonfinite_rows"] == result["n_gt"], "Prediction support is inconsistent.")
    return result


def compact_mechanism(row):
    for model in row["models"].values():
        require(model["parameters_and_buffers_unchanged"] is True and model["all_head_logits_finite"] is True, "Mechanism forward/state check failed.")
    model = row["models"]["improved3d"]
    require(model["all_group_norm_values_finite"] is True, "Nonfinite GroupNorm evidence.")
    require([h["head"] for h in model["heads"]] == ["main", "half", "quarter"], "Unexpected 3D heads.")
    heads = []
    for h in model["heads"]:
        require(h["nearest_target_matches_torch"] is True and h["formula_matches_at_1e_minus5"] is True, "Head formula/nearest check failed.")
        targets = [count(v) for v in h["nearest_target_class_counts"]]
        require(len(targets) == 6, "Six target counts required.")
        head_scores = scores(h["gt_class5"])
        require(targets[5] == head_scores["n_gt"], "Head target support disagrees.")
        heads.append({"head": h["head"], "target_counts": targets, "scores": head_scores,
                      "loss": {"ce_absolute_difference": abs(number(h["actual_ce"], False) - number(h["independent_ce"], False)),
                               "dice_absolute_difference": abs(number(h["actual_dice_loss"], False) - number(h["independent_dice_loss"], False))},
                      "loss_matches": True, "nearest_matches": True})
    loss = model["loss"]
    require(loss["actual_finite"] is True and loss["formula_matches_at_1e_minus5"] is True, "Total loss check failed.")
    backward = model["backward"]
    gradient = {"performed": backward["status"] == "completed"}
    require(gradient["performed"] == (row["key"] in BACKWARD), "Backward cases differ from the fixed three-case set.")
    if gradient["performed"]:
        parameters = backward["parameters"]
        gradient.update(parameter_tensors=len(parameters), missing=sum(not p["gradient_present"] for p in parameters),
                        nonfinite_elements=sum(count(p["nonfinite_elements"]) for p in parameters),
                        all_zero_tensors=sum(p["all_zero"] is True for p in parameters),
                        total_l2_norm=math.sqrt(sum(number(p["l2_norm"], False) ** 2 for p in parameters)))
        gradient["heads"] = [{"head": name, "present": h["gradient_present"] is True,
                               "nonfinite_elements": count(h["nonfinite_elements"]), "all_zero": h["all_zero"] is True}
                              for name, h in zip(("main", "half", "quarter"), backward["head_logits"])]
    layers = model["group_norm_layers"]
    gn = {"n_layers": len(layers), "all_values_finite": True,
          "input_variance_range": [min(number(l["input"]["variance_finite_population"], False) for l in layers), max(number(l["input"]["variance_finite_population"], False) for l in layers)],
          "output_variance_range": [min(number(l["output"]["variance_finite_population"], False) for l in layers), max(number(l["output"]["variance_finite_population"], False) for l in layers)]}
    return {"unet2d": scores(row["models"]["unet2d"]["gt_class5"]), "improved3d_heads": heads,
            "total_loss": nums(loss, ("actual", "independent_weighted_total")), "backward": gradient, "group_norm_global_moments": gn}


def target_metrics(row):
    result = counts(row, ("tp", "fp", "fn", "gt_voxels", "predicted_voxels"))
    result["dice"] = number(row["dice"])
    require(result["tp"] + result["fn"] == result["gt_voxels"] and result["tp"] + result["fp"] == result["predicted_voxels"], "D2 confusion counts disagree.")
    denominator = 2 * result["tp"] + result["fp"] + result["fn"]
    expected = 2 * result["tp"] / denominator if denominator else None
    require(result["dice"] is None and expected is None or result["dice"] is not None and expected is not None and abs(result["dice"] - expected) < 1e-12, "D2 Dice formula mismatch.")
    return result


def compact_intervention(row, mechanism, source):
    require(row["status"] == "completed", "D2 contains an incomplete case.")
    change = row["input_change"]
    require(change["baseline_reconstruction_elementwise_equal"] is True and change["changed_voxels_inside_retained_support"] == 0, "D2 single-factor baseline reconstruction failed.")
    require(row["source_image_sha256"] == source["source_file_sha256"]["image"], "D1/D2 native source identity changed.")
    for name, descriptor in source["decoded_cache_hashes_verified"].items():
        require(row["cache_identity"]["array_sha256"][name] == descriptor["sha256"], "D1/D2 cache identity changed.")
    result = {"input_change": counts(change, ("different_voxels", "total_voxels", "gt_class5_affected_voxels", "original_gt_class5_voxels", "gt_class5_outside_nearest_support")), "models": {}}
    result["input_change"].update(baseline_reconstruction_elementwise_equal=True, changed_voxels_inside_retained_support=0)
    for name in WEIGHTS:
        model = row["models"][name]
        require(model["parameters_and_buffers_unchanged"] is True, "D2 model changed.")
        variants = {}
        for public, private in (("baseline", "baseline_frozen_cache_FP32"), ("without_support_zeroing", "without_support_zeroing_FP32")):
            v = model[private]
            require(v["all_main_logits_finite"] is True, "D2 nonfinite logits.")
            metrics = target_metrics(v["class5"])
            s = scores(v["gt_class5_logits"])
            require(metrics["gt_voxels"] == s["n_gt"] and metrics["tp"] == s["gt_predicted_as_counts"][5], "D2 metrics/logit support disagrees.")
            variants[public] = {**metrics, "p5": s["mean_probabilities"][5], "margin_median": s["margin_quantiles"]["0.5"]}
        expected = mechanism["unet2d"] if name == "unet2d" else mechanism["improved3d_heads"][0]["scores"]
        require(variants["baseline"]["gt_voxels"] == expected["n_gt"] and variants["baseline"]["tp"] == expected["gt_predicted_as_counts"][5], "D1/D2 baseline GT/TP differs.")
        require(abs(variants["baseline"]["p5"] - expected["mean_probabilities"][5]) <= 1e-6, "D1/D2 baseline probability differs.")
        variants["prediction_changed_voxels"] = count(model["prediction_change"]["different_voxels"])
        result["models"][name] = variants
    return result


def distribution(values):
    defined = [number(v, False) for v in values if v is not None]
    return {"n_defined": len(defined), "min": min(defined) if defined else None,
            "median": statistics.median(defined) if defined else None, "max": max(defined) if defined else None}


def cache_summary(report):
    rows = report["cases"]
    require(report["manifest_sha256"] == FROZEN["manifest"] and len(rows) == 189, "Cache descriptor scope/manifest mismatch.")
    require(report["uses_GT_regions"] is True and report["deployment_transform_claimed"] is False, "Unexpected cache descriptor interpretation.")
    keys = [case_key(r) for r in rows]
    require(len(set(keys)) == 189 and all(r["decoded_identity_verified"] is True for r in rows), "Cache descriptor identity/duplicates failed.")
    grouped = {"train": [r for r in rows if r["split"] == "train"], "validation": [r for r in rows if r["split"] == "validation"],
               "test_M015": [r for r in rows if r["split"] == "test" and r["patient"] == "M015"]}
    require([len(grouped[k]) for k in grouped] == [143, 38, 8] and sum(map(len, grouped.values())) == 189, "Cache scope must be 143 TRAIN + 38 VAL + M0158 only.")
    require([len({r["patient"] for r in grouped[k]}) for k in grouped] == [26, 6, 1], "Cache patient scope must be TRAIN26/VAL6/M0151.")
    require({case_key(r) for r in grouped["test_M015"]} == {f"M015/Week{i}" for i in range(8)}, "Missing M015 visit.")
    result, plot = [], []
    for split, group in grouped.items():
        metrics = {key: distribution([r["contrasts"][key] for r in group]) for key in CACHE_CONTRASTS}
        item = {"split": split, "n_scans": len(group), "n_patients": len({r["patient"] for r in group}), "contrasts": metrics}
        result.append(item)
        chosen = metrics["bladder_body_iqr_standardized_median_gap"]
        ordered = sorted(group, key=case_key)
        plot.append({"split": split, "n_scans": item["n_scans"], "n_patients": item["n_patients"], **chosen,
                     "values": [number(r["contrasts"]["bladder_body_iqr_standardized_median_gap"]) for r in ordered] if split == "test_M015" else None,
                     "case_keys": [case_key(r) for r in ordered] if split == "test_M015" else None})
    m015 = [{"case": case_key(r), "contrasts": nums(r["contrasts"], CACHE_CONTRASTS)} for r in sorted(grouped["test_M015"], key=case_key)]
    return {"n_scans": 189, "stage": "frozen normalized cache only", "groups": result, "m015_cases": m015}, plot


def identity_summary(source, archive, official):
    require(len(source["headers"]) == 211, "Expected 211 pair headers.")
    header_keys = {case_key(r) for r in source["headers"]}
    require(len(header_keys) == 211, "Duplicate source header cases.")
    headers = [row[role] for row in source["headers"] for role in ("image", "mask")]
    require(all(h["geometry_matches_frozen_record"] is True for h in headers), "Source header geometry mismatch.")
    require(all(t["free_text_exported"] is False for h in headers for t in h["text_fields"].values()), "Source header free-text export was reported.")
    require(archive["manifest_sha256"] == FROZEN["manifest"] and archive["files_checked"] == 422 and archive["case_visits_checked"] == 211, "Archive scope mismatch.")
    require(archive["source_files_changed"] is False and archive["decoded_voxels_read"] is False, "Unexpected archive-check operation.")
    files = archive["files"]
    require(len(files) == 422, "Incomplete archive file report.")
    by_role = {}
    for role in ("image", "mask"):
        group = [r for r in files if r["role"] == role]
        require(len(group) == 211 and len({case_key(r) for r in group}) == 211, "Archive role coverage mismatch.")
        require({case_key(r) for r in group} == header_keys, "Archive/header case scopes differ.")
        by_role[role] = {"n_files": 211, "size_crc_candidate_matches": sum(r["candidate_match"] is True for r in group),
                         "unique_numeric_candidate_joins": sum(r["global_join_status"] == "unique_numeric_candidate" for r in group)}
    require(official["files_checked"] == 9 and len(official["rows"]) == 9 and official["all_decoded_NIfTI_bytes_equal"] is True, "Official nine-mask check incomplete.")
    wanted = {"B006/Week0", *[f"M015/Week{i}" for i in range(8)]}
    require({case_key(r) for r in official["rows"]} == wanted, "Official mask scope differs from B006 + M0158.")
    checked = []
    for r in official["rows"]:
        require(r["whole_decoded_NIfTI_bytes_equal"] is True and r["gzip_header_only_difference_confirmed"] is True, "Official mask content mismatch.")
        checked.append({"case": case_key(r), "whole_nifti_bytes": count(r["whole_decoded_NIfTI_bytes"]),
                        "whole_nifti_sha256": digest(r["whole_decoded_NIfTI_sha256"]), "deflate_sha256": digest(r["deflate_payload_sha256"]), "equal": True})
    return {"header_pairs": 211, "header_files": 422, "all_header_geometry_matches": True, "free_text_exported": False,
            "official_container_size_crc": by_role, "official_mask_whole_nifti": {"n_files": 9, "all_equal": True, "cases": sorted(checked, key=lambda r: r["case"])},
            "official_catalog_sha256": digest(archive["official_catalog_sha256"])}


def build_summary(reports, provenance):
    require(set(reports) == set(ROLES) and set(provenance) == set(ROLES), "All six reports are required.")
    safe_provenance = {}
    for role in ROLES:
        item = provenance[role]
        job = item.get("job_id")
        require(job is None or isinstance(job, str) and re.fullmatch(r"[0-9]+", job), "Job IDs must be numeric or unspecified.")
        tool = item["source_tool_sha256"]
        safe_provenance[role] = {"report_sha256": digest(item["report_sha256"]),
                                 "source_tool_sha256": digest(tool) if tool is not None else None, "job_id": job}
    provenance = safe_provenance
    source, mechanism, cache, archive, official, d2 = (reports[k] for k in ROLES)
    require(source["status"] == "completed" and source["issues"] == [], "D1 source report incomplete.")
    require(source["input_sha256"] == {"manifest": FROZEN["manifest"], "cache_identity": FROZEN["identity"], "training_sources": FROZEN["sources"]}, "D1 source frozen identity mismatch.")
    require(mechanism["status"] == "checks_passed" and all(mechanism["evidence"][k]["sha256"] == v for k, v in FROZEN.items()), "D1 mechanism frozen identity/status mismatch.")
    require(d2["status"] == "completed" and d2["protected_inputs_unchanged"] is True and d2["all_fourteen_cases_required"] is True, "D2 incomplete or protected inputs changed.")
    require(d2["d1_source_sha256"] == provenance["source"]["report_sha256"], "D2 is bound to a different D1 source report.")
    for report in (mechanism, d2):
        for name, expected in WEIGHTS.items():
            m = report["models"][name]
            require(m["checkpoint_sha256"] == expected and m["parameters_and_buffers_unchanged"] is True and m["checkpoint_file_unchanged"] is True, "Frozen model identity/state mismatch.")
    source_cases = index_fourteen(source["cases"], "case")
    mechanism_cases = index_fourteen(mechanism["cases"], "key")
    intervention_cases = index_fourteen(d2["cases"], "case")
    cache_groups, plot_groups = cache_summary(cache)
    cache_by_key = {case_key(r): r for r in cache["cases"]}
    result = {"schema_version": 1, "diagnostic_only": True, "no_replaced_v1_test": True,
              "frozen_evidence_sha256": FROZEN, "checkpoint_sha256": WEIGHTS, "input_reports": provenance,
              "scope": {"source_pixel_cases": 14, "original_diagnostic_cases": 12, "exploratory_added_train_cases": list(EXPLORATORY),
                        "cache_descriptor_scans": 189, "cache_train_scans": 143, "cache_validation_scans": 38, "cache_test_scope": "M015 eight visits only",
                        "model_mechanism_cases": 14, "support_intervention_cases": 14},
              "source_identity": identity_summary(source, archive, official), "cache_descriptors": cache_groups,
              "cases": [], "plot_data": {"contrast_groups": plot_groups, "support_intervention": []},
              "limits": ["Retrospective diagnostics after v1 TEST inspection; every one of the fourteen selected cases is reported. No replacement TEST score or model/threshold selection.",
                         "Original twelve-case source/model inspection plus two exploratory TRAIN controls selected after initial cache statistics; the added controls are not independent confirmation.",
                         "Native-source pixel/contrast evidence covers fourteen cases only. The 189-scan descriptor comparison covers cached normalized data, not 189 raw MRI examinations.",
                         "Descriptor ranges are scan-level and contain repeated patients. GT regions define these retrospective measurements; they are not a deployable normalization rule.",
                         "Official MRI identity evidence is container size plus CRC32 for 211 files, not cryptographic identity or anatomical validation. Nine label files have matching whole decompressed NIfTI/DEFLATE hashes; the other 202 label files were not content-verified.",
                         "Global body-label ROI contrast does not establish a sequence, acquisition abnormality or causal explanation. Low-contrast TRAIN examples do not establish unseen-patient generalization.",
                         "Clip-boundary fractions include values exactly at bounds; normalized zeros need not be background. Support effects and unclipped/clipped stages are separately reported.",
                         "Finite current gradients and global GroupNorm moments do not reconstruct training history, validate annotation truth or identify a GroupNorm mediation mechanism.",
                         "D2 changes only final nearest-support zeroing at fixed v1 weights in FP32. Its response cannot establish the effects of training with another policy; no improvement does not exonerate every preprocessing/design choice.",
                         "Probabilities are GT-region means, not Dice. Head ranks use strictly larger logits; rank ties share rank. All five source stages and all fourteen paired intervention cases are retained."]}
    for key in sorted(CASES):
        s, m, i = source_cases[key], mechanism_cases[key], intervention_cases[key]
        require(m["status"] == "checks_passed", "Mechanism case incomplete.")
        require(s["policy_reconstruction"]["mask_equal_frozen"] is True and s["policy_reconstruction"]["after_support_image_max_abs_error"] <= 1e-5, "D1 policy reconstruction differs.")
        stages = {}
        for stage in STAGES:
            target_stats = s["intensity_stages"][stage]["5"]
            stages[stage] = {"contrasts": nums(s["within_stage_class_contrasts"][stage], SOURCE_CONTRASTS),
                             "target_intensity": {"n": count(target_stats["n"]), "zero_fraction": number(target_stats["zero_fraction"]),
                                                  "clip_boundary_fraction": number(target_stats["at_or_beyond_clip_fraction"])}}
        compact = compact_mechanism(m)
        require(key in cache_by_key, "Selected diagnostic case missing from cache descriptor scope.")
        for source_key, cache_key in (("bladder_minus_body_median_div_body_iqr", "bladder_body_iqr_standardized_median_gap"),
                                      ("prostate_minus_body_median_div_body_iqr", "prostate_body_iqr_standardized_median_gap")):
            a = stages["cached_after_support_forcing"]["contrasts"][source_key]
            b = cache_by_key[key]["contrasts"][cache_key]
            require(a is None and b is None or a is not None and b is not None and abs(a - b) <= 1e-9, "D1 source/cache descriptor contrasts differ.")
        paired = compact_intervention(i, compact, s)
        support = counts(s["support_policy"], ("native_target_zero_support_voxels", "target_voxels", "target_outside_resampled_nonzero_support", "forced_zero_nonzero_voxels_all", "forced_zero_nonzero_voxels_target"))
        require(support["target_voxels"] == compact["unet2d"]["n_gt"] == compact["improved3d_heads"][0]["target_counts"][5], "Cross-report target support differs.")
        require(support["forced_zero_nonzero_voxels_target"] == paired["input_change"]["gt_class5_affected_voxels"] and support["forced_zero_nonzero_voxels_all"] == paired["input_change"]["different_voxels"], "D1/D2 support effect differs.")
        result["cases"].append({"case": key, "split": CASES[key], "exploratory_added_train_control": key in EXPLORATORY,
                                "source_stages": stages, "support": support,
                                "target_volume": nums(s["target_volume"], ("native_mm3_if_unit_confirmed", "cache_mm3_if_unit_confirmed", "cache_to_native_ratio")),
                                "mechanism": compact, "support_intervention": paired})
        if key.startswith("M015/"):
            plot_row = {"visit": key.split("/")[1]}
            for name in WEIGHTS:
                v = paired["models"][name]
                plot_row[name] = {"baseline_dice": v["baseline"]["dice"], "without_support_dice": v["without_support_zeroing"]["dice"]}
            result["plot_data"]["support_intervention"].append(plot_row)
    m015 = [c for c in result["cases"] if c["split"] == "test"]
    result["m015_support_totals"] = {"target_voxels": sum(c["support"]["target_voxels"] for c in m015),
                                    "forced_zero_nonzero_target_voxels": sum(c["support"]["forced_zero_nonzero_voxels_target"] for c in m015)}
    result["d2_m015_direction_counts"] = {}
    result["exploratory_train_control_descriptors"] = [
        {"case": key, "cached_bladder_body_iqr_gap": number(cache_by_key[key]["contrasts"]["bladder_body_iqr_standardized_median_gap"]),
         "main_gt_mean_p5": next(c for c in result["cases"] if c["case"] == key)["mechanism"]["improved3d_heads"][0]["scores"]["mean_probabilities"][5]}
        for key in EXPLORATORY
    ]
    for name in WEIGHTS:
        pairs = [(c["support_intervention"]["models"][name]["baseline"]["dice"], c["support_intervention"]["models"][name]["without_support_zeroing"]["dice"]) for c in m015]
        result["d2_m015_direction_counts"][name] = {"increased": sum(b > a for a, b in pairs if a is not None and b is not None),
                                                  "decreased": sum(b < a for a, b in pairs if a is not None and b is not None),
                                                  "unchanged": sum(b == a for a, b in pairs if a is not None and b is not None),
                                                  "undefined_pairs": sum(a is None or b is None for a, b in pairs)}
    return result


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_tool_sha(role, report):
    fields = {"source": "diagnostic_source_sha256", "mechanism": "tool_sha256", "cache_descriptors": "script_sha256", "archive_identity": "tool_sha256"}
    if role in fields:
        return digest(report[fields[role]])
    if role == "support_intervention":
        values = {v for p, v in report["protected_input_file_sha256"].items() if Path(p).name == "diagnose_support_intervention.py"}
        require(len(values) == 1, "D2 source-tool identity missing or ambiguous.")
        return digest(values.pop())
    return None  # The official nine-mask comparison report has no tool SHA field.


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for role in ROLES:
        parser.add_argument("--" + role.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--job-id", action="append", default=[], metavar="ROLE=NUMERIC_ID", help="Optional recorded Slurm IDs; omit roles without a documented job.")
    args = parser.parse_args(argv)
    output = args.output.expanduser().absolute()
    require(not os.path.lexists(output), "Output must be new; refusing to overwrite evidence.")
    jobs = {}
    for item in args.job_id:
        role, separator, value = item.partition("=")
        require(separator and role in ROLES and role not in jobs and re.fullmatch(r"[0-9]+", value), "Use distinct known ROLE=numeric_job_id entries.")
        jobs[role] = value
    reports, provenance = {}, {}
    for role in ROLES:
        path = getattr(args, role).expanduser().resolve(strict=True)
        require(path.is_relative_to(PROJECT / "artifacts") and path.suffix == ".json" and path.is_file(), "Inputs must be explicit artifact JSON reports, not arrays or source data.")
        report = json.loads(path.read_text(), parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON constant.")))
        reports[role] = report
        provenance[role] = {"report_sha256": file_sha256(path), "source_tool_sha256": source_tool_sha(role, report), "job_id": jobs.get(role)}
    summary = build_summary(reports, provenance)
    summary.update(created_at_utc=datetime.now(timezone.utc).isoformat(), summary_tool_sha256=file_sha256(__file__))
    encoded = (json.dumps(summary, separators=(",", ":"), allow_nan=False) + "\n").encode()
    require(len(encoded) < MAX_BYTES, "Public summary exceeds the 100KB bound.")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=output.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
        os.link(temporary, output)  # Atomic creation without replacing any file.
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f"Public diagnostic summary: {len(summary['cases'])} cases; {len(encoded)} bytes; SHA256 {file_sha256(output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
