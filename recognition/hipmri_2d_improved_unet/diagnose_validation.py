"""Post-hoc validation diagnosis; preserve the original CE-selected experiment."""

import argparse
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import HipMRIPairDataset, collate_padded, collect_pairs, load_pair
from diagnostic_figures import save_failure_figure
from metrics import SegmentationMetrics
from modules import build_model
from train import manifest_hash, seed_everything
from validation_diagnostics import class5_overlap, classwise_ce_statistics
from verify_selected_validation import state_sha256


PROJECT = Path(__file__).resolve().parent
REPO = PROJECT.parents[1]
PATTERN = re.compile(r"case_(\d+)_week_(\d+)_slice_(\d+)\.nii\.gz")
TAGS = ("baseline_best", "improved_best", "improved_last")
SELECTION_RULES = {
    "A_missed_region": "Improved best3: lowest pooled-ID5-Dice reference-positive case/week group; within it R>=20 and FN>0, maximize FN/R then FN, then numeric case/week/slice and basename.",
    "B_empty_reference_fp": "Improved best3: R=0 and FP>0, maximize FP then numeric metadata. Report FP/Nvalid, not an organ-positive Dice estimate.",
    "C_few_reference_pixels": "Improved best3: 0<R<20 and FN>0, maximize FN/R then FN, then numeric metadata. Twenty pixels is a diagnostic bin, not a clinical threshold.",
    "D_late_improvement": "R>=20, maximize positive Dice(last20)-Dice(best3), then numeric metadata.",
    "E_late_regression": "R>=20, minimize negative Dice(last20)-Dice(best3), then numeric metadata.",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def metadata(path, index):
    match = PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Unexpected validation filename: {path.name}")
    case, week, slice_id = map(int, match.groups())
    return {"index": index, "filename": path.name, "case": case, "week": week, "slice": slice_id}


def overlap_from_counts(tp, fp, fn, valid_pixels):
    reference, predicted = tp + fn, tp + fp
    return {
        "tp": tp, "fp": fp, "fn": fn, "reference_pixels": reference,
        "predicted_pixels": predicted, "valid_pixels": valid_pixels,
        "dice": 2 * tp / (reference + predicted) if reference + predicted else None,
        "precision": tp / predicted if predicted else None,
        "recall": tp / reference if reference else None,
        "miss_rate": fn / reference if reference else None,
        "false_discovery_fraction": fp / predicted if predicted else None,
        "false_positive_rate": fp / (valid_pixels - reference) if valid_pixels > reference else None,
    }


def pool_slices(rows):
    return overlap_from_counts(*(sum(row[key] for row in rows) for key in ("tp", "fp", "fn", "valid_pixels")))


def grouped(rows):
    volumes, cases = defaultdict(list), defaultdict(list)
    for row in rows:
        volumes[(row["case"], row["week"])].append(row)
        cases[row["case"]].append(row)
    volume_rows = [{"case": key[0], "week": key[1], "slices": len(items), **pool_slices(items)} for key, items in sorted(volumes.items())]
    case_rows = []
    for case, items in sorted(cases.items()):
        values = [row["dice"] for row in volume_rows if row["case"] == case and row["dice"] is not None]
        case_rows.append({"case": case, "slices": len(items), **pool_slices(items), "mean_group_dice": sum(values) / len(values) if values else None})
    means = [row["mean_group_dice"] for row in case_rows if row["mean_group_dice"] is not None]
    return {"case_week_groups": volume_rows, "case_proxy_groups": case_rows,
            "diagnostic_case_balanced_mean_group_dice": sum(means) / len(means) if means else None,
            "scope": "Post-hoc grouping proxies only; no physical-volume or millimetre interpretation, and no replacement of the original pixel-pooled metric."}


def replay(tag, checkpoint_path, expected, config, loader, names, targets):
    digest = sha(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert checkpoint["epoch"] == expected["epoch"]
    assert checkpoint["model_config"] == config
    model = build_model(checkpoint["model"], **config).to("cuda").eval()
    model.load_state_dict(checkpoint["model_state"])
    del checkpoint
    before = state_sha256(model)
    accumulator = SegmentationMetrics(6)
    support = np.zeros(6, dtype=np.int64)
    sums = np.zeros(6, dtype=np.float64)
    shared_sum, count, offset = 0.0, 0, 0
    predictions, nll_maps, rows = [], [], []
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            image, target, valid = [batch[key].to("cuda") for key in ("image", "target", "valid_mask")]
            logits = model(image)
            diagnostic = classwise_ce_statistics(logits, target, valid, return_pixel_nll=True)
            support += np.asarray(diagnostic["support"])
            sums += np.asarray(diagnostic["nll_sum"])
            shared_sum += diagnostic["ce_shared"] * diagnostic["valid_pixels"]
            count += diagnostic["valid_pixels"]
            predicted = logits.argmax(dim=1)
            accumulator.update(predicted, target, valid)
            cpu_prediction = predicted.cpu()
            for local, (height, width) in enumerate(batch["original_sizes"]):
                actual = batch["target"][local, :height, :width]
                selected_valid = batch["valid_mask"][local, :height, :width]
                pred = cpu_prediction[local, :height, :width]
                assert bool(selected_valid.all()), "Original area must remain entirely valid."
                actual_array = actual.numpy().astype(np.uint8)
                if tag == TAGS[0]:
                    targets.append(actual_array.copy())
                else:
                    assert np.array_equal(targets[offset], actual_array), "Reference changed between replays."
                nll = diagnostic["pixel_nll_cpu"][local, :height, :width].numpy().astype(np.float32)
                predictions.append(pred.numpy().astype(np.uint8))
                nll_maps.append(nll)
                overlap = class5_overlap(pred, actual, selected_valid)
                row = {**names[offset], **overlap_from_counts(overlap["tp"], overlap["fp"], overlap["fn"], overlap["valid_pixels"]),
                       "cross_entropy_reconstructed": float(nll.sum(dtype=np.float64) / nll.size)}
                rows.append(row)
                offset += 1
    assert offset == 660 and count == 21626880
    metrics = accumulator.compute()
    assert metrics["confusion_matrix"] == expected["validation"]["confusion_matrix"]
    shared_ce = shared_sum / count
    assert abs(shared_ce - expected["validation"]["cross_entropy"]) < 1e-6
    assert abs(sums.sum() / count - shared_ce) < 1e-6
    assert before == state_sha256(model) and sha(checkpoint_path) == digest
    matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    global_overlap = pool_slices(rows)
    assert global_overlap["dice"] == metrics["dice_per_class"][5]
    predicted_id5_sources = matrix[:, 5].tolist()
    fp_sources = predicted_id5_sources.copy()
    fp_sources[5] = 0
    assert sum(fp_sources) == global_overlap["fp"]
    assert predicted_id5_sources[5] == global_overlap["tp"]
    classes = [{"class_id": c, "support": int(support[c]), "pixel_fraction": float(support[c] / count),
                "nll_sum": float(sums[c]), "conditional_mean_nll": float(sums[c] / support[c]) if support[c] else None,
                "global_ce_contribution": float(sums[c] / count)} for c in range(6)]
    result = {"tag": tag, "checkpoint_epoch": expected["epoch"], "checkpoint_sha256": digest,
              "cross_entropy_shared": shared_ce, "cross_entropy_reconstructed_float64": float(sums.sum() / count),
              "ce_matches_formal_epoch": True, "confusion_matches_formal_epoch_exactly": True,
              "state_unchanged": True, "optimizer_steps": 0, "samples": offset,
              "diagnostic_elapsed_seconds": round(time.perf_counter() - started, 3),
              "metrics": metrics, "id5_overlap": global_overlap, "ce_by_true_class": classes,
              "predicted_id5_true_class_counts": predicted_id5_sources,
              "id5_fp_source_true_class_counts": fp_sources, **grouped(rows)}
    del model, logits, predicted, image, target, valid, cpu_prediction
    print(f"{tag}: CE={shared_ce:.9f}; ID5 Dice={global_overlap['dice']:.9f}; replay matches", flush=True)
    return result, rows, predictions, nll_maps


def fixed_set_diagnostics(targets, predictions, nll_maps):
    buckets = {name: {"pixels": 0, "nll_best_sum": 0.0, "nll_last_sum": 0.0,
                      "p_true_best_sum": 0.0, "p_true_last_sum": 0.0}
               for name in ("both_correct", "corrected_errors", "new_errors", "both_wrong")}
    transitions = {name: 0 for name in ("recovered_fn", "new_fn", "new_fp", "removed_fp")}
    for target, a, b, na, nb in zip(targets, predictions["improved_best"], predictions["improved_last"], nll_maps["improved_best"], nll_maps["improved_last"]):
        ca, cb = a == target, b == target
        masks = {"both_correct": ca & cb, "corrected_errors": ~ca & cb,
                 "new_errors": ca & ~cb, "both_wrong": ~ca & ~cb}
        for key, mask in masks.items():
            group = buckets[key]
            av, bv = na[mask].astype(np.float64), nb[mask].astype(np.float64)
            group["pixels"] += int(mask.sum())
            group["nll_best_sum"] += float(av.sum())
            group["nll_last_sum"] += float(bv.sum())
            group["p_true_best_sum"] += float(np.exp(-av).sum())
            group["p_true_last_sum"] += float(np.exp(-bv).sum())
        positive, pa, pb = target == 5, a == 5, b == 5
        for key, mask in {"recovered_fn": positive & ~pa & pb, "new_fn": positive & pa & ~pb,
                          "new_fp": ~positive & ~pa & pb, "removed_fp": ~positive & pa & ~pb}.items():
            transitions[key] += int(mask.sum())
    total = sum(row["pixels"] for row in buckets.values())
    assert total == 21626880
    for row in buckets.values():
        n = row["pixels"]
        row["mean_nll_best"] = row["nll_best_sum"] / n if n else None
        row["mean_nll_last"] = row["nll_last_sum"] / n if n else None
        row["mean_p_true_best"] = row.pop("p_true_best_sum") / n if n else None
        row["mean_p_true_last"] = row.pop("p_true_last_sum") / n if n else None
        row["delta_global_ce_contribution"] = (row["nll_last_sum"] - row["nll_best_sum"]) / total
    return {"multiclass_fixed_sets": buckets, "id5_binary_transitions": transitions,
            "interpretation": "Same pixel sets defined jointly by best3/last20 and reference. p_true is the probability assigned to the true class, not maximum-softmax confidence. No calibration error metric is measured."}


def choose_cases(rows, model_results):
    best, last = rows["improved_best"], rows["improved_last"]
    used, cases = set(), []
    positive_groups = [row for row in model_results["improved_best"]["case_week_groups"] if row["reference_pixels"] > 0]
    worst = min(positive_groups, key=lambda row: (row["dice"], row["case"], row["week"]))
    tie = lambda row: (row["case"], row["week"], row["slice"], row["filename"])
    for label, rule in SELECTION_RULES.items():
        if label.startswith("A"):
            candidates = [r for r in best if (r["case"], r["week"]) == (worst["case"], worst["week"]) and r["reference_pixels"] >= 20 and r["fn"] > 0]
            rank = lambda r: (-r["miss_rate"], -r["fn"], *tie(r))
        elif label.startswith("B"):
            candidates = [r for r in best if r["reference_pixels"] == 0 and r["fp"] > 0]
            rank = lambda r: (-r["fp"], *tie(r))
        elif label.startswith("C"):
            candidates = [r for r in best if 0 < r["reference_pixels"] < 20 and r["fn"] > 0]
            rank = lambda r: (-r["miss_rate"], -r["fn"], *tie(r))
        else:
            sign = 1 if label.startswith("D") else -1
            candidates = [r for r in best if r["reference_pixels"] >= 20 and sign * (last[r["index"]]["dice"] - r["dice"]) > 0]
            rank = lambda r: (-sign * (last[r["index"]]["dice"] - r["dice"]), *tie(r))
        remaining = [r for r in candidates if r["index"] not in used]
        record = {"case_label": label, "selection_rule": rule, "candidate_count_before_exclusion": len(candidates),
                  "candidate_count_after_exclusion": len(remaining), "status": "selected" if remaining else "not_available"}
        if remaining:
            selected = min(remaining, key=rank)
            index = selected["index"]
            used.add(index)
            record.update({key: selected[key] for key in ("index", "filename", "case", "week", "slice")})
            record["models"] = {tag: rows[tag][index] for tag in TAGS}
            record["delta_dice_last_minus_best"] = (
                last[index]["dice"] - selected["dice"]
                if selected["dice"] is not None and last[index]["dice"] is not None else None
            )
        cases.append(record)
    return cases, worst


def plot_aggregates(path, histories, models):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), layout="constrained")
    for model, color in (("baseline_unet", "tab:blue"), ("improved_unet", "tab:orange")):
        records = histories[model]
        x = [r["validation"]["cross_entropy"] for r in records]
        y = [r["validation"]["dice_per_class"][5] for r in records]
        axes[0].plot(x, y, ".-", color=color, label=model)
        for r, xx, yy in zip(records, x, y):
            axes[0].annotate(str(r["epoch"]), (xx, yy), fontsize=7, xytext=(2, 2), textcoords="offset points")
        best = min(records, key=lambda r: (r["validation"]["cross_entropy"], r["epoch"]))
        axes[0].scatter(best["validation"]["cross_entropy"], best["validation"]["dice_per_class"][5], marker="*", s=150, color=color, edgecolor="black", zorder=4)
    axes[0].set(xlabel="Validation CE (lower is better)", ylabel="Pixel-pooled ID5 Dice", title="20-epoch selection trade-off\nStars retain original CE selection")
    axes[0].legend(fontsize=8)
    a, b = models["improved_best"]["ce_by_true_class"], models["improved_last"]["ce_by_true_class"]
    x = np.arange(6)
    axes[1].bar(x - .18, [r["conditional_mean_nll"] for r in a], width=.36, label="best epoch 3")
    axes[1].bar(x + .18, [r["conditional_mean_nll"] for r in b], width=.36, label="last epoch 20")
    axes[1].set(xticks=x, xlabel="True class ID", ylabel="Mean -log(p_true) within class", title="Improved: class-conditional CE")
    axes[1].legend(fontsize=8)
    delta = [rb["global_ce_contribution"] - ra["global_ce_contribution"] for ra, rb in zip(a, b)]
    axes[2].bar(x, delta, color=["tab:red" if value > 0 else "tab:green" for value in delta])
    axes[2].axhline(0, color="black", lw=.6)
    axes[2].set(xticks=x, xlabel="True class ID", ylabel="Delta contribution to global CE", title="Epoch 20 minus epoch 3\nContributions sum to global CE change")
    for i, value in enumerate(delta):
        axes[2].annotate(f"{value:+.4f}", (i, value), fontsize=8, ha="center", va="bottom" if value >= 0 else "top")
    fig.suptitle("Post-hoc validation diagnosis; source organ names assume retained IDs; no test evaluation", fontsize=12)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=PROJECT.parents[3] / "Final Report/HipMRI_2D_Improved_UNet_学习/data/keras_slices_data")
    parser.add_argument("--output-dir", type=Path, default=REPO / "outputs/selection_diagnosis_20261026")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Use a fresh diagnosis directory; preserve prior evidence.")
    torch.set_num_threads(1)
    seed_everything(3710)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for replay matching the recorded environment.")
    formal_path = PROJECT / "evidence/formal_run_plan_20261026.json"
    comparison_path = PROJECT / "evidence/formal_comparison_20261026.json"
    formal, comparison = read(formal_path), read(comparison_path)
    protected = {formal_path: sha(formal_path), comparison_path: sha(comparison_path)}
    for name, digest in formal["source_sha256"].items():
        assert sha(PROJECT / name) == digest
        protected[PROJECT / name] = digest
    pairs = collect_pairs(args.data_root / "keras_slices_validate", args.data_root / "keras_slices_seg_validate")
    assert len(pairs) == 660 and manifest_hash(pairs) == formal["data"]["validation"]["manifest_sha256"]
    names = [metadata(path, i) for i, (path, _) in enumerate(pairs)]
    histories = {}
    specs = {}
    for model, short in (("baseline_unet", "baseline"), ("improved_unet", "improved")):
        report_path = REPO / f"outputs/{short}_formal_20261026/run_summary.json"
        report = read(report_path)
        assert report["status"] == "passed" and len(report["epochs"]) == 20
        histories[model] = report["epochs"]
        protected[report_path] = sha(report_path)
        selected = min(report["epochs"], key=lambda row: (row["validation"]["cross_entropy"], row["epoch"]))
        highest = max(report["epochs"], key=lambda row: (row["validation"]["dice_per_class"][5], -row["epoch"]))
        tag = f"{short}_best"
        specs[tag] = (REPO / f"outputs/{short}_formal_20261026/best.pt", selected, report["model_config"])
        if short == "improved":
            specs["improved_last"] = (REPO / "outputs/improved_formal_20261026/last.pt", report["epochs"][-1], report["model_config"])
        assert selected["epoch"] == comparison["results"][model]["selected_epoch"]
    for tag, (path, expected, _) in specs.items():
        model = "baseline_unet" if tag == "baseline_best" else "improved_unet"
        kind = "last" if tag == "improved_last" else "best"
        assert sha(path) == comparison["results"][model]["checkpoint_verification"][kind]["sha256"]
        protected[path] = sha(path)
    args.output_dir.mkdir(parents=True)
    analysis_plan = {"created_at": datetime.now(timezone(timedelta(hours=10))).isoformat(),
        "analysis_kind": "posthoc_validation_selection_and_failures", "new_training": False,
        "formal_selection_changed": False, "test_split_read": False, "models": list(TAGS),
        "unavailable_weight_note": "Only Improved best3 and last20 exist. Epoch17 max-ID5 is historical aggregate only and is not replayed.",
        "selection_rules": SELECTION_RULES, "duplicate_policy": "A through E in order; exclude previously selected slice; no fallback if a category has no candidate.",
        "example_scope": "Failure-mode illustrations chosen post hoc, not a random sample or failure-frequency estimate; all metrics use original full pixels.",
        "formal_plan_sha256": protected[formal_path], "formal_comparison_sha256": protected[comparison_path],
        "source_sha256": {name: sha(PROJECT / name) for name in ("diagnose_validation.py", "validation_diagnostics.py", "diagnostic_figures.py")}}
    write(args.output_dir / "analysis_plan.json", analysis_plan)
    loader = DataLoader(HipMRIPairDataset(pairs), batch_size=4, shuffle=False, num_workers=0, collate_fn=collate_padded)
    models, rows, predictions, nll_maps, targets = {}, {}, {}, {}, []
    for tag in TAGS:
        models[tag], rows[tag], predictions[tag], nll_maps[tag] = replay(tag, *specs[tag], loader, names, targets)
    assert all(models[tag]["ce_by_true_class"][c]["support"] == models[TAGS[0]]["ce_by_true_class"][c]["support"] for tag in TAGS for c in range(6))
    fixed = fixed_set_diagnostics(targets, predictions, nll_maps)
    cases, worst = choose_cases(rows, models)
    for case in cases:
        if case["status"] != "selected":
            continue
        index = case["index"]
        sample = load_pair(*pairs[index])
        height, width = sample["original_size"]
        image = sample["image"][0, :height, :width].numpy()
        case["figure"] = save_failure_figure(args.output_dir / f"{case['case_label']}.png", image, targets[index] == 5,
            {tag: predictions[tag][index] == 5 for tag in TAGS}, {tag: rows[tag][index] for tag in TAGS}, case)
    ce_delta = [b["global_ce_contribution"] - a["global_ce_contribution"] for a, b in zip(models["improved_best"]["ce_by_true_class"], models["improved_last"]["ce_by_true_class"])]
    assert abs(sum(ce_delta) - sum(row["delta_global_ce_contribution"] for row in fixed["multiclass_fixed_sets"].values())) < 1e-9
    history = {}
    for model, records in histories.items():
        best_ce = min(records, key=lambda row: (row["validation"]["cross_entropy"], row["epoch"]))
        best_dice = max(records, key=lambda row: (row["validation"]["dice_per_class"][5], -row["epoch"]))
        history[model] = {"ce_selected_epoch": best_ce["epoch"], "highest_id5_epoch_historical": best_dice["epoch"],
            "epochs": [{"epoch": row["epoch"], "validation_ce": row["validation"]["cross_entropy"], "id5_dice": row["validation"]["dice_per_class"][5]} for row in records]}
    plot_aggregates(args.output_dir / "selection_and_ce.png", histories, models)
    assert all(sha(path) == digest for path, digest in protected.items()), "Original files changed during diagnosis."
    summary = {**analysis_plan, "status": "passed", "performed_by": "Codex automation; not user verification",
        "history": history, "models": models, "improved_ce_delta_by_true_class": ce_delta,
        "fixed_set_analysis": fixed, "failure_cases": cases, "lowest_best3_group": worst,
        "original_evidence_and_weights_unchanged": True,
        "ce_definition": "Per valid pixel -log p(true class); class contribution is NLL_sum_c / all valid pixels, not class-mean CE or a sum of six class errors per pixel.",
        "label_mapping_boundary": "ID5 is interpreted as prostate only under the previously accepted unverified derivative-ID-preservation assumption.",
        "geometry_boundary": "Case/week are grouping proxies, affines/physical units are not independently validated here; no millimetre or clinical-size claims.",
        "claim_boundary": "Post-hoc validation diagnosis. No original formal selection/result was replaced, no epoch17 weights were reconstructed, and no test or new training occurred.",
        "aggregate_figure": "selection_and_ce.png"}
    write(args.output_dir / "diagnosis_summary.json", summary)
    write(args.output_dir / "per_slice_metrics.json", rows)
    print(f"Diagnosis complete: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
