"""Read-only n/s validation audit and descriptive comparison; never runs inference.

Replays fixed-threshold matching independently of evaluation.py. AP is taken from
the pinned validator, not reconstructed from the separate prediction cache.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def overlaps(gt, predictions):
    if not predictions:
        return np.empty(0)
    boxes = np.asarray([p["box"] for p in predictions], dtype=float)
    gt = np.asarray(gt, dtype=float)
    intersection = np.maximum(np.minimum(boxes[:, 2:], gt[2:]) -
                              np.maximum(boxes[:, :2], gt[:2]), 0).prod(axis=1)
    union = (boxes[:, 2:] - boxes[:, :2]).clip(0).prod(axis=1) + \
        (gt[2:] - gt[:2]).clip(0).prod() - intersection
    return np.divide(intersection, union, out=np.zeros_like(union), where=union > 0)


def replay(records, threshold, match_iou):
    """One GT per image: choose the highest-confidence eligible prediction."""
    rows = []
    counts = {c: Counter(gt=0, tp=0, fp=0, fn=0) for c in (0, 1)}
    for record in records:
        cls = record["class_id"]
        pred = sorted([p for p in record["predictions"] if p["confidence"] > threshold],
                      key=lambda p: -p["confidence"])
        ious = overlaps(record["box"], pred)
        eligible = [i for i, p in enumerate(pred)
                    if p["class_id"] == cls and ious[i] >= match_iou]
        matched = eligible[0] if eligible else None
        counts[cls]["gt"] += 1
        counts[cls]["tp"] += int(matched is not None)
        counts[cls]["fn"] += int(matched is None)
        for i, p in enumerate(pred):
            if i != matched:
                counts[p["class_id"]]["fp"] += 1
        correct = [float(ious[i]) for i, p in enumerate(pred) if p["class_id"] == cls]
        rows.append({"image_id": record["image_id"], "class_id": cls,
                     "tp": matched is not None, "alert": any(p["class_id"] == 1 for p in pred),
                     "class_correct_iou": max(correct, default=0.0),
                     "class_agnostic_iou": float(max(ious, default=0.0)),
                     "tp_iou": float(ious[matched]) if matched is not None else None,
                     "retained_predictions": len(pred)})
    for cls in (0, 1):
        c = counts[cls]
        c["precision"] = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else None
        c["recall"] = c["tp"] / c["gt"]
    return counts, rows


def close(actual, expected):
    if actual is None or expected is None:
        assert actual is expected, (actual, expected)
    else:
        assert math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10), (actual, expected)


def verify_point(records, point, match_iou):
    counts, rows = replay(records, point["threshold"], match_iou)
    for cls, name in enumerate(("Benign", "Malignant")):
        for key in ("gt", "tp", "fp", "fn", "precision", "recall"):
            close(counts[cls][key], point["classes"][name][key])
        selected = [row for row in rows if row["class_id"] == cls]
        for source, key in (("class_correct_iou", "all_gt_class_correct"),
                            ("class_agnostic_iou", "all_gt_class_agnostic"), ("tp_iou", "tp_only")):
            values = [row[source] for row in selected if row[source] is not None]
            close(float(np.mean(values)) if values else None, point["classes"][name]["iou"][key]["mean"])
            assert len(values) == point["classes"][name]["iou"][key]["denominator"]
    alarms = {"malignant_tp": sum(r["class_id"] == 1 and r["alert"] for r in rows),
              "malignant_fn": sum(r["class_id"] == 1 and not r["alert"] for r in rows),
              "benign_fp": sum(r["class_id"] == 0 and r["alert"] for r in rows),
              "benign_tn": sum(r["class_id"] == 0 and not r["alert"] for r in rows)}
    for key, value in alarms.items():
        assert value == point["image_alerts"][key]
    close(alarms["benign_fp"] / counts[0]["gt"], point["image_alerts"]["benign_false_alert_rate"])
    close(alarms["malignant_tp"] / counts[1]["gt"], point["image_alerts"]["sensitivity"])
    return rows


def analyse(root, run_dir, output, image_root=None):
    code = Path(__file__).resolve().parent
    config_file = code / "experiment_provisional.json"
    config = read(config_file)
    assert sha(config_file) == sha(run_dir / "locked_config.json")
    assert (run_dir / "COMPLETED.txt").is_file()
    manifests = {split: root / "data/known_lesion_v1" / f"{split}.csv"
                 for split in config["manifest_sha256"]}
    for split, path in manifests.items():
        assert sha(path) == config["manifest_sha256"][split]
    truth = {r["image_id"]: r for r in csv.DictReader(manifests["val"].open())}
    assert len(truth) == 165
    assert Counter(int(r["class_id"]) for r in truth.values()) == {0: 124, 1: 41}
    output.mkdir(parents=True, exist_ok=True)
    summary = {"status": "passed", "scope": "validation_only", "AP_independently_recomputed": False,
               "fixed_threshold_metrics_independently_recomputed": True,
               "config_sha256": sha(config_file), "manifest_sha256": config["manifest_sha256"],
               "model_policy": config["model_policy"], "course_acceptance_confirmed": False,
               "fresh_blind_test": False, "reassessment_model_evaluated": False,
               "models": {}, "limitations": [
                   "One seed and 41 malignant validation images; no population superiority claim.",
                   "Validation selects checkpoint and threshold; these are development results.",
                   "708 cohort images lack patient ID; only recorded identity links are constrained.",
                   "Previous test exposure remains; regrouping does not create a fresh blind test.",
                   "Cached candidates are after NMS and score floor 0.001, not full class scores.",
                   "FN partitions describe available predictions, not causes of network errors."]}
    predictions, selected_rows, curves = {}, {}, {}
    grid_export = []
    for model in ("n", "s"):
        train_dir, val_dir = run_dir / f"train_{model}", run_dir / f"val_{model}"
        train, evaluation, lock = (read(train_dir / "run.json"), read(val_dir / "evaluation.json"),
                                   read(val_dir / "selection_lock.json"))
        assert train["status"] == "completed" and evaluation["split"] == "val"
        for d in (train, evaluation, lock):
            assert d["config_sha256"] == sha(config_file)
            assert d["manifest_sha256"] == config["manifest_sha256"]
        assert train["config"] == config
        assert train["initial_weights_sha256"] == config["models"][model]["sha256"]
        assert sha(train_dir / "fit/weights/best.pt") == train["best_sha256"] == evaluation["checkpoint_sha256"] == lock["checkpoint_sha256"]
        assert sha(train_dir / "fit/weights/last.pt") == train["last_sha256"]
        assert sha(train_dir / "run.json") == evaluation["training_run_sha256"]
        assert sha(train_dir / "fit/optimizer_steps.jsonl") == train["optimizer_trace_sha256"]
        assert sha(train_dir / "fit/checkpoint_history.jsonl") == train["checkpoint_history_sha256"]
        assert sha(code / "evaluation.py") == evaluation["evaluator_source_sha256"]
        source_diffs = [name for name, digest in train["source_sha256"].items() if sha(code / name) != digest]
        # The transfer packager is not imported by the training/evaluation path.
        assert not (set(source_diffs) - {"transfer.py"}), source_diffs
        args = yaml.safe_load((train_dir / "fit/args.yaml").read_text())
        for key, value in config["train"].items():
            assert args[key] == value, (model, key, args[key], value)
        assert read(train_dir / "fit/project_model_interface.json")["nc"] == 2
        loaders = [json.loads(line) for line in (train_dir / "fit/project_loader_calls.jsonl").read_text().splitlines()]
        assert [(x["mode"], x["images"], x["batch"]) for x in loaders] == [("train", 768, 16), ("val", 165, 32)]
        assert len((train_dir / "fit/optimizer_steps.jsonl").read_text().splitlines()) == train["optimizer_steps"] == 2880
        history = [json.loads(line) for line in (train_dir / "fit/checkpoint_history.jsonl").read_text().splitlines()]
        assert len(history) == train["checkpoint_selection"]["completed_epochs"] == 60
        best = max(history, key=lambda row: (row["validation_mAP50_95"], row["epoch"]))
        assert best["epoch"] == train["checkpoint_selection"]["best_epoch"]
        curves[model] = list(csv.DictReader((train_dir / "fit/results.csv").open()))
        assert len(curves[model]) == 60
        assert all(math.isfinite(float(value)) for row in curves[model] for value in row.values())
        records = read(val_dir / "predictions.json")
        assert len(records) == len({r["image_id"] for r in records}) == len(truth)
        assert {r["image_id"] for r in records} == set(truth)
        for record in records:
            row = truth[record["image_id"]]
            assert record["class_id"] == int(row["class_id"])
            x, y, w, h = [float(row[k]) for k in ("center_x", "center_y", "box_width", "box_height")]
            np.testing.assert_allclose(record["box"], [x-w/2, y-h/2, x+w/2, y+h/2], rtol=0, atol=1e-12)
            for p in record["predictions"]:
                assert p["class_id"] in (0, 1) and config["evaluation"]["score_floor"] < p["confidence"] <= 1
        points = evaluation["operating_point"]["grid"]
        assert [p["threshold"] for p in points] == config["evaluation"]["threshold_grid"]
        for point in points:
            verify_point(records, point, config["evaluation"]["match_iou"])
            grid_export.append({"model": model, "threshold": point["threshold"],
                                **{k: point["classes"]["Malignant"][k] for k in ("tp", "fp", "fn", "precision", "recall")},
                                "benign_false_alerts": point["image_alerts"]["benign_fp"],
                                "benign_false_alert_rate": point["image_alerts"]["benign_false_alert_rate"]})
        feasible = [p for p in points if p["image_alerts"]["benign_false_alert_rate"] <= config["evaluation"]["benign_false_alert_budget"]]
        chosen = max(feasible, key=lambda p: (p["classes"]["Malignant"]["recall"],
                     p["classes"]["Malignant"]["precision"] or 0, -p["image_alerts"]["benign_fp"], p["threshold"]))
        assert chosen == evaluation["operating_point"]["selected"]
        assert chosen["threshold"] == lock["threshold"]
        selected_rows[model] = verify_point(records, chosen, config["evaluation"]["match_iou"])
        predictions[model] = records
        categories = Counter()
        row_lookup = {r["image_id"]: r for r in selected_rows[model]}
        for record in records:
            row = row_lookup[record["image_id"]]
            if record["class_id"] != 1 or row["tp"]:
                continue
            pred = record["predictions"]
            ious = overlaps(record["box"], pred)
            if any(p["class_id"] == 1 and ious[i] >= 0.5 for i, p in enumerate(pred)):
                reason = "qualified_malignant_candidate_below_operating_threshold"
            elif any(p["class_id"] == 0 and ious[i] >= 0.5 for i, p in enumerate(pred)):
                reason = "only_benign_candidates_localize_at_IoU_0.5"
            elif pred:
                reason = "no_cached_candidate_reaches_IoU_0.5"
            else:
                reason = "no_cached_candidates"
            row["fn_cache_category"] = reason
            categories[reason] += 1
        summary["models"][model] = {
            "mAP50": evaluation["mAP50"], "mAP50_95": evaluation["mAP50_95"], "ap": evaluation["ap"],
            "operating_point": chosen, "FN_cache_partition": dict(categories),
            "training_seconds": train["elapsed_seconds"], "parameters": train["parameters"],
            "peak_cuda_allocated_GiB": train["peak_cuda_allocated_bytes"] / 2**30,
            "checkpoint_selection": train["checkpoint_selection"], "best_sha256": train["best_sha256"],
            "predictions_sha256": sha(val_dir / "predictions.json"),
            "evaluation_sha256": sha(val_dir / "evaluation.json"),
            "current_source_differences": source_diffs,
            "source_difference_note": "transfer.py is packaging only, not imported by training/evaluation; all other recorded source files match.",
            "low_floor_diagnostic": {"threshold": config["evaluation"]["score_floor"],
                                      "malignant_counts": dict(replay(records, config["evaluation"]["score_floor"], .5)[0][1])}}
    assert read(run_dir / "train_n/run.json")["staged_image_content_sha256"] == read(run_dir / "train_s/run.json")["staged_image_content_sha256"]
    assert read(run_dir / "train_n/run.json")["source_sha256"] == read(run_dir / "train_s/run.json")["source_sha256"]
    paired = []
    maps = {m: {r["image_id"]: r for r in selected_rows[m]} for m in ("n", "s")}
    for image_id in sorted(truth):
        paired.append({"image_id": image_id, "class_id": int(truth[image_id]["class_id"]),
                       **{f"{m}_{k}": maps[m][image_id][k] for m in ("n", "s") for k in ("tp", "alert", "class_correct_iou", "class_agnostic_iou")},
                       **{f"{m}_fn_category": maps[m][image_id].get("fn_cache_category", "") for m in ("n", "s")}})
    summary["paired_malignant_TP"] = dict(Counter(
        "both" if r["n_tp"] and r["s_tp"] else "n_only" if r["n_tp"] else "s_only" if r["s_tp"] else "neither"
        for r in paired if r["class_id"] == 1))
    summary["paired_benign_false_alerts"] = dict(Counter(
        "both" if r["n_alert"] and r["s_alert"] else "n_only" if r["n_alert"] else "s_only" if r["s_alert"] else "neither"
        for r in paired if r["class_id"] == 0))
    summary["analysis_source_sha256"] = sha(Path(__file__))
    (output / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    for name, rows in (("threshold_grid.csv", grid_export), ("paired_cases.csv", paired)):
        with (output / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), layout="constrained")
    for ax in axes:
        for model, marker, style in (("n", "o", "-"), ("s", "s", "--")):
            rows = [p for p in grid_export if p["model"] == model]
            ax.plot([p["benign_false_alert_rate"] for p in rows], [p["recall"] for p in rows],
                    color="black", linestyle=style, marker=marker, markersize=3, label=f"YOLOv8{model}")
            p = summary["models"][model]["operating_point"]
            x, y = p["image_alerts"]["benign_false_alert_rate"], p["classes"]["Malignant"]["recall"]
            ax.plot(x, y, marker, color="black", markersize=8, markerfacecolor="white", markeredgewidth=1.5)
            ax.annotate(f'{model}: {p["threshold"]:.2f}', (x, y), xytext=(7, 5), textcoords="offset points", fontsize=8)
        ax.axvline(.1, color=".55", linestyle=":", label="10% budget")
        ax.set_xlabel("Benign image false-alert rate"); ax.set_ylabel("Malignant detection recall")
        ax.grid(alpha=.18)
    axes[0].set(xlim=(-.01, .46), ylim=(-.02, .8), title="All 14 declared thresholds")
    axes[1].set(xlim=(-.005, .15), ylim=(-.01, .45), title="Operating-region detail")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.savefig(output / "threshold_tradeoff.png", dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.7), layout="constrained")
    for ax, key, title in zip(axes, ("train/cls_loss", "val/cls_loss", "metrics/mAP50-95(B)"),
                              ("Training classification loss", "Validation classification loss", "Training-time validation mAP50:95")):
        for model, style in (("n", "-"), ("s", "--")):
            ax.plot([int(r["epoch"]) for r in curves[model]], [float(r[key]) for r in curves[model]],
                    color="black", linestyle=style, label=f"YOLOv8{model}")
        ax.set_title(title, fontsize=8); ax.set_xlabel("Epoch"); ax.grid(alpha=.18)
        if key == "val/cls_loss":
            ax.set_yscale("log")
            ax.set_title(title + " (log scale)", fontsize=8)
    axes[0].legend(fontsize=8)
    fig.savefig(output / "training_curves.png", dpi=180); plt.close(fig)
    if image_root is not None:
        from PIL import Image
        from matplotlib.patches import Rectangle
        import matplotlib.patheffects as effects
        # First image ID satisfying each stated category; descriptive examples only.
        rules = [lambda r: r["class_id"] == 1 and r["n_fn_category"] ==
                 r["s_fn_category"] == "only_benign_candidates_localize_at_IoU_0.5",
                 lambda r: r["class_id"] == 1 and r["s_tp"] and not r["n_tp"]
                 and r["n_alert"] and r["n_class_correct_iou"] < .5]
        examples = [next(r for r in paired if rule(r)) for rule in rules]
        fig, axes = plt.subplots(2, 2, figsize=(8, 5.7), layout="constrained")
        details = []
        for index, example in enumerate(examples):
            image_id = example["image_id"]
            picture = Image.open(image_root / f"{image_id}.jpg").convert("RGB")
            for j, model in enumerate(("n", "s")):
                record = next(r for r in predictions[model] if r["image_id"] == image_id)
                threshold = summary["models"][model]["operating_point"]["threshold"]
                retained = [p for p in record["predictions"] if p["confidence"] > threshold]
                ax = axes[index, j]; ax.imshow(picture, extent=(0, 1, 1, 0))
                ax.set_aspect(picture.height / picture.width); ax.set_axis_off()
                for box, linestyle in [(record["box"], "--")] + [(p["box"], "-") for p in retained]:
                    x1, y1, x2, y2 = box
                    patch = Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor="black", linewidth=1.4, linestyle=linestyle)
                    patch.set_path_effects([effects.Stroke(linewidth=3, foreground="white"), effects.Normal()])
                    ax.add_patch(patch)
                ious = overlaps(record["box"], retained)
                labels = [f'{"M" if p["class_id"] else "B"} score={p["confidence"]:.3f}, IoU={ious[i]:.3f}' for i,p in enumerate(retained)]
                ax.set_title(f'{image_id} | {model} at {threshold:.2f}\n' + ("; ".join(labels) if labels else "No retained prediction"), fontsize=8)
                details.append({"image_id": image_id, "model": model, "threshold": threshold,
                                "ground_truth": "Malignant", "retained_predictions": retained,
                                "top_cached_candidate": max(record["predictions"], key=lambda p:p["confidence"], default=None),
                                "image_sha256": sha(image_root / f"{image_id}.jpg")})
        fig.savefig(output / "validation_examples.png", dpi=180); plt.close(fig)
        (output / "validation_examples.json").write_text(json.dumps(details, indent=2) + "\n")
    print(json.dumps({"status": summary["status"], "grid_points_verified": 28,
                      "paired_malignant_TP": summary["paired_malignant_TP"],
                      "FN_partitions": {m: summary["models"][m]["FN_cache_partition"] for m in ("n", "s")}}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, help="Optional unchanged photo directory, for validation-only examples")
    options = parser.parse_args()
    analyse(options.project_root.resolve(), options.run_dir.resolve(), options.output.resolve(), options.image_root)
