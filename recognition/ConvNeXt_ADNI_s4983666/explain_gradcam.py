"""Grad-CAM figures for a trained checkpoint on its fold's validation patients.

Only the ``val`` role is allowed: calibration and test patients stay unseen.
For every validation scan the middle slice is explained (the model input is
that slice with its two neighbours). Outputs:

* ``gradcam_cases.png`` - input and Grad-CAM overlay for the most confident
  correct and incorrect scans of each group;
* ``gradcam_group_means.png`` - mean input and mean Grad-CAM per group;
* ``gradcam_summary.json`` / ``gradcam_scans.csv`` - per-scan prediction, map
  share outside the head, and per-group averages.

Groups are diagnosis x acquisition period (2011-2016 / 2017-2021, read from
the metadata file names; docs/ACQUISITION_CONFOUND.md), so the maps of AD scans
whose acquisition style is unique to AD can be compared with AD and NC scans
acquired in the shared period. The middle slices of ADNI scans are roughly
aligned, so group means are indicative only (no registration is applied).
"""

import argparse
from collections import defaultdict
import os
from pathlib import Path
import statistics
import sys

import torch

from dataset import load_fold, make_loader
from adni_data.acquisition import SHARED_PERIOD_START, acquisition_info
from adni_data.preprocessing import checkpoint_preprocessing
from engine.prediction import load_checkpoint
from evaluation.gradcam import grad_cam, outside_share, target_layer
from evaluation.inference import predict_slices, score_slices
from utils.artifacts import validate_output, write_csv, write_json
from utils.runtime import seed_everything, select_device


def group_of(scan: dict, info: dict) -> str:
    """Diagnosis and acquisition period of one scan ("unknown" if the metadata path has no date)."""
    year = info.get(scan["image_id"], {}).get("year")
    period = "unknown" if year is None else "2017-21" if year >= SHARED_PERIOD_START else "2011-16"
    return f"{'AD' if scan['label'] else 'NC'} {period}"


def run(args: argparse.Namespace) -> dict:
    """Predict the fold's validation scans, explain their middle slices and write the figures."""
    output = validate_output(args.output, args.data_root, args.splits_dir)
    device = select_device(args.device)
    model, config, controls, _ = load_checkpoint(args.checkpoint, device)
    seed_everything(config["seed"])
    torch.set_num_threads(args.threads)
    preprocessing = checkpoint_preprocessing(config)
    data = load_fold(args.data_root, args.splits_dir, int(config["fold"]))
    if data["manifest_sha256"] != config["manifest_sha256"]:
        raise ValueError("Checkpoint was trained with a different set of frozen manifests.")
    image_size = tuple(config["image_size"])
    loader = make_loader(data["val"], args.data_root, image_size, args.batch_size, args.workers, config["seed"],
                         False, device, role="val", preprocessing=preprocessing,
                         context_slices=controls.context_slices)
    slices, _ = predict_slices(model, loader, device, controls)
    _, scans = score_slices(slices, config["expected_slices"])
    info = acquisition_info(args.data_root)

    # Middle slice of every validation scan.
    by_scan = defaultdict(list)
    for row in data["val"]:
        by_scan[row["image_id"]].append(int(row["slice_index"]))
    dataset, layer = loader.dataset, target_layer(model, args.layer)
    centre = controls.context_slices // 2
    records, inputs, maps = [], [], []
    for start in range(0, len(scans), args.batch_size):
        batch_scans = scans[start:start + args.batch_size]
        items = [dataset[dataset.position[(s["image_id"], sorted(by_scan[s["image_id"]])[len(by_scan[s["image_id"]]) // 2])]]
                 for s in batch_scans]
        images = torch.stack([item["image"] for item in items]).to(device)
        cams, logits = grad_cam(model, images, layer)
        heads = images[:, centre] > -1 + 1e-6  # Background is exactly 0 in the JPEGs, i.e. -1 here.
        shares = outside_share(cams, heads)
        for scan, item, cam, logit, share in zip(batch_scans, items, cams.cpu(), logits.cpu(), shares.cpu()):
            records.append({**scan, "group": group_of(scan, info), "slice_index": item["slice_index"],
                            "slice_logit": float(logit), "map_share_outside_head": float(share),
                            "correct": int(scan["prediction"] == scan["label"])})
            inputs.append(item["image"][centre])
            maps.append(cam)

    groups = sorted({r["group"] for r in records})
    summary = {"checkpoint": str(args.checkpoint), "model_name": config["model_name"], "fold": config["fold"],
               "layer": args.layer, "role": "val", "groups": {}}
    for group in groups:
        members = [r for r in records if r["group"] == group]
        summary["groups"][group] = {
            "scans": len(members), "scan_accuracy": statistics.mean(r["correct"] for r in members),
            "mean_probability": statistics.mean(r["probability"] for r in members),
            "mean_map_share_outside_head": statistics.mean(r["map_share_outside_head"] for r in members)}
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "gradcam_scans.csv", records)
    write_json(output / "gradcam_summary.json", summary)
    plot_cases(output / "gradcam_cases.png", records, inputs, maps, args.per_group, config["model_name"])
    plot_group_means(output / "gradcam_group_means.png", records, inputs, maps, groups, config["model_name"])
    for group, values in summary["groups"].items():
        print(f"{group}: scans={values['scans']} accuracy={values['scan_accuracy']:.3f} "
              f"mean p(AD)={values['mean_probability']:.3f} map outside head={values['mean_map_share_outside_head']:.3f}",
              flush=True)
    return summary


def _pyplot(path: Path):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(path).parent / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_cases(path: Path, records: list[dict], inputs: list, maps: list, per_group: int, model_name: str) -> None:
    """Most confident correct and incorrect scans per group: input and overlay side by side."""
    chosen = []
    for group in sorted({r["group"] for r in records}):
        for correct in (1, 0):
            members = [i for i, r in enumerate(records) if r["group"] == group and r["correct"] == correct]
            members.sort(key=lambda i: abs(records[i]["probability"] - 0.5), reverse=True)
            chosen += members[:per_group]
    plt = _pyplot(path)
    figure, axes = plt.subplots(len(chosen), 2, figsize=(5.2, 2.6 * len(chosen)), squeeze=False, layout="constrained")
    for row, index in enumerate(chosen):
        record = records[index]
        image = (inputs[index] + 1) / 2
        axes[row, 0].imshow(image, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].imshow(image, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].imshow(maps[index], cmap="jet", alpha=0.45, vmin=0, vmax=1)
        verdict = "correct" if record["correct"] else "WRONG"
        axes[row, 0].set_title(f"{record['group']} | {record['patient_id']}", fontsize=7)
        axes[row, 1].set_title(f"scan p(AD)={record['probability']:.2f} ({verdict})", fontsize=7)
        for axis in axes[row]:
            axis.axis("off")
    figure.suptitle(f"Grad-CAM, {model_name}, validation middle slices", fontsize=8)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_group_means(path: Path, records: list[dict], inputs: list, maps: list, groups: list[str],
                     model_name: str) -> None:
    """Mean middle slice (top) and mean Grad-CAM (bottom) of every group."""
    plt = _pyplot(path)
    figure, axes = plt.subplots(2, len(groups), figsize=(3 * len(groups), 6), squeeze=False, layout="constrained")
    for column, group in enumerate(groups):
        members = [i for i, r in enumerate(records) if r["group"] == group]
        axes[0, column].imshow((torch.stack([inputs[i] for i in members]).mean(0) + 1) / 2, cmap="gray", vmin=0, vmax=1)
        axes[0, column].set_title(f"{group} (n={len(members)})", fontsize=8)
        axes[1, column].imshow(torch.stack([maps[i] for i in members]).mean(0), cmap="jet", vmin=0, vmax=1)
        for axis in axes[:, column]:
            axis.axis("off")
    figure.suptitle(f"Mean input and mean Grad-CAM, {model_name}", fontsize=9)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("/home/groups/comp3710/ADNI"))
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", choices=("last", "penultimate"), default="last",
                        help="Feature stage that is explained (default: last).")
    parser.add_argument("--per-group", type=int, default=2, help="Cases per group and outcome in gradcam_cases.png.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run(args)
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
