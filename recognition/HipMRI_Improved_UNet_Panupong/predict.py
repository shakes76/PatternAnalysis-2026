"""Predict with the trained HipMRI models: figures, failure cases and inference time.

Run this after ``train.py --test`` has evaluated both models. It does not
train, choose a checkpoint or tune anything: the checkpoints were already
chosen on the validation split by train.py. Here the test predictions are
only reported (slice-level and boundary metrics) and used to pick which
slices to display; nothing is selected or tuned on them.

* Visual comparison: MRI, ground truth and both models' predictions for one
  slice from each of ``--num-examples`` test series (evenly spread over the
  sorted series), using the slice with the largest ground-truth prostate area.
  Each prediction is followed by a prostate difference overlay: correctly
  found prostate (TP), predicted prostate that is not prostate (FP) and
  missed prostate (FN).
* Failure cases: the prostate-containing test slices where the Improved U-Net
  (the main model) has the lowest slice-level prostate Dice, at most one
  slice per series so the cases do not come from neighbouring slices. Many
  small edge slices score exactly 0, so ties go to the larger ground-truth
  prostate (the largest missed regions are shown first).
* Boundary analysis: the prostate 95th-percentile Hausdorff distance (HD95,
  in pixels) of every slice where both the ground truth and the prediction
  contain prostate; slice Dice, missed slices and HD95 grouped by distance
  from the nearest end of the series' prostate slice range (0, 1, 2, 3+);
  and how predicted prostate pixels are split over the true classes (and
  the reverse), to show leakage into the rectum or bladder.
* Consistency check: the per-series test prostate Dice is recomputed with
  ``metrics.py`` and printed next to the value ``train.py`` saved in
  ``test_metrics_<checkpoint>.json``.
* Inference time: forward passes only, with the whole test split already on
  the device and one warm-up pass first; the median of ``--timing-repeats``
  passes is reported for batch size 1 and for ``--batch-size``.
* 2.5D experiment: ``--models`` names the run folders to compare (the model
  type and the number of context slices are read from each checkpoint), and
  ``--split validation`` runs the same analysis on the validation split, so
  the 2.5D run can be judged without the test split. On validation the
  consistency check compares with the validation Dice saved in history.json.

Examples::

    python predict.py --data-root keras_slices_data --runs-dir runs
    python predict.py --split validation --models unet improved improved_25d
"""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dataset import CLASS_NAMES, NUM_CLASSES, PROSTATE_LABEL, HipMRISliceDataset
from metrics import SegmentationMetrics
from modules import build_model, count_parameters

# Model names used by train.py and the labels shown in figures.
MODELS = {"unet": "Baseline U-Net", "improved": "Improved U-Net"}
MAIN_MODEL = "improved"  # failure cases are chosen for this run folder
# Overlay colours for labels 1-5 (Body, Bones, Bladder, Rectum, Prostate);
# Background (0) is left transparent.
CLASS_COLOURS = ("#8d99ae", "#f4d35e", "#3a86ff", "#2a9d8f", "#e63946")
OUTLINE_COLOUR = "lime"  # ground-truth prostate outline on the prediction panels
# Prostate difference overlay (Okabe-Ito colours, readable with colour blindness).
DIFF_COLOURS = {"TP": "#009e73", "FP": "#d55e00", "FN": "#56b4e9"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-root", default="keras_slices_data")
    parser.add_argument("--runs-dir", default="runs",
                        help="folder holding the unet/ and improved/ output folders of train.py")
    parser.add_argument("--models", nargs="+", default=list(MODELS),
                        help="run folders inside --runs-dir to compare")
    parser.add_argument("--split", choices=["test", "validation"], default="test")
    parser.add_argument("--checkpoint", choices=["best", "last"], default="best",
                        help="best = validation-selected checkpoint (the primary result)")
    parser.add_argument("--out-dir", default=None,
                        help="default: <runs-dir>/predict_<checkpoint> for the test split, "
                             "<runs-dir>/predict_validation_<checkpoint> for validation")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-examples", type=int, default=4)
    parser.add_argument("--num-failures", type=int, default=5)
    parser.add_argument("--timing-repeats", type=int, default=5)
    parser.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    args = parser.parse_args()
    if min(args.batch_size, args.num_examples, args.num_failures, args.timing_repeats) < 1:
        parser.error("--batch-size, --num-examples, --num-failures and --timing-repeats must be >= 1")
    if len(set(args.models)) != len(args.models):
        parser.error("--models must not repeat a run folder")
    return args


def load_model(path, device):
    """Rebuild a model from its saved training config and load the weights saved by train.py.

    Returns the model, the checkpoint epoch, the model type and the number
    of context slices (0 for runs made before the 2.5D option existed).
    """
    state = torch.load(path, map_location=device, weights_only=True)
    model_type = state["config"]["model"]
    context_slices = state["config"].get("context_slices", 0)
    model = build_model(model_type, in_channels=2 * context_slices + 1).to(device)
    model.load_state_dict(state["model"])
    model.eval()  # dropout off; the baseline's batch norm uses its running statistics
    return model, state["epoch"], model_type, context_slices


def model_label(model_type, context_slices):
    """Figure label, e.g. "Improved U-Net" or "Improved U-Net 2.5D (3 slices)"."""
    if context_slices == 0:
        return MODELS[model_type]
    return f"{MODELS[model_type]} 2.5D ({2 * context_slices + 1} slices)"


def saved_dice(model_dir, split, checkpoint, epoch):
    """The per-series prostate Dice train.py saved for this checkpoint, or None."""
    if split == "test":
        path = model_dir / f"test_metrics_{checkpoint}.json"
        return json.loads(path.read_text())["prostate_dice_series_mean"] if path.is_file() else None
    path = model_dir / "history.json"
    if not path.is_file():
        return None
    records = [h for h in json.loads(path.read_text()) if h["epoch"] == epoch]
    return records[0]["val_prostate_dice"] if records else None


@torch.inference_mode()
def predict_labels(model, images, device, use_amp, batch_size):
    """Return predicted label maps (N, H, W) for images of shape (N, C, H, W)."""
    predictions = []
    for batch in images.split(batch_size):
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(batch.to(device))
        predictions.append(logits.argmax(dim=1).cpu())
    return torch.cat(predictions)


@torch.inference_mode()
def time_inference(model, images, device, use_amp, batch_size, repeats):
    """Median time of forward passes over the whole split, with the data already on the device."""
    batches = images.to(device).split(batch_size)

    def one_pass():
        for batch in batches:
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                model(batch)
        if device.type == "cuda":
            torch.cuda.synchronize()  # wait for the GPU before reading the clock

    one_pass()  # warm-up: cuDNN picks its algorithms for each batch shape
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()  # after warm-up, so tuning memory is not counted
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        one_pass()
        times.append(time.perf_counter() - start)
    seconds = float(np.median(times))
    return {
        "batch_size": batch_size,
        "repeats": repeats,
        "seconds_per_pass": seconds,
        "ms_per_slice": 1000 * seconds / len(images),
        "slices_per_second": len(images) / seconds,
        # Includes the model weights and the test images held on the GPU (about 0.07 GiB).
        "peak_gpu_memory_gib": (torch.cuda.max_memory_allocated() / 2 ** 30
                                if device.type == "cuda" else None),
    }


def prostate_dice_per_slice(predictions, labels):
    """Slice-level prostate Dice (None when neither map contains prostate)."""
    pred = predictions == PROSTATE_LABEL
    target = labels == PROSTATE_LABEL
    intersection = (pred & target).sum((1, 2)).tolist()
    total = (pred.sum((1, 2)) + target.sum((1, 2))).tolist()
    return [2 * i / t if t > 0 else None for i, t in zip(intersection, total)]


def boundary_pixels(mask):
    """Pixels of a binary mask (H, W) with a background pixel among their 8 neighbours.

    Pixels outside the image are ignored (max-pool pads with -inf).
    """
    m = mask.float()[None, None]
    eroded = -F.max_pool2d(-m, kernel_size=3, stride=1, padding=1)  # 3x3 erosion
    return (m - eroded)[0, 0] > 0


def hd95(pred_mask, true_mask):
    """95th-percentile Hausdorff distance in pixels between two binary masks.

    The 95th percentile of the distances from each boundary pixel to the
    other boundary is taken in both directions, and the larger value is
    returned (the same rule as MONAI's percentile=95; the 8-neighbour
    boundary used here can make values differ from MONAI's by about a
    pixel). None when either mask is empty.
    """
    if not pred_mask.any() or not true_mask.any():
        return None
    a = boundary_pixels(pred_mask).nonzero().float()
    b = boundary_pixels(true_mask).nonzero().float()
    distances = torch.cdist(a, b)
    return max(torch.quantile(distances.min(dim=1).values, 0.95).item(),
               torch.quantile(distances.min(dim=0).values, 0.95).item())


def distance_to_prostate_end(keys, labels):
    """Slices from each prostate slice to the nearest end of its series' prostate range.

    The Nifti files carry no orientation, so the two ends (base and apex)
    cannot be told apart; None for slices without prostate.
    """
    has_prostate = (labels == PROSTATE_LABEL).flatten(1).any(dim=1).tolist()
    ends = {}
    for (case, week, s), present in zip(keys, has_prostate):
        if present:
            low, high = ends.get((case, week), (s, s))
            ends[(case, week)] = (min(low, s), max(high, s))
    return [min(s - ends[(c, w)][0], ends[(c, w)][1] - s) if present else None
            for (c, w, s), present in zip(keys, has_prostate)]


def boundary_analysis(predictions, labels, dice, end_distance):
    """HD95, edge-versus-centre slice results and prostate leakage for one model."""
    pred = predictions == PROSTATE_LABEL
    target = labels == PROSTATE_LABEL
    hd = [hd95(pred[i], target[i]) if end_distance[i] is not None else None
          for i in range(len(labels))]

    def summarise(indices):
        values = [hd[i] for i in indices if hd[i] is not None]
        return {
            "slices": len(indices),
            "mean_dice": float(np.mean([dice[i] for i in indices])) if indices else None,
            "missed_slices": sum(1 for i in indices if not pred[i].any()),
            "hd95_slices": len(values),
            "hd95_px_median": float(np.median(values)) if values else None,
            "hd95_px_mean": float(np.mean(values)) if values else None,
        }

    positive = [i for i, d in enumerate(end_distance) if d is not None]
    groups = {"0": [i for i in positive if end_distance[i] == 0],
              "1": [i for i in positive if end_distance[i] == 1],
              "2": [i for i in positive if end_distance[i] == 2],
              "3+": [i for i in positive if end_distance[i] >= 3]}

    # Where predicted prostate pixels really belong, and where true prostate pixels went.
    by_true = torch.bincount(labels[pred], minlength=NUM_CLASSES) / max(int(pred.sum()), 1)
    by_pred = torch.bincount(predictions[target], minlength=NUM_CLASSES) / max(int(target.sum()), 1)
    return {
        "all_prostate_slices": summarise(positive),
        "by_slices_from_prostate_end": {name: summarise(g) for name, g in groups.items()},
        "predicted_prostate_by_true_class": dict(zip(CLASS_NAMES, by_true.tolist())),
        "true_prostate_by_predicted_class": dict(zip(CLASS_NAMES, by_pred.tolist())),
    }


def choose_examples(keys, labels, n):
    """Pick the largest ground-truth prostate slice of n evenly spread test series."""
    series = sorted({key[:2] for key in keys})
    n = min(n, len(series))
    chosen = [series[round(i * (len(series) - 1) / max(n - 1, 1))] for i in range(n)]
    areas = (labels == PROSTATE_LABEL).sum((1, 2)).tolist()
    return [max((i for i, key in enumerate(keys) if key[:2] == s), key=lambda i: areas[i])
            for s in chosen]


def choose_failures(keys, labels, dice, n):
    """Pick the lowest-Dice prostate slices of the main model, at most one per series.

    Ties (often several slices at Dice 0) go to the larger ground-truth prostate.
    """
    areas = (labels == PROSTATE_LABEL).sum((1, 2)).tolist()
    candidates = sorted((i for i in range(len(keys)) if areas[i] > 0),
                        key=lambda i: (dice[i], -areas[i]))
    picks, used_series = [], set()
    for i in candidates:
        if len(picks) == n:
            break
        if keys[i][:2] not in used_series:
            picks.append(i)
            used_series.add(keys[i][:2])
    return picks


def _format_dice(value):
    return "n/a" if value is None else f"{value:.3f}"


def difference_map(prediction, truth):
    """Prostate difference map: 0 neither, 1 TP, 2 FP (predicted only), 3 FN (missed)."""
    pred = prediction == PROSTATE_LABEL
    target = truth == PROSTATE_LABEL
    return np.select([pred & target, pred & ~target, ~pred & target], [1, 2, 3], default=0)


def save_figure(indices, images, labels, predictions, dice, keys, names, path, title):
    """Save one row per slice: MRI, ground truth, then each model's prediction and
    its prostate difference overlay (TP / FP / FN).

    ``images`` holds the network input; for 2.5D input only the middle
    (target) channel is shown. ``names`` maps each run folder to its label.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    cmap = ListedColormap(CLASS_COLOURS)
    diff_cmap = ListedColormap([DIFF_COLOURS[k] for k in ("TP", "FP", "FN")])
    columns = 2 + 2 * len(predictions)
    fig, axes = plt.subplots(len(indices), columns, figsize=(2.2 * columns, 3.9 * len(indices)),
                             squeeze=False)
    for row, index in enumerate(indices):
        image = images[index, images.shape[1] // 2].numpy()
        truth = labels[index].numpy()
        case, week, slice_index = keys[index]
        panels = [(None, None, f"Case {case} / week {week} / slice {slice_index}\n"
                               f"GT prostate {int((truth == PROSTATE_LABEL).sum())} px"),
                  (truth, None, "Ground truth")]
        for name, prediction in predictions.items():
            label_map = prediction[index].numpy()
            diff = difference_map(label_map, truth)
            panels.append((label_map, None,
                           f"{names[name]}\nprostate Dice {_format_dice(dice[name][index])}"))
            panels.append((None, diff, f"{names[name]}\nFP {int((diff == 2).sum())} px, "
                                       f"FN {int((diff == 3).sum())} px"))
        for column, (label_map, diff, caption) in enumerate(panels):
            ax = axes[row, column]
            ax.imshow(image, cmap="gray")
            if label_map is not None:
                # Labels 1-5 map onto the 5 colours; label 0 is masked out.
                ax.imshow(np.ma.masked_equal(label_map, 0), cmap=cmap, vmin=0.5, vmax=5.5,
                          alpha=0.45, interpolation="nearest")
                if column >= 2 and (truth == PROSTATE_LABEL).any():
                    ax.contour(truth == PROSTATE_LABEL, levels=[0.5], colors=OUTLINE_COLOUR,
                               linewidths=0.8)
            if diff is not None:
                ax.imshow(np.ma.masked_equal(diff, 0), cmap=diff_cmap, vmin=0.5, vmax=3.5,
                          alpha=0.8, interpolation="nearest")
            ax.set_title(caption, fontsize=8)
            ax.axis("off")

    handles = [Patch(color=colour, alpha=0.6, label=name)
               for colour, name in zip(CLASS_COLOURS, CLASS_NAMES[1:])]
    handles.append(Line2D([], [], color=OUTLINE_COLOUR, label="ground-truth prostate outline"))
    handles += [Patch(color=DIFF_COLOURS["TP"], label="prostate TP (found)"),
                Patch(color=DIFF_COLOURS["FP"], label="prostate FP (predicted, not prostate)"),
                Patch(color=DIFF_COLOURS["FN"], label="prostate FN (missed)")]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    default_name = (f"predict_{args.checkpoint}" if args.split == "test"
                    else f"predict_{args.split}_{args.checkpoint}")
    out_dir = Path(args.out_dir or runs_dir / default_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    torch.backends.cudnn.benchmark = True  # as in train.py

    # The chosen split with the same preprocessing as in train.py (no augmentation).
    # One input tensor per number of context slices; labels and keys are shared.
    inputs = {}

    def split_images(context_slices):
        if context_slices not in inputs:
            dataset = HipMRISliceDataset(args.data_root, args.split,
                                         context_slices=context_slices)
            inputs[context_slices] = [dataset[i] for i in range(len(dataset))]
        return torch.stack([s["image"] for s in inputs[context_slices]])  # (N, C, H, W)

    images = split_images(0)
    samples = inputs[0]
    labels = torch.stack([s["label"] for s in samples])  # (N, H, W)
    keys = [(s["case"], s["week"], s["slice"]) for s in samples]

    report = {
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor(),
        "torch": str(torch.__version__), "python": platform.python_version(), "amp": use_amp,
        "checkpoint": args.checkpoint, "split": args.split, "slices": len(samples), "models": {},
    }
    end_distance = distance_to_prostate_end(keys, labels)
    predictions, dice, names = {}, {}, {}
    for name in args.models:
        model_dir = runs_dir / name
        model, epoch, model_type, context_slices = load_model(
            model_dir / f"{args.checkpoint}.pt", device)
        names[name] = model_label(model_type, context_slices)
        model_images = split_images(context_slices)
        model_keys = [(s["case"], s["week"], s["slice"]) for s in inputs[context_slices]]
        if model_keys != keys:
            raise RuntimeError(f"{name}: slice order differs from the 2D dataset")
        predictions[name] = predict_labels(model, model_images, device, use_amp, args.batch_size)
        dice[name] = prostate_dice_per_slice(predictions[name], labels)

        # Recompute the primary metric to check that the reloaded model matches train.py.
        metrics = SegmentationMetrics()
        metrics.update(predictions[name], labels, *zip(*keys))
        recomputed = metrics.summary()["prostate_dice_series_mean"]
        saved = saved_dice(model_dir, args.split, args.checkpoint, epoch)

        timing = [time_inference(model, model_images, device, use_amp, batch_size,
                                 args.timing_repeats)
                  for batch_size in (1, args.batch_size)]
        boundary = boundary_analysis(predictions[name], labels, dice[name], end_distance)
        report["models"][name] = {
            "label": names[name], "model": model_type, "context_slices": context_slices,
            "checkpoint_epoch": epoch, "parameters": count_parameters(model),
            "prostate_dice_recomputed": recomputed, "prostate_dice_saved": saved,
            "timing": timing, "boundary": boundary,
        }
        print(f"{names[name]} [{name}] ({args.checkpoint}, epoch {epoch}): {args.split} "
              f"prostate Dice per-series mean {recomputed:.4f} (saved by train.py: "
              f"{'n/a' if saved is None else f'{saved:.4f}'})")
        for t in timing:
            print(f"  batch {t['batch_size']:2d}: {t['ms_per_slice']:.3f} ms/slice, "
                  f"{t['slices_per_second']:.0f} slices/s")
        groups = {"all": boundary["all_prostate_slices"], **boundary["by_slices_from_prostate_end"]}
        for group, g in groups.items():
            label = "all" if group == "all" else f"{group} from end"
            hd = "n/a" if g["hd95_px_median"] is None else f"{g['hd95_px_median']:.2f} px"
            print(f"  prostate slices {label:>10}: {g['slices']:3d} slices, mean Dice "
                  f"{_format_dice(g['mean_dice'])}, missed {g['missed_slices']:2d}, HD95 median {hd}")
        leak = boundary["predicted_prostate_by_true_class"]
        print(f"  predicted prostate pixels that are truly rectum {leak['Rectum']:.2%}, "
              f"bladder {leak['Bladder']:.2%}, body {leak['Body']:.2%}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    main_model = MAIN_MODEL if MAIN_MODEL in predictions else args.models[0]
    split_name = "Test" if args.split == "test" else "Validation"
    examples = choose_examples(keys, labels, args.num_examples)
    failures = choose_failures(keys, labels, dice[main_model], args.num_failures)
    save_figure(examples, images, labels, predictions, dice, keys, names,
                out_dir / "comparison.png",
                f"{split_name} slices with the largest ground-truth prostate area in each shown "
                f"series ({args.checkpoint} checkpoints)")
    save_figure(failures, images, labels, predictions, dice, keys, names,
                out_dir / "failures.png",
                f"Lowest {names[main_model]} prostate Dice on {split_name.lower()} slices, "
                f"one per series ({args.checkpoint} checkpoints)")

    def describe(i):
        case, week, slice_index = keys[i]
        return {"series": f"{case}_week{week}", "slice": slice_index,
                "gt_prostate_pixels": int((labels[i] == PROSTATE_LABEL).sum()),
                "pred_prostate_pixels": {n: int((p[i] == PROSTATE_LABEL).sum())
                                         for n, p in predictions.items()},
                "prostate_dice": {n: d[i] for n, d in dice.items()}}

    report["failures_chosen_for"] = main_model
    report["examples"] = [describe(i) for i in examples]
    report["failures"] = [describe(i) for i in failures]
    (out_dir / "predict_summary.json").write_text(json.dumps(report, indent=2))
    print(f"Failure cases (chosen for {names[main_model]}):")
    for f in report["failures"]:
        print(f"  {f['series']} slice {f['slice']}: GT {f['gt_prostate_pixels']} px, Dice "
              + ", ".join(f"{names[n]} {_format_dice(d)}" for n, d in f["prostate_dice"].items()))
    print(f"Saved comparison.png, failures.png and predict_summary.json to {out_dir}")


if __name__ == "__main__":
    main()
