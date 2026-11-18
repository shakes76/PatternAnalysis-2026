"""
Showing example usage of your trained model. It should load your saved model
weights, run inference on test cases, print out any results, and provide visualisations where applicable
(e.g., prediction overlays, generated samples, heatmaps). NumPy is allowed in this script for
visualization or loading data.
"""

import argparse
import glob
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save figures without a window
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
import torch
import torch.nn.functional as F

from dataset import full_load_dataset, N_LABELS, H, W
from modules import UNet2D
from train import LABELS, REGIONS, prostate_regions

LABEL_COLOURS = ListedColormap(["#8da0cb", "#e5c494", "#ffd92f", "#a6d854", "#e41a1c"])  # body .. prostate
ERROR_COLOURS = {"TP": (1.0, 0.85, 0.0), "FP": (0.85, 0.1, 0.1), "FN": (0.1, 0.45, 0.95)}


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained 2D U-Net on the HipMRI test split")
    p.add_argument("run_dirs", type=Path, nargs="+", help="run folders written by train.py")
    p.add_argument("--data-root", type=Path, required=True,
                   help="HipMRI keras_slices_data folder (holds the keras_slices_<split> and keras_slices_seg_<split> folders)")
    p.add_argument("--cache-dir", type=Path, required=True,
                   help="folder for the slice cache (~2 GB, built on the first run), keep it outside the repo")
    p.add_argument("--checkpoint", default="best.pt", help="checkpoint file inside each run folder")
    p.add_argument("--batch-size", type=int, default=16, help="slices per forward pass at inference")
    p.add_argument("--fail-dice", type=float, default=0.5, help="prostate slices below this dice are listed as failures")
    p.add_argument("--slices", nargs="+", metavar="NAME",
                   help="only draw these test slices (e.g. case_010_week_5_slice_17) into <run>/failures/slices/")
    p.add_argument("--split", choices=["test", "validate"], default="test",
                   help="validate to choose settings like --hysteresis without looking at test")
    p.add_argument("--hysteresis", type=float, nargs="+", metavar="LOW",
                   help="also evaluate with 3D hysteresis: argmax prostate grows into touching pixels, on the same or a neighbouring slice, "
                        "with prostate probability > LOW. several values sweep them, plain argmax is always evaluated first")
    args = p.parse_args()
    # PowerShell passes wildcards through unexpanded, so expand runs/* here
    run_dirs = []
    for pattern in args.run_dirs:
        matches = sorted(Path().glob(str(pattern))) if glob.has_magic(str(pattern)) else [pattern]
        run_dirs.extend(d for d in matches if d.is_dir())
        if not any(d.is_dir() for d in matches):
            p.error(f"no run folder matches {pattern} (relative to {Path.cwd()})")
    args.run_dirs = run_dirs
    return args

def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if "args" not in ckpt:
        raise ValueError(f"{checkpoint_path} has no saved args, use best.pt rather than last.pt")
    args = ckpt["args"]
    model = UNet2D(1 + 2 * args.get("context", 0), N_LABELS, conv_depths=tuple(args["conv_depths"]), skip_connected=args["skip_connected"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, args, ckpt

def val_region_dice(run_dir, epoch):
    """base/central/apex val prostate dice at the checkpoint's epoch, from train.py's history.json"""
    with open(run_dir / "history.json") as f:
        return json.load(f)["val_region_dice"][epoch - 1]

def eval_dataset(train_args, split, data_root, cache_dir):
    """the run's held-out test (or validate) patients, rebuilt from its saved seed and split sizes, no augmentation.
    the data and cache folders come from this command line, not the checkpoint, so runs still evaluate after the data moves"""
    loaders = full_load_dataset(seed=train_args["seed"], batch_size=1, augment=False, num_workers=0,
                                n_val=train_args["n_val"], n_test=train_args["n_test"], root=data_root, cache_dir=cache_dir,
                                resplit=not train_args.get("original_split", False), context=train_args.get("context", 0))
    return loaders[split].dataset

def cases_in(dataset):
    """(patient, week) -> dataset indices of that scan's slices, a case is one weekly scan"""
    cases = {}
    for k, sid in enumerate(dataset.ids):
        cases.setdefault((sid.patient, sid.week), []).append(k)
    return cases

def class_counts(pred, mask, n_classes=N_LABELS):
    """per-class overlap, predicted and true pixel counts, summed over the batch"""
    inter, pred_sum, true_sum = (torch.zeros(n_classes, device=mask.device) for _ in range(3))
    for c in range(n_classes):
        p, t = pred == c, mask == c
        inter[c], pred_sum[c], true_sum[c] = (p & t).sum(), p.sum(), t.sum()
    return inter, pred_sum, true_sum

def dice_iou(inter, pred_sum, true_sum, eps=1e-6):
    dice = (2 * inter + eps) / (pred_sum + true_sum + eps)
    iou = (inter + eps) / (pred_sum + true_sum - inter + eps)
    return dice.cpu(), iou.cpu()

def connected_components(mask):
    labels = torch.arange(1, mask[0].numel() + 1, device=mask.device, dtype=torch.float32).view(1, *mask.shape[1:])
    labels = (labels * mask).unsqueeze(1)
    keep = mask.unsqueeze(1)
    while True:
        spread = F.max_pool2d(labels, kernel_size=3, stride=1, padding=1) * keep
        if torch.equal(spread, labels):
            return labels.squeeze(1).long()
        labels = spread

def hysteresis(pred, prob, low):
    """pred [S,H,W] argmax of one scan's slices in order, prob [S,H,W] prostate probability.
    argmax prostate is the seed, it grows into touching pixels (26-connected, so across neighbouring slices too)
    with prob > low. faint pixels not touching a seed stay as they were.
    only slices where argmax found no prostate take the grown pixels, growing on found slices just inflated their outlines"""
    seed = pred == LABELS["prostate"]
    keep = ((prob > low) | seed).float()[None, None]
    grown = seed.float()[None, None]
    while True:
        spread = F.max_pool3d(grown, kernel_size=3, stride=1, padding=1) * keep
        if torch.equal(spread, grown):
            break
        grown = spread
    empty = ~seed.flatten(1).any(dim=1)  # slices argmax left without prostate
    return torch.where(grown[0, 0].bool() & empty[:, None, None], LABELS["prostate"], pred)

def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)

@torch.no_grad()
def evaluate(model, dataset, device, batch_size, low=None):
    """Dice/IoU per class over the whole split, base/central/apex prostate Dice, per-case prostate Dice,
    peak VRAM and per-case inference time (upload + forward + argmax + hysteresis, data loading not counted).
    low: hysteresis threshold, None for plain argmax"""
    regions = prostate_regions(dataset)
    inter, pred_sum, true_sum = (torch.zeros(N_LABELS, device=device) for _ in range(3))
    reg_inter, reg_total = (torch.zeros(len(REGIONS), device=device) for _ in range(2))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model(torch.zeros(batch_size, 1 + 2 * dataset.context, H, W, device=device))  # warm-up so cuDNN set-up isn't timed in the first case
    sync(device)

    cases, slices = [], []
    preds = np.zeros((len(dataset), H, W), dtype=np.uint8)
    for (patient, week), idx in cases_in(dataset).items():
        images, masks, names = zip(*(dataset[k] for k in idx))
        images, masks = torch.stack(images), torch.stack(masks)
        sync(device)
        t0 = time.perf_counter()
        logits = torch.cat([model(images[i:i + batch_size].to(device)) for i in range(0, len(idx), batch_size)])
        pred = logits.argmax(dim=1)
        if low is not None:
            pred = hysteresis(pred, logits.softmax(dim=1)[:, LABELS["prostate"]], low)
        sync(device)
        seconds = time.perf_counter() - t0

        masks = masks.to(device)
        i, p, t = class_counts(pred, masks)
        inter += i
        pred_sum += p
        true_sum += t
        r = torch.tensor([regions.get(n, -1) for n in names], device=device)
        keep = r >= 0
        pred_p, true_p = pred == LABELS["prostate"], masks == LABELS["prostate"]
        s_inter, s_pred, s_true = ((pred_p & true_p).flatten(1).sum(1), pred_p.flatten(1).sum(1), true_p.flatten(1).sum(1))
        reg_inter.index_add_(0, r[keep], s_inter[keep].float())
        reg_total.index_add_(0, r[keep], (s_pred + s_true)[keep].float())
        case_dice, _ = dice_iou(i, p, t)
        cases.append({"patient": patient, "week": week, "slices": len(idx), "seconds": seconds,
                      "prostate_dice": case_dice[LABELS["prostate"]].item()})

        # per-slice prostate stats for the failure autopsy, a stray blob is a predicted component not touching the true prostate
        comps, true_np = connected_components(pred_p).cpu().numpy(), true_p.cpu().numpy()
        preds[idx] = pred.cpu().numpy()
        for j, k in enumerate(idx):
            ids = np.unique(comps[j])
            ids = ids[ids > 0]
            stray = np.setdiff1d(ids, comps[j][true_np[j]])
            slices.append({"index": k, "name": names[j], "patient": patient, "week": week, "slice": dataset.ids[k].slice,
                           "region": int(r[j]), "inter": int(s_inter[j]), "pred": int(s_pred[j]), "true": int(s_true[j]),
                           "components": len(ids), "stray_blobs": len(stray),
                           "stray_px": int(np.isin(comps[j], stray).sum()) if len(stray) else 0})

    dice, iou = dice_iou(inter, pred_sum, true_sum)
    region_dice = ((2 * reg_inter + 1e-6) / (reg_total + 1e-6)).cpu()
    peak_vram = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else float("nan")
    return {"dice": dice.tolist(), "iou": iou.tolist(), "region_dice": region_dice.tolist(),
            "peak_vram_gb": peak_vram, "batch_size": batch_size, "cases": cases, "slices": slices}, preds

def print_results(name, split, results):
    seconds = np.array([c["seconds"] for c in results["cases"]])
    slices = np.array([c["slices"] for c in results["cases"]])
    case_dice = np.array([c["prostate_dice"] for c in results["cases"]])
    print(f"\n{name} {split}: {len(results['cases'])} cases, {slices.sum()} slices")
    print(f"  {'class':12} {'dice':>6} {'iou':>6}")
    for label, c in LABELS.items():
        print(f"  {label:12} {results['dice'][c]:6.3f} {results['iou'][c]:6.3f}")
    print(f"  prostate {' '.join(f'{r} {d:.3f}' for r, d in zip(REGIONS, results['region_dice']))}")
    print(f"  per-case prostate dice {case_dice.mean():.3f} +/- {case_dice.std():.3f} (min {case_dice.min():.3f})")
    print(f"  peak VRAM {results['peak_vram_gb']:.2f} GB, inference {1000 * seconds.mean():.0f} +/- {1000 * seconds.std():.0f} ms per case "
          f"({1000 * seconds.sum() / slices.sum():.1f} ms per slice, batch {results['batch_size']})")

def print_comparison(all_results):
    w = max(len(name) for name in all_results)
    print(f"\n{'run':{w}} {'pros dice':>9} {'pros iou':>8} {'base':>6} {'central':>7} {'apex':>6} {'case dice':>13} "
          f"{'prec':>6} {'recall':>6} {'FP px':>6} {'VRAM GB':>7} {'ms/case':>7}")
    for name, r in all_results.items():
        case_dice = np.array([c["prostate_dice"] for c in r["cases"]])
        ms = 1000 * np.mean([c["seconds"] for c in r["cases"]])
        pr = LABELS["prostate"]
        f = r["failure_summary"]
        print(f"{name:{w}} {r['dice'][pr]:9.3f} {r['iou'][pr]:8.3f} "
              f"{r['region_dice'][0]:6.3f} {r['region_dice'][1]:7.3f} {r['region_dice'][2]:6.3f} "
              f"{case_dice.mean():6.3f}+/-{case_dice.std():.3f} {f['precision']:6.3f} {f['recall']:6.3f} {f['fp_px']['total']:6d} "
              f"{r['peak_vram_gb']:7.2f} {ms:7.0f}")

def slice_dice(s, eps=1e-6):
    return (2 * s["inter"] + eps) / (s["pred"] + s["true"] + eps)

def beyond_side(slices):
    """slice index -> 'base'/'apex' for slices without true prostate: the region of the nearest prostate slice in that scan"""
    side = {}
    for case in {(s["patient"], s["week"]) for s in slices}:
        rows = [s for s in slices if (s["patient"], s["week"]) == case]
        gland = [s for s in rows if s["region"] >= 0]
        for s in rows:
            if s["region"] < 0 and gland:
                nearest = min(gland, key=lambda g: abs(g["slice"] - s["slice"]))
                side[s["index"]] = REGIONS[nearest["region"]]
    return side

def failure_summary(slices):
    side = beyond_side(slices)
    gland = [s for s in slices if s["true"] > 0]
    missed = [s for s in gland if s["pred"] == 0]
    beyond = [s for s in slices if s["true"] == 0 and s["pred"] > 0]
    stray = [s for s in gland if s["stray_blobs"] > 0]
    summary = {
        "gland_slices": len(gland),
        "missed": {r: sum(s["region"] == k for s in missed) for k, r in enumerate(REGIONS)},
        "beyond_gland": {r: sum(side.get(s["index"]) == r for s in beyond) for r in ("base", "apex")},
        "beyond_gland_median_px": float(np.median([s["pred"] for s in beyond])) if beyond else 0.0,
        "stray_blob_slices": len(stray),
        "median_slice_dice": {r: float(np.median([slice_dice(s) for s in gland if s["region"] == k])) for k, r in enumerate(REGIONS)},
        **false_positives(slices, side),
    }
    print(f"  failure summary: {len(gland)} slices with prostate, {len(missed)} missed entirely "
          f"({', '.join(f'{r} {n}' for r, n in summary['missed'].items())})")
    print(f"    {len(beyond)} slices with prostate predicted beyond the gland "
          f"({', '.join(f'past {r} {n}' for r, n in summary['beyond_gland'].items())}, median {summary['beyond_gland_median_px']:.0f} px)")
    print(f"    {len(stray)} prostate slices with stray blobs, median slice dice "
          f"{' '.join(f'{r} {d:.3f}' for r, d in summary['median_slice_dice'].items())}")
    fp = summary["fp_px"]
    print(f"    false positive prostate: {fp['total']} px, {summary['fp_share']:.1%} of predicted prostate "
          f"(precision {summary['precision']:.3f}, recall {summary['recall']:.3f}) | "
          f"on gland slices {' '.join(f'{r} {fp[r]}' for r in REGIONS)} | past base {fp['past base']} past apex {fp['past apex']}")
    return summary

def false_positives(slices, side):
    """prostate predicted where there is none, in pixels: spill-over on gland slices by region, and on slices past the gland"""
    fp_px = {r: 0 for r in (*REGIONS, "past base", "past apex")}
    for s in slices:
        fp = s["pred"] - s["inter"]
        if s["region"] >= 0:
            fp_px[REGIONS[s["region"]]] += fp
        elif s["index"] in side:
            fp_px[f"past {side[s['index']]}"] += fp
    inter, pred, true = (sum(s[k] for s in slices) for k in ("inter", "pred", "true"))
    fp_px["total"] = pred - inter  # also counts scans with no prostate at all, which none of the groups above do
    return {"fp_px": fp_px, "fp_share": fp_px["total"] / max(pred, 1),
            "precision": inter / max(pred, 1), "recall": inter / max(true, 1)}

def pick_failures(slices):
    picks, used = [], set()
    def take(kind, candidates, key):
        candidates = sorted(candidates, key=key)
        fresh = [s for s in candidates if (s["patient"], s["week"]) not in used]
        if fresh or candidates:
            s = (fresh or candidates)[0]
            picks.append((kind, s))
            used.add((s["patient"], s["week"]))
    worst = lambda s: (slice_dice(s), -s["true"])  # lowest dice, the bigger gland breaks ties
    take("base", [s for s in slices if s["region"] == 0], worst)
    take("apex", [s for s in slices if s["region"] == 2], worst)
    # a visible (10+ px) stray blob next to a gland that was found, otherwise any slice with one
    stray = [s for s in slices if s["true"] > 0 and s["stray_blobs"] > 0]
    take("stray blob", [s for s in stray if s["inter"] > 0 and s["stray_px"] >= 10] or stray, lambda s: -s["stray_px"])
    take("beyond the gland", [s for s in slices if s["true"] == 0 and s["pred"] > 0], lambda s: -s["pred"])
    return picks

def error_overlay(image, pred, mask):
    grey = (image - image.min()) / (np.ptp(image) + 1e-8)
    rgb = np.repeat(grey[..., None], 3, axis=-1)
    p, t = pred == LABELS["prostate"], mask == LABELS["prostate"]
    for name, where in (("TP", p & t), ("FP", p & ~t), ("FN", ~p & t)):
        rgb[where] = 0.35 * rgb[where] + 0.65 * np.array(ERROR_COLOURS[name])
    return rgb

def zoom_box(pred, mask, margin=12):
    rows, cols = np.nonzero((pred == LABELS["prostate"]) | (mask == LABELS["prostate"]))
    if len(rows) == 0:
        return slice(None), slice(None)
    return (slice(max(rows.min() - margin, 0), rows.max() + margin + 1),
            slice(max(cols.min() - margin, 0), cols.max() + margin + 1))

def plot_failure_row(axes, kind, s, image, mask, pred):
    overlay = error_overlay(image, pred, mask)
    r, c = zoom_box(pred, mask)
    panels = [("input", None), ("ground truth", mask), ("prediction", pred), ("prostate errors", "overlay"), ("errors, zoomed", "zoom")]
    for ax, (title, layer) in zip(axes, panels):
        if isinstance(layer, str) and layer == "overlay":
            ax.imshow(overlay.transpose(1, 0, 2))
        elif isinstance(layer, str) and layer == "zoom":
            ax.imshow(overlay[r, c].transpose(1, 0, 2))
        else:
            ax.imshow(image.T, cmap="gray")
            if layer is not None:
                ax.imshow(np.ma.masked_equal(layer, 0).T, cmap=LABEL_COLOURS, vmin=1, vmax=N_LABELS - 1, alpha=0.55, interpolation="nearest")
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    axes[0].text(-0.05, 0.5, f"{kind}\n{s['name']}\ndice {slice_dice(s):.2f}\ntrue {s['true']} px, pred {s['pred']} px"
                 + (f"\n{s['stray_blobs']} stray blob(s), {s['stray_px']} px" if s["stray_blobs"] else ""),
                 transform=axes[0].transAxes, ha="right", va="center", fontsize=8)

def failure_row(kind, s, dataset, preds):
    image, mask = dataset[s["index"]][:2]
    image, mask = image[dataset.context].numpy(), mask.numpy()  # centre slice of a 2.5D stack
    return kind, s, image, mask, preds[s["index"]]

def save_failure_figure(rows, path):
    legend = [Patch(color=ERROR_COLOURS[k]) for k in ERROR_COLOURS]
    legend += [Patch(color=LABEL_COLOURS(i), alpha=0.55) for i in range(N_LABELS - 1)]
    labels = ["prostate TP", "prostate FP", "prostate FN"] + [n for n in LABELS if n != "background"]
    fig, axes = plt.subplots(len(rows), 5, figsize=(17, 2.6 * len(rows) + 0.6), squeeze=False)
    for ax_row, row in zip(axes, rows):
        plot_failure_row(ax_row, *row)
    fig.legend(legend, labels, loc="lower center", ncol=len(labels), fontsize=8, frameon=False)
    fig.tight_layout(rect=(0.08, 0.06, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)

def plot_failures(picks, dataset, preds, out_dir):
    out_dir.mkdir(exist_ok=True)
    rows = [failure_row(kind, s, dataset, preds) for kind, s in picks]
    save_failure_figure(rows, out_dir / "failures.png")
    for k, row in enumerate(rows, 1):
        save_failure_figure([row], out_dir / f"failure_{k}_{row[0].replace(' ', '_')}.png")
    print(f"  failure figures: {', '.join(f'{kind} {s['name']} (dice {slice_dice(s):.2f})' for kind, s in picks)} -> {out_dir}")

def failure_kinds(s, fail_dice, min_stray_px=10):
    kinds = []
    if s["region"] >= 0 and slice_dice(s) < fail_dice:
        kinds.append(REGIONS[s["region"]])
    if s["true"] > 0 and s["stray_px"] >= min_stray_px:
        kinds.append("stray blob")
    if s["true"] == 0 and s["pred"] > 0:
        kinds.append("beyond the gland")
    return kinds

def failure_order(kind, s):
    if kind == "stray blob":
        return -s["stray_px"]
    if kind == "beyond the gland":
        return -s["pred"]
    return slice_dice(s) - 1e-6 * s["true"]  # the bigger gland breaks dice ties

def all_failures(slices, fail_dice):
    failures = {kind: [] for kind in (*REGIONS, "stray blob", "beyond the gland")}
    for s in slices:
        kinds = failure_kinds(s, fail_dice)
        for kind in kinds:
            failures[kind].append((" + ".join(kinds), s))
    return {kind: sorted(rows, key=lambda row: failure_order(kind, row[1])) for kind, rows in failures.items()}

def plot_slices(names, slices, dataset, preds, out_dir, fail_dice):
    by_name = {s["name"]: s for s in slices}
    missing = [n for n in names if n not in by_name]
    if missing:
        raise ValueError(f"not in this run's test split: {', '.join(missing)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        s = by_name[name]
        label = " + ".join(failure_kinds(s, fail_dice)) or "slice"
        save_failure_figure([failure_row(label, s, dataset, preds)], out_dir / f"{name}.png")
        print(f"  {name} ({label}, dice {slice_dice(s):.2f}) -> {out_dir / f'{name}.png'}")

if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets, all_results = {}, {}
    for run_dir in args.run_dirs:
        model, train_args, ckpt = load_model(run_dir / args.checkpoint, device)
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"{run_dir.name}: epoch {ckpt['epoch']}, {n_params:.2f}M params, "
              f"skip {train_args['skip_connected']}, depths {train_args['conv_depths']}, "
              f"seed {train_args['seed']}, val prostate dice {ckpt['val_dice'][LABELS['prostate']]:.3f} "
              f"({' '.join(f'{name} {d:.3f}' for name, d in zip(REGIONS, val_region_dice(run_dir, ckpt['epoch'])))})")

        if train_args.get("original_split", False):
            split = "original"
        else:
            split = (train_args["seed"], train_args["n_val"], train_args["n_test"])
        split = (split, train_args.get("context", 0))  # 2D and 2.5D runs need their own datasets
        if split not in datasets:  # runs with the same split share the test set
            datasets[split] = eval_dataset(train_args, args.split, args.data_root, args.cache_dir)
        dataset = datasets[split]
        for low in [None, *(args.hysteresis or [])]:
            # plain test keeps the old file names, validate and hysteresis get their own so nothing is overwritten
            tag = f"_hyst{low:g}" if low is not None else ""
            out_tag = ("" if args.split == "test" else f"_{args.split}") + tag
            results, preds = evaluate(model, dataset, device, args.batch_size, low)
            if args.slices:  # only draw the chosen slices, leave the saved results and figures alone
                plot_slices(args.slices, results["slices"], dataset, preds, run_dir / f"failures{out_tag}" / "slices", args.fail_dice)
                continue
            name = run_dir.name + tag
            print_results(name, args.split, results)
            results["failure_summary"] = failure_summary(results["slices"])
            picks = pick_failures(results["slices"])
            plot_failures(picks, dataset, preds, run_dir / f"failures{out_tag}")
            results["failures"] = [{"kind": kind, "name": s["name"]} for kind, s in picks]
            failures = all_failures(results["slices"], args.fail_dice)
            results_file = f"{args.split}_results{tag}.json"
            print(f"  all failures: {', '.join(f'{kind} {len(rows)}' for kind, rows in failures.items())} -> {results_file}")
            results["all_failures"] = {kind: [{"kinds": label, "name": s["name"], "dice": slice_dice(s)} for label, s in rows]
                                       for kind, rows in failures.items()}
            with open(run_dir / results_file, "w") as f:
                json.dump(results, f, indent=1)
            all_results[name] = results
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if len(all_results) > 1:
        print_comparison(all_results)
