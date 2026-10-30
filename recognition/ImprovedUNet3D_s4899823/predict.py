import argparse,json,time,torch
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import torch.nn.functional as F

from dataset import DEFAULT_ROOT, find_pairs, patient_split, HipMRIDataset3D
from modules import ImprovedUNet


CLASS_NAMES = ["background", "body", "bone", "bladder", "rectum", "prostate"]
BLADDER, RECTUM, PROSTATE = 3, 4, 5
SEG_CMAP = ListedColormap(["black", "tan", "yellow", "deepskyblue", "orange", "red"])
OUT_DIR = Path("outputs")
IMAGES_DIR = Path("images")
REGION_FRAC = 0.2     
N_FAILURES = 4 

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Improved U-Net on the HipMRI test set")
    p.add_argument("--root", default=DEFAULT_ROOT)
    p.add_argument("--dims", type=int, choices=[2, 3], default=2)
    p.add_argument("--checkpoint", default=None, help="default: outputs/best_model_{dims}d.pt")
    p.add_argument("--scale", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--subset", type=int, default=0, help="only use this many test scans (0 = all)")
    
    return p.parse_args()

def load_model(check_point_path, dims,device):

    check_point= torch.load(check_point_path,map_location=device,weights_only=False)
    if check_point["dims"] != dims:
        raise ValueError(f"Checkpoint is {check_point['dims']}")
    model= ImprovedUNet(in_channels=1,num_classes=6,dims=dims)

    model.load_state_dict(check_point["model"])
    model.to(device).eval()
    return model,check_point

@torch.no_grad
def predict_volume_2d(model,image,device,batch_size = 32):
    slices = image.permute(3, 0, 1, 2) 
    preds =  []
    for i in range(0, slices.shape[0], batch_size):
        x = slices[i:i + batch_size].to(device)
        preds.append(model(x).argmax(dim=1).cpu())
    return torch.cat(preds).permute(1,2,0)    

@torch.no_grad
def predict_volume_3d(model,image,device,multiple=16):
    x = image.unsqueeze(0).to(device)
    H, W, D = x.shape[2:]
    pad = [(-s) % multiple for s in (H, W, D)] 
    x = F.pad(x, (0, pad[2], 0, pad[1], 0, pad[0]))
    pred = model(x).argmax(dim=1)[0] 

    return pred[:H, :W, :D].cpu()  


@torch.no_grad
def predict_volume(model, image, dims, device):
    if dims == 2:
        return predict_volume_2d(model, image, device)
    return predict_volume_3d(model, image, device)


# metrics
def dice_iou(pred,label,c):
    p = pred == c
    t = label == c
    inter = (p & t).sum().item()
    p_sum, t_sum = p.sum().item(), t.sum().item()
    if p_sum + t_sum == 0:                              
        return float("nan"), float("nan")
    return 2 * inter / (p_sum + t_sum), inter / (p_sum + t_sum - inter)

def base_apex_slices(label,frac):
    zs = torch.where((label == PROSTATE).any(dim =0).any(dim=0))[0].tolist()
    if not zs:
        return [],[]
    k = max(1,round(frac * len(zs)))
    return zs[-k:], zs[:k]

def evaluate_volume(pred, label, frac):
    r = {}
    for c, name in enumerate(CLASS_NAMES):
        r[f"dice_{name}"], r[f"iou_{name}"] = dice_iou(pred, label, c)

    base, apex = base_apex_slices(label, frac)
    r["dice_prostate_base"] = dice_iou(pred[..., base], label[..., base], PROSTATE)[0] if base else float("nan")
    r["dice_prostate_apex"] = dice_iou(pred[..., apex], label[..., apex], PROSTATE)[0] if apex else float("nan")

    pred_p = pred == PROSTATE
    n_pred = pred_p.sum().item()
    r["leak_into_bladder"] = (pred_p & (label == BLADDER)).sum().item() / n_pred if n_pred else 0.0
    r["leak_into_rectum"] = (pred_p & (label == RECTUM)).sum().item() / n_pred if n_pred else 0.0

    r["base_slices"], r["apex_slices"] = base, apex
    return r

def worst_slices(pred,label,zs):
    scores = [dice_iou(pred[..., z], label[..., z], PROSTATE)[0] for z in zs]
    scores = [1.0 if np.isnan(s) else s for s in scores]
    return zs[int(np.argmin(scores))], min(scores)


def plot_failure(image, label, pred, r, path):
    show = np.rot90                                     # rotate for display; change if sideways
    rows = []
    if r["base_slices"]:
        rows.append(("Base",) + worst_slices(pred, label, r["base_slices"]))
    if r["apex_slices"]:
        rows.append(("Apex",) + worst_slices(pred, label, r["apex_slices"]))
    if not rows:
        return

    fig, axes = plt.subplots(len(rows), 4, figsize=(14, 3.6 * len(rows)), squeeze=False)
    for row, (region, z, slice_dice) in enumerate(rows):
        img = image[0, :, :, z].numpy()
        gt = label[:, :, z].numpy()
        pr = pred[:, :, z].numpy()
        ax = axes[row]

        ax[0].imshow(show(img), cmap="gray")
        ax[0].set_title(f"{region} slice z={z}: image")

        for a, seg, title in [(ax[1], gt, "Ground truth"), (ax[2], pr, "Prediction")]:
            a.imshow(show(img), cmap="gray")
            a.imshow(show(np.ma.masked_where(seg < 2, seg)), cmap=SEG_CMAP,
                     vmin=0, vmax=5, alpha=0.5, interpolation="nearest")
            a.set_title(title)

        overlay = np.zeros(gt.shape + (4,))
        overlay[(gt == PROSTATE) & (pr == PROSTATE)] = [0, 1, 0, 0.6]     # correct: green
        overlay[(gt == PROSTATE) & (pr != PROSTATE)] = [0, 0.4, 1, 0.9]   # missed (FN): blue
        overlay[(gt != PROSTATE) & (pr == PROSTATE)] = [1, 0, 0, 0.9]     # false positive: red
        ax[3].imshow(show(img), cmap="gray")
        ax[3].imshow(show(overlay), interpolation="nearest")
        ax[3].set_title(f"Prostate difference (slice Dice {slice_dice:.2f})")

    for a in axes.flat:
        a.axis("off")
    fig.suptitle(f"{r['scan']}: prostate Dice {r['dice_prostate']:.3f} "
                 f"(base {r['dice_prostate_base']:.3f}, apex {r['dice_prostate_apex']:.3f})\n"
                 "Overlays: bone yellow, bladder blue, rectum orange, prostate red | "
                 "Difference: green correct, blue missed, red false positive")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_history(history_path, images_dir, dims):
    if not history_path.exists():
        print(f"No {history_path}, skipping training curves")
        return
    with open(history_path) as f:
        h = json.load(f)
    epochs = range(1, len(h["train_loss"]) + 1)
    dice = np.array(h["val_dice"])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    a1.plot(epochs, h["train_loss"], label="train")
    a1.plot(epochs, h["val_loss"], label="val")
    a1.set_xlabel("Epoch"); a1.set_ylabel("Dice + CE loss"); a1.set_title(f"{dims}D loss")
    a1.legend(); a1.grid(alpha=0.3)

    for c in range(1, len(CLASS_NAMES)):
        a2.plot(epochs, dice[:, c], label=CLASS_NAMES[c])
    a2.axhline(0.7, color="grey", linestyle="--", linewidth=1, label="target 0.7")
    a2.set_xlabel("Epoch"); a2.set_ylabel("Validation Dice"); a2.set_ylim(0, 1)
    a2.set_title(f"{dims}D validation Dice"); a2.legend(); a2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(images_dir / f"training_curves_{dims}d.png", dpi=150)
    plt.close(fig)

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = OUT_DIR
    images_dir =IMAGES_DIR
    images_dir.mkdir(parents=True, exist_ok=True)
    check_point_path = Path(args.checkpoint) if args.checkpoint else out_dir / f"best_model_{args.dims}d.pt"

    model, check_point = load_model(check_point_path, args.dims, device)
    print(f"Device: {device} | {check_point_path} (epoch {check_point['epoch'] + 1}, "
          f"val prostate Dice {check_point['val_dice'][PROSTATE]:.4f})")

    split = patient_split(find_pairs(args.root), seed=args.seed)
    test_pairs = split["test"][:args.subset] if args.subset else split["test"]
    ds = HipMRIDataset3D(test_pairs, scale=args.scale, train=False)
    print(f"Test set: {len(ds)} scans from {len({p['case'] for p in test_pairs})} patients")

    # warm-up so the first timed volume doesn't include GPU start-up cost
    predict_volume(model, torch.as_tensor(ds[0]["image"]).float(), args.dims, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    results, store = [], {}
    for i in range(len(ds)):
        s = ds[i]
        image = torch.as_tensor(s["image"]).float()
        label = torch.as_tensor(s["label"]).long()

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        pred = predict_volume(model, image, args.dims, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter() - t0

        r = evaluate_volume(pred, label, REGION_FRAC)
        r.update(scan=s["scan"], case=s["case"], inference_s=t)
        results.append(r)
        store[s["scan"]] = (image, label, pred)
        print(f"{s['scan']:>14}: prostate {r['dice_prostate']:.3f} | base {r['dice_prostate_base']:.3f} "
              f"| apex {r['dice_prostate_apex']:.3f} | {t * 1000:.0f} ms")

    # ----- summary over volumes -----
    metric_keys = [k for k in results[0] if k.startswith(("dice_", "iou_", "leak_"))]
    summary = {}
    for k in metric_keys:
        vals = np.array([r[k] for r in results], dtype=float)
        summary[k] = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals))}
    summary["inference_s_per_volume"] = float(np.mean([r["inference_s"] for r in results]))
    summary["peak_vram_gb_inference"] = (torch.cuda.max_memory_allocated() / 1024**3
                                         if device.type == "cuda" else 0.0)
    summary["n_test_scans"] = len(results)

    print(f"\n{args.dims}D test results ({len(results)} volumes, mean ± std)")
    print(f"{'class':<12}{'Dice':>16}{'IoU':>16}")
    for name in CLASS_NAMES:
        d, iou = summary[f"dice_{name}"], summary[f"iou_{name}"]
        print(f"{name:<12}{d['mean']:>9.3f} ± {d['std']:.3f}{iou['mean']:>9.3f} ± {iou['std']:.3f}")
    for k in ["dice_prostate_base", "dice_prostate_apex", "leak_into_bladder", "leak_into_rectum"]:
        print(f"{k:<22}{summary[k]['mean']:.3f} ± {summary[k]['std']:.3f}")
    print(f"Inference: {summary['inference_s_per_volume'] * 1000:.0f} ms/volume | "
          f"peak VRAM {summary['peak_vram_gb_inference']:.2f} GB")

    with open(out_dir / f"test_results_{args.dims}d.json", "w") as f:
        json.dump({"checkpoint": str(check_point_path), "summary": summary, "per_volume": results}, f, indent=2)

    # ----- failure cases: lowest prostate Dice -----
    def score(r):
        return 1.0 if np.isnan(r["dice_prostate"]) else r["dice_prostate"]
    for r in sorted(results, key=score)[:N_FAILURES]:
        path = images_dir / f"failure_{args.dims}d_{r['scan']}.png"
        plot_failure(*store[r["scan"]], r, path)
        print(f"Saved {path}")

    plot_history(out_dir / f"history_{args.dims}d.json", images_dir, args.dims)
    print("Done.")


if __name__ == "__main__":
    main()
