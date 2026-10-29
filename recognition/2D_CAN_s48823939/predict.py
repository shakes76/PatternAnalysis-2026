import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

# Import modules and dataset loader
from modules import ContextAggregationModule, Simple2DUNet
from dataset import get_oasis_dataloaders

from train import PROJECT_ROOT, DATA_DIR, MODELS_DIR

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "visualisations")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_TYPE = "can"      # 'can' or 'unet'
BATCH_SIZE = 8
NUM_CLASSES = 4
CLASS_NAMES = ["Background", "CSF", "Grey Matter", "White Matter"]
TISSUE_CLASSES = [1, 2, 3]  # exclude background for the tissue-only means

# Custom color palette for OASIS 4-class segmentation
CLASS_COLORS = ['none', '#e6194b', '#3cb44b', '#4363d8']


def build_model(model_type, device):
    """Return (model, weights_filename, tag) for the chosen model type."""
    if model_type.lower() == "unet":
        model = Simple2DUNet(in_channels=1, num_classes=NUM_CLASSES, C=64)
        return model.to(device), "best_unet_model.pth", "unet"
    model = ContextAggregationModule(in_channels=1, num_classes=NUM_CLASSES, C=64)
    return model.to(device), "best_can_model.pth", "can"

def load_weights(model, weights_filename, device='cpu'):
    """Safely loads model weights, checking both script directory and project root."""
    
    weights_path = os.path.join(MODELS_DIR, weights_filename) # Inside models/ folder (matches train.py)    

    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=device)
        if 'model_state' in state_dict:
            model.load_state_dict(state_dict['model_state'])
        else:
            model.load_state_dict(state_dict)
        print(f"Successfully loaded weights from {weights_path}")
    else:
        print(f"WARNING: Weights file '{weights_filename}' not found. Running with uninitialized weights.")
    return model

def safe_ratio(num, den):
    """Return num/den, or NaN when the denominator is zero (class absent everywhere)."""
    return float(num) / float(den) if den > 0 else float("nan")

def evaluate(model, loader, device):
    """
    Runs inference on the full loader and returns a dict of metrics.
 
    Global counts per class c over the whole test set:
        I_c = |pred==c AND gt==c|     (intersection)
        P_c = |pred==c|               (prediction cardinality)
        T_c = |gt==c|                 (target cardinality)
 
        Dice_c = 2 I_c / (P_c + T_c)
        IoU_c  = I_c / (P_c + T_c - I_c)
    """
    inter = np.zeros(NUM_CLASSES, dtype=np.int64)
    pred_card = np.zeros(NUM_CLASSES, dtype=np.int64)
    tgt_card = np.zeros(NUM_CLASSES, dtype=np.int64)
    per_slice_dice = []  # rows: slices; cols: classes (NaN where undefined)
 
    with torch.no_grad():
        for images, targets in loader:
            outputs = model(images.to(device))
            preds = torch.argmax(outputs, dim=1).cpu().numpy()   # [B, H, W]
            gts = targets.numpy()                                # [B, H, W]
 
            for p, g in zip(preds, gts):
                slice_dice = np.full(NUM_CLASSES, np.nan)
                for c in range(NUM_CLASSES):
                    pc = (p == c)
                    gc = (g == c)
                    i = np.logical_and(pc, gc).sum()
                    pn, tn = pc.sum(), gc.sum()
 
                    inter[c] += i
                    pred_card[c] += pn
                    tgt_card[c] += tn
 
                    slice_dice[c] = safe_ratio(2 * i, pn + tn)
                per_slice_dice.append(slice_dice)
 
    union = pred_card + tgt_card - inter
    dice = np.array([safe_ratio(2 * inter[c], pred_card[c] + tgt_card[c]) for c in range(NUM_CLASSES)])
    iou = np.array([safe_ratio(inter[c], union[c]) for c in range(NUM_CLASSES)])
 
    per_slice_dice = np.array(per_slice_dice)  # [N_slices, NUM_CLASSES]
 
    # Per-slice mean over tissue classes, skipping undefined (NaN) entries
    slice_tissue_mean = np.nanmean(per_slice_dice[:, TISSUE_CLASSES], axis=1)
 
    return {
        "dice": dice,
        "iou": iou,
        "mean_dice_all": np.nanmean(dice),
        "mean_iou_all": np.nanmean(iou),
        "mean_dice_tissue": np.nanmean(dice[TISSUE_CLASSES]),
        "mean_iou_tissue": np.nanmean(iou[TISSUE_CLASSES]),
        "slice_dice_mean": np.nanmean(slice_tissue_mean),
        "slice_dice_std": np.nanstd(slice_tissue_mean),
        "num_slices": len(per_slice_dice),
    }
    
def sanity_check_iou_dice(dice, iou, tol=1e-6):
    """
    Identity: IoU = Dice / (2 - Dice). Derivation:
        D = 2I/(U+I) with U = P+T-I  =>  D(U+I) = 2I  =>  DU = (2-D)I  =>  I/U = D/(2-D)
    If this fails, the Dice or IoU formula has a bug.
    """
    for c in range(NUM_CLASSES):
        if np.isnan(dice[c]):
            continue
        expected = dice[c] / (2 - dice[c])
        assert abs(iou[c] - expected) < tol, (
            f"Dice/IoU mismatch for class {c}: IoU={iou[c]:.6f}, D/(2-D)={expected:.6f}"
        )
    print("Sanity check passed: IoU = Dice / (2 - Dice) for every class.")

def fmt(x):
    return "n/a" if np.isnan(x) else f"{x:.4f}"
 
def print_and_save_report(model, metrics, tag):
    d, u = metrics["dice"], metrics["iou"]
 
    header = f"{'Class':<15}{'Dice':>10}{'IoU':>10}"
    print("\n=== Test set results ({} slices, model = {}) ===".format(metrics["num_slices"], tag.upper()))
    print(header)
    print("-" * len(header))
    for c in range(NUM_CLASSES):
        print(f"{CLASS_NAMES[c]:<15}{fmt(d[c]):>10}{fmt(u[c]):>10}")
    print("-" * len(header))
    print(f"{'Mean (all 4)':<15}{fmt(metrics['mean_dice_all']):>10}{fmt(metrics['mean_iou_all']):>10}")
    print(f"{'Mean (tissue)':<15}{fmt(metrics['mean_dice_tissue']):>10}{fmt(metrics['mean_iou_tissue']):>10}")
    print(f"\nPer-slice tissue Dice: {metrics['slice_dice_mean']:.4f} +/- {metrics['slice_dice_std']:.4f}")
    print(f"Total parameter count: {determine_parameters(model=model)}")

def visualise_samples(model, loader, device, tag, num_samples=4):
    images, targets = next(iter(loader))
    num_samples = min(num_samples, images.size(0))
 
    with torch.no_grad():
        preds = torch.argmax(model(images[:num_samples].to(device)), dim=1).cpu().numpy()
 
    cmap = ListedColormap(CLASS_COLORS)
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    axes = np.atleast_2d(axes)
 
    for idx in range(num_samples):
        img = images[idx, 0].numpy()
        gt = targets[idx].numpy()
        pr = preds[idx]
 
        axes[idx, 0].imshow(img, cmap="gray")
        axes[idx, 0].set_title(f"Sample {idx + 1} (Slice Index: {idx})") 
        
        axes[idx, 1].imshow(img, cmap="gray")
        axes[idx, 1].imshow(np.ma.masked_where(gt == 0, gt), cmap=cmap, alpha=0.6, vmin=0, vmax=3)
        axes[idx, 1].set_title("Ground truth")
 
        axes[idx, 2].imshow(img, cmap="gray")
        axes[idx, 2].imshow(np.ma.masked_where(pr == 0, pr), cmap=cmap, alpha=0.6, vmin=0, vmax=3)
        axes[idx, 2].set_title("Prediction")
 
        for ax in axes[idx]:
            ax.axis("off")
 
    patches = [mpatches.Patch(color=CLASS_COLORS[i], label=CLASS_NAMES[i]) for i in range(1, NUM_CLASSES)]
    fig.legend(handles=patches, loc="lower center", ncol=3, frameon=True)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
 
    save_path = os.path.join(OUTPUT_DIR, f"oasis_{tag}_predictions.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"--> Sample figure saved to: {save_path}")

def determine_parameters(model):
    """Returns the total parameters of the loaded model."""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params
    
import time


def measure_inference_latency(model, sample_input, device, num_warmup=10, num_repeats=50):
    """Measures average inference latency per slice in milliseconds (ms)."""
    model.eval()
    single_slice_input = sample_input[:1].to(device)  # Take a batch of 1

    # Warmup steps to clear CUDA/backend kernel compilation overhead
    with torch.no_grad():
            for _ in range(num_warmup):
                _ = model(single_slice_input)

        # I haven't used Rangpur once for this project but this is here anyways. 
    if "cuda" in str(device):
        torch.cuda.synchronize(device)

    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_repeats):
            _ = model(single_slice_input)
            if "cuda" in str(device):
                torch.cuda.synchronize(device)

    total_time = time.time() - start_time
    latency_ms = (total_time / num_repeats) * 1000.0
    return latency_ms

if __name__ == "__main__":
    # Imported here so the module can be imported without DirectML installed
    from train import get_device
 
    device = get_device()
    print(f"Using inference device: {device}")
 
    # Test loader has no shuffle, so this file is deterministic, picks the same samples for visuals
    _, _, test_loader, _ = get_oasis_dataloaders(DATA_DIR, batch_size=BATCH_SIZE)
 
    model, weights_file, tag = build_model(MODEL_TYPE, device)
    model = load_weights(model, weights_file, device)
    
    sample_imgs, _ = next(iter(test_loader))
    latency = measure_inference_latency(model, sample_imgs, device)

    print(f"Inference Latency (per slice): {latency:.2f} ms")
 
    metrics = evaluate(model, test_loader, device)
    sanity_check_iou_dice(metrics["dice"], metrics["iou"])
    print_and_save_report(model, metrics, tag)
    visualise_samples(model, test_loader, device, tag)
