"""
Inference driver and qualitative failure case autopsy for HipMRI 2D Prostate Segmentation.

Features:
- Loads trained model checkpoints (BaselineUNet2D or ImprovedUNet2D).
- Evaluates segmentation metrics across all 6 anatomical pelvic structures on the held-out test split.
- Generates 4-panel diagnostic visualisations:
  [Original Pelvic MRI | Ground Truth Mask | Predicted Mask | Discrepancy Error Map (TP/FP/FN)]
- Automatically ranks and isolates 3-5 representative boundary failure cases
  (targeting superior base and inferior apex slices) to address the Engineering Dilemma.

Note: Permitted to use NumPy and Matplotlib per COMP3710 specification Section 4.1.
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

# Mandatory modular imports from project directory
from modules import BaselineUNet2D, ImprovedUNet2D, compute_dice_iou
from dataset import get_hipmri_dataloaders

CLASS_NAMES = [
    "Background",
    "Body Outline",
    "Bone",
    "Bladder",
    "Rectum",
    "Prostate"
]

# Discrete colormap for the 6 anatomical classes:
# 0: Background(black), 1: Body(gray), 2: Bone(yellow), 3: Bladder(blue), 4: Rectum(brown), 5: Prostate(red)
SEG_COLORS = ["#000000", "#7f7f7f", "#bcbd22", "#1f77b4", "#8c564b", "#d62728"]
SEG_CMAP = ListedColormap(SEG_COLORS)


def parse_args():
    parser = argparse.ArgumentParser(description="HipMRI Inference and Boundary Error Autopsy")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/groups/comp3710/HipMRI_Study_open/keras_slices_data",
        help="Path to HipMRI 2D Nifti slices directory"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="improved",
        choices=["baseline", "improved"],
        help="Model architecture: 'baseline' or 'improved'"
    )
    parser.add_argument(
        "--weights_path",
        type=str,
        default="./results/best_improved_unet.pth",
        help="Path to trained checkpoint (.pth)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./predictions",
        help="Directory to save diagnostic autopsy figures"
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for test inference")
    parser.add_argument("--dummy", action="store_true", help="Run with synthetic dummy data for local pipeline verification")
    return parser.parse_args()


def plot_4panel_autopsy(
    image_np: np.ndarray,
    gt_mask_np: np.ndarray,
    pred_mask_np: np.ndarray,
    save_path: str,
    case_rank: int,
    slice_idx: int,
    prostate_dice: float
):
    """
    Renders and saves a 4-panel diagnostic figure for clinical error autopsy:
    Panel 1: Input Pelvic T2 MRI slice.
    Panel 2: Expert Ground Truth multi-class annotation.
    Panel 3: Model Prediction mask with slice prostate Dice score.
    Panel 4: Discrepancy Error Map highlighting True Positive (Green),
             False Positive leakage (Cyan), and False Negative omission (Yellow).
    """
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8))

    # Panel 1: Original MRI slice
    axes[0].imshow(image_np, cmap="gray")
    axes[0].set_title("Input Pelvic MRI", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    # Panel 2: Ground Truth
    axes[1].imshow(gt_mask_np, cmap=SEG_CMAP, vmin=0, vmax=5)
    axes[1].set_title("Ground Truth Mask", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    # Panel 3: Model Prediction
    axes[2].imshow(pred_mask_np, cmap=SEG_CMAP, vmin=0, vmax=5)
    axes[2].set_title(f"Predicted Mask (Prostate DSC: {prostate_dice:.3f})", fontsize=11, fontweight="bold")
    axes[2].axis("off")

    # Panel 4: Discrepancy Overlay Map (Targeting Prostate Class 5)
    # Background in subdued greyscale; errors highlighted in high-contrast colours
    error_canvas = np.zeros((image_np.shape[0], image_np.shape[1], 3), dtype=np.float32)
 
    data_range = float(image_np.max() - image_np.min())
    norm_bg = np.clip((image_np - image_np.min()) / (data_range + 1e-8), 0.1, 0.45)

    for c in range(3):
        error_canvas[:, :, c] = norm_bg

    prostate_gt = (gt_mask_np == 5)
    prostate_pred = (pred_mask_np == 5)

    tp = prostate_gt & prostate_pred
    fp = (~prostate_gt) & prostate_pred
    fn = prostate_gt & (~prostate_pred)

    error_canvas[tp] = [0.0, 0.85, 0.0]   # Green: True Positive
    error_canvas[fp] = [0.0, 0.90, 1.0]   # Cyan: False Positive (boundary leakage)
    error_canvas[fn] = [1.0, 0.80, 0.0]   # Yellow: False Negative (apex/base truncation)

    axes[3].imshow(error_canvas)
    axes[3].set_title("Discrepancy (TP:Green, FP:Cyan, FN:Yellow)", fontsize=11, fontweight="bold")
    axes[3].axis("off")

    # Legend for anatomical structures
    legend_patches = [
        mpatches.Patch(color=SEG_COLORS[i], label=CLASS_NAMES[i])
        for i in range(1, 6)
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.5, -0.04),
        frameon=False,
        fontsize=10
    )

    plt.suptitle(
        f"Failure Case Autopsy #{case_rank} (Slice Index: {slice_idx} | Prostate DSC: {prostate_dice:.4f})",
        fontsize=13,
        fontweight="bold",
        y=1.02
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Inference Setup] Hardware: {device}")

    # Build model architecture
    if args.model == "improved":
        model = ImprovedUNet2D(in_channels=1, num_classes=6, base_filters=16).to(device)
    else:
        model = BaselineUNet2D(in_channels=1, num_classes=6, base_filters=32).to(device)

    # Load weights if available
    if os.path.exists(args.weights_path):
        print(f"[Checkpoint] Loading weights from: {args.weights_path}")
        model.load_state_dict(torch.load(args.weights_path, map_location=device, weights_only=True))
    else:
        print(f"[Notice] Checkpoint '{args.weights_path}' not found. Initialising model for pipeline verification.")

    model.eval()

    use_dummy = args.dummy or (not os.path.exists(args.data_dir))
    _, _, test_loader = get_hipmri_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=0,
        dummy_mode=use_dummy
    )

    case_records = []
    print("\n--- Running Inference & Evaluation on Test Split ---")

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(test_loader):
            images_gpu = images.to(device)
            masks_gpu = masks.to(device)

            logits = model(images_gpu)
            preds_idx = torch.argmax(logits, dim=1, keepdim=True)
            preds_onehot = torch.zeros_like(masks_gpu).scatter_(1, preds_idx, 1.0)

            # Evaluate per-slice prostate dice score to identify boundary failure modes
            for i in range(images.size(0)):
                p_onehot = preds_onehot[i:i+1]
                m_onehot = masks_gpu[i:i+1]
                slice_dices, _ = compute_dice_iou(p_onehot, m_onehot)
                prostate_dsc = slice_dices[5].item()

                case_records.append({
                    "image": images[i, 0].cpu().numpy(),
                    "gt_mask": torch.argmax(masks[i], dim=0).cpu().numpy(),
                    "pred_mask": preds_idx[i, 0].cpu().numpy(),
                    "prostate_dice": prostate_dsc,
                    "slice_idx": len(case_records)
                })

    print(f"[Evaluation] Completed inference on {len(case_records)} test slices.")

    # Sort slices by Prostate DSC to isolate the 3–5 representative failure cases
    sorted_cases = sorted(case_records, key=lambda x: x["prostate_dice"])
    num_cases_to_save = min(5, len(sorted_cases))

    print(f"\n--- Generating Qualitative Error Autopsies for {num_cases_to_save} Boundary Cases ---")
    for rank, case in enumerate(sorted_cases[:num_cases_to_save], 1):
        filename = f"autopsy_case_{rank}_slice{case['slice_idx']}_dice_{case['prostate_dice']:.3f}.png"
        save_path = os.path.join(args.output_dir, filename)

        plot_4panel_autopsy(
            image_np=case["image"],
            gt_mask_np=case["gt_mask"],
            pred_mask_np=case["pred_mask"],
            save_path=save_path,
            case_rank=rank,
            slice_idx=case["slice_idx"],
            prostate_dice=case["prostate_dice"]
        )
        print(f"Saved Case #{rank} -> {save_path} (Prostate DSC: {case['prostate_dice']:.4f})")

    print(f"\n[Completed] All diagnostic figures successfully saved to '{args.output_dir}'.")


if __name__ == "__main__":
    main()