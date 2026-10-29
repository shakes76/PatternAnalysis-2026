from pathlib import Path 
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch 

from dataset import get_dataloaders, NUM_CLASSES, TARGET_PROSTATE_LABEL
from modules import Baseline_UNet, Improved_UNet

"""
Functions:
1. Loads trained weights from the checkpoint for the selected model.
2. computes oficial Test Dice Similarity Coefficients (DSC) across all 6 classes
3. Generates qualitative 3-panel comparison figures (MRI Slice | Ground Truth | prediction) 
   and saves then to figures.
"""
# Select model to evaluate: "baseline" or "improved" 
MODEL_NAME = "improved"

# Configurations
PROJECT_DIR = Path(__file__).resolve().parent
FIGURES_DIR = PROJECT_DIR / "figures"

if MODEL_NAME == "improved":
    CHECKPOINT_PATH = PROJECT_DIR / "checkpoints" / "improved_unet_best.pth"
    SAMPLE_PREFIX = "improved_prediction_sample"
    PLOT_TITLE = "Predicted Segmentation (Improved U-Net)"
else:
    CHECKPOINT_PATH = PROJECT_DIR / "checkpoints" / "baseline_unet_best.pth"
    SAMPLE_PREFIX = "baseline_prediction_sample"
    PLOT_TITLE = "Predicted Segmentation (baseline U-Net)"

BATCH_SIZE = 16 
NUM_WORKERS = 1 
DICE_SMOOTH = 1e-5

# Organ names corresponding to labels 0 through 5
ORGAN_NAMES = {
    0: "Background",
    1: "Body Outline",
    2: "Bony Pelvis",
    3: "Prostate (Target)",
    4: "Bladder",
    5: "Rectum"
}

# Qualitative visualization color palette 
CUSTOM_COLORS = [ 
    "#000000", # 0: Background (Black)
    "#3498db", # 1: Body Outline (Blue) 
    "#e67e22", # 2: Bone (Orange) 
    "#e74c3c", # 3: Prostate (Bright Red) 
    "#2ecc71", # 4: Bladder (Green) 
    "#9b59b6" # 5: Rectum (Purple) 
]

SEG_CMAP = ListedColormap(CUSTOM_COLORS)

# Evaluation Metric Functions
@torch.no_grad()
def evaluate_test_set(model, test_loader, device):
    """
        Evaluates model across unseen test cohort. 
        Accumulates intersections and cardinalities to produce global Dice scores.
    """
    model.eval()

    class_intersections = np.zeros(NUM_CLASSES)
    class_cardinalities = np.zeros(NUM_CLASSES)

    for images, masks in test_loader:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        preds = torch.argmax(logits, dim=1)

        for c in range(NUM_CLASSES): 
            pred_c = (preds == c) 
            true_c = (masks == c)

            class_intersections[c] += (pred_c & true_c).sum().item()
            class_cardinalities[c] += pred_c.sum().item() + true_c.sum().item()

    # Calculate Dice per class 
    test_dice_per_class = {} 
    for c in range(NUM_CLASSES): 
        inter = class_intersections[c] 
        card = class_cardinalities[c] 
        if card == 0: 
            test_dice_per_class[c] = 1.0 if inter == 0 else 0.0 
        else: 
            test_dice_per_class[c] = (2.0 * inter + DICE_SMOOTH) / (card + DICE_SMOOTH)

    return test_dice_per_class

# Qualitative visualisation generator
@torch.no_grad()
def save_qualitative_predictions(model, test_loader, device, num_samples=5, out_dir=FIGURES_DIR):
    """
        Generates and saves qualitative 3-panel figures:
        [Input MRI Slice | Ground Truth Mask | Predicted Segmentation]
        Priorities slices containing the target prostate label (Class 3).
    """
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_count = 0

    print(f"\nGenerating {num_samples} qualitative visual overlays...")

    for images, masks in test_loader:
        images_dev = images.to(device) 
        logits = model(images_dev) 
        preds = torch.argmax(logits, dim=1).cpu()

        for idx in range(images.size(0)): 
            gt_mask = masks[idx].numpy() 
            pred_mask = preds[idx].numpy() 
            raw_slice = images[idx, 0].numpy()

            # Prioritize saving slices where the prostate is present
            has_prostate = (gt_mask == TARGET_PROSTATE_LABEL).any()
            if not has_prostate and saved_count >= 2:
                continue
            fig, axes = plt.subplots(1, 3, figsize=(15, 6))

            # Panel 1: Greyscale MRI Slice
            axes[0].imshow(raw_slice, cmap="gray", aspect="auto")
            axes[0].set_title("Input Pelvic MRI Slice", fontsize=12, fontweight="bold") 
            axes[0].axis("off")

            # Panel 2: Ground Truth Segmentation 
            axes[1].imshow(gt_mask, cmap=SEG_CMAP, vmin=0, vmax=NUM_CLASSES - 1, aspect="auto") 
            axes[1].set_title("Ground Truth Segmentation", fontsize=12, fontweight="bold") 
            axes[1].axis("off")

            # Panel 3: Model Prediction 
            axes[2].imshow(pred_mask, cmap=SEG_CMAP, vmin=0, vmax=NUM_CLASSES - 1, aspect="auto") 
            axes[2].set_title(PLOT_TITLE, fontsize=12, fontweight="bold") 
            axes[2].axis("off")

            plt.tight_layout() 
            output_file = out_dir / f"{SAMPLE_PREFIX}_{saved_count + 1}.png" 
            plt.savefig(output_file, dpi=200, bbox_inches="tight") 
            plt.close(fig)

            print(f" --> Saved: {output_file.name}") 
            saved_count += 1 
            if saved_count >= num_samples: 
                break

        if saved_count >= num_samples:
            break

    print(f"[Info] Qualitative prediction figures preserved in: {out_dir}")

def main():
    print("=" * 65)
    print(f"EVALUATE & INFERENCE: {MODEL_NAME.upper()} 2D U-NET (HipMRI)")
    print("=" * 65)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Execution device: {device}")

    # Check if trained weights exist 
    if not CHECKPOINT_PATH.exists(): 
        print(f"\n[Error] Checkpoint not found at: {CHECKPOINT_PATH}") 
        print("Please train the model first by running `python train.py` or submitting `run_train.sh`.") 
        return

    # Load DataLoaders 
    print("\nLoading unseen test split...") 
    _, _, test_loader = get_dataloaders( 
        batch_size=BATCH_SIZE, 
        num_workers=NUM_WORKERS 
    )

    # Initialize Architecture 
    if MODEL_NAME == "improved":
        model = Improved_UNet(
            in_channels=1, 
            num_classes=NUM_CLASSES,
            base_filters=16
        ).to(device)
    else:
        model = Baseline_UNet(
            in_channels=1, 
            num_classes=NUM_CLASSES,
            base_filters=16
        ).to(device)

    # Load checkpoint 
    print(f"Loading checkpoint weights from: {CHECKPOINT_PATH}") 
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device) 
    model.load_state_dict(checkpoint["model_state_dict"]) 
    saved_epoch = checkpoint.get("epoch", "N/A") 
    val_prostate_dice = checkpoint.get("best_prostate_dice", "N/A") 
    print(f"Checkpoint restored (Epoch: {saved_epoch}, Best Val Prostate DSC: {val_prostate_dice})")

    # Run Quantitative Evaluation 
    print("\n" + "=" * 65) 
    print("RUNNING QUANTITATIVE TEST SET EVALUATION...") 
    print("=" * 65)

    test_dice_scores = evaluate_test_set(model, test_loader, device)

    print(f"{'Class ID':<10} | {'Organ / Anatomical Structure':<22} | {'Test Dice (DSC)':<15}") 
    print("-" * 65) 
    for c in range(NUM_CLASSES): 
        organ_name = ORGAN_NAMES.get(c, f"Class {c}") 
        score = test_dice_scores[c] 
        marker = " <-- TARGET" if c == TARGET_PROSTATE_LABEL else "" 
        print(f"{c:<10} | {organ_name:<22} | {score:.4f}{marker}")

    # Overall Mean DSC
    all_classes_mean = np.mean(list(test_dice_scores.values())) 
    prostate_score = test_dice_scores[TARGET_PROSTATE_LABEL]

    print("-" * 65) 
    print(f"{'Overall':<10} | {'Mean Across All Classes':<22} | {all_classes_mean:.4f}") 
    print("=" * 65)

    print(f"\nTarget Prostate Benchmark Status:") 
    if prostate_score >= 0.75: 
        print(f" [PASSED] Prostate DSC ({prostate_score:.4f}) meets or exceeds the 0.75 threshold!") 
    else: 
        print(f" [BASELINE REFERENCE] Prostate DSC: {prostate_score:.4f} (Goal threshold: >= 0.75)")


    # Generate Qualitative Figures for README
    save_qualitative_predictions(
        model, test_loader, device, num_samples=5, out_dir=FIGURES_DIR
    )

    print("\n" + "=" * 65) 
    print("EVALUATION & PREDICTION PIPELINE COMPLETE") 
    print("=" * 65)

if __name__ == "__main__":
    main()
