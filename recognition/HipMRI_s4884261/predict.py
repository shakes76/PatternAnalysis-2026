# Showing example usage of your trained model. It should load your saved model weights, run inference
# on test cases, print out any results, and provide visualisations where applicable (e.g., prediction overlays, generated
# samples, heatmaps). NumPy is allowed in this script for visualisation or loading data.

import os
import time
import torch
import matplotlib.pyplot as plt

from dataset import get_pairs, split_pairs, get_loaders
from modules import Improved3DUNet
from UNet_2D import UNet2D
from train import num_classes, class_names, base_filters, dice_score

data_root = "/home/twig/Desktop/Uni/Year 3 - Sem 2/COMP3710/Proj/data"
is_3d = True # False tests the 2D UNet instead
out_dir = "3D_Model" if is_3d else "2D_Unet_output"
model_name = "unet3d" if is_3d else "unet_2d"
model_path = f"{out_dir}/{model_name}_hipmri.pt"
PROSTATE = 5
end_slices = 5 # how many slices at each end of the prostate count as the apex / base
num_examples = 3 # how many test volumes to save a picture of

def dice_and_iou(pred, target):
    # overlap of two boolean masks, a region missing from both counts as perfect
    intersection = (pred & target).sum().item()
    union = (pred | target).sum().item()
    total = pred.sum().item() + target.sum().item()
    dice = 2 * intersection / total if total > 0 else 1.0
    iou = intersection / union if union > 0 else 1.0
    return dice, iou

def prostate_regions(preds, labels):
    # dice and iou for the whole prostate, and for its first / last few slices
    # slices run along the last axis, the lowest slices are the apex (inferior) and the highest are the base (next to the bladder)
    pred = preds == PROSTATE # (H, W, D)
    target = labels == PROSTATE
    prostate_slices = target.any(dim=0).any(dim=0).nonzero().flatten() # slices where the ground truth has prostate
    apex = prostate_slices[:end_slices]
    base = prostate_slices[-end_slices:]
    return {
        "whole": dice_and_iou(pred, target),
        "apex": dice_and_iou(pred[..., apex], target[..., apex]),
        "base": dice_and_iou(pred[..., base], target[..., base]),
    }

def save_example(image, label, pred, save_prefix):
    # one image per prostate region (apex / middle / base), rows are consecutive slices, columns are input MRI | ground truth | prediction
    # apex and base also show one slice past the end of the prostate, to check the model stops predicting prostate there
    prostate_slices = (label == PROSTATE).any(dim=0).any(dim=0).nonzero().flatten()
    first, last = prostate_slices[0].item(), prostate_slices[-1].item()
    mid = prostate_slices[len(prostate_slices) // 2].item()
    regions = {
        "Apex": range(first - 1, first + 3), # 1 slice before the prostate starts + first 3 prostate slices
        "Middle": range(mid - 1, mid + 2),
        "Base": range(last - 2, last + 2), # last 3 prostate slices + 1 slice after it ends
    }

    for region, slices in regions.items():
        slices = [k for k in slices if 0 <= k < label.shape[-1]] # stay inside the volume
        fig, axes = plt.subplots(len(slices), 3, figsize=(9, 3 * len(slices)))
        for row, k in enumerate(slices):
            # prostate dice on just this slice, shows where along the prostate the prediction breaks down
            dice = dice_and_iou(pred[..., k] == PROSTATE, label[..., k] == PROSTATE)[0]
            axes[row, 0].imshow(image[..., k], cmap="gray")
            axes[row, 1].imshow(label[..., k], cmap="viridis", vmin=0, vmax=num_classes - 1) # fixed range so each class keeps the same colour
            axes[row, 2].imshow(pred[..., k], cmap="viridis", vmin=0, vmax=num_classes - 1)
            axes[row, 0].set_title(f"{region} input (slice {k})")
            axes[row, 1].set_title("ground truth" if first <= k <= last else "ground truth (no prostate)")
            axes[row, 2].set_title(f"prediction (prostate DSC {dice:.2f})")
            for ax in axes[row]:
                ax.axis("off")

        save_path = f"{save_prefix}_{region.lower()}.png"
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"saved {save_path}")

def main(root=data_root):
    # test split only, same seed as training so these patients were never seen
    splits = split_pairs(get_pairs(root))
    test_loader = get_loaders({"test": splits["test"]})["test"] # batch size 1, one volume at a time
    print(f"Test volumes: {len(splits['test'])}")

    # both models take whole volumes and give (1, C, H, W, D), so everything below is the same for both
    if is_3d:
        model = Improved3DUNet(in_channels=1, num_classes=num_classes, base_filters=base_filters)
    else:
        model = UNet2D(1, num_classes, base_filters)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    # dice and iou per class for each volume, then averaged over volumes
    total_dice = torch.zeros(num_classes)
    total_iou = torch.zeros(num_classes)
    region_scores = {"whole": [], "apex": [], "base": []} # (dice, iou) per volume for each prostate region
    times = [] # forward pass seconds per volume
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            # synchronize so the timer waits for the gpu to actually finish
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            logits = model(images.to(device))
            if device == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
            logits = logits.cpu() # back to cpu so the scoring below is unchanged
            total_dice += dice_score(logits, labels.long(), num_classes)

            preds = logits.argmax(dim=1)[0] # (H, W, D), batch size is 1
            total_iou += torch.tensor([dice_and_iou(preds == c, labels[0] == c)[1] for c in range(num_classes)])
            for region, score in prostate_regions(preds, labels[0]).items():
                region_scores[region].append(score)

            # pictures of the first few test volumes, [0, 0] drops the batch and channel dims
            # named by case (e.g. unet3d_Case_004_Week0_LFOV_apex.png), the test loader isn't shuffled so i matches splits["test"]
            if i < num_examples:
                case = os.path.basename(splits["test"][i][0]).replace(".nii.gz", "")
                save_example(images[0, 0], labels[0], preds, f"{out_dir}/predictions/{model_name}_{case}")
    test_dice = total_dice / len(test_loader)
    test_iou = total_iou / len(test_loader)

    for name, d, j in zip(class_names, test_dice, test_iou):
        print(f"Test {name}: DSC {d:.4f} | IoU {j:.4f}")
    print(f"Min class DSC: {test_dice.min():.4f} | Mean DSC: {test_dice.mean():.4f} | Mean IoU: {test_iou.mean():.4f}")

    # mean over volumes, so the apex and base can be compared against the whole prostate
    print(f"\nProstate by region ({end_slices} slices at each end):")
    for region, scores in region_scores.items():
        dice, iou = torch.tensor(scores).mean(dim=0)
        print(f"{region:>5}: DSC {dice:.4f} | IoU {iou:.4f}")

    # first case skipped, it includes one-off warm-up cost
    case_times = torch.tensor(times[1:])
    print(f"\nInference time per case ({device}): {case_times.mean():.3f} ± {case_times.std():.3f} s")
    if device == "cuda":
        print(f"Peak GPU VRAM: {torch.cuda.max_memory_allocated() / 1024**2:.0f} MB")

if __name__ == "__main__":
    main()
