"""
Loads best.pt, runs on a few test samples, saves overlays.

Usage:
    python predict.py --model unet2d --weights ./runs/unet2d_baseline/best.pt --out-dir ./predictions_baseline
    python predict.py --model improved_unet2d --weights ./runs/improved_unet2d/best.pt --out-dir ./predictions_improved
"""
import argparse
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import build_dataloaders, NUM_CLASSES
from modules import UNet2D, ImprovedUNet2D

CLASS_COLORS = np.array([
    [0.0, 0.0, 0.0], # 0 background, black
    [0.5, 0.5, 0.5], # 1 body, grey
    [1.0, 1.0, 1.0], # 2 bone, white
    [1.0, 0.0, 0.0], # 3 bladder, red
    [0.0, 1.0, 0.0], # 4 rectum, green
    [0.0, 0.0, 1.0], # 5 prostate, blue
])

def mask_to_rgb(mask):
    return CLASS_COLORS[mask]

@torch.no_grad()
def predict_batch(model, imgs, device):
    model.eval()
    imgs = imgs.to(device)
    logits = model(imgs)

    # logits shape: (B, C, H, W) -> argmax over channels -> (B, H, W)
    preds = logits.argmax(dim=1).cpu().numpy()
    return preds

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # load model
    if args.model == "unet2d":
        model = UNet2D(in_channels=1, out_channels=NUM_CLASSES).to(device)
    elif args.model == "improved_unet2d":
        model = ImprovedUNet2D(in_channels=1, out_channels=NUM_CLASSES).to(device)
    else:
        raise ValueError(f"unknown model: {args.model}")
    
    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Could not find weights at {args.weights}")
    model.load_state_dict(torch.load(args.weights, map_location=device))
    print(f"Loaded weights from {args.weights}")

    # load data 
    loaders = build_dataloaders(batch_size=args.batch_size, num_workers=2)
    test_loader = loaders["test"]

    os.makedirs(args.out_dir, exist_ok=True)

    for i, batch in enumerate(test_loader):
        if i == args.offset:
            imgs, lbls = batch
            break

    preds = predict_batch(model, imgs, device)
    
    # ground truth is one-hot, so take argmax to get class index
    gts = lbls.argmax(dim=1).numpy()
    imgs_np = imgs[:, 0].numpy() # remove channel dim

    # plot 4 samples
    num_samples = min(args.batch_size, len(imgs_np))
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))

    if num_samples == 1:
        axes = axes.reshape(1, 3)
    
    for i in range(num_samples):
        img = imgs_np[i]
        gt = gts[i]
        pred = preds[i]

        # input image
        axes[i, 0].imshow(img, cmap="gray")
        axes[i, 0].set_title(f"Input Image {i}")
        axes[i, 0].axis("off")

        # ground truth
        axes[i, 1].imshow(mask_to_rgb(gt))
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis("off")

        # prediction
        axes[i, 2].imshow(mask_to_rgb(pred))
        axes[i, 2].set_title("Prediction")
        axes[i, 2].axis("off")

    plt.tight_layout()
    out_path = os.path.join(args.out_dir, "test_predictions.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True, choices=["unet2d", "improved_unet2d"])
    p.add_argument("--weights", type=str, required=True, help="Path to best.pt")
    p.add_argument("--out-dir", type=str, default="./predictions")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--offset", type=int, default=0)
    args = p.parse_args()
    main(args)