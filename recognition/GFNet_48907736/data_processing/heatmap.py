"""
This file creates a Grad-CAM and gradient x input visual over any input image using the specified model.
Note that the model is not in the repo due to the rules.

python heatmap.py --weights GFNet_model_outputs/gfnet.pt <image location>.jpeg
"""
import argparse

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from modules import load_model

parser = argparse.ArgumentParser()
parser.add_argument("image")
parser.add_argument("--weights", default="gfnet.pt")
parser.add_argument("--out", default="GFNet_heatmap.png")
args = parser.parse_args()



device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model(args.weights, device)
transform = transforms.Compose([transforms.Grayscale(1), transforms.CenterCrop(200), transforms.Resize((224, 224)),
                                transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
image = transform(Image.open(args.image))[None].to(device).requires_grad_(True)  # (1, 1, 224, 224)

# Save the output of the last GFBlock (1, 14, 14, 256) during the forward pass
saved = {}
model.blocks.register_forward_hook(lambda module, inp, out: saved.update(tokens=out))

# One forward + backward pass: score = how much the model prefers its predicted class
logits = model(image)[0]
pred = logits.argmax().item()
saved["tokens"].retain_grad()
(logits[pred] - logits[1 - pred]).backward()

# Grad-CAM: weight each channel by its mean gradient, sum over channels, keep the positive part
tokens, grads = saved["tokens"][0], saved["tokens"].grad[0]
cam = F.relu((tokens * grads.mean(dim=(0, 1))).sum(dim=-1))                   # (14, 14)
cam = F.interpolate(cam[None, None], size=(224, 224), mode="bilinear")[0, 0]  # (224, 224)
cam = cam / cam.max()

# Gradient x input, with pixels shifted from [-1, 1] to [0, 2] so the black background is 0
gxi = (image.grad * (image + 1)).abs()[0, 0]
gxi = (gxi / torch.quantile(gxi, 0.99)).clamp(0, 1)  # clip the brightest 1% so the rest is visible

slice_ = image[0, 0].detach().cpu() * 0.5 + 0.5      # back to [0, 1] for display
fig, axes = plt.subplots(1, 3, figsize=(9, 3.8), layout="constrained")
axes[0].imshow(slice_, cmap="gray")
axes[0].set_title("Input slice")
for ax, heat, title in [(axes[1], cam, "Grad-CAM"), (axes[2], gxi, "Gradient x input")]:
    ax.imshow(slice_, cmap="gray")
    ax.imshow(heat.detach().cpu(), cmap="jet", alpha=0.45, vmin=0, vmax=1)
    ax.set_title(title)
for ax in axes:
    ax.axis("off")
fig.savefig(args.out, dpi=150, bbox_inches="tight", pad_inches=0.3)
print(f"Saved {args.out}")
