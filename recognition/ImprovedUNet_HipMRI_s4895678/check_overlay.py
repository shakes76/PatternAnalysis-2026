"""Save a picture of one slice with the prostate label outlined (sanity check)."""
import matplotlib
matplotlib.use("Agg")          # draw to a file, no screen needed
import matplotlib.pyplot as plt
import os
from dataset import load_case, PROSTATE

# Sanity check of the data pipeline: if the red outline sits on the prostate, image and label are aligned.
case = "M015_Week0"
img, lab = load_case(case)
z = int((lab == PROSTATE).sum(axis=(1, 2)).argmax())   # slice with most prostate
os.makedirs("images", exist_ok=True)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(img[z], cmap="gray"); ax[0].set_title(f"{case}, slice {z}")
ax[1].imshow(img[z], cmap="gray")
ax[1].contour(lab[z] == PROSTATE, levels=[0.5], colors="red")
ax[1].contour(lab[z] == 4, levels=[0.5], colors="cyan")
ax[1].set_title("red = label 5, cyan = label 4")
for a in ax: a.axis("off")
plt.savefig("images/overlay_check.png", dpi=100, bbox_inches="tight")
print("saved images/overlay_check.png, slice", z)
