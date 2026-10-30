import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path

root = Path("/home/groups/comp3710/HipMRI_Study_open")

mri = np.asarray(nib.load(
    root / "semantic_MRs/B006_Week0_LFOV.nii.gz"
).dataobj)

mask = np.asarray(nib.load(
    root / "semantic_labels_only/B006_Week0_SEMANTIC.nii.gz"
).dataobj)

print("MRI shape:", mri.shape)
print("Mask shape:", mask.shape)

if mri.shape != mask.shape:
    raise ValueError("MRI and mask shapes do not match")

colors = ["black", "red", "green", "blue", "orange", "purple"]
cmap = ListedColormap(colors)

# Select slices containing the most small-structure labels.
scores = [
    np.count_nonzero(mask[:, :, z] >= 5)
    for z in range(mask.shape[2])
]
selected = np.argsort(scores)[-3:]

fig, axes = plt.subplots(2, 3, figsize=(12, 8))

for col, z in enumerate(sorted(selected)):
    axes[0, col].imshow(mri[:, :, z].T, cmap="gray", origin="lower")
    axes[0, col].set_title(f"MRI slice {z}")

    axes[1, col].imshow(mri[:, :, z].T, cmap="gray", origin="lower")

    overlay = np.ma.masked_where(
        mask[:, :, z].T == 0,
        mask[:, :, z].T
    )

    axes[1, col].imshow(
        overlay,
        cmap=cmap,
        vmin=0,
        vmax=5,
        alpha=0.45,
        origin="lower"
    )
    axes[1, col].set_title(f"Segmentation overlay {z}")

for ax in axes.flat:
    ax.axis("off")

fig.tight_layout()
fig.savefig("evaluation/label_inspectio5n.png", dpi=160)
plt.close(fig)

print("Saved: evaluation/label_inspection.png")
print("Selected slices:", sorted(selected))
