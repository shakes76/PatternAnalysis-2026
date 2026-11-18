import numpy as np
from dataset import load_data_3D, get_pairs, split_pairs
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch



root = "/home/twig/Desktop/Uni/Year 3 - Sem 2/COMP3710/Proj/data/"

# dataset stats
pairs = get_pairs(root)
patients = {mr.split("/")[-1].split("_")[1] for mr, _ in pairs}
print("Scans:", len(pairs), "Patients:", len(patients))
print("MR shapes:", {nib.load(mr).shape for mr, _ in pairs}) # >1 shape means load_data_3d clips or pads
for name, split in split_pairs(pairs).items():
    print(name, len(split), "scans")

image = load_data_3D([root + "semantic_MRs_anon/Case_004_Week0_LFOV.nii.gz"])
label = load_data_3D([root + "semantic_labels_anon/Case_004_Week0_SEMANTIC_LFOV.nii.gz"], dtype=np.uint8)
print("img:", image.shape, image.dtype, "min-max:", image.min(), image.max())
print("label:", label.shape, label.dtype, "types, counts:", np.unique(label, return_counts=True))

names = ["Background", "Body", "Bone", "Bladder", "Rectum", "Prostate"]
lab = label[0] #(H,W,D)

# which slices contain each class
for c, n in enumerate(names):
    print(f"{n}: slices {np.nonzero((lab == c).any(axis=(0,1)))[0]}")

full = [k for k in range(lab.shape[2]) if len(np.unique(lab[:, :, k])) == len(names)]
print("slices with every class:", full)
k = (lab == 5).sum(axis=(0, 1)).argmax() # most prostate
print("plotting slice", k)


plt.figure(figsize=(16,8))
cmap = plt.get_cmap("tab10", len(names))
plt.subplot(2, 4, 1); plt.imshow(image[0, :, :, k], cmap="gray"); plt.title(f"MR slice {k}")
plt.subplot(2, 4, 2); plt.imshow(lab[:, :, k], cmap=cmap, vmin=0, vmax=len(names) - 1, interpolation="nearest"); plt.title("Labels")
plt.figlegend(handles=[Patch(color=cmap(i), label=n) for i, n in enumerate(names)], loc="lower center", ncol=len(names))
# one panel per class: that class's mask over the MR, so you can see which organ it is
for c, n in enumerate(names):
    plt.subplot(2, 4, c + 3); plt.imshow(image[0, :, :, k], cmap="gray")
    plt.imshow(np.ma.masked_where(lab[:, :, k] != c, lab[:, :, k]), cmap=cmap, vmin=0, vmax=len(names) - 1, alpha=0.6, interpolation="nearest")
    plt.title(n)
plt.tight_layout(rect=(0, 0.05, 1, 1))
plt.savefig("data_audit.png", bbox_inches="tight")
plt.show()