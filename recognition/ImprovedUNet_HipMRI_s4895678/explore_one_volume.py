"""Quick look at one HipMRI 3D volume and its label map (data audit, step 1)."""
import nibabel as nib
import numpy as np

DATA_DIR = "/home/groups/comp3710/HipMRI_Study_open"
case = "B040_Week0"

# nib.load only reads the header; the voxels are read later by get_fdata()
img = nib.load(f"{DATA_DIR}/semantic_MRs/{case}_LFOV.nii.gz")
lab = nib.load(f"{DATA_DIR}/semantic_labels_only/{case}_SEMANTIC.nii.gz")

print("image shape:", img.shape, "| label shape:", lab.shape)
print("voxel size (mm):", img.header.get_zooms())

x = img.get_fdata()   # MRI intensities as a NumPy array
y = lab.get_fdata()   # tissue label for every voxel
print("image min/max:", x.min(), x.max())
print("memory as float32 (MB):", x.astype(np.float32).nbytes / 1e6)

# Count how many voxels carry each label value
values, counts = np.unique(y, return_counts=True)
for v, c in zip(values, counts):
    print(f"label {int(v)}: {c} voxels ({100 * c / y.size:.2f}%)")
