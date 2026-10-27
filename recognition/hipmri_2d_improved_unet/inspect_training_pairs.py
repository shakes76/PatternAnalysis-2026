"""Inspect train/validation filenames and case groups; run from the repo root."""

from pathlib import Path

from dataset import collect_pairs, patient_ids

data_root = Path(
    "../../Final Report/HipMRI_2D_Improved_UNet_学习/data/keras_slices_data"
)
image_dir = data_root / "keras_slices_train"
mask_dir = data_root / "keras_slices_seg_train"
training_pairs = collect_pairs(image_dir, mask_dir)
reference_pair = (
    image_dir / "case_004_week_0_slice_12.nii.gz",
    mask_dir / "seg_004_week_0_slice_12.nii.gz",
)

print("Paired training filenames:", len(training_pairs))
print("Reference pair found:", reference_pair in training_pairs)

validation_pairs = collect_pairs(
    data_root / "keras_slices_validate", data_root / "keras_slices_seg_validate"
)
train_ids = patient_ids(training_pairs)
validation_ids = patient_ids(validation_pairs)
overlap = train_ids & validation_ids

print("Train case groups:", len(train_ids))
print("Validation filename pairs:", len(validation_pairs))
print("Validation case groups:", len(validation_ids))
print("Shared case IDs:", sorted(overlap))
if overlap:
    raise ValueError(f"Train/validation case IDs overlap: {sorted(overlap)}")
