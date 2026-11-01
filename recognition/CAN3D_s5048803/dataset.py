"""
dataset.py: Containing the data loader for loading and preprocessing your data, including leakagefree train/validation/test splitting functions and data augmentations.
"""

import os, glob, re, random
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

NUM_CLASSES = 6
CLASS_NAMES = ["background", "body", "skeleton", "bladder", "rectum", "prostate"]
FG_NAMES = CLASS_NAMES[1:]
RANGPUR_MODE = os.environ.get("RANGPUR_MODE", "0") == "1"
DEBUG = False

# Folder names inside --data_root for each machine
LAYOUTS = {
	"local":   dict(d2="HipMRI_2D", mr="HipMRI_3D/semantic_MRs_anon",
					lbl="HipMRI_3D/semantic_labels_anon"),
	"rangpur": dict(d2="keras_slices_data", mr="semantic_MRs",
					lbl="semantic_labels_only"),
}
SLICE_DIRS = {"train": "train", "val": "validate", "test": "test"}

def layout():
	return LAYOUTS["rangpur" if RANGPUR_MODE else "local"]


def slice_key(path):
	"""'case_004_week_0_slice_10.nii.gz' or 'seg_004_week_0_slice_10.nii.gz' -> ('004', '0', '10')."""
	return re.search(r"(\d+)_week_(\d+)_slice_(\d+)", os.path.basename(path), re.I).groups()

def get_slices(root, split):
	"""(image, label) slice pairs for split in {'train', 'val', 'test'}."""
	d2 = os.path.join(root, layout()["d2"])
	name = SLICE_DIRS[split]
	images = sorted(glob.glob(os.path.join(d2, f"keras_slices_{name}", "*.nii.gz")))  # skips stray .nii
	segs = {slice_key(p): p for p in glob.glob(os.path.join(d2, f"keras_slices_seg_{name}", "*.nii.gz"))}
	pairs = [(i, segs[slice_key(i)]) for i in images if slice_key(i) in segs]
	assert pairs, f"No 2D slices found for split '{split}' under {d2}"
	if DEBUG:
		print(split, end="")
		print(set(([slice_key(i)[0] for i in images])))
	return pairs

class HipMRI2D(Dataset):
	"""One slice per item: image (1, H, W) float, label (H, W) long.
	Every slice is decoded, padded and scaled once here, so epochs only index into memory."""
	def __init__(self, pairs, scale):
		images, labels = [], []
		for image_path, label_path in pairs:
			image = nib.load(image_path).get_fdata(caching="unchanged")
			label = nib.load(label_path).get_fdata(caching="unchanged")
			if image.ndim == 3: image = image[..., 0] # some files carry an extra axis
			if label.ndim == 3: label = label[..., 0]
			image = (image - image.mean()) / (image.std() + 1e-8) # Normalization
			image = torch.from_numpy(image.astype(np.float32))[None, None]
			label = torch.from_numpy(np.rint(label).astype(np.float32))[None, None]

			# slices come in different widths (e.g. 128 and 144): pad to a common size so batches can stack
			pad_h = max(0, 256 - image.shape[-2])
			pad_w = max(0, 144 - image.shape[-1])
			image = F.pad(image, (0, pad_w, 0, pad_h))   # pads with 0 (the mean after z-scoring)
			label = F.pad(label, (0, pad_w, 0, pad_h))   # pads with 0 = background

			if scale != 1:
				image = F.interpolate(image, scale_factor=scale, mode="bilinear", align_corners=False)
				label = F.interpolate(label, scale_factor=scale, mode="nearest")
			images.append(image[0])                          # (1, H, W)
			labels.append(label[0, 0].to(torch.uint8))       # (H, W), small in memory
		self.images, self.labels = torch.stack(images), torch.stack(labels)

	def __len__(self):
		return len(self.images)

	def __getitem__(self, i):
		return self.images[i], self.labels[i].long()

def key(path):
	"""
	3D loading
	The key is what differentiates different patients.
	'Case_004_Week0_LFOV.nii.gz' or 'M004_Week1_SEMANTIC.nii.gz' -> ('004', '0').
	"""
	return re.search(r"(?:Case_|[A-Z])(\d+)_Week(\d+)", os.path.basename(path), re.I).groups()

def patient_id(pair):
	return key(pair[0])[0]

def get_pairs(root):
	"""Match every MRI with its label by (patient, week): [(img_path, lbl_path), ...]."""
	L = layout()
	images = sorted(glob.glob(os.path.join(root, L["mr"], "*.nii.gz")))
	labels = {key(p): p for p in glob.glob(os.path.join(root, L["lbl"], "*.nii.gz"))}
	pairs = [(i, labels[key(i)]) for i in images if key(i) in labels]
	assert pairs, f"No image/label pairs under {os.path.join(root, L['mr'])}"
	return pairs

def split_pairs(root, pairs, seed=42, frac=(0.7, 0.15, 0.15)):
	"""Patient-level split. Uses the patients in the course's 2D train/validate/test
	folders so CAN2D and CAN3D share the same test patients. frac is fallback if the pre-split cant be found"""
	d2 = os.path.join(root, layout()["d2"])
	ids = {}
	for split, name in SLICE_DIRS.items():
		files = glob.glob(os.path.join(d2, f"keras_slices_{name}", "*.nii.gz"))
		ids[split] = {re.search(r"(\d+)", os.path.basename(f)).group(1) for f in files}
	if DEBUG:
		print(ids) # Print to verify 3D split

	if not all(ids.values()): # 2D folders missing -> random patient split
		print("WARNING: 2D split not found, using random patient split")
		patient_ids = sorted({patient_id(p) for p in pairs})
		random.Random(seed).shuffle(patient_ids)
		end_train = int(frac[0] * len(patient_ids))
		end_val = int((frac[0] + frac[1]) * len(patient_ids))
		ids = {"train": set(patient_ids[:end_train]), "val": set(patient_ids[end_train:end_val]), "test": set(patient_ids[end_val:])}

	# leakage check: no patient in two splits
	assert not (ids["train"] & ids["val"] or ids["train"] & ids["test"] or ids["val"] & ids["test"])
	return {s: [p for p in pairs if patient_id(p) in g] for s, g in ids.items()}

class HipMRI3D(Dataset):
	"""One volume per item: image (1, D, H, W) float, label (D, H, W) long."""
	def __init__(self, pairs, scale, augment=False):
		self.pairs, self.scale, self.augment = pairs, scale, augment

	def __len__(self):
		return len(self.pairs)

	def __getitem__(self, i):
		image_path, label_path = self.pairs[i]
		image = nib.load(image_path).get_fdata(caching="unchanged")
		label = nib.load(label_path).get_fdata(caching="unchanged")
		if image.ndim == 4: image = image[..., 0]
		if label.ndim == 4: label = label[..., 0]
		image = (image - image.mean()) / (image.std() + 1e-8) # Normalization
		# files are (H, W, D); model wants (D, H, W); add batch+channel axes for interpolate
		image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)[None, None]
		label = torch.from_numpy(np.rint(label).astype(np.float32)).permute(2, 0, 1)[None, None]
		if self.scale != 1:
			image = F.interpolate(image, scale_factor=self.scale, mode="trilinear", align_corners=False)
			label = F.interpolate(label, scale_factor=self.scale, mode="nearest")  # keeps class IDs intact
		image, label = image[0], label[0, 0].long() # image (1,D,H,W), label (D,H,W)
		if self.augment:
			if torch.rand(1) < 0.5: # flip left-right
				image, label = image.flip(-2), label.flip(-2)
			image = image * (1 + 0.1 * torch.randn(1)) + 0.1 * torch.randn(1)   # intensity scale/shift
		return image, label
