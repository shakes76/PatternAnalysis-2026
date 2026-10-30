# COMP3710 — HipMRI 3D Data Audit (Preliminary)

**Project:** 3D Improved U-Net (Hard difficulty)
**Initial baseline:** Standard 3D U-Net
**Dataset:** HipMRI Study Open
**Cluster dataset location:** `/home/groups/comp3710/HipMRI_Study_open`
**Status:** Preliminary audit — findings below distinguish dataset-wide metadata checks from one-sample checks.

## 1. Dataset inventory and pairing

| Item | Observed result | Coverage |
|---|---:|---|
| MRI files (`semantic_MRs/*_LFOV.nii.gz`) | 211 | All matching MRI file paths |
| Mask files (`semantic_labels_only/*_SEMANTIC.nii.gz`) | 211 | All matching mask file paths |
| Unique patient IDs | 38 | All MRI filenames |
| MRI paths without corresponding masks | 0 | All 211 MRI paths |
| Matched pairs with different shapes | 0 | All 211 MRI–mask pairs |

**Important qualification:** The script checked that every MRI had a matching mask, but did not separately test whether any mask had no corresponding MRI. It also did not establish the correctness of the anatomical labels.

## 2. Original image dimensions

| MRI shape | Number of scans |
|---|---:|
| `(256, 256, 128)` | 210 |
| `(256, 256, 144)` | 1 |

**Finding:** One scan has 144 voxels on the third array axis; 210 have 128. Shapes need to be made consistent for batching and standard 3D U-Net input.

### Single-sample spatial metadata: B006_Week0

| Item | Result |
|---|---|
| MRI shape | `(256, 256, 128)` |
| Mask shape | `(256, 256, 128)` |
| MRI voxel spacing | Approximately `(1.5625, 1.5625, 1.56)` mm |
| Mask voxel spacing | Approximately `(1.5625, 1.5625, 1.56)` mm |
| Shape match | True |
| Affine match (`np.allclose`) | True |

**Limit:** Affine equivalence has been directly verified for **B006_Week0 only**, not all 211 scan pairs. Identical shape or spacing does not by itself demonstrate anatomical alignment.

## 3. Labels — B006_Week0 only

The inspected mask contains integer labels `0, 1, 2, 3, 4, 5`.

| Label | Voxels | Share of sample |
|---|---:|---:|
| 0 | 4,819,047 | 57.45% |
| 1 | 3,197,231 | 38.11% |
| 2 | 314,267 | 3.75% |
| 3 | 29,725 | 0.35% |
| 4 | 18,140 | 0.22% |
| 5 | 10,198 | 0.12% |
| **Total** | **8,388,608** | **100%** |

**Finding:** Strong class imbalance is apparent in this sample, especially for labels 3–5. These are sample-specific percentages and **not dataset-wide class frequencies**. The precise anatomical mapping of class IDs remains unverified.

## 4. Patient-level split (seed=42)

| Subset | Distinct patients | MRI–mask pairs |
|---|---:|---:|
| Train | 26 | 145 |
| Validation | 6 | 39 |
| Test | 6 | 27 |
| **Total** | **38** | **211** |

The script verified that the three sets of patient IDs are disjoint (`Patient-level split: OK`). Repeated weekly scans from a given patient stay within one subset. A fixed seed makes this particular split reproducible given the same sorted input IDs and software behavior.

## 5. Implemented data preprocessing

Location: `recognition/UNet3D_s4984160/dataset.py`.

1. `get_data_pairs()`: pairs MRI paths with corresponding mask paths using filename prefixes and extracts patient IDs.
2. `split_by_patient()`: deterministically shuffles unique patient IDs, partitions them into train/validation/test (26/6/6), and assigns all scans accordingly.
3. `HipMRIDataset.__getitem__()`: loads one MRI–mask pair, checks shape and affine compatibility, converts arrays to numeric types, Z-score normalizes nonzero MRI voxels, transposes NIfTI array axes `(X,Y,Z) -> (Z,X,Y)`, resizes the MRI with trilinear interpolation and the mask with nearest-neighbor interpolation, and returns PyTorch tensors.

**Configured final per-sample shapes (observed from a successful test):**

- MRI image: `torch.Size([1, 64, 128, 128])` (channel, depth, height, width)
- Mask: `torch.Size([64, 128, 128])`
- Unique mask labels in tested returned sample: `tensor([0, 1, 2, 3, 4, 5])`

The successful `python dataset.py` test loaded **one** sample from the training subset, not every sample.

## 6. Feasibility evidence available now

- Confirmed accessible 3D HipMRI cohort on the Rangpur cluster: 211 paired volumes from 38 distinct patient IDs.
- Confirmed paired masks for every scanned MRI filename, and matching MRI–mask array shapes in all 211 pairs.
- Detected a dimension outlier requiring standardized preprocessing.
- Demonstrated a patient-disjoint 26/6/6 split with 145/39/27 scans.
- Demonstrated an initial 3D PyTorch data loading and resizing pipeline on a real MRI–mask pair.
- Identified sample-level class imbalance as a material segmentation risk.

These items support **preliminary data-pipeline feasibility**, but are not evidence of model accuracy, GPU performance, a successful 3D U-Net training run, or clinically usable segmentation.

## 7. Outstanding checks and experiments

1. Audit `np.allclose(mri.affine, mask.affine)` for **all** MRI–mask pairs, and inspect potential orientation or spacing differences.
2. Verify integer mask label IDs across **all** 211 masks and identify the source-documented anatomical label mapping.
3. Identify the filename of the single `(256,256,144)` volume and verify that it successfully passes preprocessing.
4. Visually inspect MRI–mask overlay alignment, including a small structure and boundaries, before full training.
5. Test actual full-size samples through `DataLoader`, then run standard 3D U-Net forward/backward on a Rangpur GPU; record peak allocated GPU memory, model parameters, and time.
6. Retain fixed patient splits for a fair standard 3D U-Net versus improved 3D U-Net comparison. Establish Dice/IoU metrics (per class) and review 3–5 failure cases.

## 8. Submission / engineering notes

- **PyCharm:** sole location for editing code; commit and push changes to the fork's `topic-recognition` branch.
- **GitHub:** version control; do not commit patient image data or trained model checkpoints.
- **Rangpur:** pull code, then execute tests or training on allocated Slurm CPU/GPU compute nodes (not on the login node).
- **Report:** include this audit as provisional feasibility evidence and update it when the dataset-wide checks and baseline smoke test complete.
