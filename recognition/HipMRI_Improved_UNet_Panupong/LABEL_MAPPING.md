# HipMRI prostate label: source discovery and verification

Progress recorded on **21 October 2026** (Australia/Brisbane).

## How the mapping was found

The prostate label was identified through **independent online research of the dataset publisher's own documentation**, with AI assistance. It was **not obtained from or confirmed by a tutor**. The tutor's earlier acceptance of the 2D project direction is a separate milestone.

Dataset citation: Dowling, Jason; and Greer, Peter (2021), *Labelled weekly MR images of the male pelvis*, v2, CSIRO Data Collection, [doi:10.25919/45t8-p065](https://doi.org/10.25919/45t8-p065). The collection credits CSIRO and Calvary Mater Newcastle Hospital.

The source can be traced through:

1. The [official CSIRO collection](https://data.csiro.au/collection/csiro:51392v2).
2. Its [public metadata](https://data.csiro.au/dap/ws/v2/collections/52549), which links to the [supporting-file list](https://data.csiro.au/dap/ws/v2/collections/52549/support).
3. The attached **[BriefDataDescription.txt](https://data.csiro.au/dap/ws/v2/collections/52549/support/5034)**, file ID 5034, which explicitly defines these stored values:

| Stored mask value | Anatomical meaning |
| ---: | --- |
| 0 | Background |
| 1 | Body |
| 2 | Bones |
| 3 | Bladder |
| 4 | Rectum |
| **5** | **Prostate** |

The numbered class list in [CAN3D, Section 4.1](https://arxiv.org/html/2109.05443#S4.SS1) uses numbers 1-6 and a different order. That list alone does not establish the integer values stored in these masks. The publisher's explicit stored-value table is the source for the mapping above.

## Verification against the processed 2D data

The available original mask `B006_Week0_SEMANTIC.nii.gz` has shape `(256, 256, 128)`, dtype `int32`, and values `0..5`. All **60 supplied training masks** for **Case 006 / Week 0**, named `seg_006_week_0_slice_0.nii.gz` through `seg_006_week_0_slice_59.nii.gz`, have shape `(256, 128)` and dtype `uint8`.

The matching plane window and reversal were identified through exact voxel-array matching. Direct comparison on the original integer arrays then verified this relationship for every supplied slice:

```python
np.array_equal(processed_slice_i, original_mask[::-1, 98 + i, :])
# True for each i from 0 through 59.
```

**Result: 60/60 exact matches.** Reversing an image axis and selecting a plane preserve the numeric values; no label remapping, interpolation or dtype casting was used in the equality check. This confirms **5 = Prostate for the processed Case 006 / Week 0 series**.

The local check recorded 60 supplied masks and 60 exact matches. Per-slice equality results and array hashes are retained in the local project workspace. The checked input hashes are:

- Publisher text SHA256: `dc3a99204684ced859872d79a75707af3c0ad31d00f5f3213063ce5e30727ee6`.
- Original mask file SHA256: `c325ccab8a767580dc2c4e666aae6a583ac5067e1fc7206bbd4af196dd7c2e27`.

MRI volumes, masks and model weights are stored separately from this repository.

For a local reproduction using separately obtained copies of the same inputs, the core comparison is:

```python
from pathlib import Path
import nibabel as nib
import numpy as np

raw_path = Path("/path/to/B006_Week0_SEMANTIC.nii.gz")
processed_dir = Path("/path/to/keras_slices_seg_train")
raw = np.asanyarray(nib.load(raw_path).dataobj)

assert raw.shape == (256, 256, 128)
assert np.array_equal(np.unique(raw), np.arange(6))
for i in range(60):
    path = processed_dir / f"seg_006_week_0_slice_{i}.nii.gz"
    processed = np.asanyarray(nib.load(path).dataobj)
    assert np.array_equal(processed, raw[::-1, 98 + i, :]), path.name
print("60/60 processed masks preserve the original numeric labels.")
```

This reads masks only and does not train a model or perform inference.

## Scope and remaining work

- **Confirmed:** the original release maps stored value 5 to Prostate.
- **Confirmed:** all 60 supplied masks in the checked Case 006 / Week 0 series preserve those values.
- **Pending:** individual original-to-processed comparisons for other series, the complete mask readability/shape/label audit, class counts and split/patient grouping checks.
- **Not claimed:** teaching-staff confirmation, dataset-wide verification, model implementation, training, segmentation scores or satisfaction of the target Dice score.

This progress update records source discovery and a bounded data check. The model experiments remain pending.
