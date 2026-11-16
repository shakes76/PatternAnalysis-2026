# Data access and local storage

The experiments use the course-preprocessed **2D slices**, available to authorized
COMP3710 accounts on Rangpur at:

```text
/home/groups/comp3710/HipMRI_Study_open/keras_slices_data
```

The original source is Dowling and Greer, [Labelled weekly MR images of the male
pelvis, CSIRO version 2](https://doi.org/10.25919/45t8-p065). Its original ZIP
contains 3D volumes, not the course's six slice directories, and cannot directly
replace the input to this 2D loader. The recorded original release licence is
CC BY-NC-ND 4.0. Source metadata and acquisition checks are retained in
[results/data_source.json](results/data_source.json).

## Obtain the course slices

Connect using an authorized university account (replace YOUR_UQ_USERNAME):

```bash
ssh YOUR_UQ_USERNAME@rangpur.eait.uq.edu.au
cat /home/groups/comp3710/HipMRI_Study_open/data_use_agreement.txt
```

The recorded course terms prohibit participant identification/contact and
commercial use, require acknowledgement and a listed publication citation, and
do not guarantee data correctness. Review the actual terms before use. The
[report](README.md#8-data-acknowledgement-and-references) acknowledges HipMRI and
cites Dowling et al. (2015). Course terms and the original release licence are
separate source records; neither is replaced by this summary.

From a **local** terminal in this project directory, create `private_data` and
copy the course directory there. The trailing destination is a local directory:

```bash
scp -r YOUR_UQ_USERNAME@rangpur.eait.uq.edu.au:/home/groups/comp3710/HipMRI_Study_open/keras_slices_data private_data/
```

Network access and authentication depend on university arrangements. A completed
transfer must pass the audit below; terminal progress alone is not an integrity
check. Raw images, labels, manifests, run outputs and weights are ignored by the
project `.gitignore` and must not be committed.

## Expected layout and pairing

| Split | Image directory | Mask directory |
|---|---|---|
| train | `keras_slices_train` | `keras_slices_seg_train` |
| validate | `keras_slices_validate` | `keras_slices_seg_validate` |
| test | `keras_slices_test` | `keras_slices_seg_test` |

These six directories belong beneath `private_data/keras_slices_data/`.
Images follow `case_<id>_week_<week>_slice_<slice>.nii.gz`; masks use the same
keys with prefix `seg_`. Pair by keys, not by independent sorted-list positions.
All weeks of a Case stay in one split. Case grouping follows filename metadata;
actual participant identity was not independently established.

The official [label dictionary](labels.json) is 0 background, 1 body, 2 bones,
3 bladder, 4 rectum and 5 prostate. The CSIRO supporting description is available
at [BriefDataDescription.txt](https://data.csiro.au/dap/ws/v2/collections/52549/support/5034).
Three training Cases containing all six labels were matched element-for-element
to original source image/mask planes; see [label verification](results/label_verification.json).
This supports the dictionary, not exhaustive anatomical reannotation.

## Audit and freeze the manifest

Run from this project directory using the dependencies in `requirements.txt`:

```bash
python audit_course_download.py --root private_data/keras_slices_data --output runs/reproduce_audit.json --private-output private_data/source_metadata/reproduce_audit_private.json
python prepare_course_manifest.py --audit-summary runs/reproduce_audit.json --audit-private private_data/source_metadata/reproduce_audit_private.json --report runs/reproduce_preparation.json
```

The audit checks compressed files, image/mask keys, dimensions, finite values,
integer labels, affine agreement and Case separation. Preparation rechecks source
hashes and preserves course assignments (`validate` becomes `val`) in
`private_data/manifests/course_v1.csv`. Existing artifacts are protected; use new
report paths when repeating checks. The published acquisition passed with 12,660
pairs: 11,460 train, 660 validation and 540 test. See
[audit evidence](results/course_download_audit.json) and
[preparation evidence](results/data_preparation.json).

Training, prediction and evaluation commands are maintained only in the
[main report](README.md#6-installation-reproduction-and-code-map). Selected weights
remain local in `runs/real_unet_20ep_s3710/best.pt` and
`runs/real_improved_unet_20ep_s3710/best.pt`; their hashes and frozen evaluation
contract are in [TEST_PROTOCOL.md](TEST_PROTOCOL.md). There is no public weight
download. A fresh authorized checkout can retrain from the documented settings.
