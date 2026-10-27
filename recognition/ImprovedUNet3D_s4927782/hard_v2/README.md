# Integrated Hard HipMRI pipeline

This directory packages the completed 96×192×192 Hard pipeline without changing
its scientific sources, checkpoint choice or measurements. It reuses the parent
project's residual, deeply supervised 3D model and loss; it is not a larger model
or an exact reproduction of nnU-Net. The original 2D/3D implementations and their
results remain in the parent project.

The measured common-grid patient-balanced prostate Dice on the **previously
inspected** TEST set is **0.781784**, versus 0.677191 for original 3D and 0.678982
for original 2D. There are six TEST patients and 30 scans. These are retrospective
comparisons, not a new independent blind test. Ten scans remain below 0.70;
M015 and W029 patient means are 0.665621 and 0.635941. See the parent README for
the scientific discussion, per-class tables, five failure cases and costs.

The original custom VAL-gain adoption rule is retained in the unchanged protocols
and saved JSON only as experiment history. It is not a teacher requirement or a
current condition for including the new pipeline in this assignment. Its historic
failed outcome is not relabelled as a pass. Checkpoint selection remains fixed
using the seven scheduled VAL evaluations; no TEST-driven reselection occurs.

## Files and provenance

- `v2_pipeline.py`: verified old model/loss imports, 96-grid preprocessing,
  TRAIN-only organ-balanced sampling, training and sliding-window prediction.
- `train_hard_followup.py`: the actually executed Hard-only training/evaluation
  driver after the original new-2D/new-3D pair stopped at its TRAIN-fit gate.
- `infer_v2.py` / `infer_native.py`: unchanged image-only prediction and inverse
  geometry mapping. Restoring native geometry does not restore discarded detail
  or establish native-resolution segmentation accuracy.
- `predict.py`: new integration wrapper that validates the genuine Hard-only
  selected checkpoint, then calls the unchanged image-only implementation.
- `verify_package.py`: new portable standard-library audit of file identities,
  class-count mathematics, shared targets and equal-patient aggregation. It
  reports missing external weights/cache inputs separately.
- `protocol_v2.json`, `hard_followup_protocol.json`, source freezes and
  engineering amendments: byte-exact historical protocols and source identities.
- `artifacts/hard_common_{validation,test}/comparison.json`: complete three-model
  common-grid results, all classes, patients, scans and failures.
- `artifacts/hard_followup/improved3d/{run.json,history.json,curves.png}` and
  `artifacts/hard_selection.json`: actual 30-epoch run and fixed selection.
- `artifacts/hard_results_figures/`: measured accuracy and cost figures.
- `artifacts/hard_failure_gallery/`: five distinct patients, full field of view
  and display zoom. Display ROI did not provide a model inference crop.
- `INTEGRATION_ORIGINS.json`: exact source-to-package mapping and SHA values.
- `PUBLICATION_MANIFEST.json`: explicit public file whitelist and hashes,
  including the new integration-only helpers. It excludes itself to avoid a
  circular digest. No raw MRI, masks, prediction arrays or weights enter Git.

The `.slurm` files are the unchanged **historical UQ job scripts** and contain
site-specific paths. They are evidence of the original execution, not portable
one-click launchers. Use the explicit Python arguments below when relocating.
No training or medical-image inference was performed during this integration.

## Verify saved measurements without a GPU or medical data

From the parent project directory:

```bash
python hard_v2/verify_package.py
python -m unittest discover -s hard_v2 -p test_integration.py -v
```

These checks need only Python's standard library. They recompute saved count
mathematics; they do not recalculate predictions, raw pixels or surface distances.
The historical execution/metrics audits are retained as JSON with their actual
scope. A fresh Git clone has no private cache or weight files; the verifier lists
these missing external dependencies rather than pretending that it ran a model.

The unchanged pipeline requires the parent project's pinned dependencies in
`requirements.txt` and `requirements-extra.txt` for array/model execution. The
recorded execution used Python 3.11.15, PyTorch 2.13.0+cu130, CUDA 13.0,
NumPy 2.4.6, SciPy 1.16.3, nibabel 5.3.2 and an A100-PCIE-40GB. CUDA warn-only
settings emitted unsupported-determinism warnings, so bitwise identity is not
promised. This integration did not perform a fresh-environment install.

## Obtain weights and original cache metadata

Weights remain outside Git. The parent README provides the verified release
assets and extraction instructions. The new selected weight belongs at
`artifacts/hard_followup/improved3d/best.pt` **in the parent project**, with SHA256:

```text
852e3e1ee0a8f41be8a50902b7f6f2e28ffaa8227144d6ddf7806e06670e88bf
```

The new selected weight is available as an additional asset in the existing
release: [Hard seed3711 v2 weight archive](https://github.com/Jack-hithub/PatternAnalysis-2026/releases/download/demo3-hipmri-seed42-v1/demo3_hipmri_hard_seed3711_v2.zip)
and [its metadata](https://github.com/Jack-hithub/PatternAnalysis-2026/releases/download/demo3-hipmri-seed42-v1/demo3_hipmri_hard_seed3711_v2.metadata.json).
The archive has **18,221,187 bytes** and SHA256
`2c424dc514dd28837adfb40d9799b8a103d84576eba9f76e538320bd6b0ba101`.
The original v1 release assets remain unchanged. The public Git package itself
deliberately excludes all model weights.

Run the following in the **parent project directory**, alongside `modules.py`,
not inside `hard_v2/`. Verify the archive before extracting it; `unzip -n` avoids
overwriting any existing output.

```bash
mkdir -p artifacts/releases
curl --fail --location \
  https://github.com/Jack-hithub/PatternAnalysis-2026/releases/download/demo3-hipmri-seed42-v1/demo3_hipmri_hard_seed3711_v2.zip \
  --output artifacts/releases/demo3_hipmri_hard_seed3711_v2.zip
python - <<'PY'
from pathlib import Path
import hashlib
import subprocess
p = Path('artifacts/releases/demo3_hipmri_hard_seed3711_v2.zip')
assert p.stat().st_size == 18221187, 'Archive size mismatch'
assert hashlib.sha256(p.read_bytes()).hexdigest() == '2c424dc514dd28837adfb40d9799b8a103d84576eba9f76e538320bd6b0ba101', 'Archive SHA256 mismatch'
print('Archive verified; extracting into the parent project.')
subprocess.run(['unzip', '-n', str(p), '-d', '.'], check=True)
PY
```

The original release must also populate the original 2D/3D weights and cache64
manifest/cache-identity metadata under the parent `artifacts/` directory. The
frozen training code checks all 17 original dependencies, including these inputs;
it does not silently relax a missing-data or source guard. The parent download
instructions are the authority for actual published asset availability.

```bash
python hard_v2/verify_package.py --require-external \
  --checkpoint artifacts/hard_followup/improved3d/best.pt
```

The selected checkpoint is the original epoch 30 chosen by scheduled VAL P5
strict maximum with earliest ties. It is not selected again during packaging.

## Image-only inference

Once the recorded tensor dependencies and the correct weight are available,
run from the parent project directory:

```bash
python hard_v2/predict.py \
  --image /path/to/scan.nii.gz \
  --checkpoint artifacts/hard_followup/improved3d/best.pt \
  --output-dir /path/to/fresh_prediction \
  --device cpu
```

Use `--device cuda` on an allocated GPU if available. The wrapper's default
`--project-root` is the parent project; an explicit path can be supplied. The
output directory must not already exist. This operation needs the input image,
selected weight, public protocol and verified parent `modules.py`; it does not
read a target label or data manifest. It emits the unchanged adapter's NIfTI and
receipt plus an integration Hard-only selection receipt.

The unchanged `infer_v2.py --selection-freeze` option expects the stopped pair
and therefore must not be given a fabricated second checkpoint. The new wrapper
instead checks the actual single-Hard freeze without weakening checkpoint,
configuration, model-source or history validation. Its metadata-only rejection
paths were tested; new image inference was deliberately not run for this edit.

## Historical experiment reproduction

These commands describe reproduction on authorised UQ data, **not permission to
append experiments now**. Raw inputs, caches and arrays are not public Git
contents. Use fresh output directories; do not overwrite the completed run or
reinterpret the already viewed TEST as blind data.

Set task-specific locations (examples; replace with accessible paths):

```bash
DEMO3_RAW=/home/groups/comp3710/HipMRI_Study_open
DEMO3_V2_CACHE=/path/to/fresh_cache_v2
DEMO3_V2_RUN=/path/to/fresh_hard_run
```

The original parent cache64 manifest must match its frozen SHA. A fresh,
TRAIN/VAL-only 181-record cache can be rebuilt with the actual recovery helper:

```bash
python hard_v2/parallel_prepare_v2.py \
  --project-root . --protocol hard_v2/protocol_v2.json \
  --original-manifest artifacts/cache64/manifest.json \
  --raw-root "$DEMO3_RAW" --output-dir "$DEMO3_V2_CACHE" \
  --amendment hard_v2/cache_recovery_amendment.json --workers 4
```

The helper does not prepare TEST pixels at this stage. The rebuilt immutable
manifest must match `b2c62845c2ce2eb459afd3b6201469cb45dfc43faa2a3fb6457b74bf3af46d1d`
for the unchanged fixed Hard scope; a mismatch is an investigation requirement,
not permission to replace the expected digest. This fresh rebuild was not run
again during integration. Historical failed preparation, failed new-2D fit and
actual passing Hard receipts remain recorded, not rewritten as successes.

To reproduce the already specified Hard training on that verified cache:

```bash
python hard_v2/train_hard_followup.py --mode train \
  --protocol hard_v2/protocol_v2.json \
  --scope-plan hard_v2/hard_followup_protocol.json \
  --project-root . --manifest "$DEMO3_V2_CACHE/manifest.json" \
  --qa-receipt hard_v2/artifacts/qa_v2/qa_receipt.json \
  --output-dir "$DEMO3_V2_RUN" --device cuda
```

This is the exact completed Hard scientific configuration: 30 epochs, seed 3711,
286 patches per epoch, AdamW 0.001, TRAIN flips, weighted CE+Dice with residual
and deep supervision, and seven scheduled VAL selections. It reuses the genuine
recorded TRAIN QA; it does not claim a newly executed QA on another installation.
For new-environment validation, first rerun the real QA in fresh output and
retain its actual outcome rather than fabricating a passing record.

Further historical modes are `evaluate`, `freeze` and `prepare-test` on
`train_hard_followup.py`; each requires the same explicit `--protocol`,
`--scope-plan`, `--project-root` and `--manifest`. `--mode freeze` takes
`--run-dir` and a fresh `--selection-freeze` path; `prepare-test` additionally
requires `--raw-root` and that real freeze. Evaluation takes `--checkpoint`,
`--split`, `--output-dir` and, for TEST, the genuine Hard-only freeze. The
historical GPU script documents their exact order. Recorded frozen source and
manifest identities must remain satisfied on any relocated copy.

Common-grid comparison uses the unchanged command below with completed raw
arrays/caches available on the UQ server and a fresh destination:

```bash
python hard_v2/compare_hard_followup.py \
  --review-root hard_v2 --v1-project . --artifacts-root /path/to/completed_artifacts \
  --scope-plan hard_v2/hard_followup_protocol.json --split validation \
  --output-dir /path/to/fresh_common_validation
```

For the already exposed TEST cohort the comparator additionally requires
`--split test --retrospective-test-ack`; its historical gate output remains
archival, not an assignment eligibility condition. This edit did not rerun it.
