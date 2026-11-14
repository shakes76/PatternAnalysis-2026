# Provisional Data Protocol

Recorded 2026-10-02, before generating any project split or evaluating a model.

## Split definition

Pool the available course JPEGs without moving or modifying them. Recover patient IDs through the supplied metadata. Assign whole patients to 70% training, 15% validation, and 15% test, stratified by AD/NC patient class. Sort each class's patients by SHA-256 of `seed:patient_id`, with patient ID as a tie-breaker. Allocate integer counts by the largest-remainder method; ties follow train, validation, test order. Default seed: 3710. Save the original course directory as provenance.

This is a proposed project evaluation split, pending confirmation of any course-mandated split. Do not tune the seed or proportions based on model performance. Multiple scans and slices of one patient always remain together. Ratios apply to patients, not images; slice counts need not follow exactly 70/15/15.

Validate metadata/folder label agreement (AD=2, NC=0), unique scan/slice identities and paths, consistent patient labels, patient and scan separation, and exact-file duplicates across splits. A mismatch fails preparation rather than silently dropping images. Save patient-level manifests under ignored `data/`; publish only aggregate summaries and manifest hashes. Hash equality detects identical files, not near-duplicates or all possible leakage.

## Image loading

Read the course JPEGs directly. Fully decode them during preparation to catch damaged files. Preserve aspect ratio by adding zero-valued borders to a square, then bilinearly resize to a configurable multiple of eight (initial smoke check: 256). Convert grayscale to three identical channels, then scale [0,255] to [-1,1]. Do not infer MRI physical orientation or voxel spacing from JPEG dimensions or slice indices.

The padding approach avoids stretching 256x240 images into squares. It is a provisional preprocessing choice, to be revisited based on training/validation VAE reconstruction evidence before model comparison. Apply the same documented geometric preprocessing to the baseline. No augmentation, learned normalization, random crop, or prompts are introduced at this stage. A 256-size CPU check does not settle the final SD training resolution.

## Validation boundaries

Integrity checks may inspect all files for decoding, identity and hashes. Pixel statistics, visual model debugging, and reconstruction/model quality analysis use training or validation data. Keep test model outputs and metrics reserved for the finalized protocol.

The CPU smoke check must verify manifest determinism, disjoint groups, complete dataset coverage, actual image decoding, tensor shape/range and rejection of an intentionally corrupted manifest. These are data-pipeline checks, not model-quality tests. Synthetic negative controls must be identified as such.
