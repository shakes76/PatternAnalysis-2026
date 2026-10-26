# Initial feasibility evidence: 26 October 2026

This is an environment and filename inventory, not a model result or a complete
data-integrity audit.

## Confirmed

- Authenticated access to Rangpur and read access to the course HipMRI root.
- `semantic_MRs` contains **211 NIfTI filenames**; `semantic_labels_only` also
  contains **211**. Equal counts alone do not establish correct pairing.
- The existing `torch` environment has Python **3.11.15**, PyTorch **2.13.0**,
  NumPy **2.4.6** and Matplotlib **3.11.2**, as read from package metadata.
- Nibabel and SciPy are not installed in that environment. A project-specific
  environment is being prepared on a CPU compute node.
- Slurm exists under `/opt/slurm/bin/`. The course `comp3710` partition reports
  **10 A100 GPU nodes**. No GPU was allocated or model trained by this inventory.
- Scheduler metadata advertises 1 MiB of host memory per node and an unlimited
  memory default. An explicit 4 GiB CPU-job request was rejected. Setup uses
  the cluster's configured memory default and one CPU, matching the existing
  course workflow; this metadata is not a measurement of physical RAM.
- The audit utility passed **seven local automated checks**.

## Still to validate

- Unique patient count and identities for grouping, repeated scans, exact
  image/mask correspondence, label meanings, shapes, affines and spacing.
- A disjoint patient split, spatial preprocessing and foreground coverage.
- A real-data model forward/backward test, measured memory and runtime.
- The official final deadline in Learn.UQ.

The raw inventory remains under ignored `artifacts/`; scan files and trained
weights are never committed. All development-stage model checks must use
training/validation data.

## Data acknowledgement

Data were provided by HipMRI Study. The dataset usage note requires attribution,
an appropriate study citation, research/non-commercial use, and no attempt to
identify or contact participants. The final report will include the exact
study attribution and relevant publication after checking the study source.
