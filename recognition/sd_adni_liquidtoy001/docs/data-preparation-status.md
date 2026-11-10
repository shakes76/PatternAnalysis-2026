# Data Preparation Status — 2 October 2026

## Implemented

- `dataset.py`: full-manifest validation, split-specific lazy loading, aspect-preserving square padding, bilinear resizing, replicated grayscale channels, float32 tensors in [-1,1].
- `tools/prepare_data.py`: metadata-backed patient grouping, deterministic class-stratified 70/15/15 allocation, full JPEG decode and SHA-256 checks, private CSV manifest and aggregate JSON report.
- `tools/check_data.py`: checks real source coverage, deterministic grouping, sampled training/validation tensors and collation, plus rejection of intentionally corrupted copies of the real manifest.
- `tools/prepare_data.sbatch`: bounded CPU-only preparation/validation without dependency installation or model downloads.

The [protocol](data-protocol.md) was recorded before running preparation. The split is provisional pending confirmation of the course evaluation requirements.

## Completed validation

Seven local unit tests passed using deliberately synthetic tiny grayscale images and manifest fixtures. They verify geometry and normalization, patient/content overlap rejection, path traversal and label-path mismatch rejection, resolution checks, and class-stratified counts/order independence. These are contract tests, not evidence of clinical quality or successful full-data processing. PyTorch 2.13.0+cu132 and Pillow 12.3.0 were used locally. No packages were installed.

Ruff check and formatting validation passed on the new Python files. The loader uses PyTorch and Pillow; torchvision is not required for pixel conversion.

## Remote acceptance completed

CPU job **623310** completed with **exit 0:0 in 40 seconds**. Both aggregate
outputs were retrieved and verified on 2 October 2026 (Australia/Brisbane):
[data preparation](data-preparation-2026-10-02.json) and
[loader validation](data-validation-2026-10-02.json).

- All 30,520 JPEGs fully decoded at 256 x 240, grayscale L.
- No duplicate file-hash groups or cross-split patient/scan/hash overlap.
- Train: 476 patients, 1,064 scans, 21,280 slices.
- Validation: 102 patients, 227 scans, 4,540 slices.
- Test: 102 patients, 235 scans, 4,700 slices.
- Deterministic assignment and complete source coverage passed.
- Actual train/validation tensors and batch collation passed; corrupted manifest
  controls for patient overlap, path escape, label mismatch and duplicate rows
  were rejected. Test images were decoded for integrity, not model evaluation.
- Manifest SHA-256: `bc77d11ade201eaa0acbbcd1986be565557c161bc676bc45b230dae52e557396`.

The patient-level CSV remains private on Rangpur. No image or patient identifier
is included in the committed aggregate reports. The split remains provisional
pending course-protocol confirmation, even though engineering acceptance passed.

## Reproduction

From this project directory on a machine with PyTorch and Pillow:

```bash
python -m unittest discover -s tests -v
```

On Rangpur, upload `dataset.py` and `tools/` to an isolated project directory, enter it, and submit `sbatch tools/prepare_data.sbatch`. The generator refuses to overwrite an existing manifest. Do not resubmit a completed preparation without first deciding whether to reuse the verified manifest or create a separate run directory. Check `sacct` and both JSON outputs, not merely log existence.

Home space is now approximately 12 GiB free. The next bounded diagnostic is documented in [the VAE protocol](vae-protocol.md); see the experiment ledger for its current scheduling status.
