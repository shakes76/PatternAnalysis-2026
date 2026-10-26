#!/bin/bash
#SBATCH --job-name=hipmri-label-review
#SBATCH --account=comp3710
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export TMPDIR="${SLURM_TMPDIR:-$SLURM_SUBMIT_DIR/artifacts/tmp}"
export MPLCONFIGDIR="$TMPDIR/matplotlib"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
.venv/bin/python -m unittest discover -s tests -p 'test_label_schema.py' -v
.venv/bin/python -m unittest discover -s tests -p 'test_pair_analysis.py' -v
# Read every saved array before training; a scheduler timeout during a later
# plotting step must not be mistaken for proof that the cache is complete.
.venv/bin/python -u - <<'PY'
from pathlib import Path
import numpy as np
from dataset import HipMRIVolumeDataset, load_manifest

path = Path('artifacts/cache64/manifest.json')
manifest = load_manifest(path)
assert not manifest['development_only']
assert len(manifest['records']) == manifest['source_audit_record_count'] == 211
for split in ('train', 'validation', 'test'):
    data = HipMRIVolumeDataset(path, split)
    for i in range(len(data)):
        item = data[i]
        actual = np.bincount(item['mask'].numpy().ravel(), minlength=manifest['num_classes']).tolist()
        if actual != data.records[i]['class_voxel_counts']:
            raise ValueError('Cached mask counts disagree with the frozen manifest.')
    print(f'Validated {split}: {len(data)} cached volumes', flush=True)
print('All 211 complete image/mask/spacing caches passed.')
PY
# Freeze source-confirmed semantics before the first formal checkpoint.
.venv/bin/python tools/attach_label_schema.py \
  --manifest artifacts/cache64/manifest.json --schema label_schema.json
.venv/bin/python tools/verify_label_semantics.py \
  --manifest artifacts/cache64/manifest.json --count 5 \
  --output-dir artifacts/label_review_v1
.venv/bin/python tools/inspect_cache.py --manifest artifacts/cache64/manifest.json \
  --output artifacts/alignment_official.png --target-class 5
