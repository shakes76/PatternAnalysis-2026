#!/bin/bash
#SBATCH --job-name=hipmri-model-cpu
#SBATCH --account=comp3710
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
.venv/bin/python -m unittest discover -s tests -p 'test_models.py' -v
.venv/bin/python tools/smoke_models.py --device cpu \
  --output artifacts/model_smoke_cpu.json
.venv/bin/python - <<'PY'
import nibabel as nib
import json
root = '/home/groups/comp3710/HipMRI_Study_open/'
for relative in ['semantic_MRs/B006_Week0_LFOV.nii.gz',
                 'semantic_labels_only/B006_Week0_SEMANTIC.nii.gz']:
    image = nib.load(root + relative)
    print(json.dumps({'role': relative.split('/')[0], 'shape': image.shape,
        'spacing': [float(v) for v in image.header.get_zooms()],
        'description': image.header['descrip'].item().decode(errors='replace'),
        'intent': str(image.header.get_intent())}))
PY
