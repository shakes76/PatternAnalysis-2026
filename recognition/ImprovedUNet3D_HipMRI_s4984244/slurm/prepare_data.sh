#!/bin/bash
#SBATCH --job-name=hipmri-prepare
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
# No --mem: Rangpur nodes report RealMemory=1, so any --mem request pends forever.

# Data audit, patient split (splits.json) and pre-processed cache, on a CPU
# node so the GPU jobs can start training straight away:
#     cd recognition/ImprovedUNet3D_HipMRI_s4984244 && sbatch slurm/prepare_data.sh
# Extra arguments go to the cache step, e.g. `sbatch slurm/prepare_data.sh --shape 128x256x256`.
# Build each cache here before training: concurrent jobs would write the same file.

set -euo pipefail
echo "job $SLURM_JOB_ID on $(hostname), $(date)"
cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1   # stream prints to the log as they happen

python - <<'EOF'
import sys, matplotlib, nibabel, numpy, scipy, torch
print("python", sys.version.split()[0], "| torch", torch.__version__, "| nibabel",
      nibabel.__version__, "| numpy", numpy.__version__, "| scipy", scipy.__version__,
      "| matplotlib", matplotlib.__version__)
EOF
python dataset.py
python dataset.py --prepare cache "$@"      # e.g. --shape 128x256x256
echo "done $(date)"
