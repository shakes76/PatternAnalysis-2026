#!/bin/bash
#SBATCH --job-name=sd-data-check
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --output=data_check_%j.out
#SBATCH --error=data_check_%j.err
set -euo pipefail

# Submit from the project directory. Uses existing CPU/PyTorch dependencies.
# Host-memory directives await clarification of Rangpur's placeholder memory TRES.
cd "$SLURM_SUBMIT_DIR"
audit_python="${AUDIT_PYTHON:-$HOME/miniconda3/envs/torch/bin/python}"
audit_root="${ADNI_ROOT:-/home/groups/comp3710/ADNI}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
mkdir -p data
"$audit_python" tools/prepare_data.py --root "$audit_root" \
    --output data/manifest-seed3710.csv > data/preparation.json
"$audit_python" tools/check_data.py --root "$audit_root" \
    --manifest data/manifest-seed3710.csv > data/validation.json
