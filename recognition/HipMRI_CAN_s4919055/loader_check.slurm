#!/bin/bash
# CPU data preparation: no GPU is needed for NIfTI/tensor validation.
#SBATCH --job-name=hipmri-loader-check
#SBATCH --partition=cpu
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --output=runs/loader_check_%j.out
#SBATCH --error=runs/loader_check_%j.err

set -e
cd "$SLURM_SUBMIT_DIR"
source "$HOME/miniconda3/bin/activate"
conda activate torch
export OMP_NUM_THREADS=1
python -u dataset.py --loader-check "runs/loader_check_${SLURM_JOB_ID}.json" --samples 3
