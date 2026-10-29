#!/bin/bash
# First real CAN3D volume feasibility check; do not launch a full fit yet.
#SBATCH --job-name=hipmri-can3d-smoke
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=1
#SBATCH --output=runs/can3d_smoke_%j.out
#SBATCH --error=runs/can3d_smoke_%j.err

set -e
cd "$SLURM_SUBMIT_DIR"
source "$HOME/miniconda3/bin/activate"
conda activate torch
export OMP_NUM_THREADS=1
python -u can3d_smoke.py --output "runs/can3d_smoke_${SLURM_JOB_ID}"
