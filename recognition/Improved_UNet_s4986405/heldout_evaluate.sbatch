#!/bin/bash
#SBATCH --job-name=hipmri-heldout-v1
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --export=ALL
#SBATCH --output=heldout-evaluate-%j.log
set -euo pipefail
: "${SLURM_JOB_ID:?Submit this script through Slurm}"
: "${HIPMRI_HELDOUT_COMMIT:?Export the full committed held-out runner SHA}"
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
hostname
nvidia-smi
"$HOME/miniconda3/envs/torch/bin/python" -u run_heldout.py \
  --runner-commit "$HIPMRI_HELDOUT_COMMIT" \
  --output-dir "$HOME/hipmri_runs/heldout_v1_${SLURM_JOB_ID}"
