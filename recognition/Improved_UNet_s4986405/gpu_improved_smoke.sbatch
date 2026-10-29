#!/bin/bash
#SBATCH --job-name=hipmri-improved-gpu-smoke
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --output=improved-gpu-smoke-%j.log

set -euo pipefail
: "${SLURM_JOB_ID:?Submit this script using sbatch}"
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

hostname
nvidia-smi
run_dir="$HOME/hipmri_runs/improved_gpu_smoke_${SLURM_JOB_ID}"
python_bin="$HOME/miniconda3/envs/torch/bin/python"

"$python_bin" -u train.py \
  --data-root /home/groups/comp3710/HipMRI_Study_open/keras_slices_data \
  --output-dir "$run_dir" \
  --architecture ImprovedUNet2D \
  --epochs 2 --batch-size 2 --base-channels 16 --device cuda --augment \
  --max-train-batches 2 --max-val-batches 1

"$python_bin" -u check_checkpoint.py --run-dir "$run_dir"
cat "$run_dir/gpu_profile.json"
printf '\nImproved U-Net GPU smoke check: PASS\n'
