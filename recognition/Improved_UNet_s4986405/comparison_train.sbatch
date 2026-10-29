#!/bin/bash
#SBATCH --job-name=hipmri-comparison-v1
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --array=0-1
#SBATCH --export=ALL
#SBATCH --output=comparison-train-%A_%a.log

set -euo pipefail
: "${SLURM_ARRAY_JOB_ID:?Submit this script using sbatch}"
: "${SLURM_ARRAY_TASK_ID:?Expected a Slurm array task}"
: "${HIPMRI_SOURCE_COMMIT:?Export the full Git commit SHA used for the source snapshot}"
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

architectures=(UNet2D ImprovedUNet2D)
case "$SLURM_ARRAY_TASK_ID" in
  0|1) architecture="${architectures[$SLURM_ARRAY_TASK_ID]}" ;;
  *) echo 'Unexpected array task ID' >&2; exit 1 ;;
esac
hostname
nvidia-smi
python_bin="$HOME/miniconda3/envs/torch/bin/python"
run_dir="$HOME/hipmri_runs/comparison_v1_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${architecture}"
"$python_bin" -u run_comparison.py \
  --architecture "$architecture" \
  --source-commit "$HIPMRI_SOURCE_COMMIT" \
  --output-dir "$run_dir"
