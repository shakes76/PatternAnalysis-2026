#!/bin/bash
#SBATCH --partition=a100-test
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=full_%j.out
set -eu
cd "$SLURM_SUBMIT_DIR"
case "$1" in basic|improved) model="$1" ;; *) exit 2 ;; esac
export OMP_NUM_THREADS=2
source "$HOME/venv/bin/activate"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
/usr/bin/time -v python -u train.py --model "$model" --run "runs/${model}_full_${SLURM_JOB_ID}" \
  --epochs 20 --cache --workers 0 --time-budget 1000
