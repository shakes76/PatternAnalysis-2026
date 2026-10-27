#!/bin/bash
#SBATCH --job-name=adni_entry
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --qos=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=entry_check_%j.out
#SBATCH --error=entry_check_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate torch
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

python -u train.py \
  --test-checkpoint "$HOME/data/comp3710_a3_runs_lr1e4/cnn/full/best_model.pth" \
  --test-output-dir "$HOME/data/comp3710_a3_entry_check/cnn" \
  --batch-size 32 --workers 4

python -u train.py \
  --test-checkpoint "$HOME/data/comp3710_a3_runs_pretrained/convnext/full/best_model.pth" \
  --test-output-dir "$HOME/data/comp3710_a3_entry_check/convnext" \
  --batch-size 32 --workers 4
