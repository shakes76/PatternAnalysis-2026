#!/bin/bash
# Evaluate every finished training run on the held-out test split, then
# aggregate all seeds into the README comparison tables and figures.
#
# Usage (from this folder, after all train.slurm jobs have finished):
#   sbatch eval.slurm
#SBATCH --job-name=hipmri-eval
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

source $HOME/miniconda3/bin/activate
conda activate torch

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"

for ckpt in runs/*_s*/best.pt; do
    run=$(basename "$(dirname "$ckpt")")      # e.g. improved_s0, improved3d_s0
    echo "=== Evaluating $run ==="
    python predict.py --ckpt "$ckpt" --out "eval/$run"   # model read from the run's config.json
done

python compare.py --eval-dir eval --out images

echo "Finished $(date)"
