#!/bin/bash
#SBATCH --job-name=unet-baseline
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=baseline_%j.out
#SBATCH --error=baseline_%j.err

set -euo pipefail
source "$HOME/miniconda3/bin/activate"
conda activate torch
cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

run_dir="runs/baseline_${SLURM_JOB_ID}"
mkdir -p "$run_dir/source"

# Preserve the exact source files used for this experiment.
cp dataset.py modules.py metrics.py sampling.py train.py train_baseline.sh \
    "$run_dir/source/"
git rev-parse HEAD > "$run_dir/source/git_commit.txt"
git diff HEAD -- dataset.py modules.py metrics.py sampling.py train.py \
    > "$run_dir/source/uncommitted_changes.patch"
python -m pip freeze > "$run_dir/source/environment.txt"
nvidia-smi > "$run_dir/source/gpu_info.txt"

python -u train.py \
    --epochs 20 \
    --batch-size 8 \
    --learning-rate 0.001 \
    --seed 42 \
    --output "$run_dir/results"
