#!/bin/bash
#SBATCH --job-name=adni-eval20
#SBATCH --partition=a100-test
#SBATCH --qos=test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:08:00
#SBATCH --output=eval20-%j.out

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate comp3710-unet
cd "$SLURM_SUBMIT_DIR"

python -u eval_full_val20.py

echo "=== FULL VALIDATION COMPLETED ==="
