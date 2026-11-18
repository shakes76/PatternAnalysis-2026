#!/bin/bash
#SBATCH --job-name=adni-plan
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out
# Run (part of) an experiment plan on one GPU. Examples, from the project folder:
#   sbatch slurm/run_plan.sbatch config/experiments_strong_aug.json --smoke
#   sbatch slurm/run_plan.sbatch config/experiments_strong_aug.json --cases S3_lite_context_seed3710
# Override paths with ADNI_SPLITS / ADNI_RUNS and the environment with ADNI_CONDA_ENV.
set -euo pipefail
source "${ADNI_CONDA_ROOT:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate "${ADNI_CONDA_ENV:-torch}"
python -u slurm/run_plan.py "$@" \
    --splits-dir "${ADNI_SPLITS:-$HOME/comp3710/adni_splits_v1}" \
    --runs-root "${ADNI_RUNS:-$HOME/comp3710/runs}"
