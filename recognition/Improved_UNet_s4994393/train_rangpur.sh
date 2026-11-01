#!/bin/bash
# Resource syntax follows the supplied COMP3710-Rangpur course guide.
# Adjust wall time after a short pilot; 12 hours is a request, not a runtime estimate.
#SBATCH --job-name=hipmri-3d
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=hipmri_%j.out
#SBATCH --error=hipmri_%j.err

set -euo pipefail
# Resolve relative data/output paths from the directory where sbatch was called.
cd "${SLURM_SUBMIT_DIR:?Submit from the project code directory}"
source "$HOME/miniconda3/bin/activate"
conda activate "${CONDA_ENV_NAME:-torch}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

architecture="${1:-baseline}"
# Consume an optional architecture; forward the remaining training options below.
shift "$(( $# > 0 ? 1 : 0 ))"
data_root="${DATA_ROOT:-/home/groups/comp3710/HipMRI_Study_open}"
output_root="${OUTPUT_ROOT:-runs}"
python -u train.py --architecture "$architecture" --device cuda \
  --data-root "$data_root" --output-dir "$output_root/$architecture" \
  --patch-size 64 64 64 --batch-size 1 --workers 2 "$@"
