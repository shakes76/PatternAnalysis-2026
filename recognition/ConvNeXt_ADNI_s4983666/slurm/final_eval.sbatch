#!/bin/bash
#SBATCH --job-name=adni-final-eval
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out
# Final evaluation of ONE frozen model (run once, after all model choices are made):
#   sbatch slurm/final_eval.sbatch <output dir> <run dir with final.pt> [<run dir> ...]
# Several run directories form an ensemble (mean logit), e.g. the five fold models.
# 1) fit Platt scaling + referral threshold on calibration patients;
# 2) apply both unchanged to the test patients.
set -euo pipefail
output="$1"; shift
checkpoints=()
for run_dir in "$@"; do checkpoints+=("$(cd -- "$run_dir" && pwd -P)/final.pt"); done
splits="${ADNI_SPLITS:-$HOME/comp3710/adni_splits_v1}"
source "${ADNI_CONDA_ROOT:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate "${ADNI_CONDA_ENV:-torch}"
mkdir -p "$output"
python -u predict.py --checkpoint "${checkpoints[@]}" --splits-dir "$splits" \
    --role calibration --output "$output/calibration"
python -u predict.py --checkpoint "${checkpoints[@]}" --splits-dir "$splits" \
    --role test --calibration-file "$output/calibration/calibration.json" --output "$output/test"
