#!/bin/bash
#SBATCH --job-name=hipmri_eval
#SBATCH --partition=a100-test
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --output=evaluate_%j.out
set -eu
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=2
source "$HOME/venv/bin/activate"
if [ "$#" -ne 2 ]; then
  echo "Usage: sbatch submit_evaluate.sh basic|improved /path/to/best.pt" >&2
  exit 2
fi
case "$1" in basic|improved) model="$1" ;; *) echo "Unknown model: $1" >&2; exit 2 ;; esac
checkpoint="$2"
test -f "$checkpoint" || { echo "Checkpoint not found: $checkpoint" >&2; exit 2; }
python -u train.py --model "$model" --evaluate "$checkpoint" \
  --run "runs/${model}_test_${SLURM_JOB_ID}" --split test --workers 0
python -u analyse_failures.py --checkpoint "$checkpoint" \
  --output "runs/${model}_validation_${SLURM_JOB_ID}"
