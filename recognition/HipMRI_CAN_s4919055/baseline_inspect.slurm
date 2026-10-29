#!/bin/bash
# Diagnostic CPU inference only: inspect one training and one validation case.
# These small previews do not train, tune or evaluate the final test split.
#SBATCH --job-name=hipmri-inspect
#SBATCH --partition=cpu
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --output=runs/baseline_inspect_%j.out
#SBATCH --error=runs/baseline_inspect_%j.err

set -e
cd "$SLURM_SUBMIT_DIR"
source "$HOME/miniconda3/bin/activate"
conda activate torch
export OMP_NUM_THREADS=1
export MPLCONFIGDIR="$SLURM_SUBMIT_DIR/runs/mpl-cache"

# Inspect the last model, rather than best.pt, which may still be epoch1 at Dice0.
inspection_checkpoint="${INSPECT_CHECKPOINT:-runs/baseline_balance_704488/last.pt}"
test -f "$inspection_checkpoint"
python -u predict.py --checkpoint "$inspection_checkpoint" --device cpu \
    --split train --max-cases 1 --output "runs/baseline_inspect_${SLURM_JOB_ID}/train"
python -u predict.py --checkpoint "$inspection_checkpoint" --device cpu \
    --split validation --max-cases 1 --output "runs/baseline_inspect_${SLURM_JOB_ID}/validation"
echo "BASELINE_INSPECTION_COMPLETED (two diagnostic previews; not a final benchmark)."
