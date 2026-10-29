#!/bin/bash
# Continue verified run708618 from epoch3 to8; retain the matched experiment recipe.
#SBATCH --job-name=hipmri-dice-ce-resume
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=1
#SBATCH --output=runs/baseline_dice_ce_resume_%j.out
#SBATCH --error=runs/baseline_dice_ce_resume_%j.err

set -e
cd "$SLURM_SUBMIT_DIR"
source "$HOME/miniconda3/bin/activate"
conda activate torch
export OMP_NUM_THREADS=1
export MPLCONFIGDIR="$SLURM_SUBMIT_DIR/runs/mpl-cache"

test -f runs/baseline_dice_ce_708618/last.pt
test -f runs/baseline_dice_ce_708618/best.pt
experiment_dir="runs/baseline_dice_ce_resume_${SLURM_JOB_ID}"
python -u train.py --mode fit --device cuda --epochs 8 --batch-size 8 \
    --learning-rate 0.001 --seed 42 --slice-sampling prostate-balanced --loss dice-ce \
    --manifest runs/split_seed42.json --resume runs/baseline_dice_ce_708618/last.pt \
    --output "$experiment_dir"

# Inspect the validation-selected best model without opening any test cases.
python -u predict.py --device cuda --checkpoint "$experiment_dir/best.pt" \
    --split train --max-cases 1 --output "$experiment_dir/preview_train"
python -u predict.py --device cuda --checkpoint "$experiment_dir/best.pt" \
    --split validation --max-cases 1 --output "$experiment_dir/preview_validation"

# One small diagnostic download: reports and figures, without checkpoint weights.
tar -czf "${experiment_dir}_diagnostics.tar.gz" -C "$experiment_dir" \
    config.json history.json validation_latest.json curves.png preview_train preview_validation
echo "DICE_CE_RESUME_AND_PREVIEWS_COMPLETED (test cases were not evaluated)."
