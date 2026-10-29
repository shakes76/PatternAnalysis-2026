#!/bin/bash
# Continue verified CAN3D709710 from epoch3 to8 with the same recipe.
#SBATCH --job-name=hipmri-can3d-resume
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=1
#SBATCH --output=runs/can3d_resume_%j.out
#SBATCH --error=runs/can3d_resume_%j.err

set -e
cd "$SLURM_SUBMIT_DIR"
source "$HOME/miniconda3/bin/activate"
conda activate torch
export OMP_NUM_THREADS=1
export MPLCONFIGDIR="$SLURM_SUBMIT_DIR/runs/mpl-cache"
test -f runs/can3d_pilot_709710/last.pt
test -f runs/can3d_pilot_709710/best.pt
output_dir="runs/can3d_resume_${SLURM_JOB_ID}"
python -u train.py --model can3d --mode fit --device cuda --epochs 8 \
    --batch-size 1 --learning-rate 0.001 --seed 42 --loss dice-ce \
    --flip-probability 0.5 --manifest runs/split_seed42.json \
    --resume runs/can3d_pilot_709710/last.pt --output "$output_dir"

# Recheck the selected weights, instead of mixing best and last metrics.
# The hash also ties the following previews to exactly this model file.
checkpoint_sha=$(sha256sum "$output_dir/best.pt" | cut -d ' ' -f 1)
python -u validate_best.py --device cuda --checkpoint "$output_dir/best.pt" \
    --expected-sha256 "$checkpoint_sha" --output "$output_dir/best_validation"
python -u predict.py --device cuda --checkpoint "$output_dir/best.pt" \
    --split train --max-cases 1 --output "$output_dir/preview_train"
python -u predict.py --device cuda --checkpoint "$output_dir/best.pt" \
    --split validation --max-cases 1 --output "$output_dir/preview_validation"
tar -czf "${output_dir}_diagnostics.tar.gz" -C "$output_dir" \
    config.json history.json validation_latest.json curves.png best_validation preview_train preview_validation
echo "CAN3D_RESUME_AND_BEST_VALIDATION_COMPLETED (test cases were not evaluated)."
