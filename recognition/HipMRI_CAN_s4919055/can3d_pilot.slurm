#!/bin/bash
# First whole-volume CAN3D pilot: three epochs, training + validation only.
#SBATCH --job-name=hipmri-can3d-pilot
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --output=runs/can3d_pilot_%j.out
#SBATCH --error=runs/can3d_pilot_%j.err

set -e
cd "$SLURM_SUBMIT_DIR"
source "$HOME/miniconda3/bin/activate"
conda activate torch
export OMP_NUM_THREADS=1
export MPLCONFIGDIR="$SLURM_SUBMIT_DIR/runs/mpl-cache"
output_dir="runs/can3d_pilot_${SLURM_JOB_ID}"
python -u train.py --model can3d --mode fit --device cuda --epochs 3 \
    --batch-size 1 --learning-rate 0.001 --seed 42 --loss dice-ce \
    --flip-probability 0.5 --manifest runs/split_seed42.json --output "$output_dir"
python -u predict.py --device cuda --checkpoint "$output_dir/best.pt" \
    --split train --max-cases 1 --output "$output_dir/preview_train"
python -u predict.py --device cuda --checkpoint "$output_dir/best.pt" \
    --split validation --max-cases 1 --output "$output_dir/preview_validation"
cp runs/can3d_smoke_709556/smoke.json "$output_dir/gpu_smoke_709556.json"
tar -czf "${output_dir}_diagnostics.tar.gz" -C "$output_dir" \
    config.json history.json validation_latest.json curves.png preview_train preview_validation gpu_smoke_709556.json
echo "CAN3D_PILOT_AND_PREVIEWS_COMPLETED (test cases were not evaluated)."
