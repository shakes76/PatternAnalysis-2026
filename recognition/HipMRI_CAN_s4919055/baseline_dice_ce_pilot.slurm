#!/bin/bash
# Change only the objective relative to the fresh balanced-slice three-epoch pilot.
# Dice and categorical cross-entropy are covered in Contact 6/7; their sum is an adaptation.
#SBATCH --job-name=hipmri-dice-ce-pilot
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --output=runs/baseline_dice_ce_%j.out
#SBATCH --error=runs/baseline_dice_ce_%j.err

set -e
cd "$SLURM_SUBMIT_DIR"
source "$HOME/miniconda3/bin/activate"
conda activate torch
export OMP_NUM_THREADS=1
export MPLCONFIGDIR="$SLURM_SUBMIT_DIR/runs/mpl-cache"

# Fresh seed42 model, same data/grid/sampling/Adam; preserve all earlier checkpoints.
python -u train.py --mode fit --device cuda --epochs 3 --batch-size 8 \
    --learning-rate 0.001 --seed 42 --slice-sampling prostate-balanced --loss dice-ce \
    --manifest runs/split_seed42.json --output "runs/baseline_dice_ce_${SLURM_JOB_ID}"
