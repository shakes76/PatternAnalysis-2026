#!/bin/bash
#SBATCH --job-name=eval-validate
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=eval_validate_%j.out
#SBATCH --error=eval_validate_%j.err

set -euo pipefail
source "$HOME/miniconda3/bin/activate"
conda activate torch
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
run_dir="runs/eval_validate_${SLURM_JOB_ID}"
mkdir -p "$run_dir"
mkdir "$run_dir/source"
cp evaluate_models.py modules.py dataset.py metrics.py check_final_evaluation.sh "$run_dir/source/"
python -m pip freeze > "$run_dir/environment.txt"
nvidia-smi > "$run_dir/gpu_info.txt"
python -u "$run_dir/source/evaluate_models.py" \
    --standard runs/ce_dice_711775/results/best.pt \
    --improved runs/improved_712829/results/best.pt \
    --split validate --repeats 3 \
    --output "$run_dir/results"
