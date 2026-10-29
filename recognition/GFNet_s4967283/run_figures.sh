#!/bin/bash
# Regenerate every report figure from the strong-augmentation checkpoints.
#
#   sbatch run_figures.sh
#
# Why this script exists
# ----------------------
# The figures in figures/ came from the 80-epoch normal-augmentation runs. The
# README now reports the strong-augmentation numbers. A figure that disagrees
# with the table beside it is worse than no figure, so every one is rebuilt
# from the strong-augmentation checkpoints.
#
# This also rebuilds failures_baseline.png and failures_gfnet.png, which are
# the input to the error autopsy section.
#
# Five python commands share one GPU allocation. Each one on its own would
# wait in the queue separately, so one job is both faster and politer.

#SBATCH --job-name=adni_figures
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Inference only, no training. Measured at 12,415 img/s for the baseline and
# 5,672 img/s for GFNet, so 9,000 slices takes about a second of compute. Data
# loading dominates. Thirty minutes is generous and keeps the job eligible for
# backfill.

set -e  # Stop at the first failure. A half-rebuilt figures folder is worse
        # than a failed job, because the mix of old and new is invisible.

source ~/venvs/comp3710/bin/activate

BASE_CKPT=outputs_baseline_seed42_strong/best_baseline.pt
GFNET_CKPT=outputs_gfnet_seed42_strong/best_gfnet.pt

echo "=== environment check ==="
which python
python -c "import torch; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())"
nvidia-smi --query-gpu=name --format=csv,noheader
echo "baseline checkpoint: $BASE_CKPT"
echo "gfnet checkpoint:    $GFNET_CKPT"
echo "========================="
echo

echo ">>> 1 of 5: baseline, test split, with figures"
python predict.py --model baseline --checkpoint "$BASE_CKPT" \
    --split test --output-dir figures

echo
echo ">>> 2 of 5: gfnet, test split, with figures"
python predict.py --model gfnet --checkpoint "$GFNET_CKPT" \
    --split test --output-dir figures

# The validation predictions exist only to fit the temperature. No figures are
# needed from them, so --no-figures keeps the folder clean.
echo
echo ">>> 3 of 5: baseline, validation split, predictions only"
python predict.py --model baseline --checkpoint "$BASE_CKPT" \
    --split validation --no-figures --output-dir figures

echo
echo ">>> 4 of 5: gfnet, validation split, predictions only"
python predict.py --model gfnet --checkpoint "$GFNET_CKPT" \
    --split validation --no-figures --output-dir figures

echo
echo ">>> 5 of 5: calibration, reject option and temperature scaling"
python evaluate_calibration.py \
    --predictions figures/test_predictions_baseline.npz figures/test_predictions_gfnet.npz \
    --val-predictions figures/validation_predictions_baseline.npz figures/validation_predictions_gfnet.npz \
    --names baseline gfnet --output-dir figures

echo
echo "=== done. figures/ now matches the strong-augmentation runs ==="
ls -la figures/
