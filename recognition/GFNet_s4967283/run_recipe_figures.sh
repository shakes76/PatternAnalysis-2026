#!/bin/bash
# Rebuild the report figures and the calibration numbers for the recipe run.
#
#   sbatch run_recipe_figures.sh
#
# The best GFNet result is the recipe configuration at patch size 8 with depth
# 12, so the figures and every calibration number have to come from that
# checkpoint rather than the earlier one.
#
# BOTH shape-affecting arguments must reach predict.py. Patch size sets the
# token grid and depth sets the block count. Leave either out and the
# checkpoint refuses to load, which is the behaviour I want: a loud failure
# rather than a silently different model with random weights in four blocks.

#SBATCH --job-name=adni_recipe_fig
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e

source ~/venvs/comp3710/bin/activate

GFNET_CKPT=outputs_gfnet_seed42_strong_p8_recipe_mix/best_gfnet.pt
GFNET_ARGS="--model gfnet --checkpoint $GFNET_CKPT --patch-size 8 --depth 12"

echo "=== environment check ==="
which python
python -c "import torch; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())"
nvidia-smi --query-gpu=name --format=csv,noheader
echo "checkpoint: $GFNET_CKPT"
echo "args:       $GFNET_ARGS"
echo "========================="
echo

# Keep the old GFNet figures. The report quotes numbers that came from those
# npz files, and this run overwrites them.
if [ ! -d figures_pre_recipe ]; then
    cp -r figures figures_pre_recipe
    echo "backed up figures/ to figures_pre_recipe/"
fi

echo
echo ">>> 1 of 3: gfnet, test split, with figures"
python predict.py $GFNET_ARGS --split test --output-dir figures

echo
echo ">>> 2 of 3: gfnet, validation split, predictions only"
python predict.py $GFNET_ARGS --split validation --no-figures --output-dir figures

# The baseline predictions in figures/ are still current, so they are reused.
echo
echo ">>> 3 of 3: calibration, reject option and temperature scaling"
python evaluate_calibration.py \
    --predictions figures/test_predictions_baseline.npz figures/test_predictions_gfnet.npz \
    --val-predictions figures/validation_predictions_baseline.npz figures/validation_predictions_gfnet.npz \
    --names baseline gfnet --output-dir figures

echo
echo "=== done ==="
