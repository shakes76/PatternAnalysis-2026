#!/bin/bash
# Measure what test-time augmentation buys on the best GFNet checkpoint.
#
#   sbatch run_tta.sh
#
# Every slice is scored twice, as it is and mirrored left to right, and the two
# probability vectors are averaged. No retraining and no change to the weights,
# so this costs one extra forward pass per batch.
#
# predict.py prints the accuracy with and without the averaging, so the gain is
# measured rather than assumed. It writes to its own output folder, which keeps
# the untouched numbers in figures/ intact until the gain is known to be worth
# having.

#SBATCH --job-name=adni_tta
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e

source ~/venvs/comp3710/bin/activate

# The 80-epoch run is the best GFNet result at 0.7450. The 150-epoch version of
# the same configuration overwrote this folder name, so the 80-epoch weights
# live under the _80ep suffix.
CKPT=outputs_gfnet_seed42_normal_p8_recipe_80ep/best_gfnet.pt
ARGS="--model gfnet --checkpoint $CKPT --patch-size 8 --depth 12"

echo "=== environment check ==="
which python
nvidia-smi --query-gpu=name --format=csv,noheader
echo "checkpoint: $CKPT"
echo "========================="

if [ ! -f "$CKPT" ]; then
    echo "MISSING: $CKPT"
    echo "Available GFNet output folders:"
    ls -d outputs_gfnet* 2>/dev/null
    exit 1
fi

mkdir -p figures_tta

echo
echo ">>> baseline for comparison: the same checkpoint with no averaging"
python predict.py $ARGS --split test --no-figures --output-dir figures_tta

echo
echo ">>> with test-time augmentation"
python predict.py $ARGS --split test --tta --no-figures --output-dir figures_tta

echo
echo "=== done ==="
