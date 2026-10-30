#!/bin/bash
# Official-split ConvNeXt with a longer training recipe:
# 30 epochs, 1-epoch warmup + per-step cosine LR, EMA weights (0.999),
# label smoothing 0.1, same mild augmentation as before.
# Choose input with INPUT_MODE (single | context3) and the folder with RUN_NAME:
#   sbatch --export=ALL,INPUT_MODE=single,RUN_NAME=official_cosine_ema_single_s42 run_recipe.slurm
# Resubmit the identical command to resume after a time limit.
#SBATCH --job-name=convnext_recipe
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=recipe_%j.out
#SBATCH --error=recipe_%j.err
#SBATCH --signal=USR1@120

INPUT_MODE=${INPUT_MODE:-single}
SEED=${SEED:-42}
RUN_NAME=${RUN_NAME:-official_cosine_ema_${INPUT_MODE}_s${SEED}}
echo "Job $SLURM_JOB_ID on $(hostname), started $(date); input=$INPUT_MODE seed=$SEED run=$RUN_NAME"
nvidia-smi

source $HOME/miniconda3/bin/activate
conda activate ${CONDA_ENV_NAME:-torch117}

ARGS="--model convnext --split-mode official --input-mode $INPUT_MODE \
  --epochs 30 --batch-size 8 --learning-rate 0.0003 --optimizer adamw --weight-decay 0.03 \
  --augment --scheduler cosine --warmup-epochs 1 --ema-decay 0.999 --label-smoothing 0.1 \
  --num-workers 4 --seed $SEED"

RUN=checkpoints/convnext/$RUN_NAME
if [ -f "$RUN/last.pth" ]; then
  python -u train.py $ARGS --resume "$RUN/last.pth"
else
  python -u train.py $ARGS --run-name "$RUN_NAME"
fi
echo "Finished $(date)"
