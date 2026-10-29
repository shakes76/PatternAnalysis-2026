#!/bin/bash
# Full training run on one A100. Takes the model name and the seed as arguments
# so I do not need a separate script per run.
#
#   sbatch run_train.sh baseline 42
#   sbatch run_train.sh gfnet 42
#
# Arguments: MODEL SEED EPOCHS AUG PATCH SLICES
#   sbatch run_train.sh gfnet 42 80 normal 8      patch size 8, finer tokens
#   sbatch run_train.sh gfnet 42 80 normal 16 3   2.5D, three adjacent slices
#
# Full argument list: MODEL SEED EPOCHS AUG PATCH SLICES DEPTH DROPPATH WARMUP OPTIM
#
# The paper's training recipe, which is AdamW with decoupled weight decay,
# a linear warmup, stochastic depth, and the depth of GFNet-Ti:
#   sbatch run_train.sh gfnet 42 80 normal 16 1 12 0.1 5 adamw
#
# Full list: MODEL SEED EPOCHS AUG PATCH SLICES DEPTH DROPPATH WARMUP OPTIM MIXUP SMOOTH
#
# The anti-memorisation run. The training history showed train accuracy at
# 0.9954 against validation at 0.8649, so the model had memorised the training
# set. Mixup stops it seeing the same slice twice and label smoothing stops it
# driving its outputs to 0 and 1:
#   sbatch run_train.sh gfnet 42 80 strong 8 1 12 0.3 5 adamw 0.8 0.1
#
# For the three-seed comparison later:
#   for s in 42 1 2; do sbatch run_train.sh baseline $s; done
#   for s in 42 1 2; do sbatch run_train.sh gfnet $s; done
#
# Each run writes into its own output folder, outputs_<model>_seed<seed>, so
# runs never overwrite each other's weights or metrics.

#SBATCH --job-name=adni_train
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# SLURM settings, all learned the hard way on Demo 2:
#   - comp3710 is the partition allocated to this course. The shared a100
#     partition puts me behind cosc3500 and a100-grind users on the same nodes.
#   - --gres must be typed as gpu:a100:1. A bare gpu:1 does not match.
#   - No --mem line at all. Slurm reports RealMemory=1 on these nodes, so any
#     memory request gets the job rejected.
#   - --time=02:00:00 because the smoke test measured 16 s/epoch for the
#     baseline, and GFNet should land somewhere near 90 s/epoch. Thirty epochs
#     of GFNet is therefore well under an hour, and two hours leaves room for a
#     slow queue without over-reserving. The partition default is three hours,
#     which blocks backfill and schedules later.

MODEL=${1:-baseline}
SEED=${2:-42}
EPOCHS=${3:-30}
AUG=${4:-normal}
PATCH=${5:-16}
SLICES=${6:-1}
DEPTH=${7:-8}
DROPPATH=${8:-0.0}
WARMUP=${9:-0}
OPTIM=${10:-adam}
MIXUP=${11:-0.0}
SMOOTH=${12:-0.0}

# Patch size only affects GFNet. 16 gives a 14x14 token grid, which rfft2 turns
# into 14x8, so 112 frequency components. 8 gives 28x28 and therefore 28x15,
# which is 420 components, at roughly four times the compute per epoch.
#
# Runs at the default patch size keep the old folder names, so nothing that
# already exists is overwritten. Anything else gets a _p<size> suffix.
OUTPUT_DIR="outputs_${MODEL}_seed${SEED}_${AUG}"
if [ "$PATCH" != "16" ]; then
    OUTPUT_DIR="${OUTPUT_DIR}_p${PATCH}"
fi
if [ "$SLICES" != "1" ]; then
    OUTPUT_DIR="${OUTPUT_DIR}_s${SLICES}"
fi
# The paper's training recipe gets its own folder name, so the earlier results
# stay exactly where they are.
if [ "$OPTIM" != "adam" ] || [ "$DEPTH" != "8" ] || [ "$DROPPATH" != "0.0" ]; then
    OUTPUT_DIR="${OUTPUT_DIR}_recipe"
fi
# Mixup and label smoothing get their own folder too, so the runs without them
# keep the numbers already in the report.
if [ "$MIXUP" != "0.0" ] || [ "$SMOOTH" != "0.0" ]; then
    OUTPUT_DIR="${OUTPUT_DIR}_mix"
fi

# A fresh shell on the compute node does not carry my activated venv.
source ~/venvs/comp3710/bin/activate

echo "=== environment check ==="
which python
python -c "import torch; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "model=$MODEL seed=$SEED epochs=$EPOCHS aug=$AUG patch=$PATCH slices=$SLICES"
echo "depth=$DEPTH drop_path=$DROPPATH warmup=$WARMUP optimiser=$OPTIM"
echo "mixup=$MIXUP label_smoothing=$SMOOTH"
echo "output=$OUTPUT_DIR"
echo "========================="
echo

# Learning rate 3e-4 rather than the 1e-3 I used in the smoke test. At 1e-3 the
# baseline's validation loss went 0.708 -> 0.462 -> 1.247 over three epochs and
# the model fell back to predicting a single class, which is the signature of
# steps that are too large rather than of overfitting.
python train.py \
    --model "$MODEL" \
    --epochs "$EPOCHS" \
    --batch-size 64 \
    --learning-rate 3e-4 \
    --seed "$SEED" \
    --num-workers 8 \
    --augmentation "$AUG" \
    --weight-decay 1e-3 \
    --patch-size "$PATCH" \
    --slice-context "$SLICES" \
    --depth "$DEPTH" \
    --drop-path "$DROPPATH" \
    --warmup-epochs "$WARMUP" \
    --optimiser "$OPTIM" \
    --mixup "$MIXUP" \
    --label-smoothing "$SMOOTH" \
    --output-dir "$OUTPUT_DIR"
