#!/bin/bash
# Result A: ConvNeXt trained on the course train folder as provided (--split-mode official).
# Same recipe as convnext_aug_adamw_run01; only the training data changes.
#SBATCH --job-name=convnext_official
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=convnext_official_%j.out
#SBATCH --error=convnext_official_%j.err
#SBATCH --signal=USR1@120   # train.py saves last.pth 2 minutes before the time limit

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi

source $HOME/miniconda3/bin/activate
conda activate ${CONDA_ENV_NAME:-torch117}   # override: sbatch --export=ALL,CONDA_ENV_NAME=torch run_official_split.slurm

RUN=checkpoints/convnext/convnext_official_aug_adamw_run01
if [ -f "$RUN/last.pth" ]; then
  python -u train.py --model convnext --split-mode official --epochs 8 --batch-size 8 \
    --learning-rate 0.0003 --optimizer adamw --weight-decay 0.03 --augment \
    --resume "$RUN/last.pth"
else
  python -u train.py --model convnext --split-mode official --epochs 8 --batch-size 8 \
    --learning-rate 0.0003 --optimizer adamw --weight-decay 0.03 --augment \
    --run-name convnext_official_aug_adamw_run01
fi
echo "Finished $(date)"
