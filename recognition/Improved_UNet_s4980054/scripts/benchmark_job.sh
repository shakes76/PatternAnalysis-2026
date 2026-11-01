#!/bin/bash
#SBATCH --job-name=bench-unet
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=bench_%j.out
#SBATCH --error=bench_%j.err

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi
source $HOME/miniconda3/bin/activate
conda activate torch
cd $HOME/improved_unet
python benchmark.py --root /home/groups/comp3710/OASIS --model unet --base-channels 16 --num-workers 4
python benchmark.py --root /home/groups/comp3710/OASIS --model improved_unet --base-channels 16 --num-workers 4
