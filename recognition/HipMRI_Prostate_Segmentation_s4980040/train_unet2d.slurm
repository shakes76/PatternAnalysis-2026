#!/bin/bash
# Slurm job script: trains the 2D U-Net on a Rangpur GPU.
# Submit with:  sbatch train_unet2d.slurm
# Check it with: squeue -u $USER      Live log:  tail -f logs/unet2d_<jobid>.out
# Note: no --mem request, because the comp3710 partition rejects memory requests.

#SBATCH --job-name=unet2d
#SBATCH --partition=comp3710        # the course's GPU partition (A100s)
#SBATCH --gres=gpu:1                # ask for 1 GPU
#SBATCH --cpus-per-task=4           # CPU cores for data loading (matches --num-workers)
#SBATCH --time=02:00:00             # max run time before Slurm kills the job
#SBATCH --output=logs/%x_%j.out     # %x = job name, %j = job ID

# Make `conda activate` work inside a batch job, then switch to our env
source ~/miniforge3/etc/profile.d/conda.sh
conda activate comp3710

cd ~/PatternAnalysis-2026/recognition/HipMRI_Prostate_Segmentation_s4980040
nvidia-smi   # prints which GPU we got into the log
python train.py --model unet2d --epochs 30 --batch-size 16 --preload