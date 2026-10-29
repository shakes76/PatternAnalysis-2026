#!/bin/bash
#SBATCH --job-name=HIPMRI_ERRORS
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --qos=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=3D_ERRORS_%j.out
#SBATCH --error=3D_ERRORS_%j.err

set -e

source $HOME/miniconda3/bin/activate
conda activate torch

cd $SLURM_SUBMIT_DIR

echo "Started: $(date)"

# C032
python -u -c "from predict import visualise_boundary_errors; visualise_boundary_errors('C032_Week0_LFOV.nii.gz', slice_mode='first')"
python -u -c "from predict import visualise_boundary_errors; visualise_boundary_errors('C032_Week0_LFOV.nii.gz', slice_mode='last')"
python -u -c "from predict import visualise_boundary_errors; visualise_boundary_errors('C032_Week0_LFOV.nii.gz', slice_mode='max_error')"

# R024
python -u -c "from predict import visualise_boundary_errors; visualise_boundary_errors('R024_Week2_LFOV.nii.gz', slice_mode='first')"
python -u -c "from predict import visualise_boundary_errors; visualise_boundary_errors('R024_Week2_LFOV.nii.gz', slice_mode='last')"
python -u -c "from predict import visualise_boundary_errors; visualise_boundary_errors('R024_Week2_LFOV.nii.gz', slice_mode='max_error')"

# W012
python -u -c "from predict import visualise_boundary_errors; visualise_boundary_errors('W012_Week0_LFOV.nii.gz', slice_mode='first')"
python -u -c "from predict import visualise_boundary_errors; visualise_boundary_errors('W012_Week0_LFOV.nii.gz', slice_mode='last')"
python -u -c "from predict import visualise_boundary_errors; visualise_boundary_errors('W012_Week0_LFOV.nii.gz', slice_mode='max_error')"

# M013
python -u -c "from predict import visualise_boundary_errors; visualise_boundary_errors('M013_Week0_LFOV.nii.gz', slice_mode='first')"
python -u -c "from predict import visualise_boundary_errors; visualise_boundary_errors('M013_Week0_LFOV.nii.gz', slice_mode='last')"
python -u -c "from predict import visualise_boundary_errors; visualise_boundary_errors('M013_Week0_LFOV.nii.gz', slice_mode='max_error')"

echo "Finished: $(date)"
