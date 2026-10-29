#!/bin/bash
#SBATCH --job-name=dataset_check
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --output=dataset_%j.out
#SBATCH --error=dataset_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=s4801682@student.uq.edu.au

source ~/miniconda3/etc/profile.d/conda.sh
conda activate torch

# Point Linux to Conda's c++ librarys (fixes GLIBCXX error)
export LD_LIBRARY_PATH=$HOME/miniconda3/envs/torch/lib:$LD_LIBRARY_PATH

# Print Execution Metadata
echo "-----------------------"
echo "Running Dataset Pipeline Sanity Test: $(hostname)"
echo "Start Time: $(date)"
echo "-----------------------"

python dataset.py

echo "-----------------------"
echo "Dataset test finished at: $(date)"
echo "-----------------------"
