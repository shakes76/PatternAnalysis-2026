#!/bin/bash
#SBATCH --job-name=run_verification
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=verify_%j.out
#SBATCH --error=verify_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=s4801682@student.uq.edu.au

source ~/miniconda3/etc/profile.d/conda.sh
conda activate torch

# Point Linux to Conda's c++ librarys (fixes GLIBCXX error)
export LD_LIBRARY_PATH=$HOME/miniconda3/envs/torch/lib:$LD_LIBRARY_PATH

# Print Execution Metadata
echo "-----------------------"
echo "Running Feasibility verification on node: $(hostname)"
echo "Start Time: $(date)"
echo "-----------------------"

python run_verification.py

echo "-----------------------"
echo "Verification finished at: $(date)"
echo "-----------------------"
