#!/bin/bash
#SBATCH --job-name=labels_audit
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=label_%j.out
#SBATCH --error=label_%j.err
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

python labels_audit.py

echo "-----------------------"
echo "Verification finished at: $(date)"
echo "-----------------------"
