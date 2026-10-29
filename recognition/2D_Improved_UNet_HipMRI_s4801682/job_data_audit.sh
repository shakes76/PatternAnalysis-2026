#!/bin/bash
#SBATCH --job-name=data_audit
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=data_audit_%j.out
#SBATCH --error=data_audit_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=s4801682@student.uq.edu.au

source ~/miniconda3/etc/profile.d/conda.sh
conda activate torch

# Print Execution Metadata
echo "-----------------------"
echo "Running Data Audit on node: $(hostname)"
echo "Start Time: $(date)"
echo "-----------------------"

python data_audit.py

echo "-----------------------"
echo "Data Audit finished at: $(date)"
echo "-----------------------"
