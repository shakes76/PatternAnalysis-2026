#!/bin/bash
#SBATCH --job-name=adni-smoke-test
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:05:00
#SBATCH --output=smoke_test_%j.out
#SBATCH --error=smoke_test_%j.err

source "$HOME/miniconda3/bin/activate"
conda activate torch
set -e

cd "$HOME/comp3710_adni_project"
python -u smoke_test.py