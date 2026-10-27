#!/bin/bash
#SBATCH --job-name=adni-baseline-bench
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:05:00
#SBATCH --output=baseline_bench_%j.out
#SBATCH --error=baseline_bench_%j.err

source "$HOME/miniconda3/bin/activate"
conda activate torch
set -e

cd "$HOME/comp3710_adni_project"
python -u benchmark_baseline.py