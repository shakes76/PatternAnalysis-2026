#!/bin/bash
#SBATCH --job-name=adni_smoke
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source ~/venvs/comp3710/bin/activate

# Confirm the right interpreter and that the GPU is visible, before the real work.
# If the job dies, these three lines tell me why without any guessing.
which python
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
nvidia-smi --query-gpu=name,memory.total --format=csv

python train.py --model baseline --epochs 3
