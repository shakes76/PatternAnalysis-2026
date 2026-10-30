#!/bin/bash
#SBATCH --job-name=flan-full
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=logs/full_%j.out
#SBATCH --error=logs/full_%j.err

set -euo pipefail

# Activate the Conda environment.
source "$HOME/miniconda3/bin/activate"
conda activate torch

# Move to the project directory.
cd "$HOME/comp3710/PatternAnalysis-2026/recognition/FLAN_T5_s5012476"

# Record job information.
echo "========================================"
echo "FLAN-T5 Full Fine-Tuning"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "========================================"

# Display GPU information.
nvidia-smi

# Verify CUDA before training.
python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable. Stopping training.")

print("GPU:", torch.cuda.get_device_name(0))
PY

# Run Full Fine-Tuning.
python -u train.py \
    --method full \
    --epochs 3 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --seed 42 \
    --max_train_steps 0 \
    --max_train_samples 0 \
    --max_val_samples 0 \
    --output_dir outputs/final

echo "========================================"
echo "Full Fine-Tuning completed: $(date)"
echo "========================================"
