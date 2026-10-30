#!/bin/bash
#SBATCH --job-name=flan-lora
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=logs/lora_%j.out
#SBATCH --error=logs/lora_%j.err

set -euo pipefail

# Activate the existing Conda environment.
source "$HOME/miniconda3/bin/activate"
conda activate torch

# Move to the FLAN-T5 project directory.
cd "$HOME/comp3710/PatternAnalysis-2026/recognition/FLAN_T5_s5012476"

# Record job information.
echo "========================================"
echo "FLAN-T5 LoRA Training"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "========================================"

# Display GPU information.
nvidia-smi

# Verify CUDA before starting training.
python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable. Stopping training.")

print("GPU:", torch.cuda.get_device_name(0))
PY

# Run LoRA fine-tuning.
python -u train.py \
    --method lora \
    --epochs 3 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --seed 42 \
    --max_train_steps 0 \
    --max_train_samples 0 \
    --max_val_samples 0 \
    --output_dir outputs/final

echo "========================================"
echo "Training completed: $(date)"
echo "========================================"
