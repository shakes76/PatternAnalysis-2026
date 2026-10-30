#!/bin/bash
#SBATCH --job-name=flan-eval
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

set -euo pipefail

# Select evaluation method.
METHOD="${1:-}"

if [[ "$METHOD" != "zero_shot" && "$METHOD" != "full" && "$METHOD" != "lora" ]]; then
    echo "Usage: sbatch evaluate.slurm {zero_shot|full|lora}" >&2
    exit 1
fi

# Activate Conda environment.
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate torch

# Move to project directory.
cd "$HOME/comp3710/PatternAnalysis-2026/recognition/FLAN_T5_s5012476"

# Print job information.
echo "========================================"
echo "FLAN-T5 Evaluation"
echo "Method: $METHOD"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "========================================"

# Display GPU information.
nvidia-smi

# Verify CUDA availability.
python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable. Stopping evaluation.")

print("GPU:", torch.cuda.get_device_name(0))
PY

# Common evaluation configuration.
ARGS=(
    --method "$METHOD"
    --evaluate
    --max_eval_samples 0
    --eval_batch_size 8
    --max_new_tokens 128
    --num_beams 4
    --run_name final
)

# Set checkpoint for fine-tuned models.
if [[ "$METHOD" == "full" ]]; then

    CHECKPOINT="outputs/final/full/best_checkpoint"

    if [[ ! -f "$CHECKPOINT/model.safetensors" ]]; then
        echo "Error: Full FT checkpoint not found."
        exit 1
    fi

    ARGS+=(--checkpoint "$CHECKPOINT")

elif [[ "$METHOD" == "lora" ]]; then

    CHECKPOINT="outputs/final/lora/best_checkpoint"

    if [[ ! -f "$CHECKPOINT/adapter_model.safetensors" ]]; then
        echo "Error: LoRA checkpoint not found."
        exit 1
    fi

    ARGS+=(--checkpoint "$CHECKPOINT")

fi

# Run evaluation.
python -u predict.py "${ARGS[@]}"

echo "========================================"
echo "Evaluation completed"
echo "Method: $METHOD"
echo "End time: $(date)"
echo "========================================"
