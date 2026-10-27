#!/bin/bash
#SBATCH --job-name=adni-gpu-check
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:05:00
#SBATCH --output=gpu_check_%j.out
#SBATCH --error=gpu_check_%j.err

source "$HOME/miniconda3/bin/activate"
conda activate torch
set -e

hostname
nvidia-smi

python -u - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise SystemExit("GPU unavailable in this job")

props = torch.cuda.get_device_properties(0)
print("GPU:", props.name)
print("GPU memory GiB:", round(props.total_memory / 1024**3, 2))

x = torch.randn(512, 512, device="cuda")
y = x @ x
torch.cuda.synchronize()

print("GPU calculation passed:", torch.isfinite(y).all().item())
PY