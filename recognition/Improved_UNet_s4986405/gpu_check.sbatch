#!/bin/bash
#SBATCH --job-name=hipmri-cuda-check
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --output=gpu-check-%j.log

set -euo pipefail

hostname
nvidia-smi

"$HOME/miniconda3/envs/torch/bin/python" -u - <<'PY'
import torch

torch.set_num_threads(1)
print("PyTorch:", torch.__version__)
print("PyTorch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable on the allocated node")

print("GPU:", torch.cuda.get_device_name(0))

layer = torch.nn.Conv2d(1, 4, kernel_size=3, padding=1).cuda()
x = torch.randn(1, 1, 32, 16, device="cuda", requires_grad=True)
y = layer(x)
y.square().mean().backward()
torch.cuda.synchronize()

if tuple(y.shape) != (1, 4, 32, 16):
    raise RuntimeError(f"Unexpected output shape: {tuple(y.shape)}")
if not torch.isfinite(y).all():
    raise RuntimeError("Non-finite convolution output")

for name, tensor in [("input", x)] + list(layer.named_parameters()):
    if tensor.grad is None:
        raise RuntimeError(f"Missing gradient: {name}")
    if not torch.isfinite(tensor.grad).all():
        raise RuntimeError(f"Non-finite gradient: {name}")

print("CUDA convolution forward/backward: PASS")
PY
