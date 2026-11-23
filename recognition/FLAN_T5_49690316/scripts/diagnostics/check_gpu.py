import torch


print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable in this GPU job")

print("GPU:", torch.cuda.get_device_name(0))

values = torch.tensor([1.0, 2.0, 3.0], device="cuda")
result = values.square()

print("GPU calculation:", result.cpu().tolist())