
import time

import torch
import torch.nn as nn

from dataset import get_data_pairs, split_by_patient, HipMRIDataset
from modules import UNet3D

# 只用 一份真实 MRI 做一次参数更新，不是完整训练。GPU 显存数值是 PyTorch 统计的峰值张量分配量，不代表 GPU 的全部占用显存
def main():
    # 1. Prepare one real MRI-Mask pair
    pairs = get_data_pairs()
    train, val, test = split_by_patient(pairs)

    dataset = HipMRIDataset(train)
    image, mask = dataset[0]

    # 2. Add batch dimension
    image = image.unsqueeze(0)
    mask = mask.unsqueeze(0)

    print("Input:", image.shape)
    print("Target:", mask.shape)

    # 3. Select GPU
    if not torch.cuda.is_available():
        raise RuntimeError("This test requires a GPU node.")

    device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0))

    image = image.to(device)
    mask = mask.to(device)

    # 4. Create standard 3D U-Net
    model = UNet3D(
        in_channels=1,
        num_classes=6,
        base_channels=8
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # 5. Forward pass
    model.train()
    optimizer.zero_grad(set_to_none=True)

    #start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()

    output = model(image)
    loss = criterion(output, mask)

    print("Output:", output.shape)
    print("Loss:", loss.item())

    # 6. Backward and one optimizer step
    loss.backward()
    optimizer.step()

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    print("Step time (seconds):", round(elapsed, 3))
    print(
        "Peak GPU allocated memory (GB):",
        round(torch.cuda.max_memory_allocated() / 1024**3, 3)
    )
    print("Real-data forward/backward: OK")


if __name__ == "__main__":
    main()
