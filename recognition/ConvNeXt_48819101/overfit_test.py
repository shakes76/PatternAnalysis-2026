# overfit_test.py
import torch
import torch.nn as nn
from dataset import get_dataloaders
from modules import ConvNeXt

device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    train_loader, _, _ = get_dataloaders(batch_size=64)
    x, y = next(iter(train_loader))
    x, y = x.to(device), y.float().to(device)

    # sanity check the data
    print("shape:", x.shape, "| mean:", x.mean().item(), "| std:", x.std().item(), "| label mean:", y.mean().item())

    model = ConvNeXt(in_channels=x.shape[1], num_classes=1, drop_path_rate=0.0).to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for i in range(150):
        opt.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(model(x).squeeze(1), y)
        loss.backward()
        opt.step()
        if i % 10 == 0:
            print(i, loss.item())

if __name__ == "__main__":
    main()