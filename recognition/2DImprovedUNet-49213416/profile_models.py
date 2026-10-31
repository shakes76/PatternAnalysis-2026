
import time
 
import torch
 
from dataset import get_dataloaders, HIPMRI_ROOT, NUM_CLASSES
from modules import UNet2D, ImprovedUNet2D
from train import dice_ce_loss, predict_all, BATCH_SIZE, LR, EPOCHS, device
 
torch.manual_seed(35)
train_loader, val_loader, _ = get_dataloaders(HIPMRI_ROOT, BATCH_SIZE, early_stop=False)
 
print(f"\n| Model | Peak training VRAM (GB) | Time per epoch (s) | Estimated time, {EPOCHS} epochs (min) |")
print("|---|---|---|---|")
for name, model_class in [("baseline", UNet2D), ("improved", ImprovedUNet2D)]:
    model = model_class(in_channels=1, num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.time()
 
    model.train()
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        loss = dice_ce_loss(model(images), masks)
        loss.backward()
        optimizer.step()
    predict_all(model, val_loader)
 
    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.time() - start
    vram = torch.cuda.max_memory_allocated() / 1024 ** 3 if device.type == "cuda" else 0.0
    print(f"| {name} | {vram:.2f} | {seconds:.1f} | {seconds * EPOCHS / 60:.1f} |")
    del model, optimizer
 