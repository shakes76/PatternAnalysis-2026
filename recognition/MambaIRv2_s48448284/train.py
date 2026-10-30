"""
Training the models
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
 
import torch
import torch.optim as optim
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

import lpips as lpips_lib
 
from dataset import get_dataloaders, denormalise_L, denormalise_ab, lab_to_rgb
from modules import MambaIRv2Frozen, MambaIRv2FineTuned, L1_Saturation_Loss, LPIPS_Loss
from predict import visualise_results


matplotlib.use('Agg')

MODELS_DIR  = Path("recognition/MambaIRv2_s48448284/models")
FIGURES_DIR = Path("recognition/MambaIRv2_s48448284/figures")
CKPT_DIR    = Path("recognition/MambaIRv2_s48448284/checkpoints")
RESULTS_DIR = Path("recognition/MambaIRv2_s48448284/results")
 
PRETRAINED_CKPT = Path("recognition/MambaIRv2_s48448284/checkpoints/mambairv2_lightSR_x2.pth") 


# Evaluation functions

def compute_metrics(L, ab_true, ab_pred):
    L_d = denormalise_L(L)
    ab_true_d = denormalise_ab(ab_true)
    ab_pred_d = denormalise_ab(ab_pred)

    psnrs, ssims = [],[]

    for i in range(L_d.shape[0]):
        # Build HxWx3 Lab arrays
        lab_true = np.concatenate([L_d[i], ab_true_d[i]],      axis=0).transpose(1, 2, 0)
        lab_pred = np.concatenate([L_d[i], ab_pred_d[i]], axis=0).transpose(1, 2, 0)
 
        rgb_true = lab_to_rgb(lab_true[..., 0], lab_true[..., 1:])
        rgb_pred = lab_to_rgb(lab_pred[..., 0], lab_pred[..., 1:])
 
        psnrs.append(psnr_fn(rgb_true, rgb_pred))
        ssims.append(ssim_fn(rgb_true, rgb_pred, channel_axis=-1))
 
    return psnrs, ssims

def compute_colorfulness(rgb):
    R = rgb[:, :, 0].astype(float)
    G = rgb[:, :, 1].astype(float)
    B = rgb[:, :, 2].astype(float)
    rg = R - G
    yb = 0.5 * (R + G) - B
    std_root  = np.sqrt(rg.std()**2  + yb.std()**2)
    mean_root = np.sqrt(rg.mean()**2 + yb.mean()**2)
    return std_root + 0.3 * mean_root

def evaluate(model, test_loader, device, run_name):
    lpips_fn = lpips_lib.LPIPS(net='alex').to(device)
    lpips_fn.eval()
 
    model.eval()
    psnr_scores, ssim_scores, lpips_scores = [], [], []
    colorfulness_pred, colorfulness_true   = [], []
 
    with torch.no_grad():
        for L, ab in test_loader:
            L, ab = L.to(device), ab.to(device)
            ab_pred = model(L)
 
            L_d       = denormalise_L(L.cpu().numpy())
            ab_d      = denormalise_ab(ab.cpu().numpy())
            ab_pred_d = denormalise_ab(ab_pred.cpu().numpy())
 
            for i in range(L.size(0)):
                lab_true = np.concatenate([L_d[i], ab_d[i]],      axis=0).transpose(1, 2, 0)
                lab_pred = np.concatenate([L_d[i], ab_pred_d[i]], axis=0).transpose(1, 2, 0)
 
                rgb_true = lab_to_rgb(lab_true[..., 0], lab_true[..., 1:])
                rgb_pred = lab_to_rgb(lab_pred[..., 0], lab_pred[..., 1:])
 
                psnr_scores.append(psnr_fn(rgb_true, rgb_pred))
                ssim_scores.append(ssim_fn(rgb_true, rgb_pred, channel_axis=-1))
                colorfulness_pred.append(compute_colorfulness(rgb_pred))
                colorfulness_true.append(compute_colorfulness(rgb_true))
 
                # LPIPS expects (1,3,H,W) in [-1,1]
                def to_lpips(img):
                    t = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)
                    return (t / 127.5 - 1.0).to(device)
                lpips_scores.append(lpips_fn(to_lpips(rgb_true), to_lpips(rgb_pred)).item())
 
    metrics = {
        'test_psnr': float(np.mean(psnr_scores)),
        'test_ssim': float(np.mean(ssim_scores)),
        'test_lpips': float(np.mean(lpips_scores)),
        'test_colorfulness_pred': float(np.mean(colorfulness_pred)),
        'test_colorfulness_true': float(np.mean(colorfulness_true)),
    }
 
    print(f"  PSNR: {metrics['test_psnr']:.4f} dB")
    print(f"  SSIM: {metrics['test_ssim']:.4f}")
    print(f"  LPIPS: {metrics['test_lpips']:.4f}  (lower is better)")
    print(f"  Colorfulness pred: {metrics['test_colorfulness_pred']:.2f}")
    print(f"  Colorfulness true: {metrics['test_colorfulness_true']:.2f}")
 
    # Save per-run test results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{run_name}_test_metrics.json"
    with open(out, 'w') as f:
        json.dump(metrics, f, indent=2)
 
    return metrics


def save_training_plot(history: dict, run_name: str):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
 
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Training curves — {run_name}", fontsize=14)
 
    metrics = [
        ('train_loss', 'val_loss',  'Loss',  'Loss'),
        ('val_psnr',  None, 'Val PSNR', 'dB'),
        ('val_ssim',  None, 'Val SSIM', ''),
        ('epoch_time', None, 'Epoch time', 's'),
    ]
 
    for ax, (train_key, val_key, title, ylabel) in zip(axes.flat, metrics):
        if train_key in history:
            ax.plot(history[train_key], label='train')
        if val_key and val_key in history:
            ax.plot(history[val_key], label='val')
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)
 
    plt.tight_layout()
    out = FIGURES_DIR / f"training_{run_name}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved training plot → {out}")

 


# Training loop

def train(
        frozen=True, 
        loss="l1_sat", 
        batch_size=4, 
        num_workers=4,
        lr=1e-4,
        epochs=50,
        resume = None,
        log_every = 5
        ):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
 
    run_name  = f"{'frozen' if frozen else 'fruit_finetuned'}_{loss}"
    best_path = MODELS_DIR / f"{run_name}_best.pth"
    last_path = MODELS_DIR / f"{run_name}_last.pth"

    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
    )

    pretrained = str(PRETRAINED_CKPT) if PRETRAINED_CKPT.exists() else None
    if pretrained is None:
        print("  WARNING: pretrained checkpoint not found — initialising from scratch.")
        print(f"  Expected at: {PRETRAINED_CKPT}")
        print(" wget https://github.com/csguoh/MambaIR/releases/download/v1.0/mambairv2_lightSR_x2.pth -o recognition/MambaIRv2_s48448284/checkpoints/mambairv2_lightSR_x2.pth")
 


    ModelClass = MambaIRv2Frozen if frozen else MambaIRv2FineTuned
    model = ModelClass(checkpoint_path=pretrained, device=device)

    counts = model.parameter_counts()
    print(f"  Model: {ModelClass.__name__}")
    print(f"  Total params:     {counts['total']:,}")
    print(f"  Trainable params: {counts['trainable']:,}")
    print(f"  Frozen params:    {counts['frozen']:,}")

    criterion = (L1_Saturation_Loss() if loss=="l1_sat" else LPIPS_Loss(device=device))

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable, lr=lr)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)


    start_epoch   = 0
    best_val_loss = float('inf')
    history = {
        'train_loss': [], 'val_loss': [],
        'val_psnr':   [], 'val_ssim': [],
        'epoch_time': [],
    }
 
    if resume:
        resume_path = Path(resume)
        if resume_path.exists():
            print(f"  Resuming from {resume_path}")
            ckpt = torch.load(resume_path, map_location=device)
            model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            scheduler.load_state_dict(ckpt['scheduler'])
            start_epoch   = ckpt['epoch'] + 1
            best_val_loss = ckpt.get('best_val_loss', float('inf'))
            history       = ckpt.get('history', history)
            print(f"  Resumed at epoch {start_epoch}, best val loss so far: {best_val_loss:.6f}")
        else:
            print(f"  WARNING: resume path not found ({resume_path}), starting fresh.")
 
    # ── Training loop
    print(f"\nTraining {run_name} for {epochs} epochs...\n")

    for epoch in range(start_epoch, epochs):
        t0 = time.time()

        model.train()
        train_loss = 0
        for batch_idx, (L, ab) in enumerate(train_loader):
            L, ab = L.to(device), ab.to(device)
            torch.cuda.synchronize()

            optimizer.zero_grad(set_to_none=True)

            ab_pred = model(L)

            torch.cuda.synchronize()

            batch_loss = criterion(ab_pred, ab)
            train_loss += batch_loss.item() * L.size(0)
            batch_loss.backward()
            optimizer.step()


        train_loss/= len(train_loader.dataset)
        scheduler.step()

        # validate
        model.eval()
        val_loss = 0.0
        all_psnr, all_ssim = [], []

        with torch.no_grad():
            for L, ab in val_loader:
                L, ab = L.to(device), ab.to(device)
                ab_pred = model(L)
                batch_loss = criterion(ab_pred, ab)
                val_loss+= batch_loss.item() * L.size(0)

                psnrs, ssims = compute_metrics(
                    L.cpu().numpy(),
                    ab.cpu().numpy(),
                    ab_pred.cpu().numpy(),
                )
                all_psnr.extend(psnrs)
                all_ssim.extend(ssims)

        val_loss /= len(val_loader.dataset)
        val_psnr  = float(np.mean(all_psnr))
        val_ssim  = float(np.mean(all_ssim))
        elapsed   = time.time() - t0
 
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_psnr'].append(val_psnr)
        history['val_ssim'].append(val_ssim)
        history['epoch_time'].append(elapsed)
 
        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'history': history,
                'args': {'frozen': frozen, 'loss': loss, 'epochs': epochs, 'lr': lr, 'batch_size': batch_size},
            }, best_path)
 
        # ── Print progress
        if (epoch + 1) % log_every == 0 or epoch == 0:
            lr_now = scheduler.get_last_lr()[0]
            print(
                f"Epoch [{epoch+1:3d}/{epochs}] "
                f"train={train_loss:.5f}  val={val_loss:.5f}  "
                f"PSNR={val_psnr:.2f}  SSIM={val_ssim:.4f}  "
                f"lr={lr_now:.2e}  t={elapsed:.1f}s"
            )

    print("Done training!")
    
    torch.save({
        'epoch': epochs - 1,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'best_val_loss': best_val_loss,
        'history': history,
        'args': {'frozen': frozen, 'loss': loss, 'epochs': epochs, 'lr': lr, 'batch_size': batch_size},
    }, last_path)
    
    history_path = MODELS_DIR / f"history_{run_name}.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)


    save_training_plot(history, run_name)

    ckpt_best = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt_best['model'])
    evaluate(model, test_loader=test_loader, device=device, run_name=f"{run_name}_A")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        'run': run_name,
        'frozen': frozen,
        'loss': loss,
        'epochs': epochs,
        'lr': lr,
        'batch_size': batch_size,
        'best_val_loss': best_val_loss,
        'final_val_loss': history['val_loss'][-1],
        'final_val_psnr': history['val_psnr'][-1],
        'final_val_ssim': history['val_ssim'][-1],
        'best_val_psnr': max(history['val_psnr']),
        'best_val_ssim': max(history['val_ssim']),
        'total_train_time_s': sum(history['epoch_time']),
        'mean_epoch_time_s': float(np.mean(history['epoch_time'])),
    }
    results_path = RESULTS_DIR / f"{run_name}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
 
    return model, history


if __name__ == "__main__":
    loss_name = "l1_sat"
    frozen = False
    resume = "recognition/MambaIRv2_s48448284/models/fruit_finetuned_lpips.pth"
    for i in range(1):
       
        loss_name = "lpips"
        resume = None


        
        model, history = train(frozen, loss_name, batch_size=1, num_workers=0, epochs= 25, resume= resume)

        final_psnr = history['val_psnr'][-1]
        final_ssim = history['val_ssim'][-1]
        best_psnr  = max(history['val_psnr'])
        print(f"\n{'='*55}")
        model_str = 'frozen' if frozen else 'fruit_finetuned'
        print(f"Run: {model_str}_{loss_name}")
        print(f"Final PSNR: {final_psnr:.4f} dB   Best PSNR: {best_psnr:.4f} dB")
        print(f"Final SSIM: {final_ssim:.4f}")
        print(f"{'='*55}")


        _, _, test_loader = get_dataloaders()
        visualise_results(model, test_loader, "recognition/MambaIRv2_s48448284/models/fruit_finetuned_lpips.pth", save_path="recognition/MambaIRv2_s48448284/figures")
        

    print("ALLLLL DONE!")

 

