import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn
from pathlib import Path
import matplotlib.pyplot as plt

from dataset import get_dataloaders, denormalise_L, denormalise_ab, lab_to_rgb, rgb_to_lab, normalise_L, normalise_ab

import lpips

loss_fn_lpips = lpips.LPIPS(net='alex')

# I just copy pasted all the code from baseline testing

def compute_lpips(rgb_true: np.ndarray, rgb_pred: np.ndarray, device='cpu') -> float:
    # LPIPS expects (1, 3, H, W) float tensors in [-1, 1]
    def to_tensor(img):
        t = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
        return t.to(device)

    with torch.no_grad():
        score = loss_fn_lpips(to_tensor(rgb_true), to_tensor(rgb_pred))
    return score.item()


def compute_colorfulness(rgb: np.ndarray) -> float:
    R = rgb[:, :, 0].astype(float)
    G = rgb[:, :, 1].astype(float)
    B = rgb[:, :, 2].astype(float)

    rg = R - G
    yb = 0.5 * (R + G) - B

    std_rg,  mean_rg  = rg.std(),  rg.mean()
    std_yb,  mean_yb  = yb.std(),  yb.mean()

    std_root  = np.sqrt(std_rg**2  + std_yb**2)
    mean_root = np.sqrt(mean_rg**2 + mean_yb**2)

    return std_root + 0.3 * mean_root

def test_model(model, test_loader, checkpoint_path, device='cuda'):
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("model", ckpt)  # falls back to ckpt itself if no "model" key
    model.load_state_dict(state_dict)    
    model = model.to(device)
    model.eval()

    psnr_scores, ssim_scores, lpips_scores = [], [], []
    colorfulness_pred, colorfulness_true = [], []
    with torch.no_grad():
        for L, ab in test_loader:
            L, ab = L.to(device), ab.to(device)
            ab_pred = model(L)

            # Calculate PSNR and SSIM
            L_denorm = denormalise_L(L.cpu().numpy())
            ab_denorm = denormalise_ab(ab.cpu().numpy())
            ab_pred_denorm = denormalise_ab(ab_pred.cpu().numpy())

            for i in range(L.size(0)):
                lab_true = np.concatenate([L_denorm[i], ab_denorm[i]], axis=0).transpose(1, 2, 0)
                lab_pred = np.concatenate([L_denorm[i], ab_pred_denorm[i]], axis=0).transpose(1, 2, 0)
                
                rgb_true = lab_to_rgb(lab_true[..., 0], lab_true[..., 1:])
                rgb_pred = lab_to_rgb(lab_pred[..., 0], lab_pred[..., 1:])
                
                psnr_scores.append(psnr_fn(rgb_true, rgb_pred))
                ssim_scores.append(ssim_fn(rgb_true, rgb_pred, channel_axis=-1))
                lpips_scores.append(compute_lpips(rgb_true, rgb_pred))          
                colorfulness_pred.append(compute_colorfulness(rgb_pred))      
                colorfulness_true.append(compute_colorfulness(rgb_true))

    metrics = {
        'psnr':               np.mean(psnr_scores),
        'ssim':               np.mean(ssim_scores),
        'lpips':              np.mean(lpips_scores),       # lower is better
        'colorfulness_pred':  np.mean(colorfulness_pred),  # your model's vividness
        'colorfulness_true':  np.mean(colorfulness_true),  # ground truth vividness
    }
    print(f'Test PSNR: {metrics["psnr"]:.4f}')
    print(f'Test SSIM: {metrics["ssim"]:.4f}')
    print(f'Test LPIPS: {metrics["lpips"]:.4f}')
    print(f'Colorfulness — Predicted: {metrics["colorfulness_pred"]:.2f}  '
        f'Ground Truth: {metrics["colorfulness_true"]:.2f}')
    return metrics



# Visualise results
def visualise_results(model, test_loader, checkpoint_path, device='cuda', num_images=5, save_path="recognition/MambaIRv2_s48448284/figures"):
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("model", ckpt)  
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    Path(save_path).mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(num_images, 3, figsize=(10, 3 * num_images))
    fig.suptitle("Colourisation Results", fontsize=16)
    col_titles = ["Grayscale (L)", "Predicted (ab)", "Ground Truth (ab)"]
    
    for ax, col in zip(axes[0], col_titles):
        ax.set_title(col, fontsize=14)


    with torch.no_grad():
        for i, (L, ab) in enumerate(test_loader):
            if i >= num_images:
                break
            L, ab = L.to(device), ab.to(device)
            ab_pred = model(L)

            # Denormalise
            L_denorm = denormalise_L(L.cpu().numpy())
            ab_denorm = denormalise_ab(ab.cpu().numpy())
            ab_pred_denorm = denormalise_ab(ab_pred.cpu().numpy())

            for j in range(L.size(0)):
                lab_true = np.concatenate([L_denorm[j], ab_denorm[j]], axis=0).transpose(1, 2, 0)
                lab_pred = np.concatenate([L_denorm[j], ab_pred_denorm[j]], axis=0).transpose(1, 2, 0)
                
                rgb_true = lab_to_rgb(lab_true[..., 0], lab_true[..., 1:])
                rgb_pred = lab_to_rgb(lab_pred[..., 0], lab_pred[..., 1:])

                # Add images to the plot
                axes[i, 0].imshow(L_denorm[j][0], cmap='gray')
                axes[i, 1].imshow(rgb_pred)
                axes[i, 2].imshow(rgb_true)
                
                for ax in axes[i]:
                    ax.axis('off')
    plt.tight_layout()
    plt.savefig(f"{save_path}/base_colourisation_results.png")

# also want to be able to visualise an entire folder
def run_colourise(model, checkpoint_path, image_folder, results_folder, device='cuda'):
    IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}

    image_dir = Path(image_folder)
    if not image_dir.is_dir():
        raise FileNotFoundError(
            f"Input folder not found: {image_folder}. "
            "Create it and add images, or check path"
        )

    image_paths = sorted(
        path 
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_paths:
        raise FileNotFoundError(
            f"No supported image files found in {image_folder}."
        )

    results_dir = Path(results_folder)
    results_dir.mkdir(parents= True, exist_ok= True)
    
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("model", ckpt)  
    model.load_state_dict(state_dict)
    model = model.to(device)

    result_paths = []
    from tqdm import tqdm
    from PIL import Image

    IMG_SIZE = 256
    for image_path in tqdm(image_paths, desc="Colourising"):
        img = Image.open(image_path).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))

        lab = rgb_to_lab(img)
        L = lab[:, :, 0]                    
        ab = lab[:, :, 1:]                  

        L_norm = normalise_L(L)              
        ab_norm = normalise_ab(ab)           
        
        L_tensor = torch.from_numpy(L_norm).unsqueeze(0).unsqueeze(0).float().to(device)
        with torch.no_grad():
            ab_pred = model(L_tensor)  # (1, 2, H, W)

        ab_pred_np = ab_pred.squeeze(0).cpu().numpy().transpose(1, 2, 0)  # (H, W, 2)

        # Denormalise back to LAB ranges
        L_denorm = denormalise_L(L_norm)          # -> (H, W), range 0-100
        ab_denorm = denormalise_ab(ab_pred_np)    # -> (H, W, 2), range -128 to 127

        # Reconstruct LAB and convert to RGB
        lab_pred = np.concatenate([L_denorm[..., np.newaxis], ab_denorm], axis=-1)  # (H, W, 3)
        rgb_pred = lab_to_rgb(L_denorm, ab_denorm)  # already returns uint8

        # save image — remove the *255 conversion
        out_img = Image.fromarray(rgb_pred)
        out_path = results_dir / (image_path.stem + "_colourised.png")
        out_img.save(out_path)
        result_paths.append(out_path)

    print(f"Saved {len(result_paths)} colourised images to {results_dir}")
    return result_paths

if __name__ == "__main__":
    from modules import MambaIRv2Frozen, MambaIRv2FineTuned

    checkpoint_path = "recognition/MambaIRv2_s48448284/models/finetuned_l1_sat_last.pth"
    model = MambaIRv2FineTuned(None, device="cuda")


    _, _, test_loader = get_dataloaders()
    run_colourise(model, checkpoint_path, "recognition/MambaIRv2_s48448284/test-data/test-dataset", "recognition/MambaIRv2_s48448284/test-data/results-data-l1_sat")
    visualise_results(model, test_loader, checkpoint_path, save_path="recognition/MambaIRv2_s48448284/figures")

        