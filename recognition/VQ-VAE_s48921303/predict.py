"""
==============================================================================
File:        predict.py
Author:      Jonathan Lu
Course:      COMP3710 / COMP7710 - Pattern Recognition
Date:        October 2026
Description: Inference and evaluation driver script for Hip MRI generative models:
             - Loads trained model weights for Baseline VAE or VQ-VAE + PixelCNN.
             - Computes test set reconstruction fidelity using SSIM.
             - Measures generative distribution quality via FID and KID.
             - Profiles computational resource efficiency (batch latency and peak VRAM).
             - Conducts an exhaustive nearest-neighbour audit against the training set.
             - Generates and exports comparison image figures to the 'eval/' directory.
==============================================================================
"""

import dataset
import modules
import torch
import argparse
import os
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from datetime import datetime
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
import time
import numpy as np
import random


def set_seed(seed=42):
    """
    Set random seeds across random, NumPy, and PyTorch for reproducible evaluation.

    Args:
        seed (int): Integer seed value. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    """
    Select the optimal available compute device (CUDA GPU, Apple MPS, or CPU).

    Returns:
        torch.device: Selected computation device.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    return device


def count_parameters(model):
    """
    Count the total number of trainable parameters in a PyTorch model.

    Args:
        model (torch.nn.Module): Target PyTorch network.

    Returns:
        int: Total number of learnable parameters.
    """
    total_params = 0
    for p in model.parameters():
        if p.requires_grad:
            total_params += p.numel()
    return total_params


def load_data(data_dir, batch_size=32, test=False):
    """
    Load test split data loader without shuffling to ensure deterministic evaluation.

    Args:
        data_dir (str): Root dataset directory containing 'keras_slices_test'.
        batch_size (int): Batch size for evaluation. Defaults to 32.
        test (bool): If True, enables early stopping for quick testing. Defaults to False.

    Returns:
        torch.utils.data.DataLoader: DataLoader yielding test slice tensors.
    """
    dir = os.path.expanduser(os.path.join(data_dir, 'keras_slices_test'))
    data = dataset.HipMRIDataset(dir, early_stop=test)
    return DataLoader(data, batch_size=batch_size, shuffle=False)
    
    
def get_sample_images(test_loader, samples=8, seed=42):
    """
    Deterministically select a fixed set of test slices for qualitative inspection.

    Args:
        test_loader (DataLoader): DataLoader for the test set.
        samples (int): Number of sample images to retrieve. Defaults to 8.
        seed (int): Seed for pseudo-random slice selection. Defaults to 42.

    Returns:
        torch.Tensor: Stacked tensor of selected test images [samples, 1, 256, 128].
    """
    total_slices = len(test_loader.dataset)
    rng = np.random.RandomState(seed)
    chosen_indices = sorted(rng.choice(total_slices, size=min(samples, total_slices), replace=False))
    sampled_tensors = [test_loader.dataset[i] for i in chosen_indices]
    return torch.stack(sampled_tensors)


def load_vae(args, device):
    """
    Instantiate and load checkpoint weights for the baseline VAE.

    Args:
        args (Namespace): Parsed command-line arguments containing weights path.
        device (torch.device): Target device to load model tensors onto.

    Returns:
        modules.VAE: Loaded VAE model ready for evaluation.
    """
    vae = modules.VAE(embedding_dim=args.embedding_dim).to(device)
    weights = torch.load(os.path.join(args.path, 'VAE.pth'), weights_only=True, map_location=device)
    vae.load_state_dict(weights)
    return vae
    
    
def load_vqvae(args, device):
    """
    Instantiate and load checkpoint weights for the VQ-VAE model.
    Dynamically detects codebook hyper-parameters from the checkpoint weights.

    Args:
        args (Namespace): Parsed command-line arguments.
        device (torch.device): Target computation device.

    Returns:
        modules.VQVAE: Loaded VQ-VAE model.
    """
    weights = torch.load(os.path.join(args.path, 'VQVAE.pth'), weights_only=True, map_location=device)
    num_embeddings = weights['vq.embedding.weight'].shape[0] if 'vq.embedding.weight' in weights else args.num_embeddings
    embedding_dim = weights['vq.embedding.weight'].shape[1] if 'vq.embedding.weight' in weights else args.embedding_dim
    vqvae = modules.VQVAE(num_embeddings=num_embeddings, embedding_dim=embedding_dim).to(device)
    vqvae.load_state_dict(weights)
    return vqvae

    
def load_pixelcnn(args, device):
    """
    Instantiate and load checkpoint weights for the Gated PixelCNN prior.
    Dynamically detects hidden dimension and layer depth from saved weights.

    Args:
        args (Namespace): Parsed command-line arguments.
        device (torch.device): Target computation device.

    Returns:
        modules.PixelCNN: Loaded PixelCNN model ready for generation.
    """
    weights = torch.load(os.path.join(args.path, 'pixelCNN.pth'), weights_only=True, map_location=device)
    hidden_dim = weights['embedding.weight'].shape[1] if 'embedding.weight' in weights else args.embedding_dim
    b_layers = max(1, len(set(int(k.split('.')[1]) for k in weights.keys() if k.startswith('net.'))) - 1)
    pixelcnn = modules.PixelCNN(num_embeddings=args.num_embeddings, hidden_dim=hidden_dim, B_layers=b_layers).to(device)
    pixelcnn.load_state_dict(weights)
    return pixelcnn


def images_convert_fid(images):
    """
    Convert single-channel float tensors in [0, 1] to 3-channel uint8 in [0, 255]
    as required by torchmetrics Inception-v3 FID/KID metrics.

    Args:
        images (torch.Tensor): Greyscale image batch of shape (B, 1, H, W).

    Returns:
        torch.Tensor: Formatted uint8 RGB tensor of shape (B, 3, H, W).
    """
    images = torch.clamp(images, 0.0, 1.0)
    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)
    return (images * 255.0).to(torch.uint8)


def init_fid(test_loader, feature, device):
    """
    Initialise Frechet Inception Distance (FID) and register real test images.

    Args:
        test_loader (DataLoader): DataLoader for real test images.
        feature (int): Inception feature layer dimensionality (e.g. 2048 or 192).
        device (torch.device): Target computation device.

    Returns:
        FrechetInceptionDistance: Configured FID metric pre-populated with real features.
    """
    fid = FrechetInceptionDistance(feature=feature).to(device)
    
    for images in test_loader:
        images = images.to(device)
        converted_images = images_convert_fid(images)
        fid.update(converted_images, real=True)
        
    return fid
    

def init_kid(test_loader, device):
    """
    Initialise Kernel Inception Distance (KID) and register real test images.

    Args:
        test_loader (DataLoader): DataLoader for real test images.
        device (torch.device): Target computation device.

    Returns:
        KernelInceptionDistance: Configured KID metric pre-populated with real features.
    """
    num_samples = len(test_loader.dataset)
    subset_size = min(50, max(2, num_samples - 1))
    kid = KernelInceptionDistance(subset_size=subset_size).to(device)
    
    for images in test_loader:
        images = images.to(device)
        converted_images = images_convert_fid(images)
        kid.update(converted_images, real=True)
        
    return kid


def audit_memorisation(generated_samples, data_dir, path, device, test=False):
    """
    Perform an exhaustive nearest-neighbour audit comparing each generated scan
    against all real training slices using pixel MSE and SSIM to disprove memorisation.

    Args:
        generated_samples (torch.Tensor): Batch of synthetic slices [N, 1, H, W].
        data_dir (str): Root dataset directory containing 'keras_slices_train'.
        path (str): Output folder to save the comparative visual figure.
        device (torch.device): Target computation device.
        test (bool): If True, limits training slice comparison for quick debugging.
    """
    train_dir = os.path.expanduser(os.path.join(data_dir, 'keras_slices_train'))
    train_dataset = dataset.HipMRIDataset(train_dir, early_stop=test)
    train_imgs = torch.from_numpy(train_dataset.imgs).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    nearest_list = []
    mse_list = []
    ssim_list = []

    with torch.no_grad():
        for i in range(len(generated_samples)):
            gen_i = generated_samples[i:i + 1]
            # Calculate pixel MSE across the entire training set to find the closest scan
            diffs = (train_imgs - gen_i).pow(2).mean(dim=[1, 2, 3])
            closest_idx = torch.argmin(diffs).item()
            closest_scan = train_imgs[closest_idx:closest_idx + 1]

            nearest_list.append(closest_scan)
            mse_list.append(diffs[closest_idx].item())
            ssim_list.append(ssim(gen_i, closest_scan).item())

    # Build side-by-side comparison: generated slices (top row) vs nearest training matches (bottom row)
    nearest_tensor = torch.cat(nearest_list, dim=0)
    cmp = torch.cat([generated_samples, nearest_tensor], dim=0)
    save_image(cmp, os.path.join(path, 'memorisation_audit.png'), nrow=len(generated_samples))

    avg_mse = sum(mse_list) / len(mse_list)
    avg_ssim = sum(ssim_list) / len(ssim_list)
    print(f"Nearest training neighbour audit saved to: {os.path.join(path, 'memorisation_audit.png')}")
    print(f"Mean Nearest-Neighbour MSE to Training Set: {avg_mse:.6f} | Mean SSIM: {avg_ssim:.4f}")


def vae_recon_gen(vae, test_loader, fid_2048, fid_192, kid, device, data_dir, samples=8, fid_samples=200, batch_size=32, test=False, seed=42):
    """
    Evaluate baseline VAE: profiles latency/VRAM, computes SSIM/FID/KID, and audits memorisation.

    Args:
        vae (modules.VAE): Trained baseline VAE model.
        test_loader (DataLoader): DataLoader for evaluation test slices.
        fid_2048 (FrechetInceptionDistance): Initialised FID metric (feature=2048).
        fid_192 (FrechetInceptionDistance): Initialised FID metric (feature=192).
        kid (KernelInceptionDistance): Initialised KID metric.
        device (torch.device): Computation device.
        data_dir (str): Root dataset directory.
        samples (int): Number of qualitative visual samples. Defaults to 8.
        fid_samples (int): Total generated samples for FID/KID computation. Defaults to 200.
        batch_size (int): Generation batch size for latency benchmarking. Defaults to 32.
        test (bool): Fast debug flag. Defaults to False.
        seed (int): Seed for sampling reproducibility. Defaults to 42.
    """
    vae.eval()
    os.makedirs('eval', exist_ok=True)
    path = os.path.join('eval', datetime.now().strftime('%H-%M-%S_%d-%m-%y'))
    os.makedirs(path, exist_ok=True)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    
    print('==============================================================================')
    print('VAE RESOURCE PROFILING')
    print(f'Total Trainable Parameters: {count_parameters(vae):,}')

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

    # Benchmark generation latency for a batch of 32
    start_time = time.time()
    _ = vae.sample(batch_size, device)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    latency_32 = (time.time() - start_time) * 1000.0

    if device.type == 'cuda':
        inference_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    else:
        inference_vram = 0.0

    print(f'Batch Inference Latency (Batch Size={batch_size}): {latency_32:.2f} ms')
    print(f'Peak Inference VRAM: {inference_vram:.2f} MB')
    print('==============================================================================')

    with torch.no_grad():
        # Qualitative reconstruction on a fixed test batch
        images = get_sample_images(test_loader, samples=samples, seed=seed).to(device)
    
        pred, _, _ = vae(images)
        cmp = torch.cat([images, pred])
        save_image(cmp, os.path.join(path, 'recon.png'), nrow=samples)
        print("Average SSIM of test sample batch is", ssim(pred, images).item())
    
        # Sample synthetic scans from Gaussian prior
        generated_samples = vae.sample(samples, device)
        save_image(generated_samples, os.path.join(path, 'generated.png'), nrow=samples)

        print('Reconstructed and generated images saved in', path, 'directory')
        # Audit synthetic samples against training set to verify privacy preservation
        audit_memorisation(generated_samples, data_dir, path, device, test=test)
        
        # Batch generation loop for computing FID and KID distributional metrics
        generated_count = 0
        while generated_count < fid_samples:
            curr_batch = min(batch_size, fid_samples - generated_count)
            generated = vae.sample(curr_batch, device)
            fid_2048.update(images_convert_fid(generated), real=False)
            fid_192.update(images_convert_fid(generated), real=False)
            kid.update(images_convert_fid(generated), real=False)
            generated_count += curr_batch
    
        fid_2048_score = fid_2048.compute().item()
        fid_192_score = fid_192.compute().item()
        kid_mean, kid_std = kid.compute()
        
        print('For VAE:')
        print('FID with feature=2048 is', fid_2048_score)
        print('FID with feature=192 is', fid_192_score)
        print('KID is', kid_mean.item(), '+-', kid_std.item())


def vqvae_recon_gen(vqvae, pixelcnn, test_loader, fid_2048, fid_192, kid, device, data_dir, samples=8, fid_samples=200, batch_size=32, temperature=0.9, top_k=0, test=False, seed=42):
    """
    Evaluate VQ-VAE + PixelCNN: profiles latency/VRAM, computes SSIM/FID/KID, and audits memorisation.

    Args:
        vqvae (modules.VQVAE): Trained VQ-VAE model.
        pixelcnn (modules.PixelCNN): Trained Gated PixelCNN prior.
        test_loader (DataLoader): DataLoader for evaluation test slices.
        fid_2048 (FrechetInceptionDistance): Initialised FID metric (feature=2048).
        fid_192 (FrechetInceptionDistance): Initialised FID metric (feature=192).
        kid (KernelInceptionDistance): Initialised KID metric.
        device (torch.device): Computation device.
        data_dir (str): Root dataset directory.
        samples (int): Number of qualitative visual samples. Defaults to 8.
        fid_samples (int): Total generated samples for FID/KID computation. Defaults to 200.
        batch_size (int): Generation batch size for latency benchmarking. Defaults to 32.
        temperature (float): Sampling temperature for PixelCNN. Defaults to 0.9.
        top_k (int): Top-k sampling restriction. Defaults to 0 (unrestricted).
        test (bool): Fast debug flag. Defaults to False.
        seed (int): Seed for sampling reproducibility. Defaults to 42.
    """
    vqvae.eval()
    pixelcnn.eval()
    os.makedirs('eval', exist_ok=True)
    path = os.path.join('eval', datetime.now().strftime('%H-%M-%S_%d-%m-%y'))
    os.makedirs(path, exist_ok=True)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    
    total_params = count_parameters(vqvae) + count_parameters(pixelcnn)
    print('==============================================================================')
    print('VQ-VAE + PIXELCNN RESOURCE PROFILING')
    print(f'VQ-VAE Parameters: {count_parameters(vqvae):,}')
    print(f'PixelCNN Parameters: {count_parameters(pixelcnn):,}')
    print(f'Total Pipeline Parameters: {total_params:,}')

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

    # Benchmark generation latency for a batch of 32
    start_time = time.time()
    _ = pixelcnn.sample(vqvae, batch_size, device, temperature=temperature, top_k=top_k)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    latency_32 = (time.time() - start_time) * 1000.0

    if device.type == 'cuda':
        inference_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    else:
        inference_vram = 0.0

    print(f'Batch Inference Latency (Batch Size={batch_size}): {latency_32:.2f} ms ({latency_32 / batch_size:.2f} ms/sample)')
    print(f'Peak Inference VRAM: {inference_vram:.2f} MB')
    print('==============================================================================')

    with torch.no_grad():
        # Qualitative reconstruction on a fixed test batch
        images = get_sample_images(test_loader, samples=samples, seed=seed).to(device)
    
        pred, _, _ = vqvae(images)
        cmp = torch.cat([images, pred])
        save_image(cmp, os.path.join(path, 'recon.png'), nrow=samples)
        print("Average SSIM of test sample batch is", ssim(pred, images).item())
    
        # Sample synthetic scans via Gated PixelCNN prior
        print(f"Sampling {samples} synthetic scans via PixelCNN prior (temperature={temperature}, top_k={top_k})...")
        generated_samples = pixelcnn.sample(vqvae, samples, device, temperature=temperature, top_k=top_k)
        save_image(generated_samples, os.path.join(path, 'generated.png'), nrow=samples)

        print('Reconstructed and generated images saved in', path, 'directory')
        # Audit synthetic samples against training set to verify privacy preservation
        audit_memorisation(generated_samples, data_dir, path, device, test=test)
        
        # Batch generation loop for computing FID and KID distributional metrics
        generated_count = 0
        while generated_count < fid_samples:
            curr_batch = min(batch_size, fid_samples - generated_count)
            generated = pixelcnn.sample(vqvae, curr_batch, device, temperature=temperature, top_k=top_k)
            fid_2048.update(images_convert_fid(generated), real=False)
            fid_192.update(images_convert_fid(generated), real=False)
            kid.update(images_convert_fid(generated), real=False)
            generated_count += curr_batch
    
        fid_2048_score = fid_2048.compute().item()
        fid_192_score = fid_192.compute().item()
        kid_mean, kid_std = kid.compute()
        
        print('For VQ-VAE + PixelCNN:')
        print('FID with feature=2048 is', fid_2048_score)
        print('FID with feature=192 is', fid_192_score)
        print('KID is', kid_mean.item(), '+-', kid_std.item())


def eval_vae(args, test_loader, fid_2048, fid_192, kid, device):
    """
    Convenience wrapper to load and evaluate the baseline VAE model.

    Args:
        args (Namespace): Command-line options.
        test_loader (DataLoader): DataLoader for test split.
        fid_2048 (FrechetInceptionDistance): Metric instance (dim 2048).
        fid_192 (FrechetInceptionDistance): Metric instance (dim 192).
        kid (KernelInceptionDistance): Metric instance for KID.
        device (torch.device): Computation device.
    """
    vae = load_vae(args, device)
    vae_recon_gen(vae, test_loader, fid_2048, fid_192, kid, device, data_dir=args.data_dir, samples=args.samples, fid_samples=args.fid_samples, batch_size=args.batch_size, test=args.test, seed=args.seed)
    
    
def eval_vqvae(args, test_loader, fid_2048, fid_192, kid, device):
    """
    Convenience wrapper to load and evaluate the VQ-VAE and PixelCNN pipeline.

    Args:
        args (Namespace): Command-line options.
        test_loader (DataLoader): DataLoader for test split.
        fid_2048 (FrechetInceptionDistance): Metric instance (dim 2048).
        fid_192 (FrechetInceptionDistance): Metric instance (dim 192).
        kid (KernelInceptionDistance): Metric instance for KID.
        device (torch.device): Computation device.
    """
    vqvae = load_vqvae(args, device)
    pixelcnn = load_pixelcnn(args, device)
    vqvae_recon_gen(vqvae, pixelcnn, test_loader, fid_2048, fid_192, kid, device, data_dir=args.data_dir, samples=args.samples, fid_samples=args.fid_samples, batch_size=args.batch_size, temperature=args.temperature, top_k=args.top_k, test=args.test, seed=args.seed)


def main(args):
    """
    Parse execution arguments, initialise metric modules, and dispatch evaluation.

    Args:
        args (Namespace): Parsed command-line arguments.
    """
    set_seed(args.seed)
    device = get_device()
    test_loader = load_data(args.data_dir, batch_size=args.batch_size, test=args.test)
    fid_2048 = init_fid(test_loader, 2048, device)
    fid_192 = init_fid(test_loader, 192, device)
    kid = init_kid(test_loader, device)
    
    if args.model == 'vae':
        eval_vae(args, test_loader, fid_2048, fid_192, kid, device)
    elif args.model == 'vqvae':
        eval_vqvae(args, test_loader, fid_2048, fid_192, kid, device)
    else:
        raise ValueError("--model argument must either be 'vae' or 'vqvae'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate trained Hip MRI generative models (VAE or VQ-VAE + PixelCNN).")
    parser.add_argument('--model', required=True, choices=['vae', 'vqvae'], help="Model architecture to evaluate.")
    parser.add_argument('--path', required=True, help="Path to folder containing saved .pth weights.")
    parser.add_argument('--data_dir', required=True, type=str, help="Path to HipMRI root directory containing keras_slices_*.")
    parser.add_argument('--num_embeddings', type=int, default=32, help="Number of codebook embeddings K.")
    parser.add_argument('--embedding_dim', type=int, default=64, help="Embedding vector dimensionality D.")
    parser.add_argument('--samples', type=int, default=8, help="Number of qualitative visual samples to export.")
    parser.add_argument('--fid_samples', type=int, default=200, help="Number of generated samples for FID/KID evaluation.")
    parser.add_argument('--batch_size', type=int, default=32, help="Evaluation and generation batch size.")
    parser.add_argument('--temperature', type=float, default=0.8, help="PixelCNN sampling temperature.")
    parser.add_argument('--top_k', type=int, default=0, help="Top-k sampling truncation (0 = disabled).")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument('--test', action='store_true', help="Fast debug flag using subset of dataset.")
    args = parser.parse_args()
    
    main(args)
