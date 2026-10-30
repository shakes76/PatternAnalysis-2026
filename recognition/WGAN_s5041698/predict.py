import os
import re
import time
import torch
import torchvision.utils as vutils
import numpy as np
from torchmetrics.image.fid import FrechetInceptionDistance
from skimage.metrics import structural_similarity as compute_ssim
from sklearn.decomposition import PCA
import umap
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataset import HipMRIDataset
from modules import Generator

def load_generator(checkpoint_path, latent_dim=128, device="cuda", type="WGAN-GP"):
    """Loads the trained WGAN-GP generator from a saved checkpoint file."""
    print(f"[*] Loading generator weights from: {checkpoint_path}")
    if type == "WGAN-GP":
        generator = Generator(latent_dim=latent_dim, img_channels=1, base_filters=64).to(device)
    elif type == "VAE":
        from modules import ConvVAE
        generator = ConvVAE(in_channels=1, latent_dim=latent_dim).to(device)
    else:
        raise ValueError(f"Unsupported model type: {type}. Choose 'WGAN-GP' or 'VAE'.")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    state_dict = checkpoint['generator_state_dict'] if 'generator_state_dict' in checkpoint else checkpoint
    generator.load_state_dict(state_dict)
    
    generator.eval()
    return generator

def generate_synthetic_cohort(generator, num_samples=500, latent_dim=128, batch_size=64, device="cuda"):
    """
    Unified helper method to generate synthetic medical slices and return them in [0, 1] range.
    """
    print(f"[*] Generating synthetic cohort of {num_samples} samples...")
    synthetic_images_list = []

    generation_time = 0.0

    is_cuda = (device.type == "cuda") if isinstance(device, torch.device) else str(device).startswith("cuda")
    if torch.cuda.is_available() and is_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        generated_count = 0
        while generated_count < num_samples:
            curr_batch_size = min(batch_size, num_samples - generated_count)
            z = torch.randn(curr_batch_size, latent_dim, device=device)

            # Check if it's a VAE or a GAN Generator
            if hasattr(generator, "decode"):
                start_time = time.time()
                fake_imgs = generator.decode(z)
                end_time = time.time()
                generation_time += (end_time - start_time)
                # VAE outputs in [0,1] (Sigmoid) -> convert to [-1,1] for compatibility
                fake_imgs = fake_imgs * 2.0 - 1.0
            else:
                start_time = time.time()
                fake_imgs = generator(z)
                end_time = time.time()
                generation_time += (end_time - start_time)

            # Rescale from generator output range [-1, 1] to [0, 1]
            fake_imgs = fake_imgs * 0.5 + 0.5
            synthetic_images_list.append(fake_imgs.cpu().numpy())
            generated_count += curr_batch_size

    peak_vram_gb = 0.0
    if torch.cuda.is_available() and is_cuda:
        peak_vram_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

    print(f"[*] Total generation time: {generation_time:.6f} seconds | Peak VRAM: {peak_vram_gb:.2f} GB")
    return np.concatenate(synthetic_images_list, axis=0)


def generate_image_grid(generator, grid_size=(5, 5), latent_dim=128, device="cuda", save_individual=False, save_dir="./plots"):
    """
    Generates a grid of synthetic images and returns the grid tensor along with 
    the latent vectors (z) used to reproduce them.
    Optionally saves each image separately and saves z vectors.
    """
    num_samples = grid_size[0] * grid_size[1]
    cols = grid_size[1]
        
    generator.eval()

    print(f"[*] Generating image grid of size {grid_size} ({num_samples} samples)...")
    
    is_cuda = (device.type == "cuda") if isinstance(device, torch.device) else str(device).startswith("cuda")
    if torch.cuda.is_available() and is_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        # Sample latent vectors
        z = torch.randn(num_samples, latent_dim, device=device)

        # Check if it's a VAE or a GAN Generator
        if hasattr(generator, "decode"):
            fake_imgs = generator.decode(z)
            fake_imgs = fake_imgs * 2.0 - 1.0
        else:
            fake_imgs = generator(z)

        # Rescale from generator output range [-1, 1] to [0, 1]
        fake_imgs = fake_imgs * 0.5 + 0.5
        
        # Arrange images into a grid
        grid_tensor = vutils.make_grid(fake_imgs, nrow=cols, normalize=False)

    peak_vram_gb = 0.0
    if torch.cuda.is_available() and is_cuda:
        peak_vram_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    print(f"[*] Grid generation complete | Peak VRAM: {peak_vram_gb:.2f} GB")

    os.makedirs(save_dir, exist_ok=True)
    
    if save_individual:
        os.makedirs(os.path.join(save_dir, "individual"), exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)
        
        # Save latent vectors
        torch.save(z, os.path.join(save_dir, "latent_vectors.pt"))
        print(f"[+] Saved latent vectors to {save_dir}/latent_vectors.pt")
        
        # Save each image separately
        for i in range(num_samples):
            img = fake_imgs[i]  # [1, H, W]
            img_np = img.cpu().squeeze(0).numpy()  # [H, W]
            plt.imsave(os.path.join(save_dir, "individual", f"image_{i:03d}.png"), 
                       img_np, cmap="gray", vmin=0, vmax=1)
        
        print(f"[+] Saved {num_samples} individual images to {save_dir}/individual/")

    # Save the grid for reference
    grid_np = grid_tensor.cpu().permute(1, 2, 0).numpy()
    plt.figure(figsize=(16, 16))
    plt.imshow(grid_np, cmap="gray" if grid_np.shape[2] == 1 else None)
    plt.axis("off")
    plt.savefig(f"{save_dir}/synthetic_grid.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved synthetic image grid to {save_dir}/synthetic_grid.png")

    return grid_tensor, z

def load_dataset_as_array(data_dir, max_samples=None, return_case_ids=False):
    """
    Unified helper method to load a dataset directory into a clean numpy array [N, H, W] in [0, 1].
    Optionally extracts and returns patient case IDs parsed from filenames.
    """
    print(f"[*] Loading dataset from {data_dir}...")
    dataset = HipMRIDataset(data_dir=data_dir)
    images_list = []
    case_ids = []
    
    limit = max_samples if max_samples else len(dataset)
    for i in tqdm(range(min(len(dataset), limit)), desc=f"Loading {os.path.basename(data_dir)}"):
        img = dataset[i]
        if isinstance(img, (tuple, list)):
            img = img[0]
        # Rescale from [-1, 1] to [0, 1]
        img_rescaled = img.numpy() * 0.5 + 0.5
        images_list.append(img_rescaled[0])  # Drop channel dim to get [H, W]

        if return_case_ids:
            file_name = os.path.basename(dataset.file_paths[i])
            # Match pattern like 'case_004' from 'case_004_week_0_slice_0.nii.gz'
            match = re.search(r'(case_\d+)', file_name)
            case_id = match.group(1) if match else file_name.split('_')[0]
            case_ids.append(case_id)
        
    imgs_array = np.stack(images_list, axis=0)
    if return_case_ids:
        return imgs_array, np.array(case_ids)
    return imgs_array


def compute_inter_patient_train_nearest_neighbors(
    train_data_dir="/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_train",
    train_array=None,
    case_ids=None,
    max_samples=None,
    batch_size=200
):
    """
    Calculates the nearest neighbor Euclidean distance between scans of distinct patients in the training set.
    For each scan, considers only other scans belonging to a different case/patient ID.
    Returns:
        dict: Summary statistics including mean, min, and array of nearest neighbor distances.
    """
    print("[*] Computing nearest neighbor distances between scans of distinct patients in train set...")
    if train_array is None or case_ids is None:
        train_array, case_ids = load_dataset_as_array(train_data_dir, max_samples=max_samples, return_case_ids=True)

    N = train_array.shape[0]
    train_flat = train_array.reshape(N, -1).astype(np.float32)

    # Compute squared norms for efficient batch Euclidean distance
    norms_sq = np.sum(train_flat ** 2, axis=1)
    min_inter_patient_dists = np.full(N, np.inf, dtype=np.float32)

    for i in tqdm(range(0, N, batch_size), desc="Computing inter-patient NN"):
        end_i = min(i + batch_size, N)
        dots = train_flat[i:end_i] @ train_flat.T
        dists_sq = norms_sq[i:end_i, None] + norms_sq[None, :] - 2 * dots
        np.maximum(dists_sq, 0, out=dists_sq)
        dists = np.sqrt(dists_sq)

        # Mask out scans from the same patient/case
        same_case_mask = (case_ids[i:end_i, None] == case_ids[None, :])
        dists[same_case_mask] = np.inf

        min_inter_patient_dists[i:end_i] = np.min(dists, axis=1)

    mean_nn_dist = float(np.mean(min_inter_patient_dists))
    min_nn_dist = float(np.min(min_inter_patient_dists))

    print(f"[*] Inter-patient Train Nearest Neighbors -> Mean: {mean_nn_dist:.4f}, Min: {min_nn_dist:.4f}")
    return {
        'mean': mean_nn_dist,
        'min': min_nn_dist,
        'all_dists': min_inter_patient_dists
    }

def evaluate_cohort_ssim(
    generator, 
    test_data_dir="/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_test", 
    num_samples=100, 
    latent_dim=128, 
    device="cuda"
):
    """
    Computes SSIM between synthetic cohort and reference test slices 
    using SSIM-based maximization for structural nearest-neighbor matching.
    """
    # 1. Generate synthetic cohort and ensure [0, 1] range
    synthetic_cohort = generate_synthetic_cohort(generator, num_samples=num_samples, latent_dim=latent_dim, device=device)
    if synthetic_cohort.min() < 0.0 or synthetic_cohort.max() > 1.0:
        synthetic_cohort = synthetic_cohort * 0.5 + 0.5  # Rescale from [-1, 1] to [0, 1] if needed

    # 2. Load real test cohort and ensure [0, 1] range
    real_cohort = load_dataset_as_array(test_data_dir, max_samples=num_samples)
    if real_cohort.min() < 0.0 or real_cohort.max() > 1.0:
        real_cohort = real_cohort * 0.5 + 0.5  # Rescale from [-1, 1] to [0, 1] if needed

    print("[*] Computing SSIM using Structural (SSIM-maximized) Nearest-Neighbor Matching...")
    ssim_scores = []
    ssim_output_dir = "./plots/ssim_pairs"
    os.makedirs(ssim_output_dir, exist_ok=True)

    for i in tqdm(range(num_samples), desc="Calculating Optimized SSIM"):
        synth_slice = synthetic_cohort[i, 0] if synthetic_cohort.ndim == 4 else synthetic_cohort[i]
        
        best_score = -1.0
        best_real_slice = None
        # Find the real test slice that maximizes SSIM (true structural match)
        for j in range(len(real_cohort)):
            real_slice = real_cohort[j, 0] if real_cohort.ndim == 4 else real_cohort[j]
            
            # Compute SSIM directly against candidate real slice
            score = compute_ssim(real_slice, synth_slice, data_range=1.0)
            if score > best_score:
                best_score = score
                best_real_slice = real_slice
                
        ssim_scores.append(best_score)

    mean_ssim = np.mean(ssim_scores)
    std_ssim = np.std(ssim_scores)

    print("\n" + "="*40)
    print(f"Optimized SSIM Evaluation Results (n={num_samples})")
    print("="*40)
    print(f"Mean SSIM : {mean_ssim:.4f}")
    print(f"Std SSIM  : {std_ssim:.4f}")
    print("="*40)

    return mean_ssim, std_ssim


def audit_nearest_neighbors(
    generator,
    train_data_dir="/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_train",
    num_synthetic_samples=500,
    top_k_synthetic=5,
    num_neighbors=5,
    latent_dim=128,
    output_path="./plots/memorization_audit_grid.png",
    device="cuda",
    compute_inter_patient_baseline=True
):
    """
    Audits a batch of synthetic images against training images for memorization.
    Screens num_synthetic_samples generated images to identify the top_k_synthetic 
    samples that are closest to any training image, then plots each with its 
    nearest training neighbors.
    """
    synthetic_array = generate_synthetic_cohort(generator, num_samples=num_synthetic_samples, latent_dim=latent_dim, device=device)
    train_array, case_ids = load_dataset_as_array(train_data_dir, return_case_ids=True)

    # Flatten for fast L2 distance computation
    train_flat = train_array.reshape(train_array.shape[0], -1)
    synthetic_flat = synthetic_array.reshape(synthetic_array.shape[0], -1)

    # Compute inter-patient nearest neighbor benchmark in train set
    inter_patient_stats = None
    if compute_inter_patient_baseline:
        inter_patient_stats = compute_inter_patient_train_nearest_neighbors(
            train_array=train_array,
            case_ids=case_ids
        )

    print(f"[*] Finding closest training images across {num_synthetic_samples} synthetic candidates...")
    min_synth_dists = np.empty(num_synthetic_samples, dtype=np.float32)
    cached_nearest_train_idx = []
    cached_nearest_train_dists = []

    for i in tqdm(range(num_synthetic_samples)):
        synth_vec = synthetic_flat[i]
        distances = np.linalg.norm(train_flat - synth_vec, axis=1)
        nearest_indices = np.argsort(distances)[:num_neighbors]
        nearest_distances = distances[nearest_indices]

        min_synth_dists[i] = nearest_distances[0]
        cached_nearest_train_idx.append(nearest_indices)
        cached_nearest_train_dists.append(nearest_distances)

    # Select the top_k synthetic samples with the smallest minimum distance to the training set (highest memorization risk)
    top_synth_indices = np.argsort(min_synth_dists)[:top_k_synthetic]

    audit_results = []
    print(f"[*] Top {top_k_synthetic} closest synthetic samples to training set:")
    for rank, synth_idx in enumerate(top_synth_indices):
        nearest_indices = cached_nearest_train_idx[synth_idx]
        nearest_distances = cached_nearest_train_dists[synth_idx]

        audit_results.append({
            'synthetic_idx': synth_idx,
            'synthetic_img': synthetic_array[synth_idx, 0] if synthetic_array.ndim == 4 else synthetic_array[synth_idx],
            'neighbor_imgs': [train_array[idx, 0] if train_array.ndim == 4 else train_array[idx] for idx in nearest_indices],
            'distances': nearest_distances
        })
        print(f"  -> Rank {rank+1} (Synthetic #{synth_idx+1}): Closest Training Neighbor Distance = {nearest_distances[0]:.4f}")

    print(f"[*] Saving visual audit grid to {output_path}...")
    num_rows = len(audit_results)
    fig, axes = plt.subplots(num_rows, num_neighbors + 1, figsize=(14, 3 * num_rows))
    if num_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, res in enumerate(audit_results):
        axes[i, 0].imshow(res['synthetic_img'], cmap='gray')
        axes[i, 0].set_title(f"Synthetic #{res['synthetic_idx']+1}\n(Closest in batch)", fontsize=9)
        axes[i, 0].axis('off')
        
        for spine in axes[i, 0].spines.values():
            spine.set_edgecolor('blue')
            spine.set_linewidth(2)

        for j, (n_img, dist) in enumerate(zip(res['neighbor_imgs'], res['distances'])):
            ax = axes[i, j + 1]
            ax.imshow(n_img, cmap='gray')
            ax.set_title(f"Neighbor {j+1}\n(Dist: {dist:.1f})" if i == 0 else f"Dist: {dist:.1f}", fontsize=8)
            ax.axis('off')

    plt.suptitle("Privacy & Memorization Audit: Closest Synthetic Slices vs. Nearest Training Scans", fontsize=14, y=0.98)
    if inter_patient_stats is not None:
        subtitle_text = (
            f"Train Set Inter-Patient Nearest Neighbor Distance (distinct cases): "
            f"Mean = {inter_patient_stats['mean']:.2f} | Min = {inter_patient_stats['min']:.2f}"
        )
        plt.figtext(0.5, 0.95, subtitle_text, ha='center', fontsize=10, style='italic', color='#333333')
        plt.tight_layout(rect=[0, 0, 1, 0.94])
    else:
        plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    print("[*] Audit complete!")

def evaluate_fid(
    generator, 
    test_data_dir="/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_test", 
    num_samples=1000, 
    latent_dim=128, 
    device="cuda"
):
    """
    Computes Fréchet Inception Distance (FID) between synthetic cohort 
    and reference test slices using Inception-v3 features.
    """
    print("[*] Initializing FID metric (Inception-v3 feature extractor)...")
    fid = FrechetInceptionDistance(feature=2048).to(device)

    # 1. Generate synthetic cohort and ensure [0, 1] range
    synthetic_cohort = generate_synthetic_cohort(generator, num_samples=num_samples, latent_dim=latent_dim, device=device)
    if synthetic_cohort.min() < 0.0 or synthetic_cohort.max() > 1.0:
        synthetic_cohort = synthetic_cohort * 0.5 + 0.5  # Rescale from [-1, 1] to [0, 1] if needed

    # 2. Load real test cohort and ensure [0, 1] range
    real_cohort = load_dataset_as_array(test_data_dir, max_samples=num_samples)
    if real_cohort.min() < 0.0 or real_cohort.max() > 1.0:
        real_cohort = real_cohort * 0.5 + 0.5  # Rescale from [-1, 1] to [0, 1] if needed

    print("[*] Processing and updating FID distributions...")
    
    # Process real images for Inception-v3 (expects 3-channel RGB uint8 [0, 255])
    real_tensor = torch.from_numpy(real_cohort).float()
    if real_tensor.ndim == 3:  # [N, H, W] -> [N, 1, H, W]
        real_tensor = real_tensor.unsqueeze(1)
    
    real_uint8 = (real_tensor.clamp(0, 1) * 255).byte()
    if real_uint8.shape[1] == 1:
        real_uint8 = real_uint8.repeat(1, 3, 1, 1)
        
    # Feed real images in batches to avoid VRAM overload
    batch_size = 16
    for i in range(0, len(real_uint8), batch_size):
        batch_real = real_uint8[i:i+batch_size].to(device)
        fid.update(batch_real, real=True)

    # Process synthetic images for Inception-v3
    synth_tensor = torch.from_numpy(synthetic_cohort).float()
    if synth_tensor.ndim == 3:  # [N, H, W] -> [N, 1, H, W]
        synth_tensor = synth_tensor.unsqueeze(1)
        
    synth_uint8 = (synth_tensor.clamp(0, 1) * 255).byte()
    if synth_uint8.shape[1] == 1:
        synth_uint8 = synth_uint8.repeat(1, 3, 1, 1)

    for i in range(0, len(synth_uint8), batch_size):
        batch_synth = synth_uint8[i:i+batch_size].to(device)
        fid.update(batch_synth, real=False)

    print("[*] Computing final FID score...")
    fid_score = float(fid.compute().item())

    print("\n" + "="*40)
    print(f"FID Evaluation Results (n={num_samples})")
    print("="*40)
    print(f"FID Score : {fid_score:.4f}")
    print("="*40)

    return fid_score

def plot_umap_distribution(
    generator,
    data_dir="/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_train",
    num_samples=2000,
    latent_dim=128,
    pca_components=50,
    n_neighbors=15,
    min_dist=0.1,
    random_state=42,
    output_path="./plots/umap_distribution_comparison.png",
    device="cuda"
):
    """
    Computes and plots a 2D UMAP projection comparing real dataset slices (blue)
    and generated images (red), using PCA first for initial dimensionality reduction.
    """
    print("\n[*] Starting UMAP distribution comparison...")
    # 1. Generate synthetic images
    synth_imgs = generate_synthetic_cohort(generator, num_samples=num_samples, latent_dim=latent_dim, device=device)
    # Ensure shape [N, C, H, W] or [N, H, W] -> flatten to [N, pixels]
    N_synth = synth_imgs.shape[0]
    synth_flat = synth_imgs.reshape(N_synth, -1)

    # 2. Load real dataset samples
    real_imgs = load_dataset_as_array(data_dir, max_samples=num_samples)
    N_real = real_imgs.shape[0]
    real_flat = real_imgs.reshape(N_real, -1)

    print(f"[*] Collected {N_real} real samples and {N_synth} synthetic samples for UMAP analysis.")

    # 3. Combine samples for a shared projection space
    combined_flat = np.concatenate([real_flat, synth_flat], axis=0)

    # Step 1: PCA to reduce noise and dimensionality
    print(f"[*] Running PCA reduction to {pca_components} components...")
    pca = PCA(n_components=pca_components, random_state=random_state)
    combined_pca = pca.fit_transform(combined_flat)

    # Step 2: UMAP projection to 2D
    print(f"[*] Running UMAP projection to 2D...")
    umap_reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric='euclidean',
        random_state=random_state
    )
    combined_2d = umap_reducer.fit_transform(combined_pca)

    real_2d = combined_2d[:N_real]
    synth_2d = combined_2d[N_real:]

    # Step 3: Plot the distributions
    print(f"[*] Plotting UMAP distribution comparison to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(9, 7))
    plt.scatter(
        real_2d[:, 0], real_2d[:, 1],
        c='royalblue', alpha=0.5, s=12, label='Real Dataset'
    )
    plt.scatter(
        synth_2d[:, 0], synth_2d[:, 1],
        c='red', alpha=0.5, s=12, label='Generated'
    )
    plt.title("2D UMAP Distribution: Real Dataset vs. Generated Images", fontsize=13, fontweight='bold')
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.legend(loc='best', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"[+] Saved UMAP plot to '{output_path}'")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_file = f"./checkpoints/wgan_gp_epoch_49.pth"
    if os.path.exists(checkpoint_file):
        # Load generator once for both evaluation tasks
        generator = load_generator(checkpoint_file, device=device)
        
        # 1. Run SSIM Evaluation
        evaluate_cohort_ssim(generator, device=device)
        
        # 2. Run Memorization Nearest-Neighbor Audit & Plot Grid
        audit_nearest_neighbors(generator, device=device)

        # 3. Run FID Evaluation
        evaluate_fid(generator, device=device)

        # 4. Generate grid and save individual images with latent vectors
        generate_image_grid(generator, grid_size=(8, 8), device=device, save_individual=True, save_dir="./plots/WGAN_synth")

        # 5. Run UMAP Distribution Comparison
        plot_umap_distribution(generator, device=device)

    else:
        print(f"[!] Checkpoint not found at {checkpoint_file}.")

    print("\n")

    checkpoint_file_vae = f"./checkpoints/oasis_vae.pth"
    if os.path.exists(checkpoint_file_vae):
        # Load VAE generator for evaluation
        vae_generator = load_generator(checkpoint_file_vae, type="VAE", device=device)
        
        # 1. Run SSIM Evaluation for VAE comparison
        evaluate_cohort_ssim(vae_generator, device=device)

        # 2. Run FID Evaluation for VAE comparison
        evaluate_fid(vae_generator, device=device)

        # 3. Generate grid and save individual images with latent vectors
        generate_image_grid(vae_generator, grid_size=(8, 8), device=device, save_individual=False, save_dir="./plots/VAE_synth")

    else:
        print(f"[!] Checkpoint not found at {checkpoint_file_vae}.")
