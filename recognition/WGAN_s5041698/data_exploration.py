import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from scipy import stats
from sklearn.decomposition import PCA
import umap
from dataset import HipMRIDataset

BASE_PATH = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"

train_dir = os.path.join(BASE_PATH, "keras_slices_train")
val_dir   = os.path.join(BASE_PATH, "keras_slices_validate")
test_dir  = os.path.join(BASE_PATH, "keras_slices_test")

# Create PyTorch Datasets
train_dataset = HipMRIDataset(data_dir=train_dir)
val_dataset   = HipMRIDataset(data_dir=val_dir)
test_dataset  = HipMRIDataset(data_dir=test_dir)

# Create DataLoaders
BATCH_SIZE = 128

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# ==========================================
# UMAP DATA AUDIT EXTENSION
# ==========================================

print("\nCollecting samples for UMAP data audit...")
max_samples = 2000  # Adjust based on memory; 2000-5000 is plenty for a 2D map
collected_images = []
total_count = 0

for batch in train_loader:
    imgs = batch[0] if isinstance(batch, (tuple, list)) else batch
    collected_images.append(imgs.detach().cpu().numpy())
    total_count += imgs.shape[0]
    if total_count >= max_samples:
        break

# Concatenate and flatten: [N, 1, H, W] -> [N, H * W]
X_real = np.concatenate(collected_images, axis=0)[:max_samples]
N, C, H, W = X_real.shape
X_flat = X_real.reshape(N, C * H * W)

print(f"Flattened shape for auditing: {X_flat.shape}")

# Step 1: PCA to reduce noise and dimensionality (e.g., to 50 dimensions)
print("Running PCA...")
pca = PCA(n_components=50, random_state=42)
X_pca = pca.fit_transform(X_flat)

# Step 2: UMAP projection to 2D
print("Running UMAP...")
umap_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
embedding_2d = umap_reducer.fit_transform(X_pca)

# Step 3: Plot the 2D Real Data Baseline
plt.figure(figsize=(8, 6))
plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], alpha=0.5, s=10, c='royalblue')
plt.title("WGAN Data Audit: Real Hip MRI 2D UMAP Baseline")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig("plots/real_mri_umap_audit.png", dpi=300, bbox_inches='tight')
plt.show()
print("Saved baseline UMAP plot to 'plots/real_mri_umap_audit.png'")

# ==========================================
# MEAN, VARIANCE, SKEWNESS, KURTOSIS AUDIT
# ==========================================

print("Collecting pixel data for statistical audit...")
max_samples_stats = 2000  
all_pixels = []

num_to_load = min(max_samples_stats, len(train_dataset))
for i in range(num_to_load):
    img = train_dataset[i]
    if isinstance(img, (tuple, list)):
        img = img[0]
    
    all_pixels.append(img.detach().cpu().numpy().flatten())

pixel_array = np.concatenate(all_pixels)

mean_val = np.mean(pixel_array)
var_val = np.var(pixel_array)
std_val = np.std(pixel_array)
skew_val = stats.skew(pixel_array)
kurt_val = stats.kurtosis(pixel_array)

print("\n" + "="*40)
print("   HIP MRI DATASET STATISTICAL AUDIT  ")
print("="*40)
print(f"Total Pixels Analyzed : {len(pixel_array):,}")
print(f"Mean                  : {mean_val:.4f}")
print(f"Variance              : {var_val:.4f}")
print(f"Standard Deviation    : {std_val:.4f}")
print(f"Skewness              : {skew_val:.4f}")
print(f"Excess Kurtosis       : {kurt_val:.4f}")
print("="*40)

# ==========================================
# OUTLIER DETECTION AUDIT
# ==========================================

print("Scanning dataset for per-image means...")
image_means = []

for i in range(len(train_dataset)):
    img = train_dataset[i]
    if isinstance(img, (tuple, list)):
        img = img[0]
    image_means.append(img.mean().item())

image_means = np.array(image_means)

dataset_mean_val = image_means.mean()
distances = np.abs(image_means - dataset_mean_val)
outlier_indices = np.argsort(distances)[-10:]  

print(f"Top 10 outlier image indices: {outlier_indices}")

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.flatten()

for plot_idx, dataset_idx in enumerate(outlier_indices):
    sample = train_dataset[dataset_idx]
    img_tensor = sample[0] if isinstance(sample, (tuple, list)) else sample
    
    img_np = img_tensor.squeeze(0).cpu().numpy()
    
    axes[plot_idx].imshow(img_np, cmap='gray')
    axes[plot_idx].set_title(f"Idx: {dataset_idx}\nMean: {image_means[dataset_idx]:.3f}", fontsize=10)
    axes[plot_idx].axis('off')

plt.suptitle("WGAN Data Audit: Top 10 Outlier MRI Slices", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("plots/mri_outliers_audit.png", dpi=300, bbox_inches='tight')
plt.show()
print("Saved outlier comparison grid to 'plots/mri_outliers_audit.png'")

# =========================================
#  Random Sampling
# =========================================

# Randomly sample 10 unique indices from the dataset
dataset_len = len(train_dataset)
sampled_indices = np.random.choice(dataset_len, size=10, replace=False)

print(f"Randomly sampled image indices: {sampled_indices}")

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.flatten()

for plot_idx, dataset_idx in enumerate(sampled_indices):
    sample = train_dataset[dataset_idx]
    img_tensor = sample[0] if isinstance(sample, (tuple, list)) else sample
    
    img_np = img_tensor.squeeze(0).cpu().numpy()
    
    axes[plot_idx].imshow(img_np, cmap='gray')
    
    # Optional: Display the mean if image_means is available, or calculate it directly from the tensor
    mean_val = image_means[dataset_idx] if 'image_means' in locals() else img_np.mean()
    axes[plot_idx].set_title(f"Idx: {dataset_idx}\nMean: {mean_val:.3f}", fontsize=10)
    axes[plot_idx].axis('off')

plt.suptitle("WGAN Data Audit: Randomly Sampled MRI Slices", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("plots/mri_random_samples.png", dpi=300, bbox_inches='tight')
plt.show()

print("Saved random sample comparison grid to 'plots/mri_random_samples.png'")