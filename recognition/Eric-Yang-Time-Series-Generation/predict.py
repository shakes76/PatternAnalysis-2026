import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def compute_kl_divergence(p_data: torch.Tensor, q_data: torch.Tensor, num_bins: int = 50) -> float:
    """computes KL divergence between two PyTorch tensors"""
    min_val = min(p_data.min().item(), q_data.min().item())
    max_val = max(p_data.max().item(), q_data.max().item())
    
    if min_val == max_val:
        max_val += 1.0

    bins = torch.linspace(min_val, max_val, num_bins + 1, device=p_data.device)
    
    p_indices = torch.bucketize(p_data, bins[:-1]) - 1
    q_indices = torch.bucketize(q_data, bins[:-1]) - 1
    
    p_indices = torch.clamp(p_indices, 0, num_bins - 1)
    q_indices = torch.clamp(q_indices, 0, num_bins - 1)
    
    p_hist = torch.zeros(num_bins, device=p_data.device)
    q_hist = torch.zeros(num_bins, device=q_data.device)
    
    p_hist.scatter_add_(0, p_indices, torch.ones_like(p_indices, dtype=torch.float32))
    q_hist.scatter_add_(0, q_indices, torch.ones_like(q_indices, dtype=torch.float32))

    eps = 1e-8
    p_hist = p_hist + eps
    q_hist = q_hist + eps

    p_hist = p_hist / p_hist.sum()
    q_hist = q_hist / q_hist.sum()

    kl = torch.sum(p_hist * torch.log(p_hist / q_hist))
    return float(kl.item())

# used https://www.ibm.com/think/topics/autocorrelation for clarification
def compute_autocorrelation(series: torch.Tensor, max_lag: int = 20) -> torch.Tensor:
    """computes autocorrelation function (ACF)"""
    n = series.numel()
    mean = series.mean()
    var = series.var(unbiased=False) + 1e-8
    
    acf = []
    for lag in range(max_lag):
        if lag == 0:
            acf.append(1.0)
        else:
            covariance = torch.sum((series[:-lag] - mean) * (series[lag:] - mean)) / n
            acf.append((covariance / var).item())
            
    return torch.tensor(acf, device=series.device)

# calculation done with the help of https://en.wikipedia.org/wiki/Structural_similarity_index_measure 
def compute_ssim(real_depth: torch.Tensor, synth_depth: torch.Tensor) -> float:
    """computes a structural similarity (SSIM) proxy between mean volume depth heatmaps"""
    real_mean = real_depth.mean(dim=0)
    synth_mean = synth_depth.mean(dim=0)

    mu_x = real_mean.mean()
    mu_y = synth_mean.mean()

    var_x = real_mean.var(unbiased=False)
    var_y = synth_mean.var(unbiased=False)
    
    flat_real = real_mean.flatten()
    flat_synth = synth_mean.flatten()
    cov_xy = torch.mean((flat_real - mu_x) * (flat_synth - mu_y))

    c1 = (0.01 * 1.0) ** 2
    c2 = (0.03 * 1.0) ** 2

    ssim_val = ((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) / ((mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2))
    return float(ssim_val.item())

def microstructure_invariant_violations(synthetic_data: torch.Tensor) -> float:
    """checks percentage of samples violating microstructure invariant (P_bid >= P_ask)"""
    violations = 0
    total_samples = synthetic_data.size(0)
    
    for seq in synthetic_data:
        ask_returns = seq[:, 0]
        bid_returns = seq[:, 1]
        
        if torch.any(bid_returns > ask_returns + 1e-4):
            violations += 1
            
    return (violations / total_samples) * 100.0


def main():
    synthetic_path = 'output/synthetic_lob_train.pt'
    real_path = 'output/real_lob_test.pt'

    if not os.path.exists(synthetic_path) or not os.path.exists(real_path):
        print("Error: PyTorch output files not found. Run train.py")
        return

    synthetic_data = torch.load(synthetic_path, map_location=torch.device('cpu'))    
    real_data = torch.load(real_path)

    print(f"Loaded {synthetic_data.size(0)} synthetic windows and {real_data.size(0)} real test windows.")
        
    real_returns = real_data[:, :, 0].reshape(-1)
    synth_returns = synthetic_data[:, :, 0].reshape(-1)

    kl_returns = compute_kl_divergence(real_returns, synth_returns)
    print(f"\nKL divergence on returns: {kl_returns:.4f}")

    real_acf = compute_autocorrelation(real_returns[:5000], max_lag=20)
    synth_acf = compute_autocorrelation(synth_returns[:5000], max_lag=20)
    
    acf_mse = torch.mean((real_acf - synth_acf) ** 2).item()
    print(f"\nautocorrelation MSE: {acf_mse:.6f}")
    
    ssim_score = compute_ssim(real_data, synthetic_data)
    print(f"\ndepth heatmap SSIM: {ssim_score:.4f}")

    violation_pct = microstructure_invariant_violations(synthetic_data)
    print(f"\nmicrostructure invariant violations: {violation_pct:.2f}%")

    os.makedirs('output', exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(real_acf.cpu().numpy(), label='Real Returns ACF', marker='o', color='blue')
    plt.plot(synth_acf.cpu().numpy(), label='Synthetic Returns ACF', marker='x', color='orange')
    plt.title('Autocorrelation Function (ACF) Comparison')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('output/acf_comparison.png', dpi=300)
    plt.close()

if __name__ == '__main__':
    main()