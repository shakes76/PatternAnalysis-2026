# Results (200 steps, toy task where each code = left neighbour + 1, uniform guess 4.85):
#   before fix: cpu 0.392 | mps 4.853 (4-D loss) | mps 4.853 (flattened loss)
#   after .contiguous() fix in PixelCNNPrior.forward: mps 0.392

import torch, torch.nn.functional as F
from modules import PixelCNNPrior

K, H, W = 128, 32, 16
def run(device, flat):
    torch.manual_seed(0)
    prior = PixelCNNPrior(num_codes=K, channels=64, n_layers=4).to(device)
    opt = torch.optim.Adam(prior.parameters(), lr=3e-4)
    for step in range(1, 201):
        start = torch.randint(0, K, (32, H, 1))
        ids = ((start + torch.arange(W)) % K).to(device)
        logits = prior(ids)
        if flat:   # (B,K,H,W) -> (B*H*W, K): the ordinary 1-D cross-entropy
            loss = F.cross_entropy(logits.permute(0, 2, 3, 1).reshape(-1, K), ids.reshape(-1))
        else:
            loss = F.cross_entropy(logits, ids)
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"{device}, {'flattened' if flat else '4-D'} loss: CE after 200 steps {loss.item():.3f}")

# run("cpu", flat=False)
run("mps", flat=False)
# run("mps", flat=True)