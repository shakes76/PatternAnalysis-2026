import time
import torch 
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure 
import matplotlib.pyplot as plt 

from dataset import get_dataloaders, IMG_DIRS, _find, load_slices, pad_multiple
from modules import BaseModel, BaseVAE, VQVAE, CodePrior
from predict import GRID, PRIOR_CKPT, ROOT, load_model

ROOT = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"
#ROOT = "C:/Users/danfu/PatternAnalysis-2026/data/"
EPOCHS = 80
BATCH_SIZE = 64
SUBSET = None 
NUM_WORKERS = 0
TARGET_BETA = 1e-1
WARMUP = 0


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #standard device def 

def main(model, ckpt_path):
    train_loader, validate_loader, test_loader = get_dataloaders(ROOT, batch_size=BATCH_SIZE, subset=SUBSET, num_workers=NUM_WORKERS) #grab data using dataset.py
    #send given model, ssim to device and define adam optimiser
    model = model.to(device) 
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    #reset the tracked memory stats so measurement is accurate 
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    best_ssim = 0.0

    #training loop. One pass over the data per epoch 
    for epoch in range (EPOCHS):
        model.beta = TARGET_BETA * min(1.0, epoch / 15)
        start = time.time()
        model.train() #model in training mode 
        total = 0.0 
        #loads a batch from the dataloader (subset of 32 shuffled training images)
        for x in train_loader:
            x = x.to(device)
            recon, extra_loss = model(x)
            loss = F.mse_loss(recon, x) + extra_loss #MSE of original from model reconstruction 
            optimiser.zero_grad() #clear old gradients 
            #backpropogate 
            loss.backward() 
            optimiser.step()

            #adds loss weighted by batch size (so the smaller batch at the end doesn't skew data)
            total += loss.item() * len(x)
        train_loss = total / len(train_loader.dataset)        
        model.eval() #eval mode
        ssim.reset()

        #model validation after each epoch is done 
        with torch.no_grad():
            #loads batches from validation loader and generates SSIM 
            for x in validate_loader:
                x = x.to(device)
                recon, _ = model(x)
                ssim.update(recon, x)
        val_ssim = ssim.compute().item()
        if epoch == 0 or (epoch >= WARMUP and val_ssim > best_ssim):
            best_ssim = val_ssim
            torch.save(model.state_dict(), ckpt_path)


        print(f"epoch: {epoch + 1}: train MSE {train_loss:.4f}, validate SSIM {ssim.compute():.3f}, {time.time() - start:.1f} seconds")

    ssim.reset()
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    #model testing after the training is done 
    with torch.no_grad():
        #loads batches from test loader and generates SSIM 
        for x  in test_loader:
            x = x.to(device)
            recon, _ = model(x)
            ssim.update(recon, x) 
    print(f"test SSIM: {ssim.compute():.3f}")

    #stats if i ran it on RANGPUR
    if device.type == "cuda":
        print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

def save_samples(model, device, path="samples.png", n=6, seed=0):
    model.eval()
    torch.manual_seed(seed)

    with torch.no_grad():
        images = model.sample(n, device=device).cpu()

    fig, axes = plt.subplots(1, n, figsize=(2*n, 4))
    for ax, img in zip(axes, images):
        ax.imshow(img.squeeze(), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    plt.suptitle("generated samples")
    plt.savefig(path)
    plt.close()
    
def train_basemodel():
    main(BaseModel(latent_dim=64), "base_best.pth")

def train_vae():
    model = BaseVAE(latent_dim=8)
    main(model, "vae_best.pth")
    save_samples(model, device, path="vae_sample.png")

def train_vq_vae():
    main(VQVAE(), VQ_CKPT)

VQ_CKPT = "vqvae_best.pth"
PRIOR_CKPT = "prior_best.pth"
PRIOR_EPOCHS = 60
PRIOR_BATCH_SIZE = 64
PRIOR_N_SAMPLES = 16
SEED = 0

@torch.no_grad()
def encode_split(vq, loader):
    #Turn every image in a split into its (H*W,) sequence of code indices.
    out = []
    for x in loader:
        out.append(vq.encode_indices(x.to(device)).flatten(1).cpu())
    return torch.cat(out)
 
#depreciated
# def save_grid(imgs, path, title):
#     cols = 8
#     rows = (len(imgs) + cols - 1) // cols
#     fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 4 * rows), squeeze=False)
#     for ax in axes.ravel():
#         ax.axis("off")
#     for ax, img in zip(axes.ravel(), imgs):
#         ax.imshow(img.squeeze(), cmap="gray", vmin=0, vmax=1)
#     plt.suptitle(title)
#     plt.savefig(path, bbox_inches="tight")
#     plt.close()
 
def save_grid(imgs, path, titles=None, cols=8, suptitle=""):
    rows = (len(imgs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2.3 * rows * 2), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for k, (ax, img) in enumerate(zip(axes.ravel(), imgs)):
        ax.imshow(img.squeeze(), cmap="gray", vmin=0, vmax=1)
        if titles:
            ax.set_title(titles[k], fontsize=7)
    plt.suptitle(suptitle)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print("saved", path)

 
def train_prior():
    torch.manual_seed(SEED)
    train_loader, val_loader, _ = get_dataloaders(ROOT, batch_size=PRIOR_BATCH_SIZE)
 
    vq = VQVAE().to(device)
    vq.load_state_dict(torch.load(VQ_CKPT, map_location=device))
    vq.eval()
 
    # encode once; the prior trains on code sequences, not images
    train_codes, val_codes = encode_split(vq, train_loader), encode_split(vq, val_loader)
    seq_len = train_codes.shape[1]
    sample_x = next(iter(val_loader)).to(device)
    h, w = vq.encode_indices(sample_x[:1]).shape[1:]  # latent grid size (e.g. 32 x 16)
    print(f"codes: train {tuple(train_codes.shape)}, val {tuple(val_codes.shape)}, grid {h}x{w}")
 
    prior = CodePrior(num_codes=vq.quant.num_codes, seq_len=seq_len).to(device)
    opt = torch.optim.Adam(prior.parameters(), lr=3e-4)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
 
    best = float("inf")
    for epoch in range(PRIOR_EPOCHS):
        start = time.time()
        prior.train()
        perm = torch.randperm(len(train_codes))
        total = 0.0
        for i in range(0, len(perm), PRIOR_BATCH_SIZE):
            codes = train_codes[perm[i:i + PRIOR_BATCH_SIZE]].to(device)
            loss = prior.loss(codes)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(prior.parameters(), 1.0)  # gradient hygiene
            opt.step()
            total += loss.item() * len(codes)
 
        prior.eval()
        with torch.no_grad():
            val = sum(prior.loss(val_codes[i:i + PRIOR_BATCH_SIZE].to(device)).item() * len(val_codes[i:i + PRIOR_BATCH_SIZE])
                      for i in range(0, len(val_codes), PRIOR_BATCH_SIZE)) / len(val_codes)
        if val < best:
            best = val
            torch.save(prior.state_dict(), PRIOR_CKPT)
        print(f"epoch {epoch + 1}: train CE {total / len(train_codes):.3f}, "
              f"val CE {val:.3f}, {time.time() - start:.1f}s")
 
    if device.type == "cuda":
        print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
 
    # generate with fixed seed (uncurated)
    prior.load_state_dict(torch.load(PRIOR_CKPT, map_location=device))
    prior.eval()
    torch.manual_seed(SEED)
    t0 = time.time()
    codes = prior.sample(PRIOR_N_SAMPLES, device=device).view(PRIOR_N_SAMPLES, h, w)
    imgs = vq.decode_indices(codes).cpu()
    print(f"generation time for {PRIOR_N_SAMPLES} images: {time.time() - t0:.1f}s")
    save_grid(imgs, "vqvae_generated.png", f"VQ-VAE + prior samples (seed {SEED})")
    print("saved vqvae_generated.png")



#----------------------EVALUATION STUFF-----------------
from tqdm import tqdm
import json 
import glob
import sys
from torchmetrics.functional.image import structural_similarity_index_measure as ssim_fn

def load_split(split, step=1):
    paths = _find(ROOT, IMG_DIRS[split])[::step]
    return pad_multiple(torch.from_numpy(load_slices(paths)), 8)


@torch.no_grad()
def nearest_ssim(queries, ref, chunk=256):
    #For each query: the highest SSIM against any slice in ref, and that slice's index.
    best_s = torch.full((len(queries),), -1.0)
    best_i = torch.zeros(len(queries), dtype=torch.long)
    for qi in tqdm(range(len(queries)), desc="nearest neighbour", leave=False):
        q = queries[qi:qi + 1].to(device)
        for s in range(0, len(ref), chunk):
            r = ref[s:s + chunk].to(device)
            sc = ssim_fn(q.expand(len(r), -1, -1, -1), r, data_range=1.0, reduction="none")
            m, j = sc.max(0)
            if m.item() > best_s[qi]:
                best_s[qi], best_i[qi] = m.item(), s + j.item()
    return best_s, best_i

@torch.no_grad()
def mean_pairwise_ssim(x):
    #Mean SSIM over all distinct pairs in x (high = samples look alike = low diversity).
    x = x.to(device)
    vals = [ssim_fn(x[i:i + 1].expand(len(x) - i - 1, -1, -1, -1), x[i + 1:],
                    data_range=1.0, reduction="none") for i in range(len(x) - 1)]
    return torch.cat(vals).mean().item()


@torch.no_grad()
def recon_ssim(model, test):
    #Paired metric: each unseen test slice vs ITS OWN reconstruction (SSIM target specified by wei dai on ed).
    s, m = [], []
    for i in range(0, len(test), 64):
        x = test[i:i + 64].to(device)
        r, _ = model(x)
        s.append(ssim_fn(r, x, data_range=1.0, reduction="none").cpu())
        m.append(((r - x) ** 2).mean(dim=(1, 2, 3)).cpu())
    return torch.cat(s).mean().item(), torch.cat(m).mean().item()



@torch.no_grad()
def fid_score(real, fake):
    #FID via torchmetrics (pretrained Inception-v3 used ONLY as the evaluation feature extractor; all models are trained from scratch).
    from torchmetrics.image.fid import FrechetInceptionDistance
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)  # inputs in [0, 1]
    for imgs, is_real in ((real, True), (fake, False)):
        for i in range(0, len(imgs), 50):
            fid.update(imgs[i:i + 50].to(device).repeat(1, 3, 1, 1), real=is_real)  # grey -> 3 channels
    return fid.compute().item()


@torch.no_grad()
def codebook_usage(vq, test):
    K = vq.quant.num_codes
    counts = torch.zeros(K, device=device)
    for i in range(0, len(test), 64):
        counts += torch.bincount(vq.encode_indices(test[i:i + 64].to(device)).flatten(), minlength=K).float()
    p = counts / counts.sum()
    return int((counts > 0).sum()), torch.exp(-(p * torch.log(p + 1e-10)).sum()).item()


# ---------------------------------------------------------------- generation
@torch.no_grad()
def sample(kind, model, prior, n, seed, temperature=1.0):
    torch.manual_seed(seed)  # fixed seed -> uncurated and reproducible
    if kind == "vqvae":
        codes = prior.sample(n, temperature=temperature, device=device).view(n, *GRID)
        return model.decode_indices(codes).cpu()
    return model.decoder(torch.randn(n, model.latent_dim, *GRID, device=device)).cpu()


@torch.no_grad()
def interpolate(kind, model, a, b, steps=8):
    #Linear interpolation between two real slices in latent space. VQ-VAE: interpolate encoder outputs, then snap to codes (discrete, so changes come in steps).
    x = torch.stack([a, b]).to(device)
    z = model.encoder(x).chunk(2, dim=1)[0] if kind == "vae" else model.encoder(x)
    t = torch.linspace(0, 1, steps, device=device).view(-1, 1, 1, 1)
    zi = (1 - t) * z[0:1] + t * z[1:2]
    if kind == "vqvae":
        zi = model.quant(zi)[0]
    return model.decoder(zi).cpu()


# ---------------------------------------------------------------- figures



def save_pairs(gen, nn_s, nn_i, train, k, path):
    #The k generated images closest to the training set, next to their nearest training slice.
    order = nn_s.argsort(descending=True)[:k]
    fig, axes = plt.subplots(2, k, figsize=(2.2 * k, 5.5), squeeze=False)
    for c, g in enumerate(order):
        axes[0, c].imshow(gen[g].squeeze(), cmap="gray", vmin=0, vmax=1)
        axes[1, c].imshow(train[nn_i[g]].squeeze().cpu(), cmap="gray", vmin=0, vmax=1)
        axes[0, c].set_title(f"generated #{int(g)}", fontsize=8)
        axes[1, c].set_title(f"nearest train\nSSIM {nn_s[g]:.2f}", fontsize=8)
        axes[0, c].axis("off")
        axes[1, c].axis("off")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print("saved", path)


def save_interpolations(kind, model, train, tag):
    g = torch.Generator().manual_seed(0)
    pairs = torch.randperm(len(train), generator=g)[:6].view(3, 2)  # random pairs, not hand-picked
    fig, axes = plt.subplots(3, 8, figsize=(16, 9))
    for r, (i, j) in enumerate(pairs):
        for c, img in enumerate(interpolate(kind, model, train[i], train[j])):
            axes[r, c].imshow(img.squeeze(), cmap="gray", vmin=0, vmax=1)
            axes[r, c].axis("off")
    plt.suptitle(f"{kind} latent interpolation (left = real slice A, right = real slice B)")
    plt.savefig(f"{tag}_interpolation.png", bbox_inches="tight")
    plt.close()
    print(f"saved {tag}_interpolation.png")


# ---------------------------------------------------------------- full evaluation
def run(kind, temperature=1.0, quick=False, n_gen=64): #quick for running on local cqu (still takes ages but good estimator)
    tag = f"{kind}_t{temperature}" if kind == "vqvae" else kind
    model = load_model(kind)
    prior = None
    if kind == "vqvae":
        prior = CodePrior(num_codes=model.quant.num_codes, seq_len=GRID[0] * GRID[1]).to(device)
        prior.load_state_dict(torch.load(PRIOR_CKPT, map_location=device))
        prior.eval()

    train = load_split("train", step=10 if quick else 1).to(device)
    test = load_split("test")
    n_test_q = 50 if quick else 200
    test_q = test[torch.linspace(0, len(test) - 1, n_test_q).long()]  # spread across test patients
    res = {"model": tag, "n_gen": n_gen, "n_train_ref": len(train), "n_test_queries": n_test_q}

    # reconstruction fidelity
    res["recon_ssim_test"], res["recon_mse_test"] = recon_ssim(model, test)
    if kind == "vqvae":
        res["codes_used"], res["perplexity"] = codebook_usage(model, test)
        res["codebook_size"] = model.quant.num_codes

    # generation (and its runtime)
    t0 = time.time()
    gen = sample(kind, model, prior, n_gen, seed=0, temperature=temperature)
    if device.type == "cuda":
        torch.cuda.synchronize()
    res["gen_time_s"] = time.time() - t0

    # 1. SSIM of generated samples against their nearest real (training) slice
    gen_nn, gen_idx = nearest_ssim(gen, train)
    res["gen_nn_ssim_mean"], res["gen_nn_ssim_std"] = gen_nn.mean().item(), gen_nn.std().item()

    # 2. memorisation audit: held-out test slices' nearest-training SSIM is the "not memorised" baseline
    test_nn, _ = nearest_ssim(test_q, train)
    res["test_nn_ssim_mean"], res["test_nn_ssim_std"] = test_nn.mean().item(), test_nn.std().item()
    p95 = torch.quantile(test_nn, 0.95).item()
    res["test_nn_ssim_p95"] = p95
    res["gen_frac_above_test_p95"] = (gen_nn > p95).float().mean().item()  # ~0.05 expected if no memorisation
    res["gen_nn_ssim_max"] = gen_nn.max().item()
    save_pairs(gen, gen_nn, gen_idx, train, 5, f"{tag}_nearest_pairs.png")

    # 3. diversity: pairwise SSIM among samples vs among real slices
    res["pairwise_ssim_gen"] = mean_pairwise_ssim(gen)
    res["pairwise_ssim_real"] = mean_pairwise_ssim(test[torch.linspace(0, len(test) - 1, n_gen).long()])

    # FID: generated samples vs the unseen test set (pretrained Inception for evaluation only).
    # FID is biased for small n, so use the same n for every model and compare relatively.
    n_fid = 100 if quick else 500
    fake = torch.cat([sample(kind, model, prior, 50, 1000 + k, temperature) for k in range(n_fid // 50)])
    res["fid"], res["fid_n"] = fid_score(test, fake), n_fid

    # 6. uncurated grids: all n_gen samples labelled by index + nearest-train SSIM (pick 3-5 flawed ones
    #    for the autopsy by index), plus two more seeds
    save_grid(gen, f"{tag}_samples_seed0.png", cols=8,
              titles=[f"#{i} nn={s:.2f}" for i, s in enumerate(gen_nn)],
              suptitle=f"{tag}: {n_gen} uncurated samples (seed 0)")
    for seed in (1, 2):
        save_grid(sample(kind, model, prior, 16, seed, temperature), f"{tag}_samples_seed{seed}.png",
                  suptitle=f"{tag}: 16 uncurated samples (seed {seed})")

    # 7. latent interpolation between random real training slices
    save_interpolations(kind, model, train, tag)

    json.dump(res, open(f"results_{tag}.json", "w"), indent=2)
    for k, v in res.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")


def table():
    files = sorted(glob.glob("results_*.json"))
    rs = [json.load(open(f)) for f in files]
    rows = [("Reconstruction SSIM (test, paired)", "recon_ssim_test"),
            ("Reconstruction MSE (test, paired)", "recon_mse_test"),
            ("FID, generated vs test (lower = better)", "fid"),
            ("[memorisation audit] Generated -> nearest train SSIM", "gen_nn_ssim_mean"),
            ("[memorisation audit] Test -> nearest train SSIM (baseline)", "test_nn_ssim_mean"),
            ("[memorisation audit] Generated above test 95th pct", "gen_frac_above_test_p95"),
            ("[diversity only] Pairwise SSIM, generated", "pairwise_ssim_gen"),
            ("[diversity only] Pairwise SSIM, real", "pairwise_ssim_real"),
            ("Codes used / perplexity", None),
            ("Generation time for n samples (s)", "gen_time_s")]
    print("| Metric | " + " | ".join(r["model"] for r in rs) + " |")
    print("|---|" + "---|" * len(rs))
    for name, key in rows:
        cells = []
        for r in rs:
            if key is None:
                cells.append(f"{r['codes_used']}/{r['codebook_size']} / {r['perplexity']:.0f}" if "perplexity" in r else "n/a")
            else:
                cells.append(f"{r[key]:.3f}")
        print(f"| {name} | " + " | ".join(cells) + " |")
    print("\nFill in from your training logs: peak VRAM and training time per model (and the prior).")


#when training vae set the warmup to 15

if __name__ == "__main__":
    # WARMUP = 15
    # train_vae()
    # WARMUP = 0
    # train_vq_vae()
    # train_prior()
    run("vae", quick=True)

    for t in (1.0, 0.9):
        run("vqvae", temperature=t, quick=True)
    table()