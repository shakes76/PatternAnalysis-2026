from modules import ConstraintAwareLOBHeadV52, check_raw_constraints
'\nCOMP3710 TimeGAN - formal training entry point.\n\nImplemented stage:\n    depth: V5.29 probabilistic depth network.\n\nAdditional TimeGAN and price-head stages will be\nintegrated after their dependencies are verified.\n'
import argparse

def train_depth(smoke_test=False, data_dir='data', output_dir=None):
    import json
    import random
    from pathlib import Path
    import numpy as np
    import torch
    import torch.nn.functional as F
    from dataset import prepare_dataset_v4
    from modules import ProbabilisticDepthNet, mixture_nll
    SEED = 371029
    EPOCHS = 1 if smoke_test else 15
    BATCH = 2048
    LR = 0.001
    THRESHOLD = 1e-05
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    out = Path(output_dir or ('outputs_smoke_depth' if smoke_test else 'outputs_v5_29_mdn'))
    out.mkdir(exist_ok=True)
    data = prepare_dataset_v4(data_dir)
    norm = data['normalizer']

    def extract(x):
        return np.maximum(np.expm1(x[..., 1:40:2]), 0)
    train = extract(norm.inverse_transform(np.asarray(data['train'], dtype=np.float64)))
    val = extract(norm.inverse_transform(np.asarray(data['val'][:512], dtype=np.float64)))

    def transitions(x):
        log = np.log1p(x)
        current = log[:, :-1]
        following = log[:, 1:]
        changed = (np.abs(np.diff(x, axis=1)) >= THRESHOLD).astype(np.float32)
        return (current.reshape(-1, 20).astype(np.float32), following.reshape(-1, 20).astype(np.float32), changed.reshape(-1, 20))
    xtr, ytr, mtr = transitions(train[:64] if smoke_test else train)
    xva, yva, mva = transitions(val[:32] if smoke_test else val)
    mu = xtr.mean(axis=0)
    sd = xtr.std(axis=0) + 1e-06
    xtr = (xtr - mu) / sd
    ytr = (ytr - mu) / sd
    xva = (xva - mu) / sd
    yva = (yva - mu) / sd
    X = torch.from_numpy(xtr)
    Y = torch.from_numpy(ytr)
    M = torch.from_numpy(mtr)
    XV = torch.from_numpy(xva)
    YV = torch.from_numpy(yva)
    MV = torch.from_numpy(mva)
    model = ProbabilisticDepthNet(levels=20, hidden=128, components=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    def objective(state, target, mask):
        logits, weights, means, log_scales = model(state)
        mask_loss = F.binary_cross_entropy_with_logits(logits, mask)
        per_level_nll = mixture_nll(target, weights, means, log_scales)
        value_loss = (per_level_nll * mask).sum() / mask.sum().clamp_min(1)
        total = mask_loss + 0.5 * value_loss
        return (total, mask_loss, value_loss)
    best_loss = float('inf')
    best_epoch = None
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = torch.randperm(len(X))
        training_losses = []
        for ids in order.split(BATCH):
            loss, _, _ = objective(X[ids], Y[ids], M[ids])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            training_losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            val_loss, mask_loss, value_loss = objective(XV, YV, MV)
        record = {'epoch': epoch, 'train_loss': float(np.mean(training_losses)), 'val_loss': float(val_loss), 'val_mask_loss': float(mask_loss), 'val_value_nll': float(value_loss)}
        history.append(record)
        print(f"Epoch {epoch:02d} | train={record['train_loss']:.4f} | val={record['val_loss']:.4f} | mask={record['val_mask_loss']:.4f} | mdn_nll={record['val_value_nll']:.4f}", flush=True)
        if record['val_loss'] < best_loss:
            best_loss = record['val_loss']
            best_epoch = epoch
            torch.save({'model_state_dict': model.state_dict(), 'mu': torch.as_tensor(mu), 'sd': torch.as_tensor(sd), 'epoch': epoch, 'components': 3, 'hidden': 128}, out / 'best_mdn.pt')
    with open(out / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    print('\n=== V5.29 TRAINING COMPLETE ===')
    print('Best epoch:', best_epoch)
    print('Best validation loss:', best_loss)
    print('Checkpoint:', out / 'best_mdn.pt')
import copy
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from dataset import prepare_dataset_v4
from modules import TimeGANV42

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def return_aware_loss(prediction, target, zero_normalized, return_weight, move_weight):
    lob_loss = nn.functional.mse_loss(prediction[..., :40], target[..., :40])
    squared_error = (prediction[..., 40] - target[..., 40]).square()
    moved = (target[..., 40] - zero_normalized).abs() > 0.0001
    weights = torch.where(moved, move_weight, 1.0).to(squared_error.dtype)
    weighted_return_loss = (weights * squared_error).sum() / weights.sum()
    return lob_loss + return_weight * weighted_return_loss

def reconstruct_metrics(model, data, normalizer, device):
    model.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(data), 128):
            x = torch.from_numpy(data[start:start + 128]).float().to(device)
            outputs.append(model.reconstruct(x).cpu().numpy())
    reconstruction = np.concatenate(outputs)
    real = normalizer.inverse_transform(data)[..., -1].reshape(-1)
    predicted = normalizer.inverse_transform(reconstruction)[..., -1].reshape(-1)
    if np.std(real) and np.std(predicted):
        corr = float(np.corrcoef(real, predicted)[0, 1])
    else:
        corr = float('nan')
    moved = np.abs(real) > 1e-12
    return {'corr': corr, 'real_std': float(np.std(real)), 'recon_std': float(np.std(predicted)), 'mae': float(np.mean(np.abs(real - predicted))), 'move_mae': float(np.mean(np.abs(real[moved] - predicted[moved]))) if moved.any() else float('nan'), 'move_fraction': float(moved.mean()), 'lob_mse': float(np.mean((data[..., :40] - reconstruction[..., :40]) ** 2))}

def train_stage1(args):
    if args.epochs < 1 or args.return_weight <= 0 or args.move_weight < 1:
        parser.error('epochs >= 1, return-weight > 0 and move-weight >= 1 required')
    seed_everything(args.seed)
    device = torch.device(args.device)
    data = prepare_dataset_v4(args.data_dir)
    if args.smoke_test:
        data['train'] = data['train'][:64]
        data['val'] = data['val'][:64]
        data['test'] = data['test'][:64]
    norm = data['normalizer']
    zero_normalized = -float(norm.return_mean) / float(norm.return_std)
    print('Raw zero return normalized as:', zero_normalized)
    print('Train / val / test:', *(data[s].shape for s in ('train', 'val', 'test')))
    seed_everything(args.seed)
    reference = TimeGANV42(feature_dim=41, hidden_dim=24, noise_dim=41, num_layers=2)
    initial_state = copy.deepcopy(reference.state_dict())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for variant in ('ordinary_mse', 'weighted_returns'):
        seed_everything(args.seed)
        model = TimeGANV42(feature_dim=41, hidden_dim=24, noise_dim=41, num_layers=2).to(device)
        model.load_state_dict(initial_state)
        opt = torch.optim.Adam(list(model.embedder.parameters()) + list(model.recovery.parameters()), lr=0.001)
        generator = torch.Generator().manual_seed(args.seed)
        loader = DataLoader(TensorDataset(torch.from_numpy(data['train']).float()), batch_size=args.batch_size, shuffle=True, generator=generator)
        print(f'\n=== {variant} ===', flush=True)
        for epoch in range(args.epochs):
            model.train()
            total = 0.0
            for x, in loader:
                x = x.to(device)
                prediction = model.reconstruct(x)
                if variant == 'ordinary_mse':
                    loss = nn.functional.mse_loss(prediction, x)
                else:
                    loss = return_aware_loss(prediction, x, zero_normalized, args.return_weight, args.move_weight)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                total += loss.item() * len(x)
            val_metrics = reconstruct_metrics(model, data['val'], norm, device)
            print(f"Epoch {epoch + 1}/{args.epochs} | Train objective {total / len(data['train']):.6f} | Val corr {val_metrics['corr']:.5f} | Val return std {val_metrics['recon_std']:.8g}", flush=True)
        test_metrics = reconstruct_metrics(model, data['test'], norm, device)
        print('Test metrics (do not tune using test set):')
        for key, value in test_metrics.items():
            print(f'  {key}: {value:.10g}')
        ckpt = output_dir / f'{variant}.pt'
        torch.save({'model_state_dict': model.state_dict(), 'model_version': 'v4.2', 'dataset_version': 'v4', 'stage': 'embedding', 'variant': variant, 'return_scale': float(norm.return_scale), 'return_mean': float(norm.return_mean), 'return_std': float(norm.return_std), 'epochs': args.epochs, 'seed': args.seed}, ckpt)
        print('Saved:', ckpt)
    print('\nStage-1 comparison completed. No Stage 2/3 performed.')
import time

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def create_loader(data, batch_size, shuffle=False):
    tensor = torch.from_numpy(data).float()
    return DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=shuffle)

def reconstruction_loss(model, batch, criterion):
    reconstructed = model.reconstruct(batch)
    return criterion(reconstructed, batch)

def validate_embedding(model, loader, device, criterion):
    model.eval()
    total = 0.0
    count = 0
    for batch, in loader:
        batch = batch.to(device)
        loss = reconstruction_loss(model, batch, criterion)
        total += loss.item() * len(batch)
        count += len(batch)
    return total / count

def train_embedding(model, train_loader, val_loader, device, epochs=10, lr=0.001):
    optimizer = torch.optim.Adam(list(model.embedder.parameters()) + list(model.recovery.parameters()), lr=lr)
    criterion = nn.MSELoss()
    history = []
    print('\n=== Stage 1: Embedding Pretraining ===', flush=True)
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for batch, in train_loader:
            batch = batch.to(device)
            loss = reconstruction_loss(model, batch, criterion)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
        train_loss = total / len(train_loader.dataset)
        val_loss = validate_embedding(model, val_loader, device, criterion)
        history.append({'epoch': epoch + 1, 'train_loss': train_loss, 'val_loss': val_loss})
        print(f'Epoch {epoch + 1}/{epochs} | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}', flush=True)
    return (history, optimizer)

def supervised_loss(model, batch, criterion):
    with torch.no_grad():
        h = model.embedder(batch)
    h_pred = model.supervisor(h)
    return criterion(h_pred[:, :-1, :], h[:, 1:, :])

def validate_supervisor(model, loader, device, criterion):
    model.eval()
    total = 0.0
    count = 0
    for batch, in loader:
        batch = batch.to(device)
        loss = supervised_loss(model, batch, criterion)
        total += loss.item() * len(batch)
        count += len(batch)
    return total / count

def train_supervisor(model, train_loader, val_loader, device, epochs=10, lr=0.001):
    optimizer = torch.optim.Adam(model.supervisor.parameters(), lr=lr)
    criterion = nn.MSELoss()
    history = []
    print('\n=== Stage 2: Supervisor Pretraining ===', flush=True)
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for batch, in train_loader:
            batch = batch.to(device)
            loss = supervised_loss(model, batch, criterion)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
        train_loss = total / len(train_loader.dataset)
        val_loss = validate_supervisor(model, val_loader, device, criterion)
        history.append({'epoch': epoch + 1, 'train_loss': train_loss, 'val_loss': val_loss})
        print(f'Epoch {epoch + 1}/{epochs} | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}', flush=True)
    return (history, optimizer)

def moment_loss(real, fake):
    """Match feature-wise means and standard deviations."""
    real_mean = real.mean(dim=(0, 1))
    fake_mean = fake.mean(dim=(0, 1))
    real_std = real.std(dim=(0, 1), unbiased=False)
    fake_std = fake.std(dim=(0, 1), unbiased=False)
    mean_error = torch.mean(torch.abs(real_mean - fake_mean))
    std_error = torch.mean(torch.abs(real_std - fake_std))
    return mean_error + std_error

def joint_losses(model, real, noise):
    """
    Compute the main TimeGAN joint-training losses.

    This function calculates losses but does not update weights.
    """
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()
    h = model.embedder(real)
    x_tilde = model.recovery(h)
    h_supervised = model.supervisor(h)
    e_hat = model.generator(noise)
    h_hat = model.supervisor(e_hat)
    x_hat = model.recovery(h_hat)
    y_real = model.discriminator(h)
    y_fake = model.discriminator(h_hat)
    y_fake_e = model.discriminator(e_hat)
    loss_recon = mse(real, x_tilde)
    loss_sup = mse(h_supervised[:, :-1, :], h[:, 1:, :])
    loss_d_real = bce(y_real, torch.ones_like(y_real))
    loss_d_fake = bce(y_fake, torch.zeros_like(y_fake))
    loss_d_fake_e = bce(y_fake_e, torch.zeros_like(y_fake_e))
    loss_d = loss_d_real + loss_d_fake + loss_d_fake_e
    loss_g_adv = bce(y_fake, torch.ones_like(y_fake)) + bce(y_fake_e, torch.ones_like(y_fake_e))
    loss_moment = moment_loss(real, x_hat)
    return {'reconstruction': loss_recon, 'supervised': loss_sup, 'discriminator': loss_d, 'generator_adversarial': loss_g_adv, 'moment': loss_moment}

def set_requires_grad(network, enabled):
    """Control which network parameters accumulate gradients."""
    for parameter in network.parameters():
        parameter.requires_grad_(enabled)

def joint_train_step(model, real, optimizers, generator_updates=1, zero_normalized=0.0, return_weight=1.0, move_weight=5.0):
    """One joint batch: G/S repeated, then E/R once and D once."""
    if generator_updates < 1:
        raise ValueError('generator_updates must be >= 1')
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()
    opt_g, opt_er, opt_d = optimizers
    batch_size, seq_len, _ = real.shape

    def sample_noise():
        return torch.rand(batch_size, seq_len, model.noise_dim, device=real.device)
    set_requires_grad(model.discriminator, False)
    set_requires_grad(model.embedder, False)
    set_requires_grad(model.recovery, False)
    set_requires_grad(model.generator, True)
    set_requires_grad(model.supervisor, True)
    with torch.no_grad():
        h_real = model.embedder(real)
    g_total = adv_total = sup_total = moment_total = 0.0
    for _ in range(generator_updates):
        z = sample_noise()
        e_fake = model.generator(z)
        h_fake = model.supervisor(e_fake)
        x_fake = model.recovery(h_fake)
        h_pred = model.supervisor(h_real)
        logits_h = model.discriminator(h_fake)
        logits_e = model.discriminator(e_fake)
        adv = bce(logits_h, torch.ones_like(logits_h)) + bce(logits_e, torch.ones_like(logits_e))
        sup = mse(h_pred[:, :-1, :], h_real[:, 1:, :])
        moments = moment_loss(real, x_fake)
        loss_g = adv + 100 * torch.sqrt(sup + 1e-08) + 100 * moments
        opt_g.zero_grad(set_to_none=True)
        loss_g.backward()
        opt_g.step()
        opt_g.zero_grad(set_to_none=True)
        g_total += loss_g.item()
        adv_total += adv.item()
        sup_total += sup.item()
        moment_total += moments.item()
    set_requires_grad(model.generator, False)
    set_requires_grad(model.supervisor, False)
    set_requires_grad(model.embedder, True)
    set_requires_grad(model.recovery, True)
    h_real = model.embedder(real)
    x_recon = model.recovery(h_real)
    h_pred = model.supervisor(h_real)
    recon = return_aware_loss(x_recon, real, zero_normalized, return_weight, move_weight)
    sup_er = mse(h_pred[:, :-1, :], h_real[:, 1:, :])
    loss_er = 10 * torch.sqrt(recon + 1e-08) + 0.1 * sup_er
    opt_er.zero_grad(set_to_none=True)
    loss_er.backward()
    opt_er.step()
    opt_er.zero_grad(set_to_none=True)
    set_requires_grad(model.embedder, False)
    set_requires_grad(model.recovery, False)
    set_requires_grad(model.discriminator, True)
    with torch.no_grad():
        h_real = model.embedder(real)
        e_fake = model.generator(sample_noise())
        h_fake = model.supervisor(e_fake)
    real_logits = model.discriminator(h_real)
    fake_logits = model.discriminator(h_fake)
    fake_e_logits = model.discriminator(e_fake)
    loss_d = bce(real_logits, torch.ones_like(real_logits)) + bce(fake_logits, torch.zeros_like(fake_logits)) + bce(fake_e_logits, torch.zeros_like(fake_e_logits))
    opt_d.zero_grad(set_to_none=True)
    loss_d.backward()
    opt_d.step()
    opt_d.zero_grad(set_to_none=True)
    for network in (model.generator, model.supervisor, model.embedder, model.recovery, model.discriminator):
        set_requires_grad(network, True)
    return {'generator': g_total / generator_updates, 'generator_adversarial': adv_total / generator_updates, 'supervised': sup_total / generator_updates, 'moment': moment_total / generator_updates, 'embedder': loss_er.item(), 'discriminator': loss_d.item()}

def measure_diversity(model, device, seq_len, noise_dim, samples=64):
    """Compare independent generated samples using a fixed local RNG seed."""
    was_training = model.training
    model.eval()
    try:
        with torch.random.fork_rng(devices=[device] if device.type == 'cuda' else []):
            torch.manual_seed(12345)
            if device.type == 'cuda':
                torch.cuda.manual_seed_all(12345)
            z1 = torch.rand(samples, seq_len, noise_dim, device=device)
            z2 = torch.rand(samples, seq_len, noise_dim, device=device)
            x1 = model.generate(z1)
            x2 = model.generate(z2)
        return {'output_difference': float((x1 - x2).abs().mean().item()), 'sample_std': float(x1.std(dim=0, unbiased=False).mean().item()), 'temporal_std': float(x1.std(dim=1, unbiased=False).mean().item())}
    finally:
        model.train(was_training)

def train_joint(model, train_loader, device, epochs=5, lr=0.001, generator_updates=1, zero_normalized=0.0, return_weight=1.0, move_weight=5.0):
    """Run joint adversarial training over multiple epochs."""
    opt_g = torch.optim.Adam(list(model.generator.parameters()) + list(model.supervisor.parameters()), lr=lr)
    opt_er = torch.optim.Adam(list(model.embedder.parameters()) + list(model.recovery.parameters()), lr=lr)
    opt_d = torch.optim.Adam(model.discriminator.parameters(), lr=lr)
    history = []
    print('Generator updates per batch:', generator_updates)
    print('Stage 3 learning rate:', lr)
    print('\n=== Stage 3: Joint Adversarial Training ===', flush=True)
    for epoch in range(epochs):
        model.train()
        total_g = 0.0
        total_er = 0.0
        total_d = 0.0
        total_adv = 0.0
        total_sup = 0.0
        total_moment = 0.0
        count = 0
        for batch, in train_loader:
            batch = batch.to(device)
            losses = joint_train_step(model, batch, (opt_g, opt_er, opt_d), generator_updates=generator_updates, zero_normalized=zero_normalized, return_weight=return_weight, move_weight=move_weight)
            n = len(batch)
            total_g += losses['generator'] * n
            total_er += losses['embedder'] * n
            total_d += losses['discriminator'] * n
            total_adv += losses['generator_adversarial'] * n
            total_sup += losses['supervised'] * n
            total_moment += losses['moment'] * n
            count += n
        epoch_losses = {'epoch': epoch + 1, 'generator': total_g / count, 'embedder': total_er / count, 'discriminator': total_d / count, 'generator_adversarial': total_adv / count, 'supervised': total_sup / count, 'moment': total_moment / count}
        diversity = measure_diversity(model, device, seq_len=train_loader.dataset.tensors[0].shape[1], noise_dim=model.noise_dim)
        epoch_losses.update(diversity)
        history.append(epoch_losses)
        print(f"Epoch {epoch + 1}/{epochs} | G: {epoch_losses['generator']:.4f} | ER: {epoch_losses['embedder']:.4f} | D: {epoch_losses['discriminator']:.4f} | Adv: {epoch_losses['generator_adversarial']:.4f} | Moment: {epoch_losses['moment']:.4f} | Sample Std: {epoch_losses['sample_std']:.8f} | Noise Sensitivity: {epoch_losses['output_difference']:.10f}", flush=True)
        if not all((np.isfinite(epoch_losses[k]) for k in epoch_losses if k != 'epoch')):
            raise RuntimeError('Non-finite loss detected during joint training.')
    return history

def audit_reconstruction(model, data, normalizer, device, label):
    """Report validation-only reconstruction performance after each stage."""
    metrics = reconstruct_metrics(model, data, normalizer, device)
    print(f'\n=== {label}: Validation Reconstruction ===', flush=True)
    for key, value in metrics.items():
        print(f'  {key}: {value:.10g}', flush=True)
    return metrics

def train_v43(args, parser):
    if args.supervisor_epochs < 1 or args.joint_epochs < 1 or args.generator_updates < 1 or (args.batch_size < 1) or (args.joint_lr <= 0) or (args.return_weight <= 0) or (args.move_weight < 1):
        parser.error('Invalid training parameters; epochs/updates/batch/lr must be positive')
    set_seed(args.seed)
    start = time.time()
    device = torch.device(args.device)
    result = prepare_dataset_v4(args.data_dir)
    if args.smoke_test:
        result['train'] = result['train'][:64]
        result['val'] = result['val'][:64]
    normalizer = result['normalizer']
    zero_normalized = -float(normalizer.return_mean) / float(normalizer.return_std)
    train_loader = create_loader(result['train'], args.batch_size, shuffle=True)
    val_loader = create_loader(result['val'], args.batch_size)
    ckpt_path = Path(args.stage1_checkpoint)
    if not ckpt_path.is_file():
        parser.error(f'Stage 1 checkpoint not found: {ckpt_path}')
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    if checkpoint.get('variant') != 'weighted_returns':
        parser.error("Stage 1 checkpoint must have variant='weighted_returns'")
    if checkpoint.get('model_version') != 'v4.2' or checkpoint.get('dataset_version') != 'v4':
        parser.error('Stage 1 checkpoint must use model v4.2 and dataset v4')
    if checkpoint.get('seed') != args.seed:
        parser.error('Requested seed does not match the Stage 1 checkpoint seed')
    if not np.isclose(checkpoint.get('return_scale', np.nan), normalizer.return_scale):
        parser.error('Return scale disagrees with current training dataset')
    for key in ('return_mean', 'return_std'):
        if not np.isclose(checkpoint.get(key, np.nan), getattr(normalizer, key), atol=1e-09):
            parser.error(f'{key} disagrees with current dataset')
    for key in ('return_weight', 'move_weight'):
        expected = getattr(args, key)
        if key in checkpoint and (not np.isclose(checkpoint[key], expected)):
            parser.error(f'{key} differs from saved Stage 1 hyperparameter')
    model = TimeGANV42(feature_dim=41, hidden_dim=24, noise_dim=41, num_layers=2).to(device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {'feature_dim': 41, 'hidden_dim': 24, 'noise_dim': 41, 'num_layers': 2, 'dataset_version': 'v4', 'model_version': 'v4.2', 'training_variant': 'v4.3-weighted-reconstruction', 'return_scale': float(normalizer.return_scale), 'return_mean': float(normalizer.return_mean), 'return_std': float(normalizer.return_std), 'return_weight': args.return_weight, 'move_weight': args.move_weight, 'zero_normalized': zero_normalized, 'stage1_checkpoint': str(ckpt_path), 'seed': args.seed}
    print('Dataset version: v4 | Model version: v4.2', flush=True)
    print('Loaded weighted Stage 1 checkpoint:', ckpt_path, flush=True)
    print('Train/validation sequences:', len(result['train']), len(result['val']), flush=True)
    print('Stage 3 return loss weights:', args.return_weight, args.move_weight, flush=True)
    audit_reconstruction(model, result['val'], normalizer, device, 'Stage 1 (loaded)')
    supervisor_history, supervisor_optimizer = train_supervisor(model, train_loader, val_loader, device, epochs=args.supervisor_epochs)
    audit_reconstruction(model, result['val'], normalizer, device, 'Stage 2')
    torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': supervisor_optimizer.state_dict(), 'supervisor_history': supervisor_history, 'stage': 'supervisor', **metadata}, output_dir / 'supervisor.pt')
    joint_history = train_joint(model, train_loader, device, epochs=args.joint_epochs, generator_updates=args.generator_updates, lr=args.joint_lr, zero_normalized=zero_normalized, return_weight=args.return_weight, move_weight=args.move_weight)
    val_metrics = audit_reconstruction(model, result['val'], normalizer, device, 'Stage 3')
    torch.save({'model_state_dict': model.state_dict(), 'joint_history': joint_history, 'supervisor_history': supervisor_history, 'joint_lr': args.joint_lr, 'generator_updates': args.generator_updates, 'validation_reconstruction': val_metrics, 'stage': 'joint', **metadata}, output_dir / 'timegan_joint.pt')
    print('\nSaved checkpoints:', flush=True)
    print(output_dir / 'supervisor.pt', flush=True)
    print(output_dir / 'timegan_joint.pt', flush=True)
    print('Runtime (seconds):', round(time.time() - start, 2), flush=True)
    print('V4.3 Stage 2 and Stage 3 completed!', flush=True)
import json
from torch.nn import functional as F

def v53_seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def v53_evaluate(embedder, recovery, head, arrays, minimum, scale, device, batch_size):
    """Evaluate old Recovery and new Head on the same preselected split."""
    embedder.eval()
    recovery.eval()
    head.eval()
    sums = {k: 0.0 for k in ('old_sq', 'new_sq', 'old_spread_abs', 'new_spread_abs')}
    num_elements = 0
    num_rows = 0
    passed = {key: True for key in ('positive_spread', 'ask_ordered', 'bid_ordered', 'ask_above_mid', 'bid_below_mid', 'nonnegative_sizes', 'finite')}
    with torch.inference_mode():
        for start in range(0, len(arrays), batch_size):
            x = torch.from_numpy(arrays[start:start + batch_size]).float().to(device)
            h = embedder(x)
            old = recovery(h)[..., :40]
            new = head(h)
            real = x[..., :40]
            sums['old_sq'] += (old - real).square().sum().item()
            sums['new_sq'] += (new - real).square().sum().item()
            num_elements += real.numel()
            num_rows += real.shape[0] * real.shape[1]
            true_spread = real[..., 0] * scale[0] + minimum[0] - (real[..., 2] * scale[2] + minimum[2])
            old_spread = old[..., 0] * scale[0] + minimum[0] - (old[..., 2] * scale[2] + minimum[2])
            new_spread = new[..., 0] * scale[0] + minimum[0] - (new[..., 2] * scale[2] + minimum[2])
            sums['old_spread_abs'] += (old_spread - true_spread).abs().sum().item()
            sums['new_spread_abs'] += (new_spread - true_spread).abs().sum().item()
            raw = head.forward_raw(h)
            for key, value in check_raw_constraints(raw).items():
                passed[key] = passed[key] and value
    return {'old_lob_mse_normalized': sums['old_sq'] / num_elements, 'new_lob_mse_normalized': sums['new_sq'] / num_elements, 'old_spread_mae_price_units': sums['old_spread_abs'] / num_rows, 'new_spread_mae_price_units': sums['new_spread_abs'] / num_rows, 'new_head_all_constraints': passed}

def v53_initialise_head_from_train(head, normalizer, train_norm, seed=42):
    """Use only train observations to initialize the 39 raw-output biases.

    Projection weights are deliberately small (not zero), so initial output
    lies near train means while gradient flow remains nonzero.
    """
    train_raw = normalizer.inverse_transform(train_norm)
    book = train_raw[..., :40]
    prices = book[..., 0::2]
    asks, bids = (prices[..., 0::2], prices[..., 1::2])
    spread = asks[..., 0] - bids[..., 0]
    ask_gaps = np.diff(asks, axis=-1)
    bid_gaps = -np.diff(bids, axis=-1)
    transformed_sizes = book[..., 1::2]
    targets = np.concatenate(([spread.mean()], ask_gaps.mean(axis=(0, 1)), bid_gaps.mean(axis=(0, 1)), transformed_sizes.mean(axis=(0, 1))))
    floors = np.concatenate(([head.min_spread], np.full(18, head.min_gap), np.zeros(20)))
    adjusted = targets - floors
    if targets.shape != (39,) or not np.isfinite(targets).all():
        raise ValueError('Invalid output mean targets')
    if np.any(adjusted <= 0):
        raise ValueError('Observed train mean is not above output floor')
    inverse_softplus = np.log(np.expm1(adjusted))
    with torch.no_grad():
        torch.nn.init.normal_(head.projection.weight, mean=0.0, std=0.001)
        head.projection.bias.copy_(torch.as_tensor(inverse_softplus, dtype=head.projection.bias.dtype, device=head.projection.bias.device))
    with torch.no_grad():
        zero_hidden = torch.zeros(1, 1, head.projection.in_features, device=head.projection.weight.device)
        preactivation = head.projection(zero_hidden)[0, 0]
        recovered = F.softplus(preactivation).cpu().numpy() + floors
    max_error = float(np.max(np.abs(recovered - targets)))
    print('Initialization targets: spread=%.6f ask_gap=%.6f bid_gap=%.6f' % (targets[0], targets[1:10].mean(), targets[10:19].mean()), flush=True)
    print('Initialization inverse-softplus max error:', max_error, flush=True)
    if max_error > 1e-05:
        raise RuntimeError('Initialized bias failed to recover targets')
    return {'spread_mean': float(targets[0]), 'ask_gap_mean': float(targets[1:10].mean()), 'bid_gap_mean': float(targets[10:19].mean()), 'initial_bias_max_error': max_error}

def train_v53(args, parser):
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error('epochs and batch-size must be positive; lr must be > 0')
    v53_seed_all(args.seed)
    device = torch.device(args.device)
    data = prepare_dataset_v4(args.data_dir)
    if args.smoke_test:
        data['train'] = data['train'][:64]
        data['val'] = data['val'][:32]
        data['test'] = data['test'][:32]
    book = data['normalizer'].book_normalizer
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = TimeGANV42(feature_dim=ckpt['feature_dim'], hidden_dim=ckpt['hidden_dim'], noise_dim=ckpt['noise_dim'], num_layers=ckpt['num_layers']).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    embedder = model.embedder
    recovery = model.recovery
    for p in model.parameters():
        p.requires_grad_(False)
    head = ConstraintAwareLOBHeadV52(hidden_dim=ckpt['hidden_dim'], minimum=book.minimum, scale=book.scale, min_spread=0.01, min_gap=0.01).to(device)
    init_info = v53_initialise_head_from_train(head, data['normalizer'], data['train'])
    opt = torch.optim.Adam(head.parameters(), lr=args.lr)
    loader = DataLoader(TensorDataset(torch.from_numpy(data['train']).float()), batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print('=== V5.3 Mean-Initialized Frozen-Embedder LOB Reconstruction ===', flush=True)
    print('Train/val/test windows:', *(len(data[s]) for s in ('train', 'val', 'test')), flush=True)
    print('Only new LOB Head is trainable; original V4.3 remains frozen.', flush=True)
    minimum = head.minimum
    scale = head.scale
    initial_val = v53_evaluate(embedder, recovery, head, data['val'], minimum, scale, device, args.batch_size)
    print('Initial validation new LOB MSE:', initial_val['new_lob_mse_normalized'], flush=True)
    history = []
    best_mse = float('inf')
    best_state = None
    for epoch in range(1, args.epochs + 1):
        head.train()
        total = 0.0
        count = 0
        for x, in loader:
            x = x.to(device)
            with torch.no_grad():
                h = embedder(x)
            output = head(h)
            target = x[..., :40]
            loss = nn.functional.mse_loss(output, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * len(x)
            count += len(x)
        val = v53_evaluate(embedder, recovery, head, data['val'], minimum, scale, device, args.batch_size)
        history.append({'epoch': epoch, 'train_lob_mse': total / count, **val})
        print(f"Epoch {epoch}/{args.epochs} | Train MSE={total / count:.6f} | Val old MSE={val['old_lob_mse_normalized']:.6f} | Val new MSE={val['new_lob_mse_normalized']:.6f} | Val old/new Spread MAE={val['old_spread_mae_price_units']:.6f}/{val['new_spread_mae_price_units']:.6f}", flush=True)
        if val['new_lob_mse_normalized'] < best_mse:
            best_mse = val['new_lob_mse_normalized']
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state)
    final_val = v53_evaluate(embedder, recovery, head, data['val'], minimum, scale, device, args.batch_size)
    final_test = v53_evaluate(embedder, recovery, head, data['test'], minimum, scale, device, args.batch_size)
    print('\n=== Best-head validation (selected by Val normalized LOB MSE) ===')
    print(json.dumps(final_val, indent=2))
    print('\n=== Test (evaluated once after selecting epoch) ===')
    print(json.dumps(final_test, indent=2))
    torch.save({'head_state_dict': best_state, 'hidden_dim': ckpt['hidden_dim'], 'source_checkpoint': args.checkpoint, 'normalizer_minimum': np.asarray(book.minimum).tolist(), 'normalizer_scale': np.asarray(book.scale).tolist(), 'best_val_lob_mse': best_mse, 'initialization': init_info, 'seed': args.seed}, output_dir / 'best_lob_head.pt')
    with open(output_dir / 'results.json', 'w', encoding='utf-8') as file:
        json.dump({'initialization': init_info, 'initial_validation': initial_val, 'history': history, 'validation': final_val, 'test': final_test}, file, indent=2)
    print('\nSaved:', output_dir / 'best_lob_head.pt')
    print('Saved:', output_dir / 'results.json')
    print('V5.3 head training complete.')
from modules import MLPConstraintAwareLOBHead

def v57_pair_difference(x):
    n = len(x) // 2 * 2
    return float(np.abs(x[:n:2] - x[1:n:2]).mean())

def v57_calculate_metrics(pred, target):
    mse = float(np.mean((pred - target) ** 2))
    pred_delta = np.diff(pred, axis=1)
    real_delta = np.diff(target, axis=1)
    return {'mse': mse, 'temporal_mse': float(np.mean((pred_delta - real_delta) ** 2)), 'pair_difference': v57_pair_difference(pred), 'across_sample_std': float(pred.std(axis=0).mean()), 'temporal_change': float(np.abs(pred_delta).mean()), 'temporal_change_ratio': float(np.abs(pred_delta).mean() / max(np.abs(real_delta).mean(), 1e-12))}

@torch.no_grad()
def v57_evaluate_decoder(embedder, decoder, windows):
    embedder.eval()
    decoder.eval()
    predictions = []
    targets = []
    loader = DataLoader(TensorDataset(windows), batch_size=128, shuffle=False)
    for x, in loader:
        hidden = embedder(x)
        pred = decoder(hidden)
        predictions.append(pred.cpu().numpy())
        targets.append(x[..., :40].cpu().numpy())
    return v57_calculate_metrics(np.concatenate(predictions), np.concatenate(targets))

def v58_initialise_from_train(head, data):
    """Initialise the 39 output parameters from TRAIN only."""
    normalizer = data['normalizer']
    raw = normalizer.inverse_transform(np.asarray(data['train']))[..., :40]
    prices = raw[..., 0::2]
    sizes = raw[..., 1::2]
    asks = prices[..., 0::2]
    bids = prices[..., 1::2]
    spread = asks[..., 0] - bids[..., 0]
    ask_gaps = np.diff(asks, axis=-1)
    bid_gaps = -np.diff(bids, axis=-1)
    targets = np.concatenate([[spread.mean()], ask_gaps.mean(axis=(0, 1)), bid_gaps.mean(axis=(0, 1)), sizes.mean(axis=(0, 1))])
    minimums = np.concatenate([[head.min_spread], np.full(18, head.min_gap), np.zeros(20)])
    residual = targets - minimums
    if np.any(residual <= 0):
        raise ValueError('Invalid mean initialization targets')
    bias = np.log(np.expm1(residual))
    last = head.projection[-1]
    with torch.no_grad():
        torch.nn.init.normal_(last.weight, mean=0.0, std=0.001)
        last.bias.copy_(torch.tensor(bias, dtype=torch.float32))
    print('Mean initialization: spread=', float(targets[0]), 'ask_gap=', float(targets[1:10].mean()), 'bid_gap=', float(targets[10:19].mean()), flush=True)

def v57_pair_difference(x):
    n = len(x) // 2 * 2
    return float(np.abs(x[:n:2] - x[1:n:2]).mean())

@torch.no_grad()
def v58_detail_metrics(embedder, head, windows):
    embedder.eval()
    head.eval()
    results = []
    for x, in DataLoader(TensorDataset(windows), batch_size=128, shuffle=False):
        h = embedder(x)
        results.append(head(h).cpu())
    pred = torch.cat(results).numpy()
    real = windows[..., :40].numpy()
    return {'mse': float(np.mean((pred - real) ** 2)), 'temporal_mse': float(np.mean((np.diff(pred, axis=1) - np.diff(real, axis=1)) ** 2)), 'pair_difference': v57_pair_difference(pred), 'real_pair_difference': v57_pair_difference(real), 'temporal_change': float(np.abs(np.diff(pred, axis=1)).mean()), 'real_temporal_change': float(np.abs(np.diff(real, axis=1)).mean())}

def train_v58(args):
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(4)
    data = prepare_dataset_v4('data')
    if args.smoke_test:
        data['train'] = data['train'][:64]
        data['val'] = data['val'][:32]
        data['test'] = data['test'][:32]
    train = torch.tensor(np.asarray(data['train']), dtype=torch.float32)
    val = torch.tensor(np.asarray(data['val']), dtype=torch.float32)
    test = torch.tensor(np.asarray(data['test']), dtype=torch.float32)
    ckpt = torch.load('outputs_smoke_v43/timegan_joint.pt' if args.smoke_test else 'checkpoints_v4_3_full/timegan_joint.pt', map_location='cpu', weights_only=True)
    model = TimeGANV42(feature_dim=ckpt['feature_dim'], hidden_dim=ckpt['hidden_dim'], noise_dim=ckpt['noise_dim'], num_layers=ckpt['num_layers'])
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    book = data['normalizer'].book_normalizer
    head = MLPConstraintAwareLOBHead(hidden_dim=ckpt['hidden_dim'], minimum=book.minimum, scale=book.scale, min_spread=0.01, min_gap=0.01, intermediate=128)
    v58_initialise_from_train(head, data)
    optimizer = torch.optim.Adam(head.parameters(), lr=args.lr)
    loader = DataLoader(TensorDataset(train), batch_size=args.batch_size, shuffle=True)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    baseline = v58_detail_metrics(model.embedder, head, val)
    best_score = baseline['mse'] + args.temporal_weight * baseline['temporal_mse']
    best_epoch = 0
    best_state = copy.deepcopy(head.state_dict())
    history = []
    print('=== V5.8 MLP Constraint-aware Reconstruction ===', flush=True)
    print('Train / Val / Test:', len(train), len(val), len(test))
    print('Initial validation:', json.dumps(baseline, indent=2), flush=True)
    for epoch in range(1, args.epochs + 1):
        head.train()
        running = 0.0
        for x, in loader:
            with torch.no_grad():
                hidden = model.embedder(x)
            prediction = head(hidden)
            target = x[..., :40]
            loss_mse = F.mse_loss(prediction, target)
            loss_temporal = F.mse_loss(prediction[:, 1:] - prediction[:, :-1], target[:, 1:] - target[:, :-1])
            loss = loss_mse + args.temporal_weight * loss_temporal
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(x)
        metrics = v58_detail_metrics(model.embedder, head, val)
        score = metrics['mse'] + args.temporal_weight * metrics['temporal_mse']
        if score < best_score:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(head.state_dict())
        history.append({'epoch': epoch, 'train_loss': running / len(train), 'validation': metrics})
        print(f"Epoch {epoch}/{args.epochs} | Train={running / len(train):.6f} | Val MSE={metrics['mse']:.6f} | Val Temp MSE={metrics['temporal_mse']:.6f} | Val Pair={metrics['pair_difference']:.6f} | Val Change={metrics['temporal_change']:.6f}", flush=True)
    head.load_state_dict(best_state)
    val_result = v58_detail_metrics(model.embedder, head, val)
    test_result = v58_detail_metrics(model.embedder, head, test)
    with torch.no_grad():
        constraints = []
        for batch in test.split(128):
            hidden = model.embedder(batch)
            raw = head.forward_raw(hidden)
            constraints.append(check_raw_constraints(raw))
    all_constraints = {key: all((result[key] for result in constraints)) for key in constraints[0]}
    result = {'experiment': 'v5.8_mlp_constrained', 'best_epoch': best_epoch, 'initial_validation': baseline, 'history': history, 'best_validation': val_result, 'test': test_result, 'constraints': all_constraints}
    torch.save({'head_state_dict': best_state, 'hidden_dim': ckpt['hidden_dim'], 'intermediate': 128, 'best_epoch': best_epoch, 'source_embedder': 'outputs_smoke_v43/timegan_joint.pt' if args.smoke_test else 'checkpoints_v4_3_full/timegan_joint.pt'}, out / 'best_head.pt')
    with open(out / 'results.json', 'w') as f:
        json.dump(result, f, indent=2)
    print('\n=== Best Validation ===')
    print(json.dumps(val_result, indent=2))
    print('\n=== Test ===')
    print(json.dumps(test_result, indent=2))
    print('\n=== Constraints ===')
    print(json.dumps(all_constraints, indent=2))
    print('\nBest epoch:', best_epoch)
    print('Saved:', out / 'best_head.pt')
    print('Saved:', out / 'results.json')
    print('V5.8 training complete!')
from modules import MLPConstraintAwareLOBHead
V59_SEED = 42
V59_EPOCHS = 3
V59_LR = 0.0001
V59_BATCH_SIZE = 64
V59_TEMPORAL_WEIGHT = 1.0
V59_VARIANCE_WEIGHT = 0.1

def v59_load_models(data, args):
    ck = torch.load('outputs_smoke_v43/timegan_joint.pt' if args.smoke_test else 'checkpoints_v4_3_full/timegan_joint.pt', map_location='cpu', weights_only=True)
    hc = torch.load('outputs_smoke_v58/best_head.pt' if args.smoke_test else 'outputs_v5_8_mlp_constrained/best_head.pt', map_location='cpu', weights_only=True)
    model = TimeGANV42(feature_dim=ck['feature_dim'], hidden_dim=ck['hidden_dim'], noise_dim=ck['noise_dim'], num_layers=ck['num_layers'])
    model.load_state_dict(ck['model_state_dict'])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    book = data['normalizer'].book_normalizer
    head = MLPConstraintAwareLOBHead(hidden_dim=ck['hidden_dim'], minimum=book.minimum, scale=book.scale, min_spread=0.01, min_gap=0.01, intermediate=hc['intermediate'])
    head.load_state_dict(hc['head_state_dict'])
    return (model, head)

def v59_pair_diff(x):
    n = len(x) // 2 * 2
    return float(np.abs(x[:n:2] - x[1:n:2]).mean())

def v59_variance_loss(pred, target):
    pred_std = pred.reshape(-1, 40).std(dim=0, unbiased=False)
    real_std = target.reshape(-1, 40).std(dim=0, unbiased=False).detach()
    denominator = real_std.clamp_min(0.02)
    return (((pred_std - real_std) / denominator) ** 2).mean()

def v59_temporal_loss(pred, target):
    return F.mse_loss(pred[:, 1:] - pred[:, :-1], target[:, 1:] - target[:, :-1])

@torch.no_grad()
def v59_evaluate(model, head, windows):
    model.eval()
    head.eval()
    predictions = []
    for batch in windows.split(128):
        hidden = model.embedder(batch)
        predictions.append(head(hidden).cpu())
    pred = torch.cat(predictions)
    real = windows[..., :40]
    pred_np = pred.numpy()
    real_np = real.numpy()
    return {'mse': float(F.mse_loss(pred, real)), 'temporal_mse': float(v59_temporal_loss(pred, real)), 'variance_loss': float(v59_variance_loss(pred, real)), 'pair_difference': v59_pair_diff(pred_np), 'real_pair_difference': v59_pair_diff(real_np), 'temporal_change': float(np.abs(np.diff(pred_np, axis=1)).mean()), 'real_temporal_change': float(np.abs(np.diff(real_np, axis=1)).mean()), 'across_sample_std': float(pred_np.std(axis=0).mean()), 'real_across_sample_std': float(real_np.std(axis=0).mean())}

def v59_train_arm(name, model, starting_state, template, train, val, variance_weight):
    head = copy.deepcopy(template)
    head.load_state_dict(starting_state)
    optimizer = torch.optim.Adam(head.parameters(), lr=V59_LR)
    rng = torch.Generator()
    rng.manual_seed(V59_SEED)
    loader = DataLoader(TensorDataset(train), batch_size=V59_BATCH_SIZE, shuffle=True, generator=rng)
    initial = v59_evaluate(model, head, val)
    best_score = initial['mse'] + V59_TEMPORAL_WEIGHT * initial['temporal_mse'] + V59_VARIANCE_WEIGHT * initial['variance_loss']
    best_state = copy.deepcopy(head.state_dict())
    best_epoch = 0
    history = []
    print(f'\n=== {name} ===', flush=True)
    print('Initial validation:', json.dumps(initial, indent=2), flush=True)
    for epoch in range(1, V59_EPOCHS + 1):
        head.train()
        total_loss = 0.0
        for x, in loader:
            with torch.no_grad():
                hidden = model.embedder(x)
            target = x[..., :40]
            pred = head(hidden)
            mse = F.mse_loss(pred, target)
            temp = v59_temporal_loss(pred, target)
            var = v59_variance_loss(pred, target)
            loss = mse + V59_TEMPORAL_WEIGHT * temp + variance_weight * var
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(x)
        val_metrics = v59_evaluate(model, head, val)
        score = val_metrics['mse'] + V59_TEMPORAL_WEIGHT * val_metrics['temporal_mse'] + V59_VARIANCE_WEIGHT * val_metrics['variance_loss']
        if score < best_score:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(head.state_dict())
        history.append({'epoch': epoch, 'train_loss': total_loss / len(train), 'validation': val_metrics})
        print(f"Epoch {epoch}/{V59_EPOCHS} | Train={total_loss / len(train):.6f} | Val MSE={val_metrics['mse']:.6f} | Val VarLoss={val_metrics['variance_loss']:.6f} | Pair={val_metrics['pair_difference']:.6f} | Change={val_metrics['temporal_change']:.6f}", flush=True)
    head.load_state_dict(best_state)
    return (head, {'name': name, 'variance_weight': variance_weight, 'best_epoch': best_epoch, 'initial_validation': initial, 'history': history, 'best_validation': v59_evaluate(model, head, val)})

def train_v59(args):
    global V59_SEED, V59_EPOCHS, V59_LR, V59_BATCH_SIZE, V59_TEMPORAL_WEIGHT, V59_VARIANCE_WEIGHT
    V59_SEED = 42
    V59_EPOCHS = 1 if args.smoke_test else 3
    V59_LR = 0.0001
    V59_BATCH_SIZE = 64
    V59_TEMPORAL_WEIGHT = 1.0
    V59_VARIANCE_WEIGHT = 0.1
    data = prepare_dataset_v4(args.data_dir)
    if args.smoke_test:
        data['train'] = data['train'][:64]
        data['val'] = data['val'][:32]
        data['test'] = data['test'][:32]
    train = torch.tensor(np.asarray(data['train']), dtype=torch.float32)
    val = torch.tensor(np.asarray(data['val']), dtype=torch.float32)
    test = torch.tensor(np.asarray(data['test']), dtype=torch.float32)
    model, template = v59_load_models(data, args)
    initial_state = copy.deepcopy(template.state_dict())
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results = {'experiment': 'v5.9_variance_aware_ablation', 'epochs': V59_EPOCHS, 'lr': V59_LR, 'batch_size': V59_BATCH_SIZE, 'temporal_weight': V59_TEMPORAL_WEIGHT, 'variance_weight': V59_VARIANCE_WEIGHT}
    for name, weight in [('control', 0.0), ('variance_aware', V59_VARIANCE_WEIGHT)]:
        head, record = v59_train_arm(name, model, initial_state, template, train, val, weight)
        record['test'] = v59_evaluate(model, head, test)
        with torch.no_grad():
            checks = []
            for batch in test.split(128):
                h = model.embedder(batch)
                raw = head.forward_raw(h)
                checks.append(check_raw_constraints(raw))
        record['constraints'] = {k: all((c[k] for c in checks)) for k in checks[0]}
        torch.save({'head_state_dict': head.state_dict(), 'hidden_dim': 24, 'intermediate': 128, 'best_epoch': record['best_epoch'], 'variance_weight': weight}, output / f'{name}_best.pt')
        results[name] = record
    with open(output / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print('\n=== V5.9 FINAL TEST COMPARISON ===')
    for name in ['control', 'variance_aware']:
        r = results[name]
        print(f"\n{name} | Best epoch: {r['best_epoch']}")
        print(json.dumps(r['test'], indent=2))
        print('Constraints:', r['constraints'])
    print('\nSaved:', output / 'results.json')
    print('V5.9 experiment complete!')

def v510_pair_difference(x):
    n = x.shape[0] // 2 * 2
    return (x[:n:2] - x[1:n:2]).abs().mean()

def v510_set_grad(module, enabled):
    for p in module.parameters():
        p.requires_grad_(enabled)

@torch.no_grad()
def v510_evaluate(model, head, test, count=512):
    model.eval()
    head.eval()
    x = test[:count]
    n, t, _ = x.shape
    with torch.random.fork_rng():
        torch.manual_seed(12345)
        z = torch.rand(n, t, model.noise_dim)
    e_fake = model.generator(z)
    h_fake = model.supervisor(e_fake)
    fake = head(h_fake)
    fake_raw = head.forward_raw(h_fake)
    real_pair = v510_pair_difference(x[..., :40]).item()
    fake_pair = v510_pair_difference(fake).item()
    prices = fake_raw[..., 0::2]
    asks = prices[..., 0::2]
    bids = prices[..., 1::2]
    spread = asks[..., 0] - bids[..., 0]
    return {'real_normalized_pair_diff': float(real_pair), 'fake_normalized_pair_diff': float(fake_pair), 'fake_across_sample_std': float(fake.std(dim=0, unbiased=False).mean()), 'fake_temporal_difference': float((fake[:, 1:] - fake[:, :-1]).abs().mean()), 'fake_spread_mean': float(spread.mean()), 'fake_spread_std': float(spread.std(unbiased=False)), 'generator_hidden_pair_diff': float(v510_pair_difference(e_fake)), 'supervisor_hidden_pair_diff': float(v510_pair_difference(h_fake))}

def train_v510(args):
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(4)
    device = torch.device('cpu')
    data = prepare_dataset_v4(args.data_dir)
    if args.smoke_test:
        data['train'] = data['train'][:64]
        data['val'] = data['val'][:32]
    normalizer = data['normalizer']
    train = torch.tensor(np.asarray(data['train']), dtype=torch.float32)
    val = torch.tensor(np.asarray(data['val']), dtype=torch.float32)
    loader = DataLoader(TensorDataset(train), batch_size=args.batch_size, shuffle=True, drop_last=True)
    ckpt = torch.load('outputs_smoke_v43/timegan_joint.pt' if args.smoke_test else 'checkpoints_v4_3_full/timegan_joint.pt', map_location='cpu', weights_only=True)
    model = TimeGANV42(feature_dim=ckpt['feature_dim'], hidden_dim=ckpt['hidden_dim'], noise_dim=ckpt['noise_dim'], num_layers=ckpt['num_layers']).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    head_ckpt = torch.load('outputs_smoke_v53/best_lob_head.pt' if args.smoke_test else 'outputs_v5_3_lob_head/best_lob_head.pt', map_location='cpu', weights_only=True)
    head = ConstraintAwareLOBHeadV52(hidden_dim=ckpt['hidden_dim'], minimum=normalizer.book_normalizer.minimum, scale=normalizer.book_normalizer.scale, min_spread=0.01, min_gap=0.01).to(device)
    head.load_state_dict(head_ckpt['head_state_dict'])
    v510_set_grad(model.embedder, False)
    v510_set_grad(model.recovery, False)
    v510_set_grad(head, False)
    opt_g = torch.optim.Adam(list(model.generator.parameters()) + list(model.supervisor.parameters()), lr=args.lr)
    opt_d = torch.optim.Adam(model.discriminator.parameters(), lr=args.lr)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()
    folder = Path(args.output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    baseline = v510_evaluate(model, head, val)
    print('=== V5.4 Diversity-aware TimeGAN ===')
    print('Train windows:', len(train))
    print('Validation windows:', len(val))
    print('Generator updates:', args.generator_updates)
    print('Diversity weight:', args.diversity_weight)
    print('Target fraction:', args.target_fraction)
    print('\n=== Before training ===')
    print(json.dumps(baseline, indent=2), flush=True)
    history = []
    for epoch in range(args.epochs):
        model.generator.train()
        model.supervisor.train()
        model.discriminator.train()
        total = {'generator_loss': 0.0, 'diversity_loss': 0.0, 'real_pair': 0.0, 'fake_pair': 0.0, 'discriminator_loss': 0.0}
        steps = 0
        for real, in loader:
            real = real.to(device)
            b, t, _ = real.shape
            with torch.no_grad():
                h_real = model.embedder(real)
                target_lob = real[..., :40]
            v510_set_grad(model.discriminator, False)
            v510_set_grad(model.generator, True)
            v510_set_grad(model.supervisor, True)
            for _ in range(args.generator_updates):
                z = torch.rand(b, t, ckpt['noise_dim'], device=device)
                e_fake = model.generator(z)
                h_fake = model.supervisor(e_fake)
                x_fake = model.recovery(h_fake)
                lob_fake = head(h_fake)
                h_pred = model.supervisor(h_real)
                logits_h = model.discriminator(h_fake)
                logits_e = model.discriminator(e_fake)
                adv = bce(logits_h, torch.ones_like(logits_h)) + bce(logits_e, torch.ones_like(logits_e))
                sup = mse(h_pred[:, :-1, :], h_real[:, 1:, :])
                moments = moment_loss(real, x_fake)
                real_pair = v510_pair_difference(h_real).detach()
                fake_pair = v510_pair_difference(h_fake)
                target = (args.target_fraction * real_pair).detach()
                diversity_loss = torch.relu((target - fake_pair) / (target + 1e-06))
                real_mean = h_real.mean(dim=(0, 1)).detach()
                real_std = h_real.std(dim=(0, 1), unbiased=False).detach()
                fake_mean = h_fake.mean(dim=(0, 1))
                fake_std = h_fake.std(dim=(0, 1), unbiased=False)
                latent_loss = (fake_mean - real_mean).abs().mean() + (fake_std - real_std).abs().mean()
                loss_g = adv + 100 * torch.sqrt(sup + 1e-08) + 100 * moments + args.diversity_weight * diversity_loss + args.latent_weight * latent_loss
                opt_g.zero_grad(set_to_none=True)
                loss_g.backward()
                opt_g.step()
                total['generator_loss'] += loss_g.item()
                total['diversity_loss'] += diversity_loss.item()
                total['real_pair'] += real_pair.item()
                total['fake_pair'] += fake_pair.detach().item()
            v510_set_grad(model.generator, False)
            v510_set_grad(model.supervisor, False)
            v510_set_grad(model.discriminator, True)
            with torch.no_grad():
                z = torch.rand(b, t, ckpt['noise_dim'], device=device)
                e_fake = model.generator(z)
                h_fake = model.supervisor(e_fake)
            logits_real = model.discriminator(h_real)
            logits_fake = model.discriminator(h_fake)
            logits_fake_e = model.discriminator(e_fake)
            loss_d = bce(logits_real, torch.ones_like(logits_real)) + bce(logits_fake, torch.zeros_like(logits_fake)) + bce(logits_fake_e, torch.zeros_like(logits_fake_e))
            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            opt_d.step()
            total['discriminator_loss'] += loss_d.item()
            steps += 1
        for key in ['generator_loss', 'diversity_loss', 'real_pair', 'fake_pair']:
            total[key] /= steps * args.generator_updates
        total['discriminator_loss'] /= steps
        validation = v510_evaluate(model, head, val)
        record = {'epoch': epoch + 1, 'training': total, 'validation': validation}
        history.append(record)
        print(f'\nEpoch {epoch + 1}/{args.epochs}')
        print(f"G Loss={total['generator_loss']:.6f} | D Loss={total['discriminator_loss']:.6f} | Diversity Loss={total['diversity_loss']:.6f}")
        print('Validation fake LOB pair difference:', validation['fake_normalized_pair_diff'])
        print('Validation fake Spread std:', validation['fake_spread_std'])
        if not all((np.isfinite(v) for v in total.values())):
            raise RuntimeError('Non-finite training metric')
    result = {'experiment': 'v5.10_latent_matching_finetuning', 'source_checkpoint': 'outputs_smoke_v43/timegan_joint.pt' if args.smoke_test else 'checkpoints_v4_3_full/timegan_joint.pt', 'epochs': args.epochs, 'lr': args.lr, 'diversity_weight': args.diversity_weight, 'target_fraction': args.target_fraction, 'baseline_validation': baseline, 'history': history}
    torch.save({'model_state_dict': model.state_dict(), 'source_checkpoint': 'outputs_smoke_v43/timegan_joint.pt' if args.smoke_test else 'checkpoints_v4_3_full/timegan_joint.pt', 'feature_dim': ckpt['feature_dim'], 'hidden_dim': ckpt['hidden_dim'], 'noise_dim': ckpt['noise_dim'], 'num_layers': ckpt['num_layers'], 'experiment': 'v5.10-latent-matching'}, folder / 'timegan_finetuned.pt')
    with open(folder / 'results.json', 'w') as f:
        json.dump(result, f, indent=2)
    print('\n=== Final comparison ===')
    print('Before pair difference:', baseline['fake_normalized_pair_diff'])
    print('After pair difference:', history[-1]['validation']['fake_normalized_pair_diff'])
    print('Before spread std:', baseline['fake_spread_std'])
    print('After spread std:', history[-1]['validation']['fake_spread_std'])
    print('\nSaved:', folder / 'timegan_finetuned.pt')
    print('Saved:', folder / 'results.json')
    print('V5.4 experiment complete!')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True, choices=['stage1', 'v43', 'v53', 'v58', 'v59', 'v510', 'v514', 'depth'])
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--smoke-test', action='store_true')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--return-weight', type=float, default=1.0)
    parser.add_argument('--move-weight', type=float, default=5.0)
    parser.add_argument('--stage1-checkpoint', default=None)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--supervisor-epochs', type=int, default=2)
    parser.add_argument('--joint-epochs', type=int, default=2)
    parser.add_argument('--joint-lr', type=float, default=0.0001)
    parser.add_argument('--generator-updates', type=int, default=2)
    parser.add_argument('--temporal-weight', type=float, default=1.0)
    parser.add_argument('--v510-epochs', type=int, default=1)
    parser.add_argument('--v510-lr', type=float, default=0.0001)
    parser.add_argument('--diversity-weight', type=float, default=0.5)
    parser.add_argument('--target-fraction', type=float, default=0.5)
    parser.add_argument('--latent-weight', type=float, default=5.0)
    args = parser.parse_args()
    if args.stage == 'v514':
        from train_multifactor import train_v514
        train_v514(
            smoke_test=args.smoke_test,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            checkpoint=args.checkpoint,
        )
    elif args.stage == 'depth':
        train_depth(smoke_test=args.smoke_test, data_dir=args.data_dir, output_dir=args.output_dir)
    elif args.stage == 'stage1':
        if args.smoke_test:
            args.epochs = 1
        if args.output_dir is None:
            args.output_dir = 'outputs_smoke_stage1' if args.smoke_test else 'checkpoints_v4_3_stage1'
        train_stage1(args)
    elif args.stage == 'v43':
        if args.smoke_test:
            args.supervisor_epochs = 1
            args.joint_epochs = 1
            args.generator_updates = 1
        if args.stage1_checkpoint is None:
            args.stage1_checkpoint = 'outputs_smoke_stage1/weighted_returns.pt' if args.smoke_test else 'checkpoints_v4_3_stage1/weighted_returns.pt'
        if args.output_dir is None:
            args.output_dir = 'outputs_smoke_v43' if args.smoke_test else 'checkpoints_v4_3_full'
        train_v43(args, parser)
    elif args.stage == 'v53':
        if args.smoke_test:
            args.epochs = 1
        if args.checkpoint is None:
            args.checkpoint = 'outputs_smoke_v43/timegan_joint.pt' if args.smoke_test else 'checkpoints_v4_3_full/timegan_joint.pt'
        if args.output_dir is None:
            args.output_dir = 'outputs_smoke_v53' if args.smoke_test else 'outputs_v5_3_lob_head'
        train_v53(args, parser)
    elif args.stage == 'v58':
        if args.smoke_test:
            args.epochs = 1
        if args.output_dir is None:
            args.output_dir = 'outputs_smoke_v58' if args.smoke_test else 'outputs_v5_8_mlp_constrained'
        train_v58(args)
    elif args.stage == 'v59':
        if args.output_dir is None:
            args.output_dir = 'outputs_smoke_v59' if args.smoke_test else 'outputs_v5_9_variance_aware'
        train_v59(args)
    elif args.stage == 'v510':
        stage_args = argparse.Namespace(smoke_test=args.smoke_test, data_dir=args.data_dir, output_dir=args.output_dir if args.output_dir is not None else 'outputs_smoke_v510' if args.smoke_test else 'outputs_v5_10_latent_matching', epochs=1 if args.smoke_test else args.v510_epochs, batch_size=args.batch_size, lr=args.v510_lr, diversity_weight=args.diversity_weight, target_fraction=args.target_fraction, generator_updates=1 if args.smoke_test else args.generator_updates, latent_weight=args.latent_weight)
        train_v510(stage_args)
if __name__ == '__main__':
    main()
