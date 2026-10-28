"""Training/validation driver and explicit, frozen-checkpoint test evaluation.

No test evaluation during training. Each invocation requires a NEW run directory.
Arguments, patient splits and checkpoint architecture are all checked before that
directory is created, so a rejected run leaves no half-built output behind.
"""
from pathlib import Path
import argparse
import json
import logging
import random
import time
import platform
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import ROOT, HipMRISlices, check_patient_splits, collate_slices
from modules import build_model, load_checkpoint
from metrics import segmentation_loss, confusion_counts, scores


def make_loader(args: argparse.Namespace, split: str) -> DataLoader:
    limit = args.limit if split == 'train' else args.val_limit
    dataset = HipMRISlices(args.root, split, limit, args.augment and split == 'train', args.cache)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=split == 'train',
                      num_workers=args.workers, pin_memory=args.device == 'cuda',
                      collate_fn=collate_slices)


def run_epoch(model: torch.nn.Module, loader: DataLoader, device: str,
              optimizer: torch.optim.Optimizer | None = None) -> dict:
    model.train(optimizer is not None)
    confusion = torch.zeros(6, 6, device=device, dtype=torch.int64)
    patient_counts = {}
    total_loss, samples = 0.0, 0
    for images, targets, names in loader:
        images, targets = images.to(device), targets.to(device)
        with torch.set_grad_enabled(optimizer is not None):
            logits = model(images)
            loss = segmentation_loss(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite loss')
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
                optimizer.step()
        confusion += confusion_counts(logits.detach(), targets)
        if optimizer is None:
            for index, name in enumerate(names):
                patient = name.split('_')[1]
                counts = confusion_counts(logits[index:index+1], targets[index:index+1])
                patient_counts[patient] = patient_counts.get(patient, 0) + counts
        total_loss += loss.item() * len(images)
        samples += len(images)
    result = {'loss': total_loss/samples, 'samples': samples, **scores(confusion.cpu())}
    result['patients'] = {patient: scores(counts.cpu())
                          for patient, counts in patient_counts.items()}
    result['confusion'] = confusion.cpu().tolist()
    return result


def save_json(path: Path, payload: dict | list) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def profile(model: torch.nn.Module, loader: DataLoader, device: str) -> dict:
    images = next(iter(loader))[0][:1].to(device)
    model.eval()
    with torch.inference_mode():
        for _ in range(5):
            model(images)
        if device == 'cuda':
            torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(30):
            model(images)
        if device == 'cuda':
            torch.cuda.synchronize()
    return {'parameters': sum(p.numel() for p in model.parameters()),
            'latency_ms_single_slice': 1000*(time.perf_counter()-started)/30,
            'latency_method': 'batch=1, 5 warmups, 30 forwards, CUDA synchronized; excludes I/O',
            'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20 if device == 'cuda' else None,
            'peak_reserved_mib': torch.cuda.max_memory_reserved()/2**20 if device == 'cuda' else None}


def plot_history(history: list[dict], output: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(9, 3))
    for split in ('train', 'validate'):
        axes[0].plot([row[split]['loss'] for row in history], label=split)
        axes[1].plot([row[split]['dice'][5] for row in history], label=split)
    axes[0].set_ylabel('CE + foreground soft Dice loss')
    axes[1].set_ylabel('Label 5 hard Dice (pooled pixels)')
    for axis in axes:
        axis.set_xlabel('Epoch index (zero based)')
        axis.legend()
    figure.tight_layout()
    figure.savefig(output/'learning_curves.png', dpi=150)
    plt.close(figure)


def train(args: argparse.Namespace, model: torch.nn.Module) -> None:
    training, validation = make_loader(args, 'train'), make_loader(args, 'validate')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    history, best = [], -1.0
    started = time.perf_counter()
    for epoch in range(1, args.epochs+1):
        train_result = run_epoch(model, training, args.device, optimizer)
        val_result = run_epoch(model, validation, args.device)
        history.append({'epoch': epoch, 'train': train_result, 'validate': val_result,
                        'elapsed_seconds': time.perf_counter()-started})
        save_json(args.run/'history.json', history)
        logging.info('epoch=%d loss=%.4f val_loss=%.4f label5_val_dice=%s elapsed=%.1f',
                     epoch, train_result['loss'], val_result['loss'], val_result['dice'][5],
                     time.perf_counter()-started)
        selection_score = val_result['dice'][5]
        if selection_score is not None and selection_score > best:
            best = selection_score
            torch.save({'state_dict': model.state_dict(), 'config': vars_serializable(args),
                        'epoch': epoch, 'validation': val_result}, args.run/'best.pt')
        if time.perf_counter()-started >= args.time_budget:
            logging.info('Time budget reached after complete epoch; stopping safely')
            break
    save_json(args.run/'resources.json', {'training_seconds': time.perf_counter()-started,
                                         **profile(model, validation, args.device)})
    plot_history(history, args.run)


def vars_serializable(args: argparse.Namespace) -> dict:
    return {key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--model', choices=['basic', 'improved'],
                        help='defaults to basic, or to the --evaluate checkpoint')
    parser.add_argument('--width', type=int,
                        help='defaults to 16, or to the --evaluate checkpoint')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--val-limit', type=int)
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--cache', action='store_true', help='cache images/uint8 labels in CPU RAM')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--time-budget', type=float, default=36000)
    parser.add_argument('--evaluate', type=Path, help='frozen checkpoint; no training')
    parser.add_argument('--split', choices=['validate', 'test'], default='validate')
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Reject impossible settings before any output directory is created."""
    for name, value in (('epochs', args.epochs), ('batch size', args.batch_size)):
        if value < 1:
            raise ValueError(f'{name} must be positive')
    for name, value in (('width', args.width), ('limit', args.limit),
                        ('validation limit', args.val_limit)):
        if value is not None and value < 1:
            raise ValueError(f'{name} must be positive')
    if args.workers < 0:
        raise ValueError('workers must not be negative')
    if args.lr <= 0:
        raise ValueError('learning rate must be positive')
    if args.time_budget <= 0:
        raise ValueError('time budget must be positive')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable; pass --device cpu')


def resolve_architecture(args: argparse.Namespace,
                         checkpoint: dict | None) -> tuple[str, int]:
    """Saved checkpoints decide the architecture; flags may only confirm it."""
    if checkpoint is None:
        return args.model or 'basic', args.width or 16
    saved = checkpoint.get('config')
    if not isinstance(saved, dict) or 'model' not in saved or 'width' not in saved:
        raise ValueError('checkpoint has no model/width configuration')
    for name, given, expected in (('--model', args.model, saved['model']),
                                  ('--width', args.width, saved['width'])):
        if given is not None and given != expected:
            raise ValueError(f'{name}={given} conflicts with checkpoint {expected}')
    return saved['model'], saved['width']


def main() -> None:
    args = parse_args()
    validate_args(args)
    checkpoint = (load_checkpoint(args.evaluate, args.device)
                  if args.evaluate else None)
    args.model, args.width = resolve_architecture(args, checkpoint)
    check_patient_splits(args.root)
    model = build_model(args.model, args.width).to(args.device)
    if checkpoint is not None:
        model.load_state_dict(checkpoint['state_dict'])
    args.run.mkdir(parents=True, exist_ok=False)
    logging.getLogger().addHandler(logging.FileHandler(args.run/'train.log', encoding='utf-8'))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if args.device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    save_json(args.run/'config.json', vars_serializable(args))
    save_json(args.run/'environment.json', {
        'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__,
        'hardware': torch.cuda.get_device_name() if args.device == 'cuda' else platform.processor(),
        'cuda': torch.version.cuda})
    if checkpoint is not None:
        args.val_limit = None
        loader = make_loader(args, args.split)
        save_json(args.run/'evaluation.json', run_epoch(model, loader, args.device))
        save_json(args.run/'resources.json', profile(model, loader, args.device))
    else:
        train(args, model)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    main()
