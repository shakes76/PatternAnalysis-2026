"""ADNI training and evaluation entry points; baseline run stays validation-only."""
import argparse
import copy
import hashlib
import shutil
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import traceback

import PIL
import torch
from dataset import ADNISliceDataset, PREPROCESSING
from modules import SimpleCNN
from check_dataloader import make_loader, require, sha256, code_version


class CountedDataset(ADNISliceDataset):
    """Count reads in the main process without retaining images (workers=0)."""
    def __init__(self, *args):
        super().__init__(*args)
        self.read_calls = 0

    def __getitem__(self, index):
        self.read_calls += 1
        return super().__getitem__(index)


def classification_metrics(labels, probabilities):
    """Slice-level binary metrics; AD positive, threshold >=0.5, tied-rank AUC."""
    require(len(labels) == len(probabilities) and len(labels) > 0, 'Empty/mismatched metrics')
    require(all(y in (0, 1) for y in labels), 'Invalid metric labels')
    require(all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities), 'Invalid probabilities')
    cm = [[0, 0], [0, 0]]
    for y, p in zip(labels, probabilities):
        cm[y][int(p >= 0.5)] += 1
    per_class = {}
    for k, name in enumerate(('NC', 'AD')):
        tp = cm[k][k]; predicted = cm[0][k] + cm[1][k]; support = sum(cm[k])
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        per_class[name] = dict(precision=precision, recall=recall,
            f1=2*precision*recall/(precision+recall) if precision+recall else 0.0, support=support)
    positives = sum(labels); negatives = len(labels)-positives
    auc = None
    if positives and negatives:
        ordered = sorted(zip(probabilities, labels)); rank_sum = 0.0; i = 0
        while i < len(ordered):
            j = i+1
            while j < len(ordered) and ordered[j][0] == ordered[i][0]: j += 1
            rank_sum += ((i+1+j)/2) * sum(y for _, y in ordered[i:j])
            i = j
        auc = (rank_sum-positives*(positives+1)/2)/(positives*negatives)
    return dict(accuracy=(cm[0][0]+cm[1][1])/len(labels),
        macro_f1=sum(v['f1'] for v in per_class.values())/2,
        auroc=auc, confusion_matrix=cm, per_class=per_class)


def class_weight_config(records, mode):
    """Compute patient-balanced weights from training records only, in NC/AD order."""
    require(mode in ('none', 'balanced'), 'Unknown class weighting mode')
    patients = {}; slice_counts = Counter()
    for row in records:
        label = row['label']; patient = row['patient_id']
        require(label in ('NC','AD'), 'Invalid training label')
        require(patients.setdefault(patient,label) == label, 'Conflicting patient label')
        slice_counts[patient] += 1
    counts = {k:sum(v == k for v in patients.values()) for k in ('NC','AD')}
    weights = None
    if mode == 'balanced':
        require(all(counts.values()), 'Balanced training requires both classes')
        require(len(set(slice_counts.values())) == 1, 'Balanced training requires equal slices per patient')
        weights = [len(patients)/(2*counts[k]) for k in ('NC','AD')]
    return dict(mode=mode, weights=weights, patient_counts=counts,
                slice_counts={k:sum(row['label']==k for row in records) for k in ('NC','AD')},
                source='training manifest only', label_order=['NC=0','AD=1'])


def state_sha256(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value = tensor.detach().cpu().contiguous().clone()
        digest.update(json.dumps([name,str(value.dtype),list(value.shape)]).encode())
        digest.update(bytes(value.untyped_storage()))
    return digest.hexdigest()


def run_epoch(model, loader, device, optimizer=None, class_weights=None):
    training = optimizer is not None
    require(training or class_weights is None, 'Weighted validation is forbidden')
    if class_weights is not None:
        require(tuple(class_weights.shape)==(2,) and bool(torch.isfinite(class_weights).all())
                and bool((class_weights>0).all()), 'Invalid class weights')
        class_weights = class_weights.to(device)
    model.train(training)
    numerator = 0.0; denominator = 0.0; plain_total = 0.0
    labels_all = []; probabilities = []; order = hashlib.sha256()
    with torch.set_grad_enabled(training):
        for batch in loader:
            # Hash ordered batch boundaries and paths without exposing identities in the summary.
            order.update((json.dumps(batch['file_path'],ensure_ascii=True)+'\n').encode())
            images = batch['image'].to(device); labels = batch['label'].to(device)
            require(bool(torch.isfinite(images).all()), 'Nonfinite images')
            if training: optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            require(tuple(logits.shape) == (len(labels), 2) and bool(torch.isfinite(logits).all()), 'Invalid logits')
            plain_loss = torch.nn.functional.cross_entropy(logits, labels)
            loss = (plain_loss if class_weights is None else
                    torch.nn.functional.cross_entropy(logits, labels, weight=class_weights))
            require(bool(torch.isfinite(loss)) and bool(torch.isfinite(plain_loss)), 'Nonfinite loss')
            if training:
                loss.backward()
                require(all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                            for p in model.parameters() if p.requires_grad), 'Invalid gradients')
                optimizer.step()
                require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), 'Invalid parameters')
            batch_denominator = (len(labels) if class_weights is None else class_weights[labels].sum().item())
            numerator += loss.item()*batch_denominator
            denominator += batch_denominator
            plain_total += plain_loss.item()*len(labels)
            labels_all.extend(labels.detach().cpu().tolist())
            probabilities.extend(logits.detach().softmax(dim=1)[:, 1].cpu().tolist())
    require(len(labels_all) == len(loader.dataset), 'Epoch did not cover full split')
    return dict(loss=numerator/denominator, unweighted_loss=plain_total/len(labels_all),
                loss_numerator=numerator, loss_denominator=denominator,
                order_sha256=order.hexdigest(), samples=len(labels_all),
                **classification_metrics(labels_all, probabilities))


def verify_checkpoint(path, probe, device, best_epoch):
    checkpoint = torch.load(path,map_location=device,weights_only=True)
    # Reload checks must not consume RNG state that a future model might use.
    devices = [torch.cuda.current_device()] if device == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        restored = SimpleCNN().to(device)
    restored.load_state_dict(checkpoint['model_state_dict']); restored.eval()
    with torch.no_grad(): actual = restored(probe).cpu()
    require(torch.allclose(actual, checkpoint['probe_logits'].cpu(),rtol=1e-5,atol=1e-6), 'Reload mismatch')
    require(checkpoint['epoch']==best_epoch, 'Best epoch mismatch')
    return True


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def run_training(manifest, summary, output_dir, device='cuda', epochs=3,
                 batch_size=16, learning_rate=0.001, seed=42, expected_lengths=None, class_weighting='none'):
    out = Path(output_dir)
    # A fresh directory is required so evidence from a previous run is never overwritten.
    out.mkdir(parents=True, exist_ok=False)
    config = dict(model='SimpleCNN', epochs=epochs, batch_size=batch_size,
        learning_rate=learning_rate if math.isfinite(learning_rate) else str(learning_rate), optimizer='Adam', seed=seed, device=device,
        num_workers=0, drop_last=False, augmentation=False,
        checkpoint_selection='minimum validation loss; strict improvement; first epoch wins ties',
        early_stopping=False, scheduler=None, class_weighting=class_weighting,
        validation_loss_weighting='none', first3_snapshot_epoch=3,
        metric_unit='slice', positive_class='AD=1', threshold=0.5,
        zero_division=0, undefined_auroc=None, preprocessing=PREPROCESSING,
        expected_lengths=expected_lengths, manifest=str(manifest), split_summary=str(summary))
    result = dict(status='running', generated_at=datetime.now(timezone.utc).isoformat(),
        config=config, code_version=code_version(), history=[],
        versions=dict(python=platform.python_version(), torch=str(torch.__version__),
                      pillow=PIL.__version__, cuda_build=torch.version.cuda))
    datasets = {}; started = time.perf_counter()
    def log(message):
        print(message, flush=True)
        with (out/'training.log').open('a') as f: f.write(message+'\n')
    try:
        write_json(out/'config.json', config)
        require(epochs > 0 and batch_size > 0 and math.isfinite(learning_rate) and learning_rate > 0,
                'epochs, batch_size and learning_rate must be positive and finite')
        require(device in ('cpu', 'cuda'), 'Use cpu or cuda')
        require(device != 'cuda' or torch.cuda.is_available(), 'CUDA requested but unavailable')
        result['code_sha256'] = {n:sha256(Path(__file__).with_name(n)) for n in
            ('train.py', 'modules.py', 'dataset.py', 'check_dataloader.py')}
        result['input_sha256'] = dict(split_samples=sha256(manifest), split_summary=sha256(summary))
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        if device == 'cuda':
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.cuda.reset_peak_memory_stats()
        result['gpu_name'] = torch.cuda.get_device_name() if device == 'cuda' else None
        datasets = {s:CountedDataset(manifest, s, summary) for s in ('train', 'validation', 'test')}
        lengths = {s:len(ds) for s,ds in datasets.items()}
        require(expected_lengths is None or lengths == expected_lengths, 'Frozen cohort lengths mismatch')
        result['dataset_lengths'] = lengths
        weighting = class_weight_config(datasets['train'].records, class_weighting)
        config['class_weight_details'] = weighting
        write_json(out/'config.json', config)
        weights = (torch.tensor(weighting['weights'],dtype=torch.float32,device=device)
                   if weighting['weights'] is not None else None)
        loaders = {s:make_loader(datasets[s], batch_size, seed) for s in ('train', 'validation')}
        model = SimpleCNN().to(device)
        result['initial_state_sha256'] = state_sha256(model)
        result['parameter_count'] = sum(p.numel() for p in model.parameters())
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        best_loss = float('inf'); best_epoch = None
        # A synthetic probe verifies serialization without additional real-image reads.
        probe = torch.linspace(0, 1, 240*256, device=device).reshape(1,1,240,256)
        for epoch in range(1, epochs+1):
            if device == 'cuda': torch.cuda.synchronize()
            epoch_start = time.perf_counter()
            train_before = [p.detach().clone() for p in model.parameters()]
            train_metrics = (run_epoch(model, loaders['train'], device, optimizer) if weights is None else
                             run_epoch(model, loaders['train'], device, optimizer, class_weights=weights))
            changed = any(not torch.equal(a,b) for a,b in zip(train_before, model.parameters()))
            require(changed, 'Training epoch did not update parameters')
            before = {k:v.detach().clone() for k,v in model.state_dict().items()}
            validation_metrics = run_epoch(model, loaders['validation'], device)
            unchanged = all(torch.equal(v, model.state_dict()[k]) for k,v in before.items())
            require(unchanged, 'Validation changed model state')
            if device == 'cuda': torch.cuda.synchronize()
            seconds = time.perf_counter()-epoch_start
            improved = validation_metrics['loss'] < best_loss
            if improved:
                best_loss = validation_metrics['loss']; best_epoch = epoch
                with torch.no_grad(): probe_logits = model(probe).cpu()
                torch.save(dict(model_state_dict=model.state_dict(), epoch=epoch,
                    validation_loss=best_loss, config=config, probe_logits=probe_logits), out/'best.pt')
            row = dict(epoch=epoch, train=train_metrics, validation=validation_metrics,
                       training_parameters_changed=changed, validation_parameters_unchanged=unchanged, saved_best=improved, seconds=seconds)
            result['history'].append(row)
            with (out/'epochs.jsonl').open('a') as f: f.write(json.dumps(row, allow_nan=False)+'\n')
            if epoch == 3:
                # Freeze the best checkpoint among epochs 1..3, not necessarily epoch 3 itself.
                snapshot_dir = out/'first3'; snapshot_dir.mkdir()
                shutil.copy2(out/'best.pt', snapshot_dir/'best.pt')
                equal = verify_checkpoint(snapshot_dir/'best.pt',probe,device,best_epoch)
                snapshot = copy.deepcopy(result)
                snapshot.update(status='passed',error_count=0,best_epoch=best_epoch,
                    best_validation_loss=best_loss,checkpoint_reload_equal=equal,
                    checkpoint_sha256=sha256(snapshot_dir/'best.pt'),
                    read_counts={k:v.read_calls for k,v in datasets.items()},
                    completed_epochs=3,scope='Best checkpoint within first 3 epochs; parent run may continue',
                    parent_run_status_at_snapshot='running')
                write_json(snapshot_dir/'training_summary.json',snapshot)
                result['first3_snapshot'] = dict(status='saved',directory='first3',best_epoch=best_epoch,
                    checkpoint_sha256=snapshot['checkpoint_sha256'],
                    training_summary_sha256=sha256(snapshot_dir/'training_summary.json'))
            log(f'Epoch {epoch}/{epochs}: train_loss={train_metrics["loss"]:.6f} '
                f'val_loss={validation_metrics["loss"]:.6f} val_macro_f1={validation_metrics["macro_f1"]:.6f} '
                f'seconds={seconds:.2f} saved_best={improved}')
        result['peak_cuda_memory'] = (dict(allocated_bytes=torch.cuda.max_memory_allocated(),
            reserved_bytes=torch.cuda.max_memory_reserved()) if device == 'cuda' else None)
        verify_checkpoint(out/'best.pt',probe,device,best_epoch)
        result.update(status='passed', error_count=0, best_epoch=best_epoch,
            best_validation_loss=best_loss, checkpoint_reload_equal=True, completed_epochs=epochs,
            checkpoint_sha256=sha256(out/'best.pt'),
            timing_scope='Epoch includes full loading, transfers, train/validation and checks; excludes checkpoint write. Not inference latency.',
            scope='C2 slice-level pilot; no test evaluation or patient-level performance claim')
        if epochs < 3:
            result['first3_snapshot'] = dict(status='not_reached',reason='Fewer than 3 epochs completed')
        log(f'PASS: best_epoch={best_epoch}; no test image reads')
    except Exception as exc:
        result.update(status='failed', error_count=1, error=f'{type(exc).__name__}: {exc}')
        log(traceback.format_exc())
        raise
    finally:
        result['elapsed_seconds'] = time.perf_counter()-started
        result['read_counts'] = {s:ds.read_calls for s,ds in datasets.items()}
        write_json(out/'training_summary.json', result)
    return result


def plot_recorded_history(history_path, output_path):
    """Plot observed epoch losses and validation patient macro-F1 when available."""
    rows = [json.loads(line) for line in Path(history_path).read_text().splitlines() if line.strip()]
    require(bool(rows) and [r['epoch'] for r in rows] == list(range(1, len(rows)+1)),
            'History must contain consecutive epochs from one')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    epochs = [r['epoch'] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.0))
    axes[0].plot(epochs, [r['train']['unweighted_loss'] for r in rows], label='Train')
    axes[0].plot(epochs, [r['validation']['loss'] for r in rows], label='Validation')
    axes[0].set(xlabel='Epoch', ylabel='Slice cross-entropy', title='Loss')
    axes[0].legend(frameon=False)
    if all('patient_metrics' in r and 'macro_f1' in r['patient_metrics'] for r in rows):
        axes[1].plot(epochs, [r['patient_metrics']['macro_f1'] for r in rows], color='#4B8268')
        axes[1].set(xlabel='Epoch', ylabel='Macro-F1', title='Validation patients', ylim=(0, 1))
    else:
        axes[1].plot(epochs, [r['validation']['macro_f1'] for r in rows], color='#4B8268')
        axes[1].set(xlabel='Epoch', ylabel='Macro-F1', title='Validation slices', ylim=(0, 1))
    for ax in axes:
        ax.grid(alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    if '--test-mode' in sys.argv[1:]:
        from predict import main as prediction_main
        prediction_main()
        return
    if '--plot-history' in sys.argv[1:]:
        parser = argparse.ArgumentParser(description='Plot saved training/validation history')
        parser.add_argument('--plot-history', action='store_true', required=True)
        parser.add_argument('--history', type=Path, required=True)
        parser.add_argument('--plot-output', type=Path, required=True)
        args = parser.parse_args()
        plot_recorded_history(args.history, args.plot_output)
        return
    if '--formal-convnext' in sys.argv[1:]:
        parser = argparse.ArgumentParser(description='Train the formal ConvNeXt model')
        parser.add_argument('--formal-convnext', action='store_true', required=True)
        for name in ('manifest', 'summary', 'weights', 'weights-summary',
                     'output-dir', 'n5-reference'):
            parser.add_argument('--' + name, type=Path, required=True)
        parser.add_argument('--r2-reference', type=Path)
        parser.add_argument('--seed', type=int, choices=(42, 43, 44), default=42)
        parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
        args = parser.parse_args()
        from run_r3_convnext import run_formal
        run_formal(args.manifest, args.summary, args.weights, args.weights_summary,
                   args.output_dir, args.device, args.seed,
                   n5_reference=args.n5_reference, r2_reference=args.r2_reference)
        plot_recorded_history(args.output_dir / 'epochs.jsonl',
                              args.output_dir / 'learning_curves.png')
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--class-weighting',choices=['none','balanced'],default='none')
    args = parser.parse_args()
    run_training(args.manifest, args.summary, args.output_dir, args.device, args.epochs,
                 args.batch_size, args.learning_rate, args.seed,
                 expected_lengths={'train':8160,'validation':2720,'test':2720}, class_weighting=args.class_weighting)
    plot_recorded_history(args.output_dir / 'epochs.jsonl',
                          args.output_dir / 'learning_curves.png')


if __name__ == '__main__': main()
