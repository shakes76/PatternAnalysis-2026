"""Evaluate frozen checkpoints with shared metrics and inference profiling."""

import argparse
import csv
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from statistics import median

import numpy as np
import torch

from dataset import HipMRIDataset, LABEL_NAMES
from metrics import positive_slice_stats, summarize_stats
from modules import UNet2D, ImprovedUNet2D


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def add(total, current):
    for key, value in current.items():
        total[key] = total.get(key, 0) + value


def finish(totals):
    return {name: summarize_stats(totals[c]) for c, name in enumerate(LABEL_NAMES)}


@torch.inference_mode()
def run(checkpoint_path, split, output, repeats):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    name = checkpoint['model_name']
    if checkpoint['num_classes'] != 6:
        raise ValueError('Expected six classes.')
    if name == 'UNet2D':
        model = UNet2D(base_channels=checkpoint['base_channels'])
    elif name == 'ImprovedUNet2D':
        model = ImprovedUNet2D(base_channels=checkpoint['base_channels'], dropout=checkpoint['dropout'])
    else:
        raise ValueError(f'Unsupported model: {name}')
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model = model.cuda().eval()
    data = HipMRIDataset(checkpoint['config']['data_root'], split=split)
    output.mkdir(parents=True, exist_ok=False)
    grouped = defaultdict(list)
    for i, record in enumerate(data.samples):
        grouped[record.case_id].append(i)
    totals = [dict() for _ in LABEL_NAMES]
    cases = {}
    profiles = {}
    fields = ['case_id', 'week', 'slice_id', 'class_id', 'class_name',
              'height', 'width', 'gt_pixels', 'predicted_pixels', 'fp_pixels',
              'fn_pixels', 'eligible', 'dice', 'iou']
    with (output / 'slice_metrics_all_classes.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for case_id, indices in sorted(grouped.items()):
            # One case is cached in CPU RAM. Disk I/O is outside inference timing.
            samples = [data[i] for i in indices]
            warmed = set()
            for sample in samples:
                shape = tuple(sample['image'].shape)
                if shape not in warmed:
                    x = sample['image'].unsqueeze(0).cuda()
                    for _ in range(10):
                        model(x).argmax(dim=1)
                    del x
                    warmed.add(shape)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            gpu_times, wall_times = [], []
            predictions = []
            for repeat in range(repeats):
                events = []
                torch.cuda.synchronize()
                start_wall = time.perf_counter()
                for sample in samples:
                    x = sample['image'].unsqueeze(0).cuda()
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    prediction = model(x).argmax(dim=1)
                    end.record()
                    cpu_prediction = prediction.cpu()
                    events.append((start, end))
                    if repeat == 0:
                        predictions.append(cpu_prediction)
                    del x, prediction, cpu_prediction
                torch.cuda.synchronize()
                wall_times.append(time.perf_counter() - start_wall)
                gpu_times.append(sum(a.elapsed_time(b) for a, b in events) / 1000)
            profiles[str(case_id)] = {
                'slices': len(samples),
                'sessions': len({s['week'] for s in samples}),
                'gpu_forward_argmax_seconds': gpu_times,
                'cached_case_wall_seconds': wall_times,
                'median_gpu_seconds': median(gpu_times),
                'median_cached_wall_seconds': median(wall_times),
                'peak_allocated_gib': torch.cuda.max_memory_allocated() / 1024**3,
                'peak_reserved_gib': torch.cuda.max_memory_reserved() / 1024**3,
            }
            case_totals = [dict() for _ in LABEL_NAMES]
            archives = {}
            for sample, prediction in zip(samples, predictions):
                target = sample['mask'].unsqueeze(0)
                key = f"week_{sample['week']}_slice_{sample['slice_id']}"
                archives[key] = prediction[0].numpy().astype(np.uint8)
                for c, label in enumerate(LABEL_NAMES):
                    stats = positive_slice_stats(prediction, target, class_id=c)
                    score = summarize_stats(stats)
                    add(totals[c], stats)
                    add(case_totals[c], stats)
                    gt, pred = target == c, prediction == c
                    writer.writerow({
                        'case_id': case_id, 'week': sample['week'], 'slice_id': sample['slice_id'],
                        'class_id': c, 'class_name': label,
                        'height': target.shape[1], 'width': target.shape[2],
                        'gt_pixels': int(gt.sum()), 'predicted_pixels': int(pred.sum()),
                        'fp_pixels': int((pred & ~gt).sum()), 'fn_pixels': int((gt & ~pred).sum()),
                        'eligible': bool(stats['eligible_slices']),
                        'dice': score['mean_dice'], 'iou': score['mean_iou'],
                    })
            cases[str(case_id)] = finish(case_totals)
            np.savez_compressed(output / f'case_{case_id:03d}_predictions.npz', **archives)
            handle.flush()
            print(f'{name}: case {case_id}, {len(samples)} slices complete', flush=True)
    summary = {
        'split': split, 'model_name': name,
        'checkpoint': str(checkpoint_path.resolve()), 'checkpoint_sha256': digest(checkpoint_path),
        'checkpoint_epoch': checkpoint['epoch'],
        'saved_validation_prostate_dice': checkpoint['validation_prostate_dice'],
        'parameter_count': sum(p.numel() for p in model.parameters()),
        'slices': len(data), 'case_ids': sorted(grouped),
        'metric_protocol': 'For each class, equal mean over GT-positive slices; empty predictions score zero; absent GT slices excluded with FP reported separately',
        'overall': finish(totals), 'per_case': cases, 'inference_profiles': profiles,
        'profiling_protocol': {
            'batch_size': 1, 'precision': 'float32', 'repeats': repeats,
            'warmup_per_shape_per_case': 10,
            'gpu_time': 'Sum of CUDA event times for forward plus argmax; excludes transfers, loading, metrics, and saving',
            'wall_time': 'CPU-cached case, including transfers and forward/argmax; excludes disk loading, normalization, metrics, and saving',
            'case_scope': 'All available weeks for each patient; report slice and session counts with time',
            'memory': 'Inference only; allocated includes model; reserved is allocator cache, not additional model memory',
        },
        'gpu': torch.cuda.get_device_name(0), 'torch_version': str(torch.__version__),
        'cuda_build': torch.version.cuda,
    }
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(name, json.dumps(summary['overall']['Prostate']), flush=True)
    del model, checkpoint
    torch.cuda.empty_cache()
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--standard', type=Path, required=True)
    parser.add_argument('--improved', type=Path, required=True)
    parser.add_argument('--split', choices=('validate', 'test'), default='validate')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1 or not torch.cuda.is_available():
        raise RuntimeError('Positive repeat count and a GPU job are required.')
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Validate identities and shared data location before processing either model.
    checks = [torch.load(p, map_location='cpu', weights_only=True)
              for p in (args.standard, args.improved)]
    assert [c['model_name'] for c in checks] == ['UNet2D', 'ImprovedUNet2D']
    assert checks[0]['config']['data_root'] == checks[1]['config']['data_root']
    del checks
    args.output.mkdir(parents=True, exist_ok=False)
    summaries = [run(path, args.split, args.output / label, args.repeats)
                 for path, label in [(args.standard, 'standard'), (args.improved, 'improved')]]
    assert summaries[0]['case_ids'] == summaries[1]['case_ids']
    assert summaries[0]['slices'] == summaries[1]['slices']
    (args.output / 'comparison.json').write_text(json.dumps([
        {k: s[k] for k in ('model_name', 'checkpoint_epoch', 'parameter_count', 'overall')}
        for s in summaries], indent=2), encoding='utf-8')
    print(f'Evaluation complete. Split: {args.split}', flush=True)


if __name__ == '__main__':
    main()
