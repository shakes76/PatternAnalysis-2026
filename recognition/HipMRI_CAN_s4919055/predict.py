"""Reload a trained baseline and save masks, hard metrics and prostate panels.

Debug checkpoints are restricted to their original training case. Held-out
test inference requires an explicit --final-test, after model choices freeze.
Saved masks use the resized RAS grid; no original-resolution claim is made.
"""

import argparse
import os
from pathlib import Path
import time

import nibabel as nib
import numpy as np
import torch

from dataset import DEFAULT_DATA_ROOT, HipMRIVolumeDataset, LABEL_NAMES
from modules import CAN3D
from experiment import (get_device, load_checkpoint, file_sha256, predict_volume,
                        hard_overlap, aggregate_cases, save_json, seed_everything)


def save_panels(sample, prediction, output):
    """First, largest and last prostate slices; FP red and FN blue."""
    os.environ.setdefault('MPLCONFIGDIR', str(Path(output).parent/'.mpl-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    truth = sample['mask'] == 5
    predicted = prediction == 5
    areas = truth.sum(dim=(1, 2))
    occupied = torch.where(areas > 0)[0]
    slices = sorted(set([int(occupied[0]), int(areas.argmax()), int(occupied[-1])])) if len(occupied) else [len(truth)//2]
    figure, axes = plt.subplots(len(slices), 4, figsize=(12, 3*len(slices)), squeeze=False)
    image = sample['image'][0].numpy()
    low, high = np.percentile(image, [1, 99])
    for row, z in enumerate(slices):
        t, p = truth[z].numpy(), predicted[z].numpy()
        axes[row, 0].imshow(image[z], cmap='gray', vmin=low, vmax=high)
        axes[row, 1].imshow(t, cmap='gray', vmin=0, vmax=1)
        axes[row, 2].imshow(p, cmap='gray', vmin=0, vmax=1)
        axes[row, 3].imshow(image[z], cmap='gray', vmin=low, vmax=high)
        overlay = np.zeros((*t.shape, 4), dtype=np.float32)
        overlay[p & ~t] = [1, 0, 0, .75]
        overlay[t & ~p] = [0, .4, 1, .75]
        axes[row, 3].imshow(overlay, interpolation='nearest')
        for ax, title in zip(axes[row], ['MRI', 'Prostate truth', 'Prostate prediction', 'FP red / FN blue']):
            ax.set_title(f'{title}: z={z}', fontsize=9)
            ax.set_axis_off()
    figure.suptitle(sample['case_id']+' (resized RAS grid; increasing row is +Y)', fontsize=10)
    figure.tight_layout()
    figure.savefig(output, dpi=140)
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=Path('runs/split_seed42.json'))
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument('--split', choices=['train', 'validation', 'test'], default='validation')
    parser.add_argument('--output', type=Path, required=True, help='New directory; refuses overwrite.')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-cases', type=int, default=1, help='Preview subset, not full test evaluation.')
    parser.add_argument('--final-test', action='store_true')
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.max_cases < 1:
        parser.error('Batch and case counts must be positive.')
    if args.split == 'test' and not args.final_test:
        parser.error('Test previews require --final-test after model choices are fixed.')
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '1')))
    device = get_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    config = checkpoint['config']
    # Match the deterministic convolution settings used in fit/validation.
    # Preview processes otherwise start with PyTorch's default CUDA policy.
    seed_everything(config['seed'])
    if file_sha256(args.manifest) != config['manifest_sha256']:
        raise ValueError('Use the exact saved split used to train this checkpoint.')
    if config['mode'] == 'debug' and args.split != 'train':
        raise ValueError('Debug checkpoints are for training-only debugging, not held-out evaluation.')
    dataset = HipMRIVolumeDataset(args.manifest, args.split, args.data_root, tuple(config['shape_dhw']))
    if config['mode'] == 'debug':
        dataset.pairs = [pair for pair in dataset.pairs if pair['case_id'] == checkpoint['case_id']]
    args.output.mkdir(parents=True, exist_ok=False)
    cases = []
    for index in range(min(args.max_cases, len(dataset))):
        sample = dataset[index]
        if device.type == 'cuda':
            # Warm up one batch before timing the complete case.
            with torch.no_grad():
                warm_input = (sample['image'][None] if isinstance(model, CAN3D) else
                              sample['image'].permute(1, 0, 2, 3)[:args.batch_size])
                model(warm_input.to(device))
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        prediction = predict_volume(model, sample['image'], device, args.batch_size).argmax(0)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        seconds = time.perf_counter()-start
        result = hard_overlap(prediction, sample['mask'])
        result.update(case_id=sample['case_id'], patient_id=sample['patient_id'],
                      inference_seconds=seconds,
                      peak_inference_allocated_mib=(torch.cuda.max_memory_allocated()/2**20
                                                    if device.type == 'cuda' else None))
        cases.append(result)
        out = args.output/sample['case_id']
        mask_xyz = prediction.numpy().transpose(2, 1, 0).astype(np.uint8)
        nib.save(nib.Nifti1Image(mask_xyz, sample['affine_xyz'].numpy()), str(out)+'_prediction_RAS.nii.gz')
        save_panels(sample, prediction, str(out)+'_prostate.png')
        print(f'{sample["case_id"]}: prostate Dice={result["dice"][5]}, seconds/full case={seconds:.4f}', flush=True)
    report = aggregate_cases(cases)
    report.update(cases=cases, label_names=LABEL_NAMES, split=args.split,
                  model=config['model'], checkpoint_epoch=checkpoint.get('epoch'),
                  preview_subset=True, debug_checkpoint=config['mode'] == 'debug',
                  checkpoint_sha256=file_sha256(args.checkpoint),
                  manifest_sha256=file_sha256(args.manifest),
                  shape_dhw=config['shape_dhw'], base_channels=config['base_channels'],
                  slice_batch_size=args.batch_size if config['model'] == 'unet2d' else None,
                  volume_batch_size=1 if config['model'] == 'can3d' else None, device=str(device),
                  gpu_name=torch.cuda.get_device_name() if device.type == 'cuda' else None,
                  torch_version=str(torch.__version__), nibabel_version=nib.__version__,
                  parameter_count=sum(p.numel() for p in model.parameters()),
                  timing_scope='preprocessed CPU input to reassembled CPU hard mask; excludes disk/plots')
    save_json(args.output/'preview_metrics.json', report)
    print('PREDICTION_PREVIEW_SAVED (subset; not a full test benchmark).')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
