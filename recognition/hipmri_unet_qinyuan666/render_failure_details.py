"""Render anonymized, shared display crops for the five fixed failure examples.

Crops use the union of GT and both predictions plus context, only for display.
Inference still uses the complete 256x256 padded input. No new examples selected.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from analyze_boundaries import select_failures
from dataset import HipMRIDataset
from metrics import confusion_matrix, scores_from_confusion
from modules import build_model
from predict import _device, _validate_checkpoint


def render(analysis, manifest, checkpoints, output, device='cuda'):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Choose a new figure directory')
    read = lambda path: json.loads(Path(path).read_text(encoding='utf-8'))
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    report = read(Path(analysis)/'summary.json')
    rows = read(Path(analysis)/'slice_details_private.json')
    assert sha(manifest) == report['manifest_sha256']
    chosen = select_failures(rows)
    data = HipMRIDataset(manifest, 'test', 256, list(range(6)), resize_mode='pad')
    arrays = {row['index']: {} for row, _ in chosen}
    target_device = _device(device)
    for checkpoint in checkpoints:
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
        config, _ = _validate_checkpoint(payload, False)
        name = config['model']
        assert sha(checkpoint) == report['checkpoints'][name]['checkpoint_sha256']
        model = build_model(name, num_classes=6, base_channels=config['base_channels']).to(target_device)
        model.load_state_dict(payload['model_state']); model.eval()
        with torch.inference_mode():
            for row, _ in chosen:
                index = row['index']
                start = index // 4 * 4
                batch = torch.stack([data[i]['image'] for i in range(start, min(start+4, len(data)))])
                pred = model(batch.to(target_device)).argmax(1).cpu()[index-start]
                actual = scores_from_confusion(confusion_matrix(pred, data[index]['mask'], 6))['per_class'][5]['dice']
                assert actual == row['models'][name]['prostate_dice'], 'Prediction changed'
                arrays[index][name] = pred.numpy() == 5
        model.to('cpu')
    output.mkdir(parents=True)
    for number, (row, _) in enumerate(chosen, 1):
        index = row['index']
        example = report['examples'][number-1]
        assert example['case_alias'] == row['case_alias']
        sample = data[index]
        image, truth = sample['image'][0].numpy(), sample['mask'].numpy() == 5
        union = truth | arrays[index]['unet'] | arrays[index]['improved_unet']
        ys, xs = np.where(union)
        size = max(64, int(max(ys.max()-ys.min()+1, xs.max()-xs.min()+1)) + 24)
        size = min(size, 256)
        top = int(np.clip((ys.min()+ys.max()+1-size)//2, 0, 256-size))
        left = int(np.clip((xs.min()+xs.max()+1-size)//2, 0, 256-size))
        crop = np.s_[top:top+size, left:left+size]
        fig, axes = plt.subplots(2, 4, figsize=(10, 6.2))
        for axrow, name, title in zip(axes, ('unet','improved_unet'), ('U-Net','Improved U-Net')):
            pred = arrays[index][name]
            overlay = np.zeros((*truth.shape, 4))
            overlay[pred & ~truth] = (1,.1,.1,.85)
            overlay[~pred & truth] = (.1,.35,1,.85)
            axrow[0].imshow(image[crop], cmap='gray', interpolation='nearest')
            axrow[1].imshow(truth[crop], cmap='gray', vmin=0, vmax=1, interpolation='nearest')
            axrow[2].imshow(pred[crop], cmap='gray', vmin=0, vmax=1, interpolation='nearest')
            axrow[3].imshow(image[crop], cmap='gray', interpolation='nearest')
            axrow[3].imshow(overlay[crop], interpolation='nearest')
            for ax, label in zip(axrow, (title+' / MRI', 'GT prostate', 'Prediction', 'FP red / FN blue')):
                ax.set_title(label, fontsize=11); ax.axis('off')
        fig.suptitle(f'F{number}: {row["case_alias"]}, GT area {row["prostate_truth_pixels"]} pixels\nShared {size} x {size} display crop; inference uses full image', fontsize=12)
        fig.text(.5,.02,'HipMRI Study / CSIRO v2, course 2D data. Selected failure; not typical performance.',ha='center',fontsize=9)
        fig.subplots_adjust(left=.02, right=.98, bottom=.09, top=.81, hspace=.38, wspace=.12)
        fig.savefig(output/f'failure_F{number}.png',dpi=170)
        plt.close(fig)
    print('Rendered five matched display crops; recorded slice Dice reproduced exactly.')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--checkpoints',type=Path,nargs=2,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    torch.set_num_threads(4)
    render(**vars(p.parse_args()))
