"""Full validation-only evaluation of custom RGB224 checkpoints; no test access."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from dataset_rgb224 import RGB224Dataset
from modules import ConvNeXtTinyRGB
from predict import classification_metrics


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=base/'runs/local_rgb224_lr3e6_imagenet_20261026_201343_178908')
    args = parser.parse_args()
    output = args.run/'validation_rgb224_evaluation'
    if output.exists():
        raise FileExistsError(f'Preserving existing output: {output}')
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    path = args.run/'best_convnext.pth'
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    config = checkpoint['config']
    if config['model'] != 'ConvNeXtTinyRGB' or config.get('diagnostic_only', True):
        raise ValueError('Expected a full-run custom RGB checkpoint')
    expected = {'size':[224,224], 'resize':'aspect-preserving bilinear with black padding',
                'channels':3, 'mean':[0.485,0.456,0.406], 'std':[0.229,0.224,0.225]}
    if config['preprocessing'] != expected or config['validation_augmentation'] is not None:
        raise ValueError('Unsupported preprocessing configuration')
    if config['selection_metric'] != 'validation_loss':
        raise ValueError('Unexpected selection rule')
    manifests = base.parents[1]/'local_data/splits_local'
    hashes = {name:sha256(manifests/f'{name}.csv') for name in ['train','val']}
    if hashes != config['manifest_sha256']:
        raise ValueError('Manifest hashes differ from checkpoint')
    train = RGB224Dataset(manifests/'train.csv', training=False)
    val = RGB224Dataset(manifests/'val.csv', training=False)
    for field in ['patient_id','scan_id','path']:
        if {r[field] for r in train.rows} & {r[field] for r in val.rows}:
            raise ValueError('Train/validation overlap: '+field)
    if len(val) != config['val_images'] or len(train) != config['train_images']:
        raise ValueError('Sample count mismatch')
    with (args.run/'training_log.csv').open(newline='') as f:
        history = list(csv.DictReader(f))
    selected = min(history,key=lambda r:float(r['val_loss']))
    if int(selected['epoch']) != checkpoint['epoch'] or not math.isclose(float(selected['val_loss']),checkpoint['val_loss'],abs_tol=1e-10,rel_tol=0):
        raise ValueError('Checkpoint does not match minimum-loss log epoch')
    model = ConvNeXtTinyRGB()
    model.load_state_dict(checkpoint['model_state_dict'],strict=True)
    model.cuda().eval()
    batch_size = config['batch_size']
    loader = DataLoader(val,batch_size=batch_size,shuffle=False,num_workers=0)
    labels_all, probs_all, preds_all = [], [], []
    loss_sum = 0.0
    print(f'Evaluating epoch {checkpoint["epoch"]}: {len(val)} validation slices',flush=True)
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.cuda())
            if not torch.isfinite(logits).all():
                raise RuntimeError('Non-finite logits')
            loss_sum += F.cross_entropy(logits,labels.cuda(),reduction='sum').item()
            labels_all.append(labels)
            probs_all.append(logits.softmax(1).cpu())
            preds_all.append(logits.argmax(1).cpu())
    labels, probs, preds = torch.cat(labels_all), torch.cat(probs_all), torch.cat(preds_all)
    if not torch.equal(labels,torch.tensor([int(r['target']) for r in val.rows])):
        raise ValueError('Label order mismatch')
    loss = loss_sum/len(val)
    metrics = classification_metrics(labels,preds,probs)
    reproduced = abs(loss-checkpoint['val_loss']) <= 1e-6
    accuracy_reproduced = abs(metrics['accuracy']-float(selected['val_accuracy'])) <= 1e-10
    metrics.update({'split':'validation','evaluation_unit':'slice','images':len(val),
                    'patients':len({r['patient_id'] for r in val.rows}),
                    'model':config['model'],'initialization':config['initialization'],
                    'checkpoint_epoch':checkpoint['epoch'],'loss':loss,
                    'recorded_val_loss':checkpoint['val_loss'],'loss_difference':loss-checkpoint['val_loss'],
                    'loss_reproduced_atol_1e6':reproduced,'accuracy_matches_training_log':accuracy_reproduced,
                    'manifest_sha256':hashes['val'],'checkpoint_sha256':sha256(path),
                    'preprocessing':expected,'probabilities_calibrated':False,
                    'torch_version':str(torch.__version__),'device':'cuda','batch_size':batch_size,
                    'row_index_definition':'zero-based data-row index in validation manifest'})
    text = json.dumps(metrics,indent=2,allow_nan=False)
    output.mkdir()
    (output/'metrics.json').write_text(text,encoding='utf-8')
    with (output/'predictions.csv').open('w',newline='',encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['row_index','target','prediction','prob_ad','confidence','correct'])
        confidence = probs.max(1).values
        for i in range(len(val)):
            writer.writerow([i,labels[i].item(),preds[i].item(),probs[i,1].item(),confidence[i].item(),int(preds[i]==labels[i])])
    print(text,flush=True)
    print('Saved:',output,flush=True)
    if not reproduced or not accuracy_reproduced:
        raise RuntimeError('Validation reproduction mismatch; diagnostics saved. Investigate before proceeding.')
    print('PASS: full validation loss and accuracy reproduced.',flush=True)


if __name__ == '__main__':
    main()
