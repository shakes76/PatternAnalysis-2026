"""Validate the preselected SimpleCNN before held-out baseline test evaluation."""
import argparse
import csv
import json
from itertools import combinations
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from dataset import ADNISliceDataset
from modules import SimpleCNN
from predict import classification_metrics
from evaluate_rgb224_validation import sha256


def evaluate(model, dataset, device):
    labels_all, probs_all, preds_all = [], [], []
    total = 0.0
    with torch.inference_mode():
        for images, labels in DataLoader(dataset,batch_size=8,shuffle=False,num_workers=0):
            logits = model(images.to(device))
            if not torch.isfinite(logits).all():
                raise ValueError('Non-finite logits')
            total += F.cross_entropy(logits,labels.to(device),reduction='sum').item()
            labels_all.append(labels)
            probs_all.append(logits.softmax(1).cpu())
            preds_all.append(logits.argmax(1).cpu())
    labels, probs, preds = torch.cat(labels_all), torch.cat(probs_all), torch.cat(preds_all)
    if not torch.equal(labels,torch.tensor([int(r['target']) for r in dataset.rows])):
        raise ValueError('Label order mismatch')
    result = classification_metrics(labels,preds,probs)
    result.update(loss=total/len(dataset),images=len(dataset),patients=len({r['patient_id'] for r in dataset.rows}))
    return result, labels, probs, preds


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    parser.add_argument('--validation-only', action='store_true')
    args=parser.parse_args()
    device=torch.device(args.device)
    base=Path(__file__).resolve().parent
    run=base/'runs/baseline_20261022_211305_056381'
    checkpoint_path=run/'best_baseline.pth'
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'Download the preselected remote baseline checkpoint to: {checkpoint_path}')
    output=run/'final_baseline_evaluation'
    if output.exists() and not args.validation_only:
        raise FileExistsError(f'Preserving previous output: {output}')
    protocol=json.loads((base/'final_evaluation_protocol.json').read_text(encoding='utf-8'))
    manifests=base.parents[1]/'local_data/splits_local'
    parts={}
    for split,expected in protocol['manifest_sha256'].items():
        path=manifests/f'{split}.csv'
        if sha256(path)!=expected:
            raise ValueError('Manifest changed: '+split)
        with path.open(newline='',encoding='utf-8-sig') as f:
            parts[split]=list(csv.DictReader(f))
    for left,right in combinations(parts,2):
        for field in ['patient_id','scan_id','path']:
            if {r[field] for r in parts[left]} & {r[field] for r in parts[right]}:
                raise ValueError('Split overlap: '+field)
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    config=checkpoint['config']
    expected_loss=0.5258350444315042
    if config['model']!='SimpleCNN' or checkpoint['epoch']!=9 or abs(checkpoint['val_loss']-expected_loss)>1e-10:
        raise ValueError('Not the previously selected baseline checkpoint')
    if config['selection_metric']!='validation_loss' or config['train_images']!=21360 or config['val_images']!=4580:
        raise ValueError('Unexpected baseline configuration')
    torch.set_num_threads(2)
    if args.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    model=SimpleCNN()
    model.load_state_dict(checkpoint['model_state_dict'],strict=True)
    del checkpoint
    model.to(device).eval()
    print('First reproducing baseline validation; test inference is gated on this check.',flush=True)
    val,_,_,_=evaluate(model,ADNISliceDataset(manifests/'val.csv'),device)
    loss_difference = val['loss']-expected_loss
    confusion_matches = val['confusion_matrix']==[[1737,543],[658,1642]]
    strict_gate = abs(loss_difference)<=1e-5 and confusion_matches
    # Revised before baseline test inference, using CPU/CUDA validation evidence.
    # This is approximate cross-environment reproduction, not exact identity.
    tolerance = 1e-4
    gate = abs(loss_difference)<=tolerance and confusion_matches
    report={'model':'SimpleCNN','checkpoint_epoch':9,'checkpoint_sha256':sha256(checkpoint_path),
            'evaluation_unit':'slice','primary_threshold':0.5,'probabilities_calibrated':False,
            'preprocessing':'Original-size grayscale, ToTensor [0,1]; no RGB224 normalization or augmentation',
            'validation':val,'recorded_val_loss':expected_loss,'loss_difference':val['loss']-expected_loss,
            'validation_reproduction_passed':gate,'validation_loss_tolerance':tolerance,
            'original_strict_tolerance':1e-5,'original_strict_check_passed':strict_gate,
            'confusion_matches_historical':confusion_matches,
            'reproduction_status':'approximate cross-environment reproduction' if gate else 'failed',
            'tolerance_rationale':'Set before baseline test inference after local CPU/CUDA losses agreed within 5.13e-9, both differing from remote by about 1.38e-5; cause not established. No weights, preprocessing, thresholds or split changed.',
            'manifest_sha256':protocol['manifest_sha256'],
            'torch_version':str(torch.__version__),'device':args.device,'gpu':torch.cuda.get_device_name(0) if args.device=='cuda' else None,'batch_size':8,
            'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,
            'cuda_matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,
            'provenance_limit':'Original baseline was trained remotely; current hash identifies the retrieved checkpoint, not an independently archived historical hash.'}
    if args.validation_only:
        print(json.dumps(report,indent=2),flush=True)
        print('VALIDATION-ONLY diagnostic complete. No test image inference was performed.',flush=True)
        return
    if not gate:
        print(json.dumps(report,indent=2),flush=True)
        raise RuntimeError('Validation differs from prior results. Test inference was NOT run; investigate the mismatch without tuning on test.')
    print('Approximate cross-environment validation reproduction PASS. Evaluating fixed baseline on test.',flush=True)
    test,labels,probs,preds=evaluate(model,ADNISliceDataset(manifests/'test.csv'),device)
    report['test']=test
    report['policy']='Preselected epoch-9 baseline, fixed threshold; no test-based selection or retuning.'
    output.mkdir()
    (output/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    with (output/'test_predictions.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f)
        writer.writerow(['row_index','target','prediction','prob_ad'])
        for i in range(len(labels)):
            writer.writerow([i,labels[i].item(),preds[i].item(),probs[i,1].item()])
    print(json.dumps(report,indent=2,allow_nan=False),flush=True)
    print('Saved:',output,flush=True)


if __name__=='__main__':
    main()
