"""Evaluate the fixed main-model checkpoint on held-out test slices once."""
import csv
import json
from itertools import combinations
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from dataset_rgb224 import RGB224Dataset
from modules import ConvNeXtTinyRGB
from predict import classification_metrics
from evaluate_rgb224_validation import sha256


def main():
    base=Path(__file__).resolve().parent
    protocol_path=base/'final_evaluation_protocol.json'
    protocol=json.loads(protocol_path.read_text(encoding='utf-8'))
    run=base/'runs'/protocol['run']
    output=run/'final_test_evaluation'
    if output.exists():
        raise FileExistsError(f'Preserving existing test evaluation: {output}')
    checkpoint_path=run/'best_convnext.pth'
    if sha256(checkpoint_path)!=protocol['checkpoint_sha256']:
        raise ValueError('Checkpoint differs from frozen protocol')
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
    verified=json.loads((run/'validation_rgb224_evaluation/metrics.json').read_text(encoding='utf-8'))
    if config['model']!=protocol['model'] or checkpoint['epoch']!=protocol['checkpoint_epoch']:
        raise ValueError('Unexpected model/epoch')
    if config['preprocessing']!=verified['preprocessing'] or verified['checkpoint_sha256']!=protocol['checkpoint_sha256']:
        raise ValueError('Validation identity mismatch')
    if not verified['loss_reproduced_atol_1e6'] or not verified['accuracy_matches_training_log']:
        raise ValueError('Validation reproduction must pass first')
    if protocol['primary_threshold']!=0.5 or protocol['secondary_threshold']!=0.21:
        raise ValueError('Unexpected predefined thresholds')
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    model=ConvNeXtTinyRGB()
    model.load_state_dict(checkpoint['model_state_dict'],strict=True)
    del checkpoint
    model.cuda().eval()
    data=RGB224Dataset(manifests/'test.csv',training=False)
    if len(data)!=4580 or len({r['patient_id'] for r in data.rows})!=104:
        raise ValueError('Unexpected test population')
    loader=DataLoader(data,batch_size=8,shuffle=False,num_workers=0)
    labels_all,probs_all,preds_all=[],[],[]
    loss_sum=0.
    print('Evaluating fixed checkpoint on 4580 held-out test slices.',flush=True)
    with torch.inference_mode():
        for images,labels in loader:
            logits=model(images.cuda())
            if not torch.isfinite(logits).all():
                raise ValueError('Non-finite logits')
            loss_sum+=F.cross_entropy(logits,labels.cuda(),reduction='sum').item()
            labels_all.append(labels)
            probs_all.append(logits.softmax(1).cpu())
            preds_all.append(logits.argmax(1).cpu())
    labels,probs,preds=torch.cat(labels_all),torch.cat(probs_all),torch.cat(preds_all)
    if not torch.equal(labels,torch.tensor([int(r['target']) for r in data.rows])):
        raise ValueError('Label order mismatch')
    secondary=(probs[:,1]>protocol['secondary_threshold']).long()
    report={'split':'test','evaluation_unit':'slice','images':len(data),'patients':104,
            'checkpoint_sha256':protocol['checkpoint_sha256'],'checkpoint_epoch':protocol['checkpoint_epoch'],
            'protocol_sha256':sha256(protocol_path),'manifest_sha256':protocol['manifest_sha256']['test'],
            'loss':loss_sum/len(data),'primary_threshold_0_50':classification_metrics(labels,preds,probs),
            'supplementary_threshold_0_21':classification_metrics(labels,secondary,probs),
            'preprocessing':config['preprocessing'],'torch_version':str(torch.__version__),
            'gpu':torch.cuda.get_device_name(0),'batch_size':8,'probabilities_calibrated':False,
            'policy':protocol['policy'],'baseline_status':'Pending separate evaluation; not included here'}
    serialized=json.dumps(report,indent=2,allow_nan=False)
    output.mkdir()
    (output/'metrics.json').write_text(serialized,encoding='utf-8')
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    with (output/'predictions.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f)
        writer.writerow(['row_index','target','prediction_primary','prediction_secondary','prob_ad'])
        for i in range(len(data)):
            writer.writerow([i,labels[i].item(),preds[i].item(),secondary[i].item(),probs[i,1].item()])
    print(serialized,flush=True)
    print('Saved:',output,flush=True)


if __name__=='__main__':
    main()
