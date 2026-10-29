"""Saved-weight ADNI inference: legacy CNN validation and test-case examples."""
import argparse
from collections import defaultdict
import csv
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
from dataset import PREPROCESSING, ConvNeXtSliceDataset
from modules import SimpleCNN, build_model
from train import CountedDataset, classification_metrics, write_json
from check_dataloader import make_loader, require, sha256, code_version


def aggregate_patients(rows, expected_slices=20):
    require(bool(rows), 'No prediction rows')
    groups=defaultdict(list); paths=set()
    for row in rows:
        require(row['file_path'] not in paths, 'Duplicate slice path')
        paths.add(row['file_path'])
        require(row['label'] in (0,1) and math.isfinite(row['p_ad']) and 0<=row['p_ad']<=1,
                'Invalid label/probability')
        groups[row['patient_id']].append(row)
    result=[]
    for patient, items in sorted(groups.items()):
        require(len(items)==expected_slices, 'Patient slice count mismatch')
        require(len({x['label'] for x in items})==1, 'Patient label mismatch')
        require(len({x['image_id'] for x in items})==1, 'Patient has multiple scans')
        probability=math.fsum(x['p_ad'] for x in items)/len(items)
        result.append(dict(patient_id=patient,image_id=items[0]['image_id'],label=items[0]['label'],
                           slice_count=len(items),p_ad=probability,prediction=int(probability>=0.5)))
    return result


def predict_pass(model, loader, device):
    model.eval(); rows=[]
    with torch.no_grad():
        for batch in loader:
            logits=model(batch['image'].to(device))
            require(tuple(logits.shape)==(len(batch['label']),2) and bool(torch.isfinite(logits).all()),
                    'Invalid prediction logits')
            probabilities=logits.softmax(dim=1)[:,1].cpu().tolist()
            for i,probability in enumerate(probabilities):
                rows.append(dict(file_path=batch['file_path'][i],patient_id=batch['patient_id'][i],
                    image_id=batch['image_id'][i],label=int(batch['label'][i]),
                    p_ad=probability,prediction=int(probability>=0.5)))
    return rows


def write_csv(path, rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def run_prediction(manifest, summary, checkpoint_path, training_summary_path, output_dir,
                   device='cuda',batch_size=16,expected_slices=20,expected_patients=136,
                   expected_lengths={'train':8160,'validation':2720,'test':2720}):
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=False)
    result=dict(status='running',generated_at=datetime.now(timezone.utc).isoformat(),
        code_version=code_version(),config=dict(device=device,batch_size=batch_size,split='validation',
        aggregation='arithmetic mean of P(AD) over all slices of one selected scan',threshold=0.5,
        expected_slices_per_patient=expected_slices,expected_patients=expected_patients,
        repeat_passes=2,num_workers=0,preprocessing=PREPROCESSING),
        versions=dict(python=platform.python_version(),torch=str(torch.__version__),
                      pillow=PIL.__version__,cuda_build=torch.version.cuda))
    datasets={};started=time.perf_counter()
    def log(message):
        print(message,flush=True)
        with (out/'prediction.log').open('a') as f:f.write(message+'\n')
    try:
        require(batch_size>0 and device in ('cpu','cuda'), 'Invalid batch size/device')
        require(device!='cuda' or torch.cuda.is_available(), 'CUDA requested but unavailable')
        result['code_sha256']={n:sha256(Path(__file__).with_name(n)) for n in
                              ('predict.py','train.py','modules.py','dataset.py','check_dataloader.py')}
        inputs=dict(split_samples=sha256(manifest),split_summary=sha256(summary))
        result['input_sha256']=inputs
        result['checkpoint_sha256']=sha256(checkpoint_path)
        result['training_summary_sha256']=sha256(training_summary_path)
        source=json.loads(Path(training_summary_path).read_text())
        require(source['status']=='passed', 'Training run has not passed')
        require(source['input_sha256']==inputs, 'Training input hash mismatch')
        require(source['checkpoint_sha256']==result['checkpoint_sha256'], 'Checkpoint hash mismatch')
        for name in ('modules.py','dataset.py'):
            require(source['code_sha256'][name]==result['code_sha256'][name], 'Training model/preprocessing source mismatch')
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        torch.manual_seed(42);torch.use_deterministic_algorithms(True)
        if device=='cuda':
            torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
            torch.cuda.reset_peak_memory_stats()
        result['gpu_name']=torch.cuda.get_device_name() if device=='cuda' else None
        datasets={s:CountedDataset(manifest,s,summary) for s in ('train','validation','test')}
        lengths={s:len(d) for s,d in datasets.items()}
        require(expected_lengths is None or lengths==expected_lengths, 'Frozen cohort lengths mismatch')
        result['dataset_lengths']=lengths
        # Validate group size and identity before reading any image.
        metadata=[dict(r,label=0 if r['label']=='NC' else 1,p_ad=0.0) for r in datasets['validation'].records]
        require(len(aggregate_patients(metadata,expected_slices))==expected_patients, 'Patient count mismatch')
        checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
        require(checkpoint['epoch']==source['best_epoch'] and
                checkpoint['validation_loss']==source['best_validation_loss'], 'Best checkpoint metadata mismatch')
        require(checkpoint['config']==source['config'], 'Checkpoint configuration mismatch')
        require(checkpoint['config']['model']=='SimpleCNN' and checkpoint['config']['preprocessing']==PREPROCESSING,
                'Checkpoint model/preprocessing mismatch')
        result['best_epoch']=checkpoint['epoch']
        model=SimpleCNN().to(device);model.load_state_dict(checkpoint['model_state_dict'])
        before={k:v.detach().clone() for k,v in model.state_dict().items()}
        loader=make_loader(datasets['validation'],batch_size,42)
        rows=predict_pass(model,loader,device)
        require(all(torch.equal(v,model.state_dict()[k]) for k,v in before.items()), 'Prediction changed model state')
        # A new model is loaded from the checkpoint and repeats every validation slice.
        del model
        model=SimpleCNN().to(device);model.load_state_dict(checkpoint['model_state_dict'])
        repeated=predict_pass(model,loader,device)
        require(len(rows)==len(repeated)==len(datasets['validation']), 'Prediction count mismatch')
        max_diff=0.0
        for a,b in zip(rows,repeated):
            require(all(a[k]==b[k] for k in ('file_path','patient_id','image_id','label','prediction')), 'Repeat identity/class mismatch')
            max_diff=max(max_diff,abs(a['p_ad']-b['p_ad']))
            require(math.isclose(a['p_ad'],b['p_ad'],rel_tol=1e-5,abs_tol=1e-6), 'Repeat probability mismatch')
        require(all(torch.equal(v,model.state_dict()[k]) for k,v in before.items()), 'Repeat changed model state')
        patients=aggregate_patients(rows,expected_slices)
        require(len(patients)==expected_patients, 'Patient count mismatch')
        write_csv(out/'slice_predictions.csv',rows);write_csv(out/'patient_predictions.csv',patients)
        metrics=lambda records:classification_metrics([r['label'] for r in records],[r['p_ad'] for r in records])
        result.update(status='passed',error_count=0,parameters_unchanged=True,repeat_equal=True,
            repeat_max_probability_difference=max_diff,slice_count=len(rows),patient_count=len(patients),
            slice_metrics=metrics(rows),patient_metrics=metrics(patients),
            output_sha256={n:sha256(out/n) for n in ('slice_predictions.csv','patient_predictions.csv')},
            c2_best_validation_metrics=source['history'][source['best_epoch']-1]['validation'],
            peak_cuda_memory=dict(allocated_bytes=torch.cuda.max_memory_allocated(),reserved_bytes=torch.cuda.max_memory_reserved()) if device=='cuda' else None,
            scope='Validation only; two full passes. Not test performance or a formal latency benchmark.')
        log(f'PASS: {len(rows)} slices, {len(patients)} patients; repeated predictions agree; test reads=0')
    except Exception as exc:
        result.update(status='failed',error_count=1,error=f'{type(exc).__name__}: {exc}')
        log(traceback.format_exc());raise
    finally:
        result['read_counts']={s:d.read_calls for s,d in datasets.items()}
        result['elapsed_seconds']=time.perf_counter()-started
        write_json(out/'prediction_summary.json',result)
    return result


def run_test_prediction(manifest, summary, checkpoint_path, output_dir, model_name,
                        device='cpu', batch_size=4, expected_slices=20):
    """Demonstrate saved-weight inference on test cases, with patient-level output.

    This standalone example does not refit a threshold or replace the nine-model
    frozen evaluation in run_r6_final.py. Individual output rows remain private.
    """
    if model_name not in ('simple_cnn', 'convnext_tiny'):
        raise ValueError('Unknown model name')
    require(device in ('cpu', 'cuda') and (device != 'cuda' or torch.cuda.is_available()),
            'Invalid or unavailable device')
    require(batch_size > 0 and expected_slices > 0, 'Invalid batch size or slice count')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    expected_model = 'SimpleCNN' if model_name == 'simple_cnn' else 'convnext_tiny'
    if checkpoint.get('config', {}).get('model') != expected_model:
        raise ValueError('Checkpoint model does not match requested model')
    model = build_model(model_name, num_classes=2)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    dataset = (CountedDataset if model_name == 'simple_cnn' else ConvNeXtSliceDataset)(
        manifest, 'test', summary)
    model = model.to(device)
    rows = predict_pass(model, make_loader(dataset, batch_size, 42), device)
    require(dataset.read_calls == len(dataset) == len(rows), 'Incomplete test inference')
    patients = aggregate_patients(rows, expected_slices)
    metrics = classification_metrics([r['label'] for r in patients],
                                     [r['p_ad'] for r in patients])
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    write_csv(out / 'slice_predictions.csv', rows)
    write_csv(out / 'patient_predictions.csv', patients)
    # The figure contains aggregate counts only; case identifiers remain in the CSVs.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(3.2, 2.9))
    ax.imshow(metrics['confusion_matrix'], cmap='Blues', vmin=0)
    ax.set(xticks=(0, 1), yticks=(0, 1), xticklabels=('CN', 'AD'),
           yticklabels=('CN', 'AD'), xlabel='Predicted', ylabel='Ground truth',
           title='Test patients')
    for truth in (0, 1):
        for prediction in (0, 1):
            ax.text(prediction, truth, str(metrics['confusion_matrix'][truth][prediction]),
                    ha='center', va='center')
    fig.tight_layout()
    fig.savefig(out / 'confusion_matrix.png', dpi=180)
    plt.close(fig)
    result = dict(status='passed', split='test', model=model_name,
                  checkpoint_sha256=sha256(checkpoint_path),
                  manifest_sha256=sha256(manifest), image_reads=dataset.read_calls,
                  slice_count=len(rows), patient_count=len(patients),
                  expected_slices=expected_slices, patient_metrics=metrics,
                  output_files=['slice_predictions.csv', 'patient_predictions.csv',
                                'confusion_matrix.png'])
    write_json(out / 'test_prediction_summary.json', result)
    return result


def main():
    if '--test-mode' in sys.argv[1:]:
        parser = argparse.ArgumentParser(description='Saved-checkpoint test inference example')
        parser.add_argument('--test-mode', action='store_true', required=True)
        for name in ('manifest', 'summary', 'checkpoint', 'output-dir'):
            parser.add_argument('--' + name, type=Path, required=True)
        parser.add_argument('--model', choices=('simple_cnn', 'convnext_tiny'), required=True)
        parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
        parser.add_argument('--batch-size', type=int, default=4)
        parser.add_argument('--expected-slices', type=int, default=20)
        args = parser.parse_args()
        run_test_prediction(args.manifest, args.summary, args.checkpoint,
                            args.output_dir, args.model, args.device,
                            args.batch_size, args.expected_slices)
        return
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','summary','checkpoint','training-summary','output-dir'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--batch-size',type=int,default=16)
    args=parser.parse_args()
    run_prediction(args.manifest,args.summary,args.checkpoint,args.training_summary,args.output_dir,
                   args.device,args.batch_size)

if __name__=='__main__':main()
