"""R4 validation-only inference benchmark of nine frozen patient-NLL checkpoints.

Run on one GPU, each checkpoint in a fresh process. FP32, batch 4, no TF32.
Pure forward excludes I/O/transfer/softmax. End-to-end includes validation JPEG
read, preprocessing, transfer, inference, CPU probabilities and patient means.
It excludes model loading, integrity audits, metric scoring and output writes.
"""
import argparse
import csv
import gc
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
import time
import traceback

import torch
from check_dataloader import require, sha256, make_loader
from dataset import ConvNeXtSliceDataset, PREPROCESSING, CONVNEXT_PREPROCESSING
from dual_checkpoint import load_selected
from modules import build_model
from predict import predict_pass, aggregate_patients, write_csv
from train import CountedDataset, write_json
from run_n3 import state_digest as state_sha256
from r4_prediction_audit import historical_settings, record_predictions

FROZEN_LOCK_SHA256='d781fb6bec3f9e378aaf6ddfa08229aadb2fc3c7361159f01e87b3be0106ebb4'


def protocol():
    return dict(version='r4-two-stage-v2', batch_size=4, precision='float32', num_workers=0, cpu_threads=2,
        split='validation', tf32=False, amp=False, compile=False,
        cudnn_benchmark=False, deterministic=True, memory_format='contiguous',
        forward_warmup=20, forward_steps=100, forward_repeats=5,
        end_to_end_warmup_passes=1, end_to_end_repeats=3,
        cache_scope='warm OS/filesystem cache, no application image cache',
        timing='synchronized perf_counter wall time; block timing for forward',
        input_scope='native trained preprocessing: CNN 1x240x256; ConvNeXt 3x224x224')


def summarize(seconds, samples, patients=None):
    require(seconds and all(math.isfinite(s) and s>0 for s in seconds), 'Invalid durations')
    require(type(samples) is int and samples>0, 'Invalid sample count')
    require(patients is None or (type(patients) is int and patients>0),'Invalid patient count')
    median=statistics.median(seconds)
    return dict(seconds=seconds, median_seconds=median, mean_seconds=statistics.mean(seconds),
        sample_sd_seconds=statistics.stdev(seconds) if len(seconds)>1 else 0.,
        min_seconds=min(seconds), max_seconds=max(seconds), samples_per_repeat=samples,
        median_ms_per_slice=median*1000/samples, median_slices_per_second=samples/median,
        median_ms_per_patient_amortized=median*1000/patients if patients else None)


def sync(device):
    if device=='cuda':torch.cuda.synchronize()


def timed(operation, synchronize, clock=time.perf_counter):
    synchronize();start=clock();value=operation();synchronize()
    return clock()-start,value


def reset_memory(device):
    if device=='cuda':torch.cuda.reset_peak_memory_stats()


def peak_memory(device):
    return dict(allocated_bytes=torch.cuda.max_memory_allocated(),
                reserved_bytes=torch.cuda.max_memory_reserved()) if device=='cuda' else None


def measure_forward(model, images, device, warmup=20, steps=100, repeats=5):
    require(all(type(n) is int and n>0 for n in (warmup,steps,repeats)), 'Invalid repetition count')
    model.eval();seconds=[]
    with torch.inference_mode():
        for _ in range(warmup):model(images)
        sync(device);reset_memory(device)
        def block():
            for _ in range(steps):model(images)
        for _ in range(repeats):
            elapsed,_=timed(block,lambda:sync(device));seconds.append(elapsed)
    result=summarize(seconds,steps*len(images))
    result.update(peak_memory=peak_memory(device),steps_per_repeat=steps,
                  median_ms_per_batch=result['median_seconds']*1000/steps)
    return result


def end_to_end_pass(model, loader, device, expected_slices=20):
    with torch.inference_mode():
        rows=predict_pass(model,loader,device)
        patients=aggregate_patients(rows,expected_slices)
    return rows,patients


def compare_patients(actual, expected):
    a={r['patient_id']:r for r in actual};b={r['patient_id']:r for r in expected}
    require(len(a)==len(actual)==len(b)==len(expected) and a.keys()==b.keys(),'Prediction cohort mismatch')
    maximum=0.
    for pid,row in a.items():
        ref=b[pid]
        require(all(str(row[k])==str(ref[k]) for k in ('image_id','label','slice_count','prediction')),'Prediction identity/class mismatch')
        p,q=float(row['p_ad']),float(ref['p_ad'])
        require(math.isfinite(p) and math.isfinite(q) and math.isclose(p,q,rel_tol=1e-5,abs_tol=1e-6),'Prediction probability mismatch')
        maximum=max(maximum,abs(p-q))
    return maximum


def source_path(root, relative):
    root=Path(root).resolve();part=Path(relative)
    require(not part.is_absolute() and '..' not in part.parts,'Unsafe source path')
    path=(root/part).resolve()
    require(path.is_relative_to(root),'Source escaped data root')
    return path


def verify_files(root, hashes):
    require(bool(hashes),'Missing frozen hashes')
    for relative,expected in hashes.items():
        require(sha256(source_path(root,relative))==expected,'Frozen file hash mismatch: '+relative)


def validate_lock(lock):
    require(lock.get('schema')==1 and lock.get('selection_rule')=='patient_nll','Invalid lock schema/rule')
    runs=lock['runs']
    require(len(runs)==9 and {(r['model'],r['seed']) for r in runs}==
        {(m,s) for m in ('CNN-A','CNN-B','ConvNeXt-F') for s in (42,43,44)},'Expected nine unique runs')
    for r in runs:
        require(type(r['epoch']) is int and 1<=r['epoch']<=30,'Invalid best epoch')
        require(bool(re.fullmatch('[0-9a-f]{64}',r['checkpoint_sha256'])),'Invalid checkpoint hash')
        require(r['files']['selection/best_patient_nll.pt']==r['checkpoint_sha256'],'Checkpoint lock mismatch')
        source_path(Path('/tmp'),r['source'])


def check_cohort(datasets):
    from run_n3 import validate_frozen_labels
    from patient_evaluation import evaluate_patients
    require({k:len(v) for k,v in datasets.items()}==dict(train=8160,validation=2720,test=2720),'Frozen cohort lengths mismatch')
    for split,ds in datasets.items():
        validate_frozen_labels(ds.records,split)
    ds=datasets['validation']
    evaluate_patients([dict(r,label=int(r['label']=='AD'),p_ad=0.) for r in ds.records],ds.records)


def load_frozen(root, entry, lock):
    source=source_path(root,entry['source']);verify_files(source,entry['files'])
    training=json.loads((source/'training_summary.json').read_text())
    require(training['status']=='passed','Training did not pass')
    config=training['config'];provenance=training['provenance']
    require(config['seed']==entry['seed'] and config['epochs']==30 and config['early_stopping'] is False,'Training protocol mismatch')
    model_name='convnext_tiny' if entry['model']=='ConvNeXt-F' else 'simple_cnn'
    require(config['model']==('SimpleCNN' if model_name=='simple_cnn' else model_name),'Model mismatch')
    if model_name=='simple_cnn':require(config['arm']==entry['model'][-1],'CNN arm mismatch')
    require(set(provenance['input_sha256'].values())==set(lock['data_files'].values()),'Manifest provenance mismatch')
    for n in ('modules.py','dataset.py'):
        require(provenance['code_sha256'][n]==sha256(Path(__file__).with_name(n)),'Model/preprocessing source changed: '+n)
    model,cp=load_selected(source/'selection','patient_nll',lambda:build_model(model_name),config,provenance)
    require(cp['epoch']==entry['epoch'],'Frozen best epoch mismatch')
    with (source/'reload/patient_nll/patient_predictions.csv').open() as f:expected=list(csv.DictReader(f))
    return model,expected



def replay_history(model,loader,device,out,expected_slices,expected,result):
    slices=predict_pass(model,loader,device)
    patients=aggregate_patients(slices)
    result['historical_replay']=record_predictions(out,'historical_replay',slices,patients,expected_slices,expected)
    write_json(Path(out)/'efficiency_summary.json',result)
    compare_patients(patients,expected)
    require(result['historical_replay']['slice_drift']['r4_exceed_count']==0 and
            result['historical_replay']['slice_drift']['class_change_count']==0,'Historical slice replay mismatch')
    result['historical_replay']['passed']=True


def check_repeat(out,attempt,slices,patients,warm_slices,warm,result):
    stats=record_predictions(out,f'unified_repeat_{attempt}',slices,patients,warm_slices,warm)
    result.setdefault('unified_repeats',[]).append(stats)
    write_json(Path(out)/'efficiency_summary.json',result)
    difference=compare_patients(patients,warm)
    require(stats['slice_drift']['r4_exceed_count']==stats['slice_drift']['class_change_count']==0,'Unified slice repeat mismatch')
    return difference

def run_worker(root, lock_path, entry_index, out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();result=dict(status='running');datasets={}
    try:
        require(sha256(lock_path)==FROZEN_LOCK_SHA256,'Frozen lock changed')
        lock=json.loads(Path(lock_path).read_text());validate_lock(lock);p=protocol()
        require(torch.cuda.is_available() and torch.cuda.device_count()==1,'Exactly one visible CUDA GPU required')
        require('A100' in torch.cuda.get_device_name(0),'Expected an A100 GPU')
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        torch.set_num_threads(p['cpu_threads']);torch.manual_seed(42)
        torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        verify_files(root,lock['data_files'])
        entry=lock['runs'][entry_index];model,expected=load_frozen(root,entry,lock)
        manifest=source_path(root,'split_seed42_60_20_20/split_samples.csv')
        summary=manifest.with_name('split_summary.json')
        kind=ConvNeXtSliceDataset if entry['model']=='ConvNeXt-F' else CountedDataset
        datasets={s:kind(manifest,s,summary) for s in ('train','validation','test')}
        check_cohort(datasets);ds=datasets['validation'];loader=make_loader(ds,4,42)
        before=state_sha256(model);parameters=sum(v.numel() for v in model.parameters())
        model=model.float().to('cuda').eval()
        result.update(entry=entry,protocol=p,lock_sha256=sha256(lock_path),
            parameter_count=parameters,input_shape=[4,3,224,224] if kind is ConvNeXtSliceDataset else [4,1,240,256],
            preprocessing=CONVNEXT_PREPROCESSING if kind is ConvNeXtSliceDataset else PREPROCESSING,
            environment=dict(python=platform.python_version(),torch=str(torch.__version__),
                cuda_build=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
                gpu_name=torch.cuda.get_device_name(0),gpu_properties=str(torch.cuda.get_device_properties(0)),
                hostname=platform.node(),slurm_job_id=os.environ.get('SLURM_JOB_ID')),
            code_sha256={n:sha256(Path(__file__).with_name(n)) for n in
                ('run_r4_efficiency.py','modules.py','dataset.py','predict.py','train.py','dual_checkpoint.py','check_dataloader.py','run_n3.py','patient_evaluation.py','r4_prediction_audit.py')})
        source=source_path(root,entry['source'])
        training=json.loads((source/'training_summary.json').read_text())
        historical=historical_settings(training)
        if entry['model']=='ConvNeXt-F':
            name='run_r3_convnext.py'
            require(sha256(Path(__file__).with_name(name))==training['provenance']['code_sha256'][name],
                    'Historical ConvNeXt backend source changed')
        reference_slice_path=source/'reload/patient_nll/slice_predictions.csv'
        slice_hash=training['reload']['patient_nll']['output_sha256']['slice_predictions.csv']
        require(sha256(reference_slice_path)==slice_hash,'Historical slice reference changed')
        with reference_slice_path.open() as f:expected_slices=list(csv.DictReader(f))
        result['historical_settings']=historical
        torch.backends.cudnn.allow_tf32=historical['cudnn_allow_tf32']
        torch.backends.cuda.matmul.allow_tf32=historical['matmul_allow_tf32']
        torch.backends.cudnn.benchmark=historical['cudnn_benchmark']
        torch.backends.cudnn.deterministic=historical['cudnn_deterministic']
        replay_history(model,make_loader(ds,historical['batch_size'],42),'cuda',out,expected_slices,expected,result)
        require(state_sha256(model)==before,'Historical replay changed model')
        # Stage two uses common settings for every model, preserving historical references.
        torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        sync('cuda');gc.collect();torch.cuda.empty_cache()
        images=next(iter(loader))['image'].to('cuda')
        result['forward']=measure_forward(model,images,'cuda',p['forward_warmup'],p['forward_steps'],p['forward_repeats'])
        del images;gc.collect();torch.cuda.empty_cache()
        # Warm a complete cohort; audit comparisons happen outside timed regions.
        warm_slices,warm=end_to_end_pass(model,loader,'cuda')
        result['unified_vs_history']=record_predictions(out,'unified_reference',warm_slices,warm,expected_slices,expected)
        write_json(out/'efficiency_summary.json',result)
        differences=[];result['unified_repeats']=[]
        sync('cuda');reset_memory('cuda');seconds=[];patients=None
        for attempt in range(1,p['end_to_end_repeats']+1):
            elapsed,(slices,patients)=timed(lambda:end_to_end_pass(model,loader,'cuda'),lambda:sync('cuda'))
            seconds.append(elapsed)
            differences.append(check_repeat(out,attempt,slices,patients,warm_slices,warm,result))
        result['end_to_end']=summarize(seconds,len(ds),len(patients))
        result['end_to_end']['peak_memory']=peak_memory('cuda')
        require(state_sha256(model)==before and all(v.grad is None for v in model.parameters()),'Inference changed weights/gradients')
        counts={k:d.read_calls for k,d in datasets.items()}
        require(counts==dict(train=0,validation=4+5*2720,test=0),'Unexpected pixel reads')
        verify_files(root,lock['data_files']);verify_files(source_path(root,entry['source']),entry['files'])
        require(sha256(reference_slice_path)==slice_hash and sha256(lock_path)==FROZEN_LOCK_SHA256,'Reference changed during measurement')
        result.update(status='passed',parameters_unchanged=True,historical_predictions_match_R3=True,unified_repeats_match_reference=True,
                      max_probability_difference=max(differences),read_counts=counts)
    except Exception as exc:
        result.update(status='failed',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        raise
    finally:
        result['elapsed_seconds']=time.perf_counter()-started
        result['read_counts']={k:d.read_calls for k,d in datasets.items()}
        write_json(out/'efficiency_summary.json',result)
    return result


def consolidate(results):
    require(len(results)==9 and all(r['status']=='passed' for r in results),'Nine successful measurements required')
    require(len({(r['entry']['model'],r['entry']['seed']) for r in results})==9,'Duplicate results')
    first=results[0]
    for r in results:
        require(r.get('historical_predictions_match_R3') is True and r.get('unified_repeats_match_reference') is True,'Both stages must pass')
        require(r['protocol']==first['protocol'] and r['environment']==first['environment'] and
                r['lock_sha256']==first['lock_sha256'] and r['code_sha256']==first['code_sha256'],'Incomparable measurement conditions')
    rows=[]
    for r in results:
        row={k:r['entry'][k] for k in ('model','seed','epoch','checkpoint_sha256')}
        row['parameters']=r['parameter_count']
        for mode in ('forward','end_to_end'):
            s=r[mode]
            for key in ('median_ms_per_slice','median_slices_per_second','sample_sd_seconds'):
                row[mode+'_'+key]=s[key]
            for key,value in s['peak_memory'].items():row[mode+'_'+key]=value
        row['end_to_end_ms_per_patient_amortized']=r['end_to_end']['median_ms_per_patient_amortized']
        rows.append(row)
    aggregate=[]
    for model in ('CNN-A','CNN-B','ConvNeXt-F'):
        group=[r for r in rows if r['model']==model]
        for metric in rows[0]:
            if metric in ('model','seed','epoch','checkpoint_sha256'):continue
            values=[r[metric] for r in group]
            aggregate.append(dict(model=model,metric=metric,mean=statistics.mean(values),sample_sd=statistics.stdev(values)))
    return rows,aggregate


def run_all(root, lock_path, out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    require(sha256(lock_path)==FROZEN_LOCK_SHA256,'Frozen lock changed')
    lock=json.loads(Path(lock_path).read_text());validate_lock(lock)
    result=dict(status='running',scope='validation inference efficiency only',completed=[])
    start=time.perf_counter()
    try:
        verify_files(root,lock['data_files'])
        for entry in lock['runs']:verify_files(source_path(root,entry['source']),entry['files'])
        # Rotate order across seeds. Each child has its own CUDA allocator/context.
        order=[('CNN-A',42),('CNN-B',42),('ConvNeXt-F',42),('CNN-B',43),('ConvNeXt-F',43),('CNN-A',43),('ConvNeXt-F',44),('CNN-A',44),('CNN-B',44)]
        results=[]
        for model,seed in order:
            index=next(i for i,r in enumerate(lock['runs']) if (r['model'],r['seed'])==(model,seed))
            name=f'{model}_{seed}';folder=out/name
            with (out/(name+'.stdout.txt')).open('w') as stdout,(out/(name+'.stderr.txt')).open('w') as stderr:
                subprocess.run([sys.executable,str(Path(__file__).resolve()),'--data-root',str(root),
                    '--lock',str(lock_path),'--output-dir',str(folder),'--worker',str(index)],
                    stdout=stdout,stderr=stderr,check=True)
            results.append(json.loads((folder/'efficiency_summary.json').read_text()))
            result['completed'].append(name);write_json(out/'r4_summary.json',result)
        rows,aggregate=consolidate(results)
        write_csv(out/'nine_run_efficiency.csv',rows);write_csv(out/'model_efficiency_mean_sd.csv',aggregate)
        write_csv(out/'nine_run_prediction_drift.csv',[dict(model=r['entry']['model'],seed=r['entry']['seed'],**r['unified_vs_history']['patient_drift']) for r in results])
        result.update(status='passed',protocol=protocol(),lock_sha256=sha256(lock_path),
            measured_order=order,test_image_reads=sum(r['read_counts']['test'] for r in results),
            output_sha256={str(f.relative_to(out)):sha256(f) for f in out.rglob('*') if f.is_file() and f.name!='r4_summary.json'})
    except Exception as exc:
        result.update(status='failed',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        result['elapsed_seconds']=time.perf_counter()-start
        result['runner_gpu_hours']=result['elapsed_seconds']/3600
        result['resource_note']='Single GPU elapsed runner time; use sacct elapsed for whole-job allocation including tests.'
        write_json(out/'r4_summary.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--lock',type=Path,default=Path(__file__).with_name('r4_frozen_bests.json'))
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--worker',type=int,choices=range(9),help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.worker is None:run_all(args.data_root,args.lock,args.output_dir)
    else:run_worker(args.data_root,args.lock,args.worker,args.output_dir)

if __name__=='__main__':main()
