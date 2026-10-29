"""CNN-A42 single-factor numerical diagnosis; never alters R4 or its lock.

Mismatch is evidence, not an execution error. Each condition runs twice in a
fresh CUDA process. Structural/provenance errors stop the diagnostic.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import traceback
import torch
from run_r4_efficiency import (require,sha256,verify_files,source_path,validate_lock,
    load_frozen,check_cohort,CountedDataset,make_loader,predict_pass,aggregate_patients,
    write_json,write_csv,state_sha256)

CHECKPOINT='2bc73f2cd6e0dd19dd303e67bd12754769141c6da0e42bd199e076351dbef106'
LOCK='d781fb6bec3f9e378aaf6ddfa08229aadb2fc3c7361159f01e87b3be0106ebb4'


def conditions():
    return [dict(name=n,batch_size=b,tf32=t,mode=m) for n,b,t,m in (
        ('A_original',16,True,'no_grad'),('B_tf32_off',16,False,'no_grad'),
        ('C_batch4',4,True,'no_grad'),('D_both',4,False,'no_grad'),
        ('E_r4_mode',4,False,'inference_mode'))]


def select_entry(lock):
    validate_lock(lock)
    entry=next(r for r in lock['runs'] if r['model']=='CNN-A' and r['seed']==42)
    require(entry['epoch']==28 and entry['checkpoint_sha256']==CHECKPOINT,'CNN-A42 frozen identity changed')
    return entry


def read_csv(path):
    with Path(path).open() as f:return list(csv.DictReader(f))


def differences(actual,reference,unit):
    require(unit in ('slice','patient'),'Invalid comparison unit')
    key='file_path' if unit=='slice' else 'patient_id'
    require(actual and reference,'Empty predictions')
    a={r[key]:r for r in actual};b={r[key]:r for r in reference}
    require(len(a)==len(actual)==len(b)==len(reference) and a.keys()==b.keys(),'Missing/duplicate/different identities')
    fields=['patient_id','image_id','label']+(['slice_count'] if unit=='patient' else [])
    rows=[]
    for identity,row in a.items():
        ref=b[identity]
        require(all(str(row[k])==str(ref[k]) for k in fields),'Metadata mismatch')
        p,q=float(row['p_ad']),float(ref['p_ad'])
        require(all(math.isfinite(v) and 0<=v<=1 for v in (p,q)),'Invalid probability')
        require(int(row['prediction'])==int(p>=.5) and int(ref['prediction'])==int(q>=.5),'Prediction/threshold inconsistency')
        delta=p-q
        rows.append(dict(identity=identity,patient_id=row['patient_id'],reference_probability=q,
            actual_probability=p,signed_difference=delta,absolute_difference=abs(delta),
            r4_probability_close=math.isclose(p,q,rel_tol=1e-5,abs_tol=1e-6),
            r3_probability_close=abs(delta)<=1e-6+1e-5*abs(q),
            reference_prediction=int(ref['prediction']),actual_prediction=int(row['prediction']),
            class_changed=int(row['prediction'])!=int(ref['prediction'])))
    return rows,dict(count=len(rows),r4_exceed_count=sum(not r['r4_probability_close'] for r in rows),
        r3_exceed_count=sum(not r['r3_probability_close'] for r in rows),
        class_change_count=sum(r['class_changed'] for r in rows),
        max_absolute_difference=max(r['absolute_difference'] for r in rows),
        mean_absolute_difference=statistics.mean(r['absolute_difference'] for r in rows))


def save_comparison(out,name,actual,reference,unit):
    # Write actual values even if a subsequent comparison finds structural errors.
    write_csv(Path(out)/(name+'_actual.csv'),actual)
    rows,summary=differences(actual,reference,unit)
    write_csv(Path(out)/(name+'_differences.csv'),rows)
    return summary


def worker(root,lock_path,index,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    result=dict(status='running');datasets={}
    try:
        require(sha256(lock_path)==LOCK,'Original R4 lock hash changed')
        lock=json.loads(Path(lock_path).read_text());entry=select_entry(lock);condition=conditions()[index]
        require(torch.cuda.is_available() and torch.cuda.device_count()==1,'One CUDA GPU required')
        require('A100' in torch.cuda.get_device_name(),'A100 required')
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        torch.set_num_threads(2);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=condition['tf32']
        verify_files(root,lock['data_files']);model,expected=load_frozen(root,entry,lock)
        source=source_path(root,entry['source']);training=json.loads((source/'training_summary.json').read_text())
        slice_path=source/'reload/patient_nll/slice_predictions.csv'
        expected_slice_hash=training['reload']['patient_nll']['output_sha256']['slice_predictions.csv']
        require(sha256(slice_path)==expected_slice_hash,'R3 slice reference hash mismatch')
        slice_reference=read_csv(slice_path)
        manifest=source_path(root,'split_seed42_60_20_20/split_samples.csv')
        datasets={s:CountedDataset(manifest,s,manifest.with_name('split_summary.json')) for s in ('train','validation','test')}
        check_cohort(datasets);before=state_sha256(model);model=model.to('cuda').eval()
        result.update(condition=condition,entry=entry,lock_sha256=sha256(lock_path),
            reference_backend_flags=training['backend_flags'],reference_batch_size=training['config']['batch_size'],
            actual_backend_flags=dict(cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_benchmark=torch.backends.cudnn.benchmark,
                cudnn_deterministic=torch.backends.cudnn.deterministic),
            environment=dict(torch=str(torch.__version__),cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
                gpu=str(torch.cuda.get_device_properties(0)),hostname=platform.node()),
            code_sha256={n:sha256(Path(__file__).with_name(n)) for n in ('run_r4_diagnostic.py','run_r4_efficiency.py','modules.py','dataset.py','predict.py','train.py','run_n3.py','check_dataloader.py','dual_checkpoint.py','patient_evaluation.py')},
            tolerance=dict(atol=1e-6,rtol=1e-5,r4='math.isclose',r3='abs(p-q)<=atol+rtol*abs(q)'),passes=[])
        previous=None
        for attempt in (1,2):
            loader=make_loader(datasets['validation'],condition['batch_size'],42)
            context=torch.inference_mode() if condition['mode']=='inference_mode' else torch.no_grad()
            with context:
                slices=predict_pass(model,loader,'cuda');patients=aggregate_patients(slices)
            # Save complete values before any numerical comparison.
            write_csv(out/f'pass{attempt}_slices_actual.csv',slices)
            write_csv(out/f'pass{attempt}_patients_actual.csv',patients)
            stats=dict(attempt=attempt,
                slices=save_comparison(out,f'pass{attempt}_slices',slices,slice_reference,'slice'),
                patients=save_comparison(out,f'pass{attempt}_patients',patients,expected,'patient'))
            if previous is not None:
                stats['repeat_patients']=save_comparison(out,'repeat_patients',patients,previous[1],'patient')
                stats['repeat_slices']=save_comparison(out,'repeat_slices',slices,previous[0],'slice')
            previous=(slices,patients);result['passes'].append(stats)
            write_json(out/'diagnostic_summary.json',result)
        require(before==state_sha256(model) and all(p.grad is None for p in model.parameters()),'Weights or gradients changed')
        require({k:d.read_calls for k,d in datasets.items()}==dict(train=0,validation=5440,test=0),'Unexpected reads')
        verify_files(root,lock['data_files']);verify_files(source,entry['files'])
        require(sha256(lock_path)==LOCK and sha256(slice_path)==expected_slice_hash,'Reference changed during diagnosis')
        result.update(status='completed',parameters_unchanged=True,
            interpretation='Completed means evidence collected, not prediction agreement or R4 acceptance.')
    except Exception as exc:
        result.update(status='failed',error=str(exc),traceback=traceback.format_exc());raise
    finally:
        result.update(read_counts={k:d.read_calls for k,d in datasets.items()},elapsed_seconds=time.perf_counter()-start)
        write_json(out/'diagnostic_summary.json',result)
    return result


def run_all(root,lock_path,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    result=dict(status='running',completed=[])
    try:
        require(sha256(lock_path)==LOCK,'Original R4 lock hash changed')
        lock=json.loads(Path(lock_path).read_text());entry=select_entry(lock)
        verify_files(root,lock['data_files']);verify_files(source_path(root,entry['source']),entry['files'])
        summaries=[]
        for index,c in enumerate(conditions()):
            name=c['name']
            with (out/(name+'.out')).open('w') as stdout,(out/(name+'.err')).open('w') as stderr:
                subprocess.run([sys.executable,str(Path(__file__).resolve()),'--data-root',str(root),
                    '--lock',str(lock_path),'--output-dir',str(out/name),'--worker',str(index)],
                    stdout=stdout,stderr=stderr,check=True)
            s=json.loads((out/name/'diagnostic_summary.json').read_text())
            require(s['status']=='completed','Incomplete worker')
            if summaries:
                require(s['environment']==summaries[0]['environment'] and s['code_sha256']==summaries[0]['code_sha256'],'Conditions differ in environment/source')
            summaries.append(s);result['completed'].append(name)
        comparisons=[]
        for left,right in [('A_original','B_tf32_off'),('A_original','C_batch4'),('C_batch4','D_both'),('B_tf32_off','D_both'),('D_both','E_r4_mode')]:
            for unit in ('patients','slices'):
                a=read_csv(out/right/f'pass1_{unit}_actual.csv');b=read_csv(out/left/f'pass1_{unit}_actual.csv')
                rows,stats=differences(a,b,'patient' if unit=='patients' else 'slice')
                write_csv(out/f'{left}_to_{right}_{unit}_differences.csv',rows)
                comparisons.append(dict(reference=left,actual=right,unit=unit,**stats))
        write_csv(out/'factor_comparisons.csv',comparisons)
        baseline=summaries[0]['passes'][0]['patients']
        result.update(status='completed',baseline_r4_agrees=baseline['r4_exceed_count']==baseline['class_change_count']==0,
            conditions=summaries,test_image_reads=sum(s['read_counts']['test'] for s in summaries),
            interpretation='Evidence collection completed; numerical mismatches remain reported, no tolerances changed.',
            output_sha256={str(p.relative_to(out)):sha256(p) for p in out.rglob('*') if p.is_file() and p.name!='diagnosis_summary.json'})
    except Exception as exc:result.update(status='failed',error=str(exc));raise
    finally:
        result['elapsed_seconds']=time.perf_counter()-start;write_json(out/'diagnosis_summary.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--lock',type=Path,default=Path(__file__).with_name('r4_frozen_bests.json'))
    p.add_argument('--worker',type=int,choices=range(5),help=argparse.SUPPRESS)
    a=p.parse_args()
    if a.worker is None:run_all(a.data_root,a.lock,a.output_dir)
    else:worker(a.data_root,a.lock,a.worker,a.output_dir)
if __name__=='__main__':main()
