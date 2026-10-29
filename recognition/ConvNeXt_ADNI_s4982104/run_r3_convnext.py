"""R3d: fresh ConvNeXt-F 30 epochs, dual patient-NLL/slice-CE selection.

Historical R2 and N5 runners remain unchanged. No test pixels are read.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import traceback
import torch
import torchvision
import PIL
from torch import nn
from dataset import ConvNeXtSliceDataset, LABEL_MAP
from modules import build_model
from check_dataloader import require, sha256, make_loader, code_version
from train import run_epoch, class_weight_config, write_json
from predict import predict_pass, write_csv
from run_n3 import initialize_model, state_digest, sync, memory, validate_frozen_labels
from run_n4 import protocol_config as n4_config, validation_with_rows
from patient_evaluation import evaluate_patients
from dual_checkpoint import DualCheckpointSelector, load_selected


from run_n5 import initialize_model, protocol_config as n5_config
from run_r2 import configure_trainable, component_digest, rng_state, same_rng, save_predictions, protocol_config as r2_config
import math

def protocol_config(seed=42):
    config=n5_config(seed)
    config['checkpoint_selection']='primary patient NLL; auxiliary slice CE; earliest tie'
    return config


def check_epoch(actual,reference,patient=False):
    def compare(a,b,path):
        if isinstance(b,dict):
            for k,v in b.items():
                require(k in a,'Missing trajectory field: '+path+'.'+k)
                compare(a[k],v,path+'.'+k)
        elif isinstance(b,float):
            require(math.isfinite(a) and math.isfinite(b) and abs(a-b)<=1e-6+1e-5*abs(b),'Trajectory mismatch: '+path)
        else:require(a==b,'Trajectory mismatch: '+path)
    require(actual['epoch']==reference['epoch'],'Trajectory epoch mismatch')
    for part in ('train','validation'):
        compare(actual[part],reference[part],part)
    if patient:compare(actual['patient_metrics'],reference['patient_metrics'],'patient_metrics')
    return dict(passed=True,epoch=actual['epoch'],atol=1e-6,rtol=1e-5)


def canonical_config(config):
    c=deepcopy(config)
    for k in ('epochs','checkpoint_selection','head_definition','training_mode','full_finetuning','split_seed'):c.pop(k,None)
    return c


def read_reference(folder,kind,seed,manifest,summary):
    folder=Path(folder);t=json.loads((folder/'training_summary.json').read_text())
    require(kind in ('N5','R2-F'),'Unknown reference kind')
    count=30 if kind=='N5' else 5
    require(t.get('status')=='passed','Reference did not pass')
    require(kind!='R2-F' or (seed==42 and t.get('arm')=='F'),'Expected seed42 R2-F')
    require(t['config']==json.loads((folder/'config.json').read_text()),'Reference config file mismatch')
    expected=n5_config(seed) if kind=='N5' else r2_config()
    require(t['config']['epochs']==count,'Reference epoch configuration mismatch')
    require(all(t['config'].get(k)==v for k,v in expected.items()),'Reference protocol mismatch')
    history=[json.loads(line) for line in (folder/'epochs.jsonl').read_text().splitlines()]
    require(history==t['history'] and [h['epoch'] for h in history]==list(range(1,count+1)),'Reference history mismatch')
    provenance=t.get('provenance',t)
    inputs=provenance['input_sha256'];wanted=dict(manifest=sha256(manifest),summary=sha256(summary)) if kind=='R2-F' else dict(split_samples=sha256(manifest),split_summary=sha256(summary))
    require(inputs==wanted,'Reference inputs mismatch')
    for n,h in provenance['code_sha256'].items():
        require(Path(n).name==n and sha256(Path(__file__).with_name(n))==h,'Reference source mismatch: '+n)
    require(bool(provenance['code_sha256']),'Missing reference source hashes')
    frozen={str(folder/n):sha256(folder/n) for n in ('training_summary.json','config.json','epochs.jsonl')}
    return t,frozen


def check_start(actual,reference):
    for key in ('initial_state_sha256','versions','gpu_name'):
        require(actual[key]==reference[key],'Reference start mismatch: '+key)
    a=json.loads(json.dumps(actual['optimizer_actual']));b=reference['optimizer_actual']
    require(isinstance(b,dict) and isinstance(a,dict),'Invalid optimizer record')
    for key,value in b.items():
        require(a.get('lr' if key=='learning_rate' else key)==value,'Reference optimizer mismatch: '+key)
    require(canonical_config(actual['config'])==canonical_config(reference['config']),'Reference runtime config mismatch')
    for key in ('trainable_parameters','trainable_count'):
        if key in reference:require(actual.get(key)==reference[key],'Reference trainable set mismatch: '+key)
    if 'trainable_parameter_count' in reference:
        require(actual['trainable_count']==reference['trainable_parameter_count']==reference['parameter_count'],'Reference full finetuning count mismatch')

def run_formal(manifest,summary,weights,weights_summary,out,device='cuda',seed=42,
               expected_lengths={'train':8160,'validation':2720,'test':2720},expected_slices=20,
               n5_reference=None,r2_reference=None):
    protocol_config(seed);arm='F';out=Path(out)
    out.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    s=dict(status='running',arm=arm,history=[],reference_checks={'N5':[],'R2-F':[]});datasets={};reload_ds={};references={};frozen={}
    try:
        require(device in ('cpu','cuda') and (device!='cuda' or torch.cuda.is_available()),'Invalid/unavailable device')
        require(expected_lengths is None or (n5_reference is not None and (seed!=42 or r2_reference is not None)),'Historical references required for real cohort')
        for kind,path in [('N5',n5_reference),('R2-F',r2_reference)]:
            if path is not None:
                references[kind],hashes=read_reference(path,kind,seed,manifest,summary);frozen.update(hashes)
        s['reference_sha256']=frozen
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        torch.manual_seed(seed);torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        if device=='cuda':torch.cuda.manual_seed_all(seed);torch.cuda.reset_peak_memory_stats()
        model,weight_hash=initialize_model(weights,weights_summary,seed)
        initial=state_digest(model);trainable=configure_trainable(model,arm)
        datasets={split:ConvNeXtSliceDataset(manifest,split,summary) for split in ('train','validation','test')}
        lengths={k:len(v) for k,v in datasets.items()}
        require(expected_lengths is None or lengths==expected_lengths,'Cohort lengths mismatch')
        for split,ds in datasets.items():
            if expected_lengths is not None:validate_frozen_labels(ds.records,split)
            evaluate_patients([dict(r,label=LABEL_MAP[r['label']],p_ad=0.) for r in ds.records],ds.records,expected_slices)
        config=protocol_config(seed);config.update(device=device,weights_sha256=weight_hash,
            class_weight_details=class_weight_config(datasets['train'].records,'balanced'))
        provenance=dict(input_sha256=dict(manifest=sha256(manifest),summary=sha256(summary)),weights_sha256=weight_hash,
            code_sha256={n:sha256(Path(__file__).with_name(n)) for n in ('run_r3_convnext.py','run_n5.py','run_r2.py','patient_evaluation.py','dual_checkpoint.py','run_n4.py','run_n3.py','train.py','predict.py','dataset.py','modules.py','check_dataloader.py','check_convnext.py')},
            arm=arm,trainable_parameters=trainable)
        s.update(config=config,provenance=provenance,code_version=code_version(),initial_state_sha256=initial,
            initial_head_sha256=component_digest(model,True),initial_backbone_sha256=component_digest(model,False),
            trainable_parameters=trainable,trainable_count=sum(p.numel() for p in model.parameters() if p.requires_grad),
            versions=dict(python=platform.python_version(),torch=str(torch.__version__),torchvision=str(torchvision.__version__),cuda_build=torch.version.cuda,pillow=PIL.__version__),
            gpu_name=torch.cuda.get_device_name() if device=='cuda' else None)
        write_json(out/'config.json',config)
        model=model.to(device)
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=3e-5,weight_decay=.01)
        s['optimizer_actual']={k:v for k,v in optimizer.param_groups[0].items() if k!='params'}
        for reference in references.values():check_start(s,reference)
        selector=DualCheckpointSelector(out/'selection',config,provenance,expected_slices)
        loaders={k:make_loader(datasets[k],4,seed) for k in ('train','validation')}
        cw=torch.tensor(config['class_weight_details']['weights'],dtype=torch.float32,device=device)
        for epoch in range(1,31):
            sync(device);t0=time.perf_counter();head_before=component_digest(model,True)
            tm=run_epoch(model,loaders['train'],device,optimizer,cw)
            no_grad=all(p.grad is None for p in model.parameters() if not p.requires_grad)
            backbone_same=component_digest(model,False)==s['initial_backbone_sha256']
            head_updated=component_digest(model,True)!=head_before
            require(head_updated,'Head did not update')
            if arm=='H':require(no_grad and backbone_same,'Frozen backbone changed or acquired gradients')
            optimizer.zero_grad(set_to_none=True);before=state_digest(model)
            vm,rows=validation_with_rows(model,loaders['validation'],device)
            require(before==state_digest(model) and all(p.grad is None for p in model.parameters()),'Validation modified model')
            rng=rng_state();counts={k:d.read_calls for k,d in datasets.items()}
            evaluation=selector.consider(epoch,model,rows,datasets['validation'].records,vm['loss'])
            diagnostic=save_predictions(out/'diagnostics'/f'epoch_{epoch}',rows,evaluation)
            require(same_rng(rng),'Diagnostics changed RNG')
            require(counts=={k:d.read_calls for k,d in datasets.items()},'Diagnostics added reads')
            sync(device)
            h=dict(epoch=epoch,train=tm,validation=vm,patient_metrics=evaluation['metrics'],diagnostics=diagnostic,
                backbone_no_grad=no_grad,backbone_unchanged=backbone_same,
                head_updated=head_updated,validation_state_unchanged=True,diagnostic_additional_reads=0,diagnostic_rng_unchanged=True,seconds=time.perf_counter()-t0)
            s['history'].append(h)
            with (out/'epochs.jsonl').open('a') as f:f.write(json.dumps(h,allow_nan=False)+'\n')
            with (out/'training.log').open('a') as f:f.write(f"epoch={epoch} train={tm['loss']} validation={vm['loss']} patient_nll={evaluation['metrics']['nll']}\n")
            write_json(out/'training_summary.json',s)
            for kind,reference in references.items():
                if epoch<=len(reference['history']):
                    s['reference_checks'][kind].append(check_epoch(h,reference['history'][epoch-1],patient=kind=='R2-F'))
        s['training_peak_memory']=memory(device)
        s['training_read_counts']={k:d.read_calls for k,d in datasets.items()}
        require(s['training_read_counts']==dict(train=30*lengths['train'],validation=30*lengths['validation'],test=0),'Training reads mismatch')
        del model,optimizer
        if device=='cuda':torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        require(sha256(manifest)==provenance['input_sha256']['manifest'] and sha256(summary)==provenance['input_sha256']['summary'],'Input changed')
        reload_ds={k:ConvNeXtSliceDataset(manifest,k,summary) for k in datasets}
        s['reload']={}
        for rule in ('patient_nll','slice_ce'):
            passes=[];times=[]
            for attempt in range(2):
                restored,cp=load_selected(out/'selection',rule,lambda:build_model('convnext_tiny',num_classes=2),config,provenance)
                restored=restored.to(device);before=state_digest(restored);sync(device);t0=time.perf_counter()
                rows=predict_pass(restored,make_loader(reload_ds['validation'],4,seed),device)
                sync(device);times.append(time.perf_counter()-t0)
                require(state_digest(restored)==before,'Reload prediction changed state')
                passes.append(rows);del restored
            original=json.loads((out/'diagnostics'/f"epoch_{cp['epoch']}"/'summary.json').read_text())
            import csv
            with (out/'diagnostics'/f"epoch_{cp['epoch']}"/'slice_predictions.csv').open() as f:expected=list(csv.DictReader(f))
            for a,b,c in zip(passes[0],passes[1],expected):
                for key in ('file_path','patient_id','image_id','label','prediction'):
                    require(str(a[key])==str(b[key])==str(c[key]),'Reload identity/class mismatch')
                require(abs(a['p_ad']-b['p_ad'])<=1e-6+1e-5*abs(b['p_ad']),'Reload repeat mismatch')
                require(abs(a['p_ad']-float(c['p_ad']))<=1e-6+1e-5*abs(float(c['p_ad'])),'Selected epoch prediction mismatch')
            for rows in passes:evaluate_patients(rows,reload_ds['validation'].records,expected_slices)
            require(len(expected)==len(passes[0]),'Selected epoch size mismatch')
            evaluation=evaluate_patients(passes[0],reload_ds['validation'].records,expected_slices)
            pr=save_predictions(out/'reload'/rule,passes[0],evaluation)
            require(abs(evaluation['metrics']['nll']-original['metrics']['nll'])<=1e-6,'Selected epoch NLL mismatch')
            pr.update(epoch=cp['epoch'],checkpoint_sha256=sha256(out/'selection'/f'best_{rule}.pt'),repeat_equal=True,
                selected_epoch_equal=True,read_count=2*len(passes[0]),full_pass_seconds=times)
            write_json(out/'reload'/rule/'summary.json',pr);s['reload'][rule]=pr
        s['reload_read_counts']={k:d.read_calls for k,d in reload_ds.items()}
        require(s['reload_read_counts']==dict(train=0,validation=4*lengths['validation'],test=0),'Reload reads mismatch')
        require(all(sha256(Path(n))==h for n,h in frozen.items()),'Historical reference changed')
        s['historical_evidence_unchanged']=True
        s['total_read_counts']={k:s['training_read_counts'][k]+s['reload_read_counts'][k] for k in datasets}
        s.update(status='passed',reload_peak_memory=memory(device),selection=deepcopy(selector.summary))
    except Exception as exc:
        s.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        (out/'error.log').write_text(traceback.format_exc());raise
    finally:
        s.update(elapsed_seconds=time.perf_counter()-started,
            actual_training_read_counts={k:d.read_calls for k,d in datasets.items()},actual_reload_read_counts={k:d.read_calls for k,d in reload_ds.items()})
        write_json(out/'training_summary.json',s)
    return s


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('manifest','summary','weights','weights-summary','output-dir','n5-reference'):
        p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--r2-reference',type=Path)
    p.add_argument('--seed',type=int,choices=(42,43,44),default=42)
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda');a=p.parse_args()
    run_formal(a.manifest,a.summary,a.weights,a.weights_summary,a.output_dir,a.device,a.seed,
               n5_reference=a.n5_reference,r2_reference=a.r2_reference)
