"""N5: fixed 30-epoch ConvNeXt formal run (one seed per invocation), then independent validation prediction.

Reuses train.run_epoch and predict's patient aggregation/metrics; CNN entry
points and historical evidence are unchanged. No test image pixels are read.
"""
import argparse
from copy import deepcopy
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import traceback
import PIL
import torch
import torchvision
from torch import nn
from dataset import ConvNeXtSliceDataset, CONVNEXT_PREPROCESSING, LABEL_MAP
from modules import build_model
from check_convnext import verify_weights, WEIGHT_NAME
from check_dataloader import require, sha256, code_version, make_loader
from train import run_epoch, class_weight_config, classification_metrics, write_json
from predict import predict_pass, aggregate_patients, write_csv


import math
import signal
from run_n3 import state_digest, sync, memory, verify_input_hashes, validate_frozen_labels, validate_rows
from run_n4 import protocol_config as n4_protocol_config, validation_with_rows, save_diagnostics, flat_metrics

# Predeclared before any formal run: integer counts/order must match exactly;
# floating metrics use |actual-reference| <= 1e-6 + 1e-5*|reference|.
PREFIX_ATOL=1e-6
PREFIX_RTOL=1e-5
ALLOWED_SEEDS=(42,43,44)


def protocol_config(seed=42):
    require(type(seed) is int and seed in ALLOWED_SEEDS,'Unsupported training seed')
    config=n4_protocol_config()
    config.update(epochs=30,seed=seed,split_seed=42)
    return config


def initialize_model(weights,weights_summary,seed=42):
    protocol_config(seed)
    digest=verify_weights(weights,weights_summary)
    model=build_model('convnext_tiny',num_classes=1000)
    model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True)
    # Match N3/N4 ordering exactly for seed42: reset AFTER loading the backbone.
    # torch.manual_seed seeds CPU and CUDA; the runner also seeds CUDA explicitly.
    torch.manual_seed(seed)
    model.classifier[-1]=nn.Linear(768,2)
    nn.init.trunc_normal_(model.classifier[-1].weight,std=0.02)
    nn.init.zeros_(model.classifier[-1].bias)
    return model,digest


def is_improvement(loss,best_loss):
    require(math.isfinite(loss),'Nonfinite selection loss')
    return loss < best_loss


def read_reference(folder,manifest,summary):
    """Read-only N4 evidence preflight; no MRI or reference model forward."""
    folder=Path(folder)
    names=['n4_summary.json','training/config.json','training/training_summary.json',
        'training/epochs.jsonl','prediction/config.json','prediction/prediction_summary.json',
        'prediction/slice_predictions.csv','prediction/patient_predictions.csv']
    names += [f'diagnostics/epoch_{e}/{n}' for e in (1,2,3)
              for n in ('summary.json','slice_predictions.csv','patient_predictions.csv')]
    hashes={n:sha256(folder/n) for n in names}
    load=lambda n:json.loads((folder/n).read_text())
    overall=load('n4_summary.json');t=load('training/training_summary.json');pr=load('prediction/prediction_summary.json')
    require(all(x['status']=='passed' for x in (overall,t,pr)),'N4 reference not passed')
    require(overall['training_summary_sha256']==hashes['training/training_summary.json']==pr['training_summary_sha256'],'N4 training summary chain mismatch')
    require(overall['prediction_summary_sha256']==hashes['prediction/prediction_summary.json'],'N4 prediction summary chain mismatch')
    require(t['config']==load('training/config.json')==pr['config']==load('prediction/config.json'),'N4 config mismatch')
    require(all(t['config'].get(k)==v for k,v in n4_protocol_config().items()),'N4 protocol mismatch')
    for name,digest in t['code_sha256'].items():
        require(sha256(Path(__file__).with_name(name))==digest,'N4 source mismatch: '+name)
    require(t['code_sha256']==pr['code_sha256'],'N4 prediction source mismatch')
    verify_input_hashes(manifest,summary,t['input_sha256'])
    require(pr['input_sha256']==t['input_sha256'],'N4 input mismatch')
    history=[json.loads(line) for line in (folder/'training/epochs.jsonl').read_text().splitlines()]
    require(history==t['history'] and [h['epoch'] for h in history]==[1,2,3],'N4 history mismatch')
    best=min(history,key=lambda h:h['validation']['loss'])
    require(t['best_epoch']==best['epoch'] and t['best_validation_loss']==best['validation']['loss'],'N4 best selection mismatch')
    require(t['checkpoint_sha256']==pr['checkpoint_sha256'],'N4 checkpoint chain mismatch')
    for name,digest in pr['output_sha256'].items():
        require(sha256(folder/'prediction'/name)==digest,'N4 prediction CSV hash mismatch')
    for h in history:
        prefix=f"diagnostics/epoch_{h['epoch']}/"
        diagnostic=load(prefix+'summary.json')
        require(diagnostic==h['diagnostics'],'N4 diagnostic summary mismatch')
        for name,digest in diagnostic['output_sha256'].items():
            require(hashes[prefix+name]==digest,'N4 diagnostic CSV mismatch')
    if (folder/'training/best.pt').exists():
        require(sha256(folder/'training/best.pt')==t['checkpoint_sha256'],'N4 weight hash mismatch')
        hashes['training/best.pt']=t['checkpoint_sha256']
    return dict(folder=folder,training=t,prediction=pr,provenance=dict(path=str(folder),file_sha256=hashes,code_sha256=t['code_sha256']))


def check_start(reference,actual,seed):
    expected=dict(reference['config']);expected.update(epochs=30,seed=seed,split_seed=42)
    require(actual['config']==expected,'N4 configuration mismatch beyond epochs/training seed')
    for key in ('versions','gpu_name','input_sha256','parameter_count','trainable_parameter_count'):
        require(actual[key]==reference[key],'N4 environment/input mismatch: '+key)
    if seed==42:
        require(actual['initial_state_sha256']==reference['initial_state_sha256'],'N4 prefix initial state mismatch')


def check_prefix_epoch(reference,row):
    old=reference['history'][row['epoch']-1]
    def compare(a,b,path):
        if isinstance(b,dict):
            require(set(a)==set(b),'N4 prefix keys mismatch: '+path)
            for key in b:compare(a[key],b[key],path+'.'+key)
        elif isinstance(b,list):
            require(len(a)==len(b),'N4 prefix length mismatch: '+path)
            for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+f'[{i}]')
        elif isinstance(b,float):
            require(isinstance(a,(int,float)) and math.isclose(a,b,rel_tol=PREFIX_RTOL,abs_tol=PREFIX_ATOL),'N4 prefix numerical mismatch: '+path)
        else:require(a==b,'N4 prefix exact mismatch: '+path)
    for split in ('train','validation'):compare(row[split],old[split],split)
    require(row['saved_best']==old['saved_best'],'N4 prefix best selection mismatch')
    return dict(epoch=row['epoch'],status='passed',atol=PREFIX_ATOL,rtol=PREFIX_RTOL,
        train_order_sha256=row['train']['order_sha256'],validation_order_sha256=row['validation']['order_sha256'])


def finish_run(out,reference,training,prediction,result,seed):
    require(len(training['history'])==30,'Incomplete formal run')
    check_start(reference['training'],training,seed)
    best=min(training['history'],key=lambda h:h['validation']['loss'])
    require(best['epoch']==training['best_epoch'],'Best epoch recomputation failed')
    for name,digest in reference['provenance']['file_sha256'].items():
        require(sha256(reference['folder']/name)==digest,'N4 reference changed during formal run')
    require(prediction['slice_metrics']=={k:best['validation'][k] for k in prediction['slice_metrics']},'Best validation/reload metric mismatch')
    write_csv(out/'metrics.csv',[dict(unit=unit,seed=seed,best_epoch=training['best_epoch'],**flat_metrics(prediction[unit+'_metrics'])) for unit in ('slice','patient')])
    result.update(metrics_sha256=sha256(out/'metrics.csv'),reference_unchanged=True,
        patient_metrics=prediction['patient_metrics'],slice_metrics=prediction['slice_metrics'],
        training_peak_cuda_memory=training['peak_cuda_memory'],prediction_peak_cuda_memory=prediction['peak_cuda_memory'],
        training_seconds=training['elapsed_seconds'],prediction_seconds=prediction['elapsed_seconds'],
        budget=dict(total_project_gpu_hours=8,job_limit_hours=2,submitted_seed_only=seed,
                    accounting='Use Slurm elapsed and allocated GPUs; queued time is excluded'),
        validation_only=True)


def load_checkpoint(path,expected_config,device):
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    require(checkpoint.get('config')==expected_config,'Checkpoint configuration mismatch')
    for key,value in protocol_config(expected_config.get('seed')).items():
        require(expected_config.get(key)==value,f'Unsupported model/preprocessing protocol: {key}')
    devices=[torch.cuda.current_device()] if device=='cuda' else []
    with torch.random.fork_rng(devices=devices):
        model=build_model(expected_config['model'],num_classes=expected_config['num_classes'])
    model.load_state_dict(checkpoint['model_state_dict'],strict=True)
    return model.to(device).eval(),checkpoint


def run_formal(manifest,summary,weights,weights_summary,output_dir,reference_dir,device='cuda',seed=42,
              expected_lengths={'train':8160,'validation':2720,'test':2720},expected_slices=20,expected_patients=136):
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=False)
    train_dir=out/'training';train_dir.mkdir();pred_dir=out/'prediction';pred_dir.mkdir()
    started=time.perf_counter();result=dict(status='running',generated_at=datetime.now(timezone.utc).isoformat())
    training=dict(status='running',history=[]);prediction=dict(status='not_started');datasets={};pdatasets={}
    def log(folder,message):
        print(message,flush=True)
        name='training.log' if folder==train_dir else 'prediction.log'
        with (folder/name).open('a') as f:f.write(message+'\n')
    try:
        protocol_config(seed)
        reference = read_reference(reference_dir, manifest, summary)
        result['reference'] = reference['provenance']
        result['n4_prefix_check']=dict(status='pending' if seed==42 else 'not_applicable',reason='Only training seed42 shares the N4 stochastic trajectory',epochs=[],atol=PREFIX_ATOL,rtol=PREFIX_RTOL)
        require(device in ('cpu','cuda'),'Invalid device')
        require(device!='cuda' or torch.cuda.is_available(),'CUDA unavailable')
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        torch.manual_seed(seed);torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        if device=='cuda':torch.cuda.manual_seed_all(seed);torch.cuda.reset_peak_memory_stats()
        config=protocol_config(seed);config['device']=device
        model,weight_digest=initialize_model(weights,weights_summary,seed)
        config['weights_sha256']=weight_digest
        datasets={s:ConvNeXtSliceDataset(manifest,s,summary) for s in ('train','validation','test')}
        lengths={s:len(d) for s,d in datasets.items()}
        require(expected_lengths is None or lengths==expected_lengths,'Frozen cohort lengths mismatch')
        for split,ds in datasets.items():
            if expected_lengths is not None:validate_frozen_labels(ds.records,split)
            metadata=[dict(r,label=LABEL_MAP[r['label']],p_ad=0.) for r in ds.records]
            patients=aggregate_patients(metadata,expected_slices)
            if split=='validation':require(len(patients)==expected_patients,'Patient count mismatch')
        config['class_weight_details']=class_weight_config(datasets['train'].records,'balanced')
        write_json(train_dir/'config.json',config)
        versions=dict(python=platform.python_version(),torch=str(torch.__version__),torchvision=str(torchvision.__version__),pillow=PIL.__version__,cuda_build=torch.version.cuda)
        code={n:sha256(Path(__file__).with_name(n)) for n in ['run_n5.py','run_n4.py','run_n3.py','train.py','predict.py','modules.py','dataset.py','check_dataloader.py','check_convnext.py']}
        inputs=dict(split_samples=sha256(manifest),split_summary=sha256(summary))
        training.update(config=config,versions=versions,code_version=code_version(),code_sha256=code,input_sha256=inputs,
            gpu_name=torch.cuda.get_device_name() if device=='cuda' else None,dataset_lengths=lengths,
            initial_state_sha256=state_digest(model),parameter_count=sum(p.numel() for p in model.parameters()),
            trainable_parameter_count=sum(p.numel() for p in model.parameters() if p.requires_grad))
        check_start(reference['training'],training,seed)
        model=model.to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
        training['optimizer_actual'] = dict(learning_rate=optimizer.param_groups[0]['lr'],weight_decay=optimizer.param_groups[0]['weight_decay'],betas=list(optimizer.param_groups[0]['betas']),eps=optimizer.param_groups[0]['eps'])
        cw=torch.tensor(config['class_weight_details']['weights'],dtype=torch.float32,device=device)
        loaders={s:make_loader(datasets[s],config['batch_size'],seed) for s in ('train','validation')}
        probe=torch.linspace(-1,1,3*224*224,device=device).reshape(1,3,224,224)
        best_loss=float('inf');best_epoch=None
        for epoch in range(1,config['epochs']+1):
            sync(device);epoch_start=time.perf_counter()
            before=state_digest(model)
            tm=run_epoch(model,loaders['train'],device,optimizer,cw)
            changed=before!=state_digest(model);require(changed,'No parameter update')
            optimizer.zero_grad(set_to_none=True)
            before=state_digest(model);vm,epoch_rows=validation_with_rows(model,loaders['validation'],device)
            validate_rows(epoch_rows,datasets['validation'].records)
            diagnostics=save_diagnostics(out,epoch,epoch_rows,vm,expected_slices,expected_patients)
            unchanged=before==state_digest(model) and all(p.grad is None for p in model.parameters())
            require(unchanged,'Validation changed state or gradients')
            sync(device);seconds=time.perf_counter()-epoch_start
            improved=is_improvement(vm['loss'],best_loss)
            if improved:
                best_loss=vm['loss'];best_epoch=epoch
                with torch.no_grad():probe_logits=model(probe).cpu()
                torch.save(dict(config=config,model_state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},epoch=epoch,validation_loss=best_loss,probe_logits=probe_logits),train_dir/'best.pt.tmp')
                (train_dir/'best.pt.tmp').replace(train_dir/'best.pt')
            row=dict(epoch=epoch,train=tm,validation=vm,diagnostics=diagnostics,training_parameters_changed=changed,validation_parameters_unchanged=unchanged,saved_best=improved,seconds=seconds)
            training['history'].append(row)
            with (train_dir/'epochs.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
            log(train_dir,f'Epoch {epoch}/30: train_loss={tm["loss"]:.6f} val_loss={vm["loss"]:.6f} seconds={seconds:.2f} best={improved}')
            training.update(completed_epochs=epoch,best_epoch=best_epoch,best_validation_loss=best_loss,
                read_counts={s:d.read_calls for s,d in datasets.items()})
            write_json(train_dir/'training_summary.json',training)
            if seed==42 and epoch<=3:
                check=check_prefix_epoch(reference['training'],row)
                result['n4_prefix_check']['epochs'].append(check)
                if epoch==3:result['n4_prefix_check']['status']='passed'
            write_json(out/'n5_summary.json',result)
        training['peak_cuda_memory']=memory(device)
        del optimizer,model
        if device=='cuda':torch.cuda.empty_cache()
        restored,cp=load_checkpoint(train_dir/'best.pt',config,device)
        with torch.no_grad():actual=restored(probe).cpu()
        require(torch.allclose(actual,cp['probe_logits'],rtol=1e-5,atol=1e-6),'Checkpoint reload mismatch')
        require(cp['epoch']==best_epoch and cp['validation_loss']==best_loss,'Best metadata mismatch')
        del restored,cp,probe
        if device=='cuda':torch.cuda.empty_cache()
        training.update(status='passed',completed_epochs=config['epochs'],best_epoch=best_epoch,best_validation_loss=best_loss,checkpoint_reload_equal=True,
            checkpoint_sha256=sha256(train_dir/'best.pt'),read_counts={s:d.read_calls for s,d in datasets.items()},elapsed_seconds=time.perf_counter()-started,
            timing_scope='Full epoch I/O, transfers, updates, validation and diagnostic checks; no inference-latency claim')
        write_json(train_dir/'training_summary.json',training)
        # Reconstruct model AND validation dataset; no training model survives into prediction.
        pred_started=time.perf_counter()
        verify_input_hashes(manifest,summary,inputs)
        pdatasets={s:ConvNeXtSliceDataset(manifest,s,summary) for s in ('train','validation','test')}
        verify_input_hashes(manifest,summary,inputs)
        val=pdatasets['validation'];loader=make_loader(val,config['batch_size'],seed)
        if device=='cuda':torch.cuda.reset_peak_memory_stats()
        passes=[];times=[]
        for attempt in range(2):
            require(sha256(train_dir/'best.pt')==training['checkpoint_sha256'],'Checkpoint hash changed')
            model,cp=load_checkpoint(train_dir/'best.pt',config,device);del cp
            before=state_digest(model);sync(device);t0=time.perf_counter()
            rows=predict_pass(model,loader,device);sync(device);times.append(time.perf_counter()-t0)
            require(before==state_digest(model),'Prediction changed state')
            validate_rows(rows,val.records);passes.append(rows);del model
        first,second=passes;maxdiff=0.
        for a,b in zip(first,second):
            require(all(a[k]==b[k] for k in ('file_path','patient_id','image_id','label','prediction')),'Repeat identities/classes mismatch')
            maxdiff=max(maxdiff,abs(a['p_ad']-b['p_ad']))
            require(abs(a['p_ad']-b['p_ad'])<=1e-6+1e-5*abs(b['p_ad']),'Repeat probabilities mismatch')
        patients=aggregate_patients(first,expected_slices);require(len(patients)==expected_patients,'Patient count mismatch')
        write_csv(pred_dir/'slice_predictions.csv',first);write_csv(pred_dir/'patient_predictions.csv',patients)
        metric=lambda rows:classification_metrics([r['label'] for r in rows],[r['p_ad'] for r in rows])
        prediction=dict(status='passed',config=config,versions=versions,code_sha256=code,input_sha256=inputs,
            checkpoint_sha256=training['checkpoint_sha256'],training_summary_sha256=sha256(train_dir/'training_summary.json'),
            slice_count=len(first),patient_count=len(patients),slice_metrics=metric(first),patient_metrics=metric(patients),
            repeat_equal=True,repeat_max_probability_difference=maxdiff,parameters_unchanged=True,manifest_exact_match=True,
            read_counts={s:d.read_calls for s,d in pdatasets.items()},peak_cuda_memory=memory(device),full_pass_seconds=times,
            timing_scope='Full validation pass with I/O/preprocessing/transfers; synchronized CUDA; not isolated inference latency',
            elapsed_seconds=time.perf_counter()-pred_started,output_sha256={n:sha256(pred_dir/n) for n in ['slice_predictions.csv','patient_predictions.csv']})
        write_json(pred_dir/'config.json',config)
        log(pred_dir,f'PASS: {len(first)} slices, {len(patients)} patients, repeat max diff={maxdiff}')
        result.update(status='passed',completed_epochs=config['epochs'],prediction_repeat_equal=True,best_epoch=best_epoch,
            total_read_counts={s:datasets[s].read_calls+pdatasets[s].read_calls for s in datasets},
            weights_sha256=weight_digest,training_summary_sha256=sha256(train_dir/'training_summary.json'))
        finish_run(out,reference,training,prediction,result,seed)
    except Exception as exc:
        result.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        if result.get('n4_prefix_check',{}).get('status')=='pending':
            result['n4_prefix_check']['status']='incomplete_or_failed'
        if pdatasets:
            prediction.update(status='failed',error=result['error'],read_counts={s:d.read_calls for s,d in pdatasets.items()})
        (out/'n5_error.log').write_text(traceback.format_exc());raise
    finally:
        if training['status']!='passed':
            training.update(status='failed',completed_epochs=len(training['history']),error=result.get('error'),read_counts={s:d.read_calls for s,d in datasets.items()})
            write_json(train_dir/'training_summary.json',training)
        write_json(pred_dir/'prediction_summary.json',prediction)
        result['elapsed_seconds']=time.perf_counter()-started
        result['prediction_summary_sha256']=sha256(pred_dir/'prediction_summary.json')
        write_json(out/'n5_summary.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ['manifest','summary','weights','weights-summary','output-dir','reference-dir']:p.add_argument('--'+arg,type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--seed',type=int,choices=ALLOWED_SEEDS,default=42);a=p.parse_args()
    def interrupted(signum,frame):raise RuntimeError(f'Interrupted by signal {signum}')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    run_formal(a.manifest,a.summary,a.weights,a.weights_summary,a.output_dir,a.reference_dir,a.device,a.seed)

if __name__=='__main__':main()
