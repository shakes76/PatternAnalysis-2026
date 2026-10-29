"""R2 finite diagnostic: F full versus H final-linear-only, five epochs each.

No V1 runner changes. Use fresh output directories. Head means the newly
initialized final Linear; classifier normalization stays frozen in H.
Both arms retain train() and stochastic depth. Reload two bests twice each.
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


def protocol_config():
    c=n4_config()
    c.update(epochs=5,checkpoint_selection='primary patient NLL; auxiliary slice CE; earliest tie',
             head_definition='final Linear only',training_mode='train including frozen backbone stochastic depth')
    # Trainable names are recorded separately; common configuration is identical.
    c.pop('full_finetuning')
    return c


def configure_trainable(model,arm):
    require(arm in ('F','H'),'Unknown arm')
    head=model.classifier[-1] if isinstance(model.classifier,nn.Sequential) else model.classifier
    require(isinstance(head,nn.Linear),'Expected final linear classification head')
    head_ids={id(p) for p in head.parameters()}
    for p in model.parameters():p.requires_grad_(arm=='F' or id(p) in head_ids)
    return [n for n,p in model.named_parameters() if p.requires_grad]


def component_digest(model,head):
    linear=model.classifier[-1] if isinstance(model.classifier,nn.Sequential) else model.classifier
    ids={id(p) for p in linear.parameters()};digest=hashlib.sha256()
    for n,p in model.named_parameters():
        if (id(p) in ids)==head:
            digest.update(n.encode());digest.update(p.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def rng_state():
    return [torch.get_rng_state().clone()]+(torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])


def same_rng(before):
    after=rng_state()
    return len(before)==len(after) and all(torch.equal(a,b) for a,b in zip(before,after))


def check_pair_start(actual,reference):
    for key in ('initial_state_sha256','initial_head_sha256','initial_backbone_sha256','config','versions','gpu_name'):
        require(actual.get(key)==reference.get(key),'F/H start mismatch: '+key)
    for key in ('input_sha256','weights_sha256','code_sha256'):
        require(actual['provenance'][key]==reference['provenance'][key],'F/H source mismatch: '+key)


def read_n4_reference(path,manifest,summary):
    reference=json.loads(Path(path).read_text())
    require(reference.get('status')=='passed','N4 reference did not pass')
    require(all(reference.get('config',{}).get(k)==v for k,v in n4_config().items()),'N4 reference protocol mismatch')
    require(reference.get('input_sha256')==dict(split_samples=sha256(manifest),split_summary=sha256(summary)),'N4 input hash mismatch')
    require(len(reference.get('history',[]))==3,'N4 history missing')
    require(bool(reference.get('code_sha256')),'N4 source hashes missing')
    for name,value in reference['code_sha256'].items():
        require(Path(name).name==name and sha256(Path(__file__).with_name(name))==value,'N4 source hash mismatch')
    return reference


def check_n4_prefix(actual,reference):
    require(actual['initial_state_sha256']==reference['initial_state_sha256'],'N4 initial state mismatch')
    require(len(actual['history'])>=3 and len(reference['history'])>=3,'N4 prefix incomplete')
    for source in (actual,reference):
        require([h.get('epoch') for h in source['history'][:3]]==[1,2,3],'N4 prefix epoch sequence mismatch')
    for a,b in zip(actual['history'][:3],reference['history'][:3]):
        for split in ('train','validation'):
            require(a[split]['order_sha256']==b[split]['order_sha256'],'N4 order mismatch')
            for metric in ('loss','unweighted_loss'):
                x,y=a[split][metric],b[split][metric]
                require(abs(x-y)<=1e-6+1e-5*abs(y),'N4 trajectory mismatch')
    return dict(passed=True,rtol=1e-5,atol=1e-6)


def save_predictions(folder,rows,evaluation):
    folder.mkdir(parents=True,exist_ok=False)
    write_csv(folder/'slice_predictions.csv',rows)
    write_csv(folder/'patient_predictions.csv',evaluation['patients'])
    summary={k:v for k,v in evaluation.items() if k!='patients'}
    summary['output_sha256']={n:sha256(folder/n) for n in ('slice_predictions.csv','patient_predictions.csv')}
    errors=[max(p['p_ad'],1-p['p_ad']) for p in evaluation['patients'] if p['prediction']!=p['label']]
    summary['error_confidence']=dict(count=len(errors),mean=sum(errors)/len(errors) if errors else None,
                                      maximum=max(errors) if errors else None,definition='max(p_AD,1-p_AD) among misclassified patients')
    write_json(folder/'summary.json',summary)
    return summary


def run_arm(arm,manifest,summary,weights,weights_summary,out,device,expected_lengths,expected_slices,pair_reference=None,n4_reference=None):
    out.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    s=dict(status='running',arm=arm,history=[]);datasets={};reload_ds={}
    try:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        torch.manual_seed(42);torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        if device=='cuda':torch.cuda.manual_seed_all(42);torch.cuda.reset_peak_memory_stats()
        model,weight_hash=initialize_model(weights,weights_summary)
        initial=state_digest(model);trainable=configure_trainable(model,arm)
        datasets={split:ConvNeXtSliceDataset(manifest,split,summary) for split in ('train','validation','test')}
        lengths={k:len(v) for k,v in datasets.items()}
        require(expected_lengths is None or lengths==expected_lengths,'Cohort lengths mismatch')
        for split,ds in datasets.items():
            if expected_lengths is not None:validate_frozen_labels(ds.records,split)
            evaluate_patients([dict(r,label=LABEL_MAP[r['label']],p_ad=0.) for r in ds.records],ds.records,expected_slices)
        config=protocol_config();config.update(device=device,weights_sha256=weight_hash,
            class_weight_details=class_weight_config(datasets['train'].records,'balanced'))
        provenance=dict(input_sha256=dict(manifest=sha256(manifest),summary=sha256(summary)),weights_sha256=weight_hash,
            code_sha256={n:sha256(Path(__file__).with_name(n)) for n in ('run_r2.py','patient_evaluation.py','dual_checkpoint.py','run_n4.py','run_n3.py','train.py','predict.py','dataset.py','modules.py','check_dataloader.py','check_convnext.py')},
            arm=arm,trainable_parameters=trainable)
        s.update(config=config,provenance=provenance,code_version=code_version(),initial_state_sha256=initial,
            initial_head_sha256=component_digest(model,True),initial_backbone_sha256=component_digest(model,False),
            trainable_parameters=trainable,trainable_count=sum(p.numel() for p in model.parameters() if p.requires_grad),
            versions=dict(python=platform.python_version(),torch=str(torch.__version__),torchvision=str(torchvision.__version__),cuda_build=torch.version.cuda,pillow=PIL.__version__),
            gpu_name=torch.cuda.get_device_name() if device=='cuda' else None)
        if pair_reference is not None:check_pair_start(s,pair_reference)
        if n4_reference is not None:
            for key in ('initial_state_sha256','versions','gpu_name'):
                require(s[key]==n4_reference[key],'N4 paired start mismatch: '+key)
            require(config['weights_sha256']==n4_reference['config']['weights_sha256'],'N4 pretrained mismatch')
        write_json(out/'config.json',config)
        model=model.to(device)
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=3e-5,weight_decay=.01)
        s['optimizer_actual']={k:v for k,v in optimizer.param_groups[0].items() if k!='params'}
        selector=DualCheckpointSelector(out/'selection',config,provenance,expected_slices)
        loaders={k:make_loader(datasets[k],4,42) for k in ('train','validation')}
        cw=torch.tensor(config['class_weight_details']['weights'],dtype=torch.float32,device=device)
        (out/'epochs').mkdir()
        for epoch in range(1,6):
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
            epoch_path=out/'epochs'/f'epoch_{epoch}.pt'
            torch.save(dict(config=config,provenance=provenance,epoch=epoch,
                patient_nll=evaluation['metrics']['nll'],slice_ce=vm['loss'],model_state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()}),epoch_path)
            require(same_rng(rng),'Diagnostics changed RNG')
            require(counts=={k:d.read_calls for k,d in datasets.items()},'Diagnostics added reads')
            sync(device)
            h=dict(epoch=epoch,train=tm,validation=vm,patient_metrics=evaluation['metrics'],diagnostics=diagnostic,
                epoch_checkpoint_sha256=sha256(epoch_path),backbone_no_grad=no_grad,backbone_unchanged=backbone_same,
                head_updated=head_updated,validation_state_unchanged=True,diagnostic_additional_reads=0,diagnostic_rng_unchanged=True,seconds=time.perf_counter()-t0)
            s['history'].append(h)
            with (out/'epochs.jsonl').open('a') as f:f.write(json.dumps(h,allow_nan=False)+'\n')
            with (out/'training.log').open('a') as f:f.write(f"epoch={epoch} train={tm['loss']} validation={vm['loss']} patient_nll={evaluation['metrics']['nll']}\n")
            write_json(out/'training_summary.json',s)
        s['training_peak_memory']=memory(device)
        s['training_read_counts']={k:d.read_calls for k,d in datasets.items()}
        require(s['training_read_counts']==dict(train=5*lengths['train'],validation=5*lengths['validation'],test=0),'Training reads mismatch')
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
                rows=predict_pass(restored,make_loader(reload_ds['validation'],4,42),device)
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
        s.update(status='passed',reload_peak_memory=memory(device),selection=deepcopy(selector.summary))
    except Exception as exc:
        s.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        (out/'error.log').write_text(traceback.format_exc());raise
    finally:
        s.update(elapsed_seconds=time.perf_counter()-started,
            actual_training_read_counts={k:d.read_calls for k,d in datasets.items()},actual_reload_read_counts={k:d.read_calls for k,d in reload_ds.items()})
        write_json(out/'training_summary.json',s)
    return s


def run_pair(manifest,summary,weights,weights_summary,output_dir,device='cuda',
             expected_lengths={'train':8160,'validation':2720,'test':2720},expected_slices=20,n4_reference=None):
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=False);started=time.perf_counter();r=dict(status='running',arms={})
    try:
        require(device in ('cpu','cuda') and (device!='cuda' or torch.cuda.is_available()),'Invalid/unavailable device')
        reference=read_n4_reference(n4_reference,manifest,summary) if n4_reference else None
        if reference:r['n4_reference']=dict(path=str(n4_reference),sha256=sha256(n4_reference))
        for arm in ('F','H'):
            s=run_arm(arm,manifest,summary,weights,weights_summary,out/arm,device,expected_lengths,expected_slices,r['arms'].get('F'),reference if arm=='F' else None);r['arms'][arm]=s
            if arm=='F' and n4_reference:
                r['n4_prefix']=check_n4_prefix(s,reference)
        if reference:require(sha256(n4_reference)==r['n4_reference']['sha256'],'N4 reference changed')
        f,h=r['arms']['F'],r['arms']['H']
        for key in ('config','initial_state_sha256','initial_head_sha256','initial_backbone_sha256','versions','gpu_name','optimizer_actual'):
            require(f[key]==h[key],'F/H mismatch: '+key)
        for key in ('input_sha256','weights_sha256','code_sha256'):require(f['provenance'][key]==h['provenance'][key],'F/H source mismatch')
        for a,b in zip(f['history'],h['history']):
            for split in ('train','validation'):require(a[split]['order_sha256']==b[split]['order_sha256'],'F/H order mismatch')
        r.update(status='passed',paired_initial_state_equal=True,paired_order_equal=True,
                 n4_prefix=r.get('n4_prefix',dict(status='not_checked',reason='No N4 reference; synthetic/local only')),
                 total_read_counts={k:sum(s['training_read_counts'][k]+s['reload_read_counts'][k] for s in (f,h)) for k in ('train','validation','test')})
    except Exception as exc:
        r.update(status='failed',error=f'{type(exc).__name__}: {exc}');(out/'r2_error.log').write_text(traceback.format_exc());raise
    finally:
        r['elapsed_seconds']=time.perf_counter()-started;write_json(out/'r2_summary.json',r)
    return r


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('manifest','summary','weights','weights-summary','output-dir','n4-reference'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--device',choices=('cuda','cpu'),default='cuda');a=p.parse_args()
    run_pair(a.manifest,a.summary,a.weights,a.weights_summary,a.output_dir,a.device,n4_reference=a.n4_reference)
