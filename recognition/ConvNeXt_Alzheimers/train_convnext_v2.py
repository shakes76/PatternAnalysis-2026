"""Version 2 development only: controlled ConvNeXt experiments and epoch recovery.

No final-test inference command exists. The old test has already been exposed;
all new scores are selected development results. Read V2_EXPERIMENT_PLAN.md.
Source assistance: ChatGPT; student review, empirical verification and disclosure
required. Network implementation remains PyTorch-only in modules.py/modules_v2.py.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import csv
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import sys
import time
import traceback
import zipfile

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import ADNISliceDataset, require, counts_for_records
from modules import ConvNeXtTiny, parameter_count
from modules_v2 import RGBTransferConvNeXt, import_torchvision_backbone
from training_utils import (aggregate_predictions, checked_bundle, choose_device,
    seed_everything, predict_validation, read_json, sha256_file, synchronize,
    binary_metrics, inference_benchmark, memory_info, reset_memory_peak)

RECIPES=('scratch_augmented','scratch_patient','transfer_patient')
SOURCE_FILES=('dataset.py','modules.py','training_utils.py','modules_v2.py',
              'train_convnext_v2.py','V2_EXPERIMENT_PLAN.md')

@dataclass(frozen=True)
class Config:
    recipe: str
    seed: int=42
    epochs: int=30
    min_epochs: int=12
    patience: int=8
    batch_size: int=8
    accumulation_steps: int=4
    rotation_degrees: float=5.0
    learning_rate: float=3e-4
    min_lr: float=1e-6
    weight_decay: float=.05
    warmup_epochs: int=3
    head_only_epochs: int=2
    transfer_backbone_lr: float=1e-5
    gradient_clip: float=5.0
    device: str='cuda'
    workers: int=0

    def validate(self):
        require(self.recipe in RECIPES, 'Unknown recipe.')
        require(self.epochs>=1 and 1<=self.min_epochs<=self.epochs and self.patience>=1,
                'Invalid epoch limits.')
        require(self.batch_size>=1 and self.accumulation_steps>=1 and self.workers==0,
                'Use positive batch sizes and zero workers for checked resume semantics.')
        require(0<=self.rotation_degrees<=5 and self.learning_rate>0 and self.min_lr>0,
                'Invalid augmentation or learning rate.')
        require(self.weight_decay>=0 and self.gradient_clip>0, 'Invalid optimization settings.')
        require(self.device in ('cpu','cuda'), 'Device must be cpu or cuda.')


def atomic_json(path, value):
    path=Path(path); temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


def save_state(path, state):
    """Save a full state atomically; keep one previous completed-epoch recovery copy."""
    path=Path(path); temp=path.with_suffix('.tmp')
    with temp.open('wb') as f:
        torch.save(state,f); f.flush(); os.fsync(f.fileno())
    if path.exists(): os.replace(path,path.with_name('PRIVATE_previous.pt'))
    os.replace(temp,path)


@contextmanager
def run_lock(folder):
    """OS file lock; released by Windows/Linux when the process exits or dies."""
    path=Path(folder)/'run.lock'
    handle=path.open('a+b')
    try:
        if path.stat().st_size==0: handle.write(b'0');handle.flush()
        handle.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX|fcntl.LOCK_NB)
    except OSError:
        handle.close();raise RuntimeError('This experiment is already locked by another process.')
    try: yield
    finally:
        handle.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def rng_state():
    n=np.random.get_state()
    return {'python':random.getstate(), 'numpy':[n[0],n[1].tolist(),int(n[2]),int(n[3]),float(n[4])],
            'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    n=state['numpy'];np.random.set_state((n[0],np.asarray(n[1],dtype=np.uint32),n[2],n[3],n[4]))
    torch.set_rng_state(state['torch'])
    if state['cuda']: torch.cuda.set_rng_state_all(state['cuda'])


def environment(device):
    import PIL, sklearn
    result={'python':platform.python_version(),'torch':str(torch.__version__),
            'numpy':np.__version__,'pillow':PIL.__version__,'sklearn':sklearn.__version__,
            'platform':platform.system(),'device':str(device),'cuda_build':torch.version.cuda}
    if device.type=='cuda': result['gpu']=torch.cuda.get_device_name(device)
    return result


def source_hashes():
    base=Path(__file__).resolve().parent
    return {name:sha256_file(base/name) for name in SOURCE_FILES}


def patient_weights(records):
    """w_i = N / (P * scans_in_patient * slices_in_scan).

    Over the complete training cohort weights sum to N, with identical total
    contribution per patient and per-scan equality within each patient. This is
    NOT class balancing. Do not renormalise separately inside each small batch.
    """
    require(bool(records) and all(r.split=='train' for r in records),'Weights use training records only.')
    scans=Counter(r.scan_id for r in records)
    patients=defaultdict(set)
    for r in records: patients[r.subject_id].add(r.scan_id)
    n,p=len(records),len(patients)
    return torch.tensor([n/(p*len(patients[r.subject_id])*scans[r.scan_id]) for r in records],
                        dtype=torch.float32)


class WeightedDataset(ADNISliceDataset):
    def __init__(self,bundle,config):
        super().__init__(bundle,'train',rotation_degrees=config.rotation_degrees)
        self.weights=patient_weights(self.records) if config.recipe!='scratch_augmented' else torch.ones(len(self))
    def __getitem__(self,index):
        sample=super().__getitem__(index);sample['loss_weight']=self.weights[index];return sample


def selection_key(metrics):
    m=metrics['patient']
    return (m['accuracy'],m['macro_f1'],-m['log_loss'])


def make_model(config):
    return RGBTransferConvNeXt() if config.recipe=='transfer_patient' else ConvNeXtTiny()


def make_optimizer(model, config):
    groups=defaultdict(list)
    for name,p in model.named_parameters():
        head=name.startswith('head.')
        groups[(head,p.ndim>1)].append(p)
    return torch.optim.AdamW([
        {'params':params,'is_head':head,
         'weight_decay':config.weight_decay if decay else 0.,'lr':config.learning_rate}
        for (head,decay),params in groups.items()])


def set_epoch(model,optimizer,config,epoch):
    transfer=config.recipe=='transfer_patient'
    head_only=transfer and epoch<=config.head_only_epochs
    for name,p in model.named_parameters(): p.requires_grad_(not head_only or name.startswith('head.'))
    for group in optimizer.param_groups:
        base=config.transfer_backbone_lr if transfer and not group['is_head'] else config.learning_rate
        start=config.head_only_epochs+1 if transfer else 1
        if head_only: lr=base
        else:
            e=epoch-start+1
            warm=config.warmup_epochs
            if e<=warm: lr=base*e/max(1,warm)
            else:
                span=max(1,config.epochs-start+1-warm-1)
                fraction=min(1.,(e-warm-1)/span)
                floor=min(config.min_lr,base)
                lr=floor+(base-floor)*.5*(1+math.cos(math.pi*fraction))
        group['lr']=lr
    return head_only


def tensor_state(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def train_epoch(model,loader,optimizer,device,config,head_only=False):
    require(loader.dataset.split=='train','Training loader is not training data.')
    # A frozen pretrained backbone uses deterministic evaluation-mode features
    # during head-only warmup. Full fine-tuning restores stochastic-depth training.
    model.eval() if head_only else model.train()
    optimizer.zero_grad(set_to_none=True)
    n_total=len(loader.dataset); effective=config.batch_size*config.accumulation_steps
    count=0; loss_sum=0.; correct=0; updates=0; t=time.perf_counter()
    for index,batch in enumerate(loader):
        require(set(batch['split'])=={'train'},'Nontraining image in optimization batch.')
        if index%config.accumulation_steps==0: window=min(effective,n_total-count)
        x=batch['image'].to(device); y=batch['label'].to(device);w=batch['loss_weight'].to(device)
        logits=model(x)
        ce=-torch.log_softmax(logits,1).gather(1,y[:,None]).squeeze(1)
        loss=(ce*w).sum()/window
        require(bool(torch.isfinite(loss)), 'Non-finite training loss.')
        loss.backward()
        n=len(y);count+=n;loss_sum+=float((ce.detach()*w).sum());correct+=int((logits.argmax(1)==y).sum())
        if (index+1)%config.accumulation_steps==0 or count==n_total:
            nn.utils.clip_grad_norm_(model.parameters(),config.gradient_clip,error_if_nonfinite=True)
            optimizer.step();optimizer.zero_grad(set_to_none=True);updates+=1
        if (index+1)%400==0: print(f'  {count}/{n_total} training images',flush=True)
    synchronize(device)
    require(count==n_total and updates==math.ceil(n_total/effective),'Incomplete training epoch.')
    return {'loss':loss_sum/count,'slice_accuracy':correct/count,'images':count,'updates':updates,
            'seconds':time.perf_counter()-t}


def load_pretrained(model,private,approval):
    require(approval is not None,'Pretrained weights require a real approval note; scratch recipes do not.')
    approval=Path(approval).resolve();private=Path(private).resolve()
    require(private in approval.parents and approval.is_file(),'Keep the approval note in your private directory.')
    text=approval.read_text(encoding='utf-8-sig').strip()
    require(len(text)>=30 and '[COMPLETE' not in text,'Record the actual approval, not an empty placeholder.')
    # The user executes this only after authorisation. No downloads occur in tests.
    torch.hub.set_dir(str(private/'pretrained-cache'))
    from torchvision.models import ConvNeXt_Tiny_Weights
    weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    source=weights.get_state_dict(progress=True, check_hash=True)
    import_torchvision_backbone(model,source)
    buffer=io.BytesIO();torch.save(source,buffer)
    return {'used':True,'provider':'torchvision ConvNeXt_Tiny_Weights.IMAGENET1K_V1',
            'approval_reported_by_user':True,'approval_sha256':sha256_file(approval),
            'loaded_state_serialization_sha256':hashlib.sha256(buffer.getvalue()).hexdigest(),
            'clinical_or_test_protocol_approval_implied':False}


def artifacts(folder,state):
    """Regenerate aggregate evidence from the authoritative resumable checkpoint."""
    config=state['configuration'];best=state['best'];history=state['history']
    report={
        'version':2,'status':'COMPLETED' if state['completed'] else 'IN_PROGRESS',
        'experiment_completed':state['completed'],'recipe':config['recipe'],
        'configuration':config,'epochs_completed':state['epoch'],
        'selected_epoch':best['epoch'] if best else None,
        'selected_validation_metrics':best['metrics'] if best else None,
        'selection':'validation patient accuracy; then macro-F1; then lower log loss; earliest exact tie',
        'operating_threshold':.5,'pretrained':state['pretrained'],
        'parameters':state['parameters'],'hardware':state['environment'],
        'source_code_sha256':state['source_hashes'],'manifest_sha256':state['manifest_hashes'],
        'training_loop_seconds_committed_epochs':sum(h['train_seconds']+h['validation_seconds'] for h in history),
        'runtime_note':'Excludes repeated work in interrupted epochs, setup/checkpoint I/O and downtime.',
        'gpu_memory':state.get('memory',{}),'stop_reason':state.get('stop_reason'),
        'resume_count':state.get('resume_count',0),'checkpoint_format':'full model/optimizer/RNG/schedule/best/history; epoch boundary',
        'precision':'float32; TF32 and mixed precision disabled',
        'training_patient_weighting':config['recipe']!='scratch_augmented',
        'validation_rotation_degrees':0.0,'rotation_review_required':True,
        'test_previously_exposed_in_v1':True,'new_test_images_decoded':0,
        'new_final_test_model_evaluation_performed':False,
        'interpretation':'Selected development results; neither a new independent test nor proof of 0.80 test accuracy.',
    }
    atomic_json(folder/'v2_summary.json',report)
    tmp=folder/'history.csv.tmp'
    with tmp.open('w',encoding='utf-8',newline='') as f:
        if history:
            writer=csv.DictWriter(f,fieldnames=list(history[0]));writer.writeheader();writer.writerows(history)
    os.replace(tmp,folder/'history.csv')
    lines=['# ConvNeXt V2 development experiment','',f'Recipe: **{config["recipe"]}**.',
           '',report['interpretation'],'','No final-test images were evaluated in this phase.',
           '',f'Completed epochs: {state["epoch"]}. Selected epoch: {report["selected_epoch"]}.',
           '',f'Settings: `{config}`.','',
           'Primary checkpoint selection is now validation patient accuracy, with macro-F1 then log loss as tie-breakers. '
           'This is a documented change from V1; it is not retrospective relabelling of V1.', '',
           'Scratch-augmented changes training-only rotation. Scratch-patient additionally changes the loss to '
           'equal patient/equal within-patient scan contribution; it is not class balancing. Transfer-patient also '
           'changes initial weights, input-channel normalisation and optimisation, so it is not a one-factor comparison.', '']
    if best:
        lines+=['| Unit | Accuracy | Macro-F1 | AUROC | AD recall |','|---|---:|---:|---:|---:|']
        for unit,m in best['metrics'].items():
            lines.append(f'| {unit} | {m["accuracy"]:.4f} | {m["macro_f1"]:.4f} | {m["auroc"]:.4f} | {m["per_class"]["AD"]["recall"]:.4f} |')
    lines+=['','One seed, repeated validation selection, no clinical validation. Keep all attempts, including negative results.',
            'ChatGPT assisted with implementation/tests and this draft; record the code review and checks personally performed.']
    (folder/'V2_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def plot_history(folder,history):
    if not history:return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for name, cols, ylabel in [('v2_loss.png',['train_loss','validation_log_loss'],'Loss'),
                               ('v2_validation.png',['patient_accuracy','patient_macro_f1','ad_recall'],'Score')]:
        fig,ax=plt.subplots(figsize=(8,5))
        for col in cols:ax.plot([r['epoch'] for r in history],[r[col] for r in history],label=col)
        ax.set(xlabel='Epoch',ylabel=ylabel,title='Development only — final test not evaluated');ax.legend();fig.tight_layout()
        fig.savefig(folder/name,dpi=150);plt.close(fig)


def train_run(folder,bundle,config,pipeline_report,private,approval=None,resume=False,
              model_factory=make_model,max_new_epochs=None):
    """The optional model_factory/max_new_epochs are testing hooks, not CLI choices."""
    config.validate();device=choose_device(config.device);seed_everything(config.seed)
    model=model_factory(config).to(device);optimizer=make_optimizer(model,config)
    hashes=source_hashes();env=environment(device);state_path=folder/'PRIVATE_resume.pt'
    reset_memory_peak(device)
    state=None
    recovered_from=None
    if resume:
        candidates=[state_path,folder/'PRIVATE_previous.pt']
        failures=[]
        for candidate in candidates:
            if not candidate.exists():continue
            try:
                state=torch.load(candidate,map_location='cpu',weights_only=True);recovered_from=candidate;break
            except Exception as exc:failures.append(type(exc).__name__)
        require(state is not None,'No readable complete checkpoint; preserve this run and report the error.')
        require(state['configuration']==asdict(config),'Configuration changed; do not silently resume a different experiment.')
        require(state['source_hashes']==hashes and state['manifest_hashes']==bundle.manifest_hashes,
                'Code or manifests changed since the checkpoint; stop.')
        require(state['environment']==env,'Software/hardware environment differs; stop before claiming exact continuation.')
        if recovered_from != state_path and state_path.exists():
            # Do not rotate a corrupt latest checkpoint over the valid fallback.
            os.replace(state_path,folder/('PRIVATE_unreadable-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f')+'.pt'))
        model.load_state_dict(state['model_state'],strict=True);optimizer.load_state_dict(state['optimizer_state'])
        state['resume_count']=state.get('resume_count',0)+1
        restore_rng(state['rng'])
    else:
        pretrained=load_pretrained(model,private,approval) if config.recipe=='transfer_patient' else {'used':False}
        state={'format_version':2,'configuration':asdict(config),'source_hashes':hashes,
               'manifest_hashes':bundle.manifest_hashes,'environment':env,'epoch':0,'history':[],
               'best':None,'bad_epochs':0,'completed':False,'pretrained':pretrained,
               'parameters':parameter_count(model),'resume_count':0,
               'model_state':tensor_state(model),'optimizer_state':optimizer.state_dict(),'rng':rng_state(),
               'schedule_state':{'last_completed_epoch':0},'loader_state':{'seed_rule':'seed+epoch, workers=0'}}
        save_state(state_path,state)
    artifacts(folder,state)
    if state['completed']:
        complete_outputs(folder,state)
        print('V2_RUN_ALREADY_COMPLETED');return state
    train_data=WeightedDataset(bundle,config)
    val_data=ADNISliceDataset(bundle,'validation')
    finished=0
    for epoch in range(state['epoch']+1,config.epochs+1):
        generator=torch.Generator().manual_seed(config.seed+epoch)
        loader=DataLoader(train_data,batch_size=config.batch_size,shuffle=True,num_workers=0,
                          generator=generator,pin_memory=device.type=='cuda',drop_last=False)
        val_loader=DataLoader(val_data,batch_size=config.batch_size,shuffle=False,num_workers=0,
                              generator=torch.Generator().manual_seed(config.seed+100000+epoch))
        head_only=set_epoch(model,optimizer,config,epoch)
        print(f'{config.recipe}: epoch {epoch}/{config.epochs}'+(' (head only)' if head_only else ''),flush=True)
        tr=train_epoch(model,loader,optimizer,device,config,head_only)
        probabilities,val_seconds=predict_validation(model,val_loader,device)
        metrics,private_predictions=aggregate_predictions(val_data.records,probabilities)
        better=state['best'] is None or selection_key(metrics)>selection_key(state['best']['metrics'])
        if better:
            state['best']={'epoch':epoch,'metrics':metrics,'model_state':tensor_state(model),
                           'private_predictions':private_predictions};state['bad_epochs']=0
        else:state['bad_epochs']+=1
        m=metrics['patient']
        state['history'].append({'epoch':epoch,'train_loss':tr['loss'],'train_slice_accuracy':tr['slice_accuracy'],
            'validation_log_loss':metrics['slice']['log_loss'],'patient_accuracy':m['accuracy'],
            'patient_macro_f1':m['macro_f1'],'patient_auroc':m['auroc'],
            'patient_log_loss':m['log_loss'],'ad_recall':m['per_class']['AD']['recall'],
            'nc_recall':m['per_class']['NC']['recall'],'train_seconds':tr['seconds'],
            'validation_seconds':val_seconds,'optimizer_updates':tr['updates'],'selected_as_best':better})
        state['epoch']=epoch;state['model_state']=tensor_state(model);state['optimizer_state']=optimizer.state_dict()
        state['rng']=rng_state();state['schedule_state']={'last_completed_epoch':epoch}
        state['loader_state']={'seed_rule':'seed+epoch, workers=0','last_generator_state':generator.get_state()}
        current_memory=memory_info(device)
        old_memory=state.get('memory',{})
        for key in ('peak_allocated_gib','peak_reserved_gib'):
            if current_memory.get(key) is not None:
                current_memory[key]=max(current_memory[key],old_memory.get(key) or 0.)
        state['memory']=current_memory
        if epoch==config.epochs:state['completed']=True;state['stop_reason']='maximum_epochs'
        elif epoch>=config.min_epochs and state['bad_epochs']>=config.patience:
            state['completed']=True;state['stop_reason']='patience'
        save_state(state_path,state) # commit full recovery state before aggregate outputs
        artifacts(folder,state)
        print(f'  validation patient accuracy {m["accuracy"]:.2%}, macro-F1 {m["macro_f1"]:.4f}, AD recall {m["per_class"]["AD"]["recall"]:.2%}; checkpoint saved',flush=True)
        finished+=1
        if state['completed'] or (max_new_epochs and finished>=max_new_epochs):break
    if state['completed']:
        complete_outputs(folder,state)
        print('V2_DEVELOPMENT_EXPERIMENT_COMPLETED',flush=True)
    return state


def complete_outputs(folder,state):
    """Idempotently rebuild final artifacts after an interrupted final write."""
    best=state['best'];config=state['configuration']
    atomic_json(folder/'PRIVATE_validation_predictions.json',best['private_predictions'])
    checkpoint={'format_version':2,
                'architecture':'RGBTransferConvNeXt' if config['recipe']=='transfer_patient' else 'ConvNeXtTiny',
                'epoch':best['epoch'],'configuration':config,'model_state':best['model_state'],
                'manifest_sha256':state['manifest_hashes'],'source_code_sha256':state['source_hashes'],
                'pretrained':state['pretrained'],'model_label_map':{'NC':0,'AD':1}}
    temp=folder/'PRIVATE_best_model.tmp'
    with temp.open('wb') as handle:
        torch.save(checkpoint,handle);handle.flush();os.fsync(handle.fileno())
    os.replace(temp,folder/'PRIVATE_best_model.pt')
    artifacts(folder,state)
    plot_history(folder,state['history'])


def pointer(private,name):
    p=Path(private)/name
    require(p.is_file(), f'Missing pointer file: {name}')
    value=p.read_text(encoding='utf-8-sig').strip()
    require(bool(value),f'Empty pointer file: {name}')
    return Path(value).resolve()


def local_context(private):
    private=Path(private).resolve();project=pointer(private,'project_path.txt')
    require(project==Path(__file__).resolve().parent,'Run the installed code in the saved assignment folder.')
    splits=pointer(private,'latest_patient_splits.txt')
    report=pointer(private,'latest_data_pipeline.txt')/'data_pipeline_summary.json'
    bundle=checked_bundle(private/'data'/'ADNI',splits,report)
    return private,project,bundle,report


def output_dir(private,name):
    p=private/(name+'-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    require(not any((parent/'.git').exists() for parent in [p,*p.parents]),'Private results must stay outside Git.')
    p.mkdir();return p


def preview(private,bundle):
    folder=output_dir(private,'v2-preview')
    selected=[]
    for label in (0,1):
        selected+= [r for r in bundle.records['train'] if r.label==label][:2]
    canvas=Image.new('RGB',(3*224, len(selected)*244),'white');draw=ImageDraw.Draw(canvas)
    from dataset import image_to_tensor
    for row,record in enumerate(selected):
        with Image.open(bundle.root/record.image_path) as image:
            image.load()
            for col,angle in enumerate((0,-5,5)):
                tensor=image_to_tensor(image,angle)
                array=((tensor[0]+1)*127.5).clamp(0,255).byte().numpy()
                canvas.paste(Image.fromarray(array).convert('RGB'),(col*224,row*244+20))
                draw.text((col*224+4,row*244+3),f'Train example {row+1}, rotation {angle}',fill='black')
    canvas.save(folder/'PRIVATE_augmentation_preview.png')
    atomic_json(folder/'preview_summary.json',{'training_examples':len(selected),'test_images_opened':0,
                                              'angles':[0,-5,5],'manual_review_required':True})
    (private/'latest_v2_preview.txt').write_text(str(folder),encoding='utf-8')
    print('V2_PRIVATE_PREVIEW_READY');print(folder)


def smoke(private,bundle,device_name):
    device=choose_device(device_name);seed_everything(42)
    config=Config('scratch_patient',device=device_name)
    data=WeightedDataset(bundle,config)
    loader=DataLoader(data,batch_size=8,shuffle=False,num_workers=0)
    batch=next(iter(loader));model=make_model(config).to(device);optimizer=make_optimizer(model,config)
    before=model.head.weight.detach().clone();losses=[]
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        logits=model(batch['image'].to(device));y=batch['label'].to(device)
        loss=(-torch.log_softmax(logits,1).gather(1,y[:,None]).squeeze(1)*batch['loss_weight'].to(device)).mean()
        require(bool(torch.isfinite(loss)),'Non-finite smoke loss.')
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);optimizer.step();losses.append(float(loss))
    require(not torch.equal(before,model.head.weight.detach()),'Parameters did not update.')
    folder=output_dir(private,'v2-smoke')
    model.eval();x=batch['image'].to(device)
    with torch.inference_mode(): expected=model(x).cpu()
    temp=folder/'PRIVATE_smoke.pt';torch.save({'model':tensor_state(model),'optimizer':optimizer.state_dict()},temp)
    loaded=torch.load(temp,map_location='cpu',weights_only=True)
    clone=make_model(config).to(device);clone.load_state_dict(loaded['model']);clone.eval()
    with torch.inference_mode(): actual=clone(x).cpu()
    require(torch.equal(expected,actual),'Checkpoint round-trip differs.')
    temp.unlink()
    atomic_json(folder/'v2_smoke_summary.json',{'passed':True,'configuration':asdict(config),'source_hashes':source_hashes(),
         'manifest_sha256':bundle.manifest_hashes,'environment':environment(device),'finite_losses':losses,
         'forward_backward_optimizer_update_passed':True,'checkpoint_roundtrip_passed':True,
         'test_images_opened':0,'pretrained_downloaded':False})
    (private/'latest_v2_smoke.txt').write_text(str(folder),encoding='utf-8')
    print('V2_GPU_SMOKE_PASSED');print(folder)


def bundle_results(private):
    folders=[]
    for recipe in RECIPES:
        p=private/f'latest_v2_{recipe}.txt'
        if p.exists():folders.append(pointer(private,p.name))
    require(bool(folders),'No V2 runs exist yet.')
    dest=private/'COMP3710_V2_Results.zip';temp=dest.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
        for folder in folders:
            for name in ('v2_summary.json','history.csv','V2_RESULTS.md'):
                path=folder/name
                if path.exists():z.write(path,f'{folder.name}/{name}')
        p=private/'latest_v2_smoke.txt'
        if p.exists():
            path=pointer(private,p.name)/'v2_smoke_summary.json'
            if path.exists():z.write(path,'v2_smoke_summary.json')
    os.replace(temp,dest);print('V2_AGGREGATE_RESULTS_READY');print(dest)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('preview','smoke','train','resume','bundle'))
    parser.add_argument('--private-root',required=True,type=Path)
    parser.add_argument('--recipe',choices=RECIPES)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--approval-note',type=Path)
    parser.add_argument('--visual-reviewed',action='store_true')
    parser.add_argument('--acknowledge-exposed-test',action='store_true')
    args=parser.parse_args()
    private,project,bundle,report=local_context(args.private_root)
    if args.command=='preview':preview(private,bundle);return
    if args.command=='smoke':smoke(private,bundle,args.device);return
    if args.command=='bundle':bundle_results(private);return
    require(args.recipe is not None,'Specify a recipe.')
    if args.command=='train':
        require(args.visual_reviewed,'Review the training-only augmentation preview before starting.')
        require(args.acknowledge_exposed_test,'Acknowledge V1 test exposure; V2 is development-only.')
        check=read_json(pointer(private,'latest_v2_smoke.txt')/'v2_smoke_summary.json')
        require(check.get('passed') is True and check['source_hashes']==source_hashes()
                and check['manifest_sha256']==bundle.manifest_hashes,'Matching passed smoke check required.')
        ptr=private/f'latest_v2_{args.recipe}.txt'
        require(not ptr.exists(),'A run for this recipe already exists. Use resume, not another attempt.')
        if args.recipe=='transfer_patient':
            require(args.approval_note is not None,'Transfer recipe requires the actual course approval note.')
        require(shutil.disk_usage(private).free >= 4*2**30,
                'Keep at least 4 GiB free for this run and its recovery checkpoints. Do not delete earlier evidence blindly.')
        config=Config(args.recipe,device=args.device)
        folder=output_dir(private,'v2-'+args.recipe)
        atomic_json(folder/'config.json',asdict(config))
        ptr.write_text(str(folder),encoding='utf-8')
    else:
        folder=pointer(private,f'latest_v2_{args.recipe}.txt')
        config=Config(**read_json(folder/'config.json'))
    with run_lock(folder):
        try:
            train_run(folder,bundle,config,report,private,args.approval_note,resume=args.command=='resume')
        except BaseException:
            (folder/'PRIVATE_error.txt').write_text(traceback.format_exc(),encoding='utf-8')
            print('V2_STOPPED: preserve this folder. Resume reloads the last complete epoch; no new test is opened.',flush=True)
            raise

if __name__=='__main__':
    try: main()
    except (Exception, KeyboardInterrupt) as exc:
        print(f'V2_STOPPED: {type(exc).__name__}: {exc}',file=sys.stderr)
        sys.exit(1)
