"""Fixed A1 schedule-endpoint experiment. No test inference or calibration.

Only the cosine endpoint changes: 30 -> 12. See SCHEDULE_ONLY_PLAN.md.
AI assistance: Codex implementation/tests; author review and disclosure required.
"""
from __future__ import annotations
import argparse
import csv
import ctypes
from ctypes import wintypes
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import traceback
import uuid

import torch
from torch.utils.data import DataLoader
import train_convnext_v2 as v2
from dataset import ADNISliceDataset, preprocessing_spec, require
from training_utils import aggregate_predictions, predict_validation, read_json
from analyse_confidence import verify_cached_predictions

NAMESPACE = 'schedule-only-end12'
PLAN = 'SCHEDULE_ONLY_PLAN.json'
NEW_FILES = ('train_schedule_only.py', 'test_schedule_only.py', 'SCHEDULE_ONLY_PLAN.md')
SEEDS = (42, 43, 44)
CAP = 10800.0
LEASE = 5.0
LOSS_ATOL = 1e-6
SCORE_ATOL = 1e-12
CONTROL_FOLDERS = {42: 'v2-scratch_augmented-20261024-161044-951159',
                   43: 'a1-stability/seed-43', 44: 'a1-stability/seed-44'}
LOSS_FIELDS = ('train_loss', 'validation_log_loss', 'patient_log_loss')
SCORE_FIELDS = ('train_slice_accuracy', 'patient_accuracy', 'patient_macro_f1',
                'patient_auroc', 'ad_recall', 'nc_recall')

def now(): return datetime.now(timezone.utc).isoformat()
def digest(x): return hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest()
def sha(p): return v2.sha256_file(Path(p))
def atomic(p, x): v2.atomic_json(Path(p), x)
def history(p):
    with Path(p).open(encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))
def sources(project):
    return {n: sha(project/n) for n in (*v2.SOURCE_FILES, *NEW_FILES)}

@dataclass(frozen=True)
class Config(v2.Config):
    decay_end_epoch: int = 12

def configuration(seed):
    require(seed in SEEDS, 'Only the three declared seeds are permitted.')
    return Config('scratch_augmented', seed=seed)

def learning_rate(epoch, config):
    require(1 <= epoch <= config.epochs, 'Epoch outside the protocol.')
    if epoch <= 3: return config.learning_rate * epoch / 3
    t = min(1., (epoch - 4) / 8)
    return config.min_lr + (config.learning_rate-config.min_lr)*(1+math.cos(math.pi*t))/2

def set_epoch(model, optimizer, config, epoch):
    for p in model.parameters(): p.requires_grad_(True)
    lr = learning_rate(epoch, config)
    for g in optimizer.param_groups: g['lr'] = lr
    return lr

def comparability(row, reference):
    differences = {}
    for name in (*LOSS_FIELDS, *SCORE_FIELDS):
        delta = abs(float(row[name])-float(reference[name]))
        tol = LOSS_ATOL if name in LOSS_FIELDS else SCORE_ATOL
        differences[name] = {'absolute_difference': delta, 'tolerance': tol, 'passed': delta <= tol}
    differences['optimizer_updates'] = {'actual': row['optimizer_updates'],
        'control': int(reference['optimizer_updates']),
        'passed': row['optimizer_updates'] == int(reference['optimizer_updates'])}
    return {'epoch': row['epoch'], 'passed': all(v['passed'] for v in differences.values()),
            'checks': differences, 'timings_excluded': True}

def context(private):
    private, project, bundle, _ = v2.local_context(private)
    controls = {}
    for seed in SEEDS:
        folder = private/CONTROL_FOLDERS[seed]
        report = read_json(folder/'v2_summary.json')
        expected = asdict(v2.Config('scratch_augmented', seed=seed))
        require(report['configuration'] == expected and report['experiment_completed'] is True
                and report['manifest_sha256'] == bundle.manifest_hashes
                and report['pretrained'] == {'used': False}, 'Control configuration/completion differs.')
        for name, expected_hash in report['source_code_sha256'].items():
            require(sha(project/name) == expected_hash, 'Historical source changed.')
        # Cached development data only. No individual test file is opened.
        verify_cached_predictions(folder/'PRIVATE_validation_predictions.json', bundle.records['validation'], report)
        controls[str(seed)] = {'folder': CONTROL_FOLDERS[seed], 'configuration': expected,
            'environment': report['hardware'], 'selected_epoch': report['selected_epoch'],
            'metrics': report['selected_validation_metrics'],
            'history_sha256': sha(folder/'history.csv'), 'summary_sha256': sha(folder/'v2_summary.json'),
            'checkpoint_sha256': sha(folder/'PRIVATE_best_model.pt'),
            'predictions_sha256': sha(folder/'PRIVATE_validation_predictions.json')}
    return private, project, bundle, controls

def payload(project, bundle, controls):
    losses = [c['metrics']['patient']['log_loss'] for c in controls.values()]
    return {'format_version': 1, 'namespace': NAMESPACE, 'seeds': list(SEEDS),
        'only_scientific_change': 'cosine endpoint 30 -> 12; hard labels, peak 3e-4 unchanged',
        'configurations': {str(s): asdict(configuration(s)) for s in SEEDS},
        'sources': sources(project), 'manifest_sha256': bundle.manifest_hashes, 'controls': controls,
        'comparability': {'loss_absolute_tolerance': LOSS_ATOL, 'score_absolute_tolerance': SCORE_ATOL,
            'optimizer_updates_exact': True, 'epochs': [1,2,3,4], 'stop_on_failure': True},
        'per_seed_active_wall_seconds': CAP, 'total_active_wall_seconds': CAP*3,
        'crash_reservation_seconds': LEASE, 'minimum_free_disk_gib': 12,
        'control_mean_patient_log_loss': math.fsum(losses)/3,
        'acceptance': {'minimum_summed_correct':251,'minimum_improved_seeds':2,
            'maximum_correct_loss_per_seed':1,'minimum_mean_ad_recall':2/3,
            'seed42_min_correct':85,'seed42_min_AD':22,'replay_absolute_tolerance':2e-5},
        'no_test_inference': True, 'no_automatic_promotion': True}

def checked(private, committed=True):
    private, project, bundle, controls = context(private)
    plan = read_json(project/PLAN)
    require(plan == payload(project,bundle,controls), 'Plan, source, controls or bindings changed.')
    if committed:
        names = (*NEW_FILES, PLAN)
        relative = Path('recognition')/project.name
        for n in (*v2.SOURCE_FILES, *names):
            raw = subprocess.check_output(['git','-C',str(project),'show',f'HEAD:{(relative/n).as_posix()}'])
            require(hashlib.sha256(raw).hexdigest() == sha(project/n), 'Commit all exact experiment inputs before training.')
    return private, project, bundle, plan

class LeaseBudget:
    """Prepaid execution intervals: a crash never refunds uncertain work.

    Parent supervisor uses monotonic time. Normal exit refunds the unused part;
    abnormal parent death keeps the last <=5 second unused reservation charged.
    Downtime is excluded. Existing sessions are never reset on resume.
    """
    def __init__(self, path, cap=CAP, quantum=LEASE, clock=time.monotonic):
        self.path=Path(path); self.cap=cap; self.quantum=quantum; self.clock=clock
        self.value=read_json(self.path) if self.path.exists() else {'cap_seconds':cap,'sessions':[],'charged_seconds':0.}
        require(self.value['cap_seconds']==cap,'Budget cap changed.')
        self.base=self.value['charged_seconds']; require(self.base < cap,'Run wall limit already exhausted.')
        self.start=clock(); self.deadline=self.start+cap-self.base
        self.session={'started_utc':now(),'charged_seconds':0.,'status':'ACTIVE'}
        self.value['sessions'].append(self.session); self.reserve()
    def reserve(self):
        elapsed=max(0.,self.clock()-self.start)
        charged=min(self.cap-self.base,elapsed+self.quantum)
        self.session['charged_seconds']=charged
        self.session['observed_elapsed_seconds']=elapsed
        self.value['charged_seconds']=self.base+charged
        atomic(self.path,self.value)
        return self.clock() < self.deadline
    def finish(self,status):
        elapsed=min(self.cap-self.base,max(0.,self.clock()-self.start))
        self.session.update(charged_seconds=elapsed,observed_elapsed_seconds=elapsed,
                            status=status,finished_utc=now())
        self.value['charged_seconds']=self.base+elapsed
        atomic(self.path,self.value)

class WindowsJob:
    """Kill the child process if its supervising process terminates."""
    def __init__(self, process):
        require(os.name=='nt','The real experiment supervisor requires the audited Windows host.')
        class BASIC(ctypes.Structure):
            _fields_=[('PerProcessUserTimeLimit',ctypes.c_int64),('PerJobUserTimeLimit',ctypes.c_int64),
                ('LimitFlags',wintypes.DWORD),('MinimumWorkingSetSize',ctypes.c_size_t),
                ('MaximumWorkingSetSize',ctypes.c_size_t),('ActiveProcessLimit',wintypes.DWORD),
                ('Affinity',ctypes.c_size_t),('PriorityClass',wintypes.DWORD),('SchedulingClass',wintypes.DWORD)]
        class IO(ctypes.Structure):
            _fields_=[(n,ctypes.c_uint64) for n in ('ReadOperationCount','WriteOperationCount',
                'OtherOperationCount','ReadTransferCount','WriteTransferCount','OtherTransferCount')]
        class EXTENDED(ctypes.Structure):
            _fields_=[('BasicLimitInformation',BASIC),('IoInfo',IO),
                ('ProcessMemoryLimit',ctypes.c_size_t),('JobMemoryLimit',ctypes.c_size_t),
                ('PeakProcessMemoryUsed',ctypes.c_size_t),('PeakJobMemoryUsed',ctypes.c_size_t)]
        self.k=ctypes.WinDLL('kernel32',use_last_error=True)
        self.k.CreateJobObjectW.argtypes=[ctypes.c_void_p,wintypes.LPCWSTR];self.k.CreateJobObjectW.restype=wintypes.HANDLE
        self.k.SetInformationJobObject.argtypes=[wintypes.HANDLE,ctypes.c_int,ctypes.c_void_p,wintypes.DWORD]
        self.k.AssignProcessToJobObject.argtypes=[wintypes.HANDLE,wintypes.HANDLE]
        self.k.CloseHandle.argtypes=[wintypes.HANDLE]
        self.handle=self.k.CreateJobObjectW(None,None)
        if not self.handle:raise ctypes.WinError(ctypes.get_last_error())
        limits=EXTENDED();limits.BasicLimitInformation.LimitFlags=0x2000
        if not self.k.SetInformationJobObject(self.handle,9,ctypes.byref(limits),ctypes.sizeof(limits)):
            self.close();raise ctypes.WinError(ctypes.get_last_error())
        if not self.k.AssignProcessToJobObject(self.handle,wintypes.HANDLE(int(process._handle))):
            self.close();raise ctypes.WinError(ctypes.get_last_error())
    def close(self):
        if self.handle:self.k.CloseHandle(self.handle);self.handle=None

def load_state(folder, config, plan, environment):
    bad=[]
    for path in (folder/'PRIVATE_resume.pt',folder/'PRIVATE_previous.pt'):
        if not path.exists():continue
        try:state=torch.load(path,map_location='cpu',weights_only=True)
        except Exception:bad.append(path);continue
        require(state['configuration']==asdict(config) and state['source_hashes']==plan['sources']
                and state['manifest_hashes']==plan['manifest_sha256'] and state['environment']==environment
                and state['study_sha256']==digest(plan), 'Recovery identity differs; stop.')
        if path.name=='PRIVATE_previous.pt' and (folder/'PRIVATE_resume.pt').exists():
            # Preserve the unreadable latest; never rotate it over the valid fallback.
            os.replace(folder/'PRIVATE_resume.pt',folder/('PRIVATE_unreadable-'+uuid.uuid4().hex+'.pt'))
        return state
    require(not (folder/'initialized.json').exists() and not (folder/'history.csv').exists(),
            'No readable checkpoint for an initialized attempt; do not start a replacement.')
    return None

def train_core(folder, bundle, config, plan, reference_history, *, model_factory=v2.make_model,
               max_new_epochs=None, enforce_comparability=True):
    """Reuse original model, optimizer, train_epoch and validation aggregation.

    Non-CLI hooks are for synthetic recovery tests only.
    """
    config.validate();device=v2.choose_device(config.device);v2.seed_everything(config.seed)
    model=model_factory(config).to(device);optimizer=v2.make_optimizer(model,config)
    environment=v2.environment(device);v2.reset_memory_peak(device)
    state=load_state(folder,config,plan,environment)
    if state is None:
        state={'format_version':4,'configuration':asdict(config),'source_hashes':plan['sources'],
            'manifest_hashes':bundle.manifest_hashes,'study_sha256':digest(plan),'environment':environment,
            'epoch':0,'history':[],'best':None,'bad_epochs':0,'completed':False,'pretrained':{'used':False},
            'parameters':v2.parameter_count(model),'resume_count':0,'model_state':v2.tensor_state(model),
            'optimizer_state':optimizer.state_dict(),'rng':v2.rng_state(),'comparability':[]}
        v2.save_state(folder/'PRIVATE_resume.pt',state)
        atomic(folder/'initialized.json',{'study_sha256':digest(plan),'seed':config.seed})
    else:
        model.load_state_dict(state['model_state'],strict=True);optimizer.load_state_dict(state['optimizer_state'])
        v2.restore_rng(state['rng'])
        if not state['completed']:state['resume_count']+=1
    if state['completed']:return state
    data=v2.WeightedDataset(bundle,config);val=ADNISliceDataset(bundle,'validation')
    new=0
    for epoch in range(state['epoch']+1,config.epochs+1):
        require(shutil.disk_usage(folder).free >= 12*2**30,'Free disk fell below 12 GiB; stop without deleting evidence.')
        generator=torch.Generator().manual_seed(config.seed+epoch)
        loader=DataLoader(data,batch_size=config.batch_size,shuffle=True,num_workers=0,
                         generator=generator,pin_memory=device.type=='cuda',drop_last=False)
        val_loader=DataLoader(val,batch_size=config.batch_size,shuffle=False,num_workers=0,
                         generator=torch.Generator().manual_seed(config.seed+100000+epoch))
        lr=set_epoch(model,optimizer,config,epoch)
        print(f'SCHEDULE seed {config.seed} epoch {epoch}/{config.epochs} lr={lr:.12g}',flush=True)
        tr=v2.train_epoch(model,loader,optimizer,device,config)
        probabilities,seconds=predict_validation(model,val_loader,device)
        metrics,private=aggregate_predictions(val.records,probabilities);m=metrics['patient']
        better=state['best'] is None or v2.selection_key(metrics)>v2.selection_key(state['best']['metrics'])
        row={'epoch':epoch,'learning_rate':lr,'train_loss':tr['loss'],'train_slice_accuracy':tr['slice_accuracy'],
            'validation_log_loss':metrics['slice']['log_loss'],'patient_accuracy':m['accuracy'],
            'patient_macro_f1':m['macro_f1'],'patient_auroc':m['auroc'],'patient_log_loss':m['log_loss'],
            'ad_recall':m['per_class']['AD']['recall'],'nc_recall':m['per_class']['NC']['recall'],
            'train_seconds':tr['seconds'],'validation_seconds':seconds,'optimizer_updates':tr['updates'],
            'selected_as_best':better}
        if epoch<=4 and enforce_comparability:
            receipt=comparability(row,reference_history[epoch-1]);state['comparability'].append(receipt)
            atomic(folder/'comparability.json',{'epochs':state['comparability'],
                'passed_so_far':all(x['passed'] for x in state['comparability'])})
            if not receipt['passed']:
                atomic(folder/'run_status.json',{'status':'MATERIAL_COMPARABILITY_FAILURE','epoch':epoch})
                raise RuntimeError('First-four-epoch comparability failed. Stop the study and investigate.')
        state['history'].append(row)
        if better:
            state['best']={'epoch':epoch,'metrics':metrics,'model_state':v2.tensor_state(model),
                           'private_predictions':private};state['bad_epochs']=0
        else:state['bad_epochs']+=1
        state.update(epoch=epoch,model_state=v2.tensor_state(model),optimizer_state=optimizer.state_dict(),
            rng=v2.rng_state(),schedule_state={'last_completed_epoch':epoch,'learning_rate':lr},
            loader_state={'seed_rule':'seed+epoch, workers=0','last_generator_state':generator.get_state()})
        memory=v2.memory_info(device)
        for k in ('peak_allocated_gib','peak_reserved_gib'):
            if memory.get(k) is not None:memory[k]=max(memory[k],state.get('memory',{}).get(k) or 0.)
        state['memory']=memory
        if epoch==config.epochs:state.update(completed=True,stop_reason='maximum_epochs')
        elif epoch>=config.min_epochs and state['bad_epochs']>=config.patience:
            state.update(completed=True,stop_reason='patience')
        v2.save_state(folder/'PRIVATE_resume.pt',state)
        v2.artifacts(folder,state)
        atomic(folder/'progress.json',{'seed':config.seed,'completed_epoch':epoch,'selected_epoch':state['best']['epoch'],
            'latest_metrics':m,'lr':lr,'utc':now()})
        print(f'  COMPLETE epoch={epoch} accuracy={m["accuracy"]:.8f} AD={m["per_class"]["AD"]["recall"]:.8f}',flush=True)
        new+=1
        if state['completed'] or (max_new_epochs is not None and new>=max_new_epochs):break
    return state

def load_best(path, config, plan, model_factory=v2.make_model):
    value=torch.load(path,map_location='cpu',weights_only=True)
    require(value['architecture']=='ConvNeXtTiny' and value['pretrained']=={'used':False}
        and value['model_label_map']=={'NC':0,'AD':1} and value['configuration']==asdict(config)
        and value['manifest_sha256']==plan['manifest_sha256'] and value['source_code_sha256']==plan['sources'],
        'Selected checkpoint metadata changed.')
    model=model_factory(config);expected=model.state_dict();actual=value['model_state']
    require(set(expected)==set(actual),'Selected checkpoint keys differ.')
    for k,t in expected.items():
        require(isinstance(actual[k],torch.Tensor) and actual[k].shape==t.shape
                and actual[k].dtype==t.dtype and bool(torch.isfinite(actual[k]).all()),
                'Selected checkpoint shape/dtype/finite check failed.')
    model.load_state_dict(actual,strict=True)
    return model.eval(),value

def finish(folder,bundle,config,plan,state,model_factory=v2.make_model):
    require(state['completed'],'Incomplete protocol cannot be completed by reporting.')
    v2.complete_outputs(folder,state)
    checkpoint=folder/'PRIVATE_best_model.pt';model,value=load_best(checkpoint,config,plan,model_factory)
    require(value['epoch']==state['best']['epoch'],'Exported epoch differs from selected epoch.')
    device=v2.choose_device(config.device);model=model.to(device)
    data=ADNISliceDataset(bundle,'validation')
    loader=DataLoader(data,batch_size=config.batch_size,shuffle=False,num_workers=0,
                      generator=torch.Generator().manual_seed(config.seed+999999))
    probabilities,seconds=predict_validation(model,loader,device)
    from selected_a1 import replay_checks
    original=state['best']['private_predictions']['slices']
    indexed={r['image_path']:r['p_ad'] for r in original}
    replay=replay_checks(data.records,[indexed[r.image_path] for r in data.records],probabilities,2e-5)
    replay.update(validation_seconds=seconds,checkpoint_sha256=sha(checkpoint),selected_epoch=value['epoch'])
    atomic(folder/'checkpoint_replay.json',replay)
    files=('PRIVATE_best_model.pt','PRIVATE_validation_predictions.json','v2_summary.json','history.csv',
           'checkpoint_replay.json','comparability.json')
    atomic(folder/'completion.json',{'status':'COMPLETED','study_sha256':digest(plan),
        'hashes':{n:sha(folder/n) for n in files if (folder/n).exists()},'selected_epoch':value['epoch']})
    atomic(folder/'run_status.json',{'status':'COMPLETED','utc':now()})

def worker(private,seed,gate):
    # A fresh unique gate is released only after assignment to the parent's job.
    deadline=time.monotonic()+60
    while not gate.exists():
        require(time.monotonic()<deadline,'Supervisor gate timed out; no training started.')
        time.sleep(.05)
    private,project,bundle,plan=checked(private)
    config=configuration(seed)
    require(v2.environment(v2.choose_device('cuda'))==plan['controls'][str(seed)]['environment'],
            'Environment differs from matched control.')
    folder=private/NAMESPACE/f'seed-{seed}'
    with v2.run_lock(folder):
        state=train_core(folder,bundle,config,plan,history(private/CONTROL_FOLDERS[seed]/'history.csv'))
        require(len(state['comparability'])==4 and all(r['passed'] for r in state['comparability']),
                'Four passing comparability epochs are required before protocol completion.')
        finish(folder,bundle,config,plan,state)

def valid_completed(folder,plan):
    receipt=read_json(folder/'completion.json')
    require(receipt['study_sha256']==digest(plan),'Completed run belongs to a different plan.')
    for n,h in receipt['hashes'].items():require(sha(folder/n)==h,'Completed output changed.')

def supervised(private,seed,resume_only=False):
    private,project,bundle,plan=checked(private)
    root=private/NAMESPACE;root.mkdir(exist_ok=True)
    with v2.run_lock(root):
        for previous in SEEDS[:SEEDS.index(seed)]:
            status=read_json(root/f'seed-{previous}'/'run_status.json')['status']
            require(status in ('COMPLETED','WALL_LIMIT_INCONCLUSIVE'),'Complete earlier seeds in order.')
        folder=root/f'seed-{seed}'
        require(not resume_only or folder.exists(),'No original attempt exists to resume.')
        folder.mkdir(exist_ok=True)
        binding={'seed':seed,'study_sha256':digest(plan)}
        if (folder/'binding.json').exists():require(read_json(folder/'binding.json')==binding,'Attempt binding changed.')
        else:atomic(folder/'binding.json',binding)
        if (folder/'completion.json').exists():
            valid_completed(folder,plan)
            # Recovery after child completion but before the parent's final write.
            # Retain the full crash reservation; never refund unknown elapsed time.
            budget=read_json(folder/'active_budget.json')
            old=read_json(folder/'run_status.json')['status']
            status='WALL_LIMIT_INCONCLUSIVE' if (old=='WALL_LIMIT_INCONCLUSIVE' or
                (old!='COMPLETED' and budget['charged_seconds']>=CAP)) else 'COMPLETED'
            atomic(folder/'run_status.json',{'status':status,'utc':now(),
                'recovered_completed_receipt':True,'crash_reservation_refunded':False})
            print('ALREADY_COMPLETED',seed,status);return
        if (folder/'run_status.json').exists():
            require(read_json(folder/'run_status.json')['status'] not in
                    ('MATERIAL_COMPARABILITY_FAILURE','WALL_LIMIT_INCONCLUSIVE'),'Stopped run cannot be restarted.')
        require(shutil.disk_usage(private).free>=12*2**30,'At least 12 GiB free disk is required.')
        used=sum(read_json(p)['charged_seconds'] for p in root.glob('seed-*/active_budget.json'))
        require(used<CAP*3,'Study budget exhausted.')
        ledger=LeaseBudget(folder/'active_budget.json')
        gate=folder/('gate-'+uuid.uuid4().hex+'.json');process=None;job=None;outcome='INTERRUPTED'
        atomic(folder/'run_status.json',{'status':'RUNNING','utc':now()})
        try:
            with (folder/'execution.log').open('ab',buffering=0) as log:
                process=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'_worker',
                    '--private-root',str(private),'--seed',str(seed),'--gate',str(gate)],
                    stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                job=WindowsJob(process)
                atomic(gate,{'released':True})
                while process.poll() is None:
                    if not ledger.reserve():
                        job.close();process.wait(timeout=30);outcome='WALL_LIMIT_INCONCLUSIVE';break
                    time.sleep(min(.5,max(0.,ledger.deadline-time.monotonic())))
                else:
                    if process.returncode==0:
                        valid_completed(folder,plan);outcome='COMPLETED'
                    else:
                        current=read_json(folder/'run_status.json')['status']
                        outcome=current if current=='MATERIAL_COMPARABILITY_FAILURE' else 'FAILED_NEEDS_INVESTIGATION'
        finally:
            if job is not None:job.close()
            if process is not None and process.poll() is None:
                process.kill();process.wait(timeout=30)
            ledger.finish(outcome)
            atomic(folder/'run_status.json',{'status':outcome,'utc':now()})
        print('SCHEDULE_SEED',seed,outcome,flush=True)
        require(outcome in ('COMPLETED','WALL_LIMIT_INCONCLUSIVE'),'Stopped; investigate before continuing.')

def score(report):
    m=report['selected_validation_metrics']['patient'];cm=m['confusion_matrix_rows_true_columns_predicted_NC_AD']
    return {'correct':cm[0][0]+cm[1][1],'AD':cm[1][1],
        **{k:m[k] for k in ('accuracy','macro_f1','auroc','log_loss')},
        'ad_recall':m['per_class']['AD']['recall'],'nc_recall':m['per_class']['NC']['recall'],
        'slice_log_loss':report['selected_validation_metrics']['slice']['log_loss']}

def trajectory(rows):
    last=rows[-5:]
    return {'epochs':[int(r['epoch']) for r in last],**{key:{'values':[float(r[key]) for r in last],
        'mean':statistics.mean(float(r[key]) for r in last),
        'population_sd':statistics.pstdev(float(r[key]) for r in last),
        'range':[min(float(r[key]) for r in last),max(float(r[key]) for r in last)]}
        for key in ('patient_accuracy','patient_log_loss')}}

def acceptance(rows,limit):
    if not all(r['status']=='COMPLETED' for r in rows):return {'assessable':False,'passed':False,'reason':'Incomplete/inconclusive run(s).'}
    new=[r['candidate'] for r in rows];old=[r['control'] for r in rows]
    changes=[a['correct']-b['correct'] for a,b in zip(new,old)]
    gates={'criterion1':sum(a['correct'] for a in new)>=251 and sum(d>0 for d in changes)>=2 and min(changes)>=-1,
        'criterion2':sum(a['AD'] for a in new)>=66 and new[0]['correct']>=85 and new[0]['AD']>=22,
        'criterion3':math.fsum(a['log_loss'] for a in new)/3<=limit,
        'criterion4':all(r['integrity_passed'] for r in rows)}
    return {'assessable':True,'passed':all(gates.values()),'gates':gates,'summed_correct':sum(a['correct'] for a in new),
        'paired_correct_changes':changes,'improved_seeds':sum(d>0 for d in changes),
        'mean_AD_recall':sum(a['AD'] for a in new)/99,
        'mean_log_loss':math.fsum(a['log_loss'] for a in new)/3,'control_exact_mean_log_loss':limit}

def compare(private):
    private,project,bundle,plan=checked(private);root=private/NAMESPACE;rows=[]
    with v2.run_lock(root):
        for seed in SEEDS:
            folder=root/f'seed-{seed}';control=plan['controls'][str(seed)]
            status=read_json(folder/'run_status.json')['status'] if (folder/'run_status.json').exists() else 'NOT_STARTED'
            row={'seed':seed,'status':status,'control':score({'selected_validation_metrics':control['metrics']}),
                'control_last_five':trajectory(history(private/CONTROL_FOLDERS[seed]/'history.csv')),
                'budget':read_json(folder/'active_budget.json') if (folder/'active_budget.json').exists() else None}
            if (folder/'v2_summary.json').exists():
                report=read_json(folder/'v2_summary.json');row.update(candidate=score(report),
                    selected_epoch=report['selected_epoch'],epochs_completed=report['epochs_completed'],
                    resources={k:report[k] for k in ('gpu_memory','training_loop_seconds_committed_epochs','stop_reason')},
                    candidate_last_five=trajectory(history(folder/'history.csv')))
                row['paired_differences']={k:row['candidate'][k]-row['control'][k] for k in row['candidate']}
            if (folder/'comparability.json').exists():row['first_four']=read_json(folder/'comparability.json')
            if status=='COMPLETED':
                valid_completed(folder,plan);row['integrity_passed']=True
                row['checkpoint_replay']=read_json(folder/'checkpoint_replay.json')
            rows.append(row)
        result={'study_sha256':digest(plan),'rows':rows,'acceptance':acceptance(rows,plan['control_mean_patient_log_loss']),
                'patients_are_102_reused_across_seeds':True,'no_test_evaluation':True,'automatic_promotion':False}
        atomic(root/'comparison.json',result)
    print(json.dumps(result['acceptance'],indent=2));return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('prepare','run','resume','run-all','compare','_worker'))
    p.add_argument('--private-root',type=Path,required=True);p.add_argument('--seed',type=int,choices=SEEDS)
    p.add_argument('--gate',type=Path,help=argparse.SUPPRESS);a=p.parse_args()
    if a.command=='prepare':
        private,project,bundle,controls=context(a.private_root);plan=payload(project,bundle,controls)
        if (project/PLAN).exists():require(read_json(project/PLAN)==plan,'Existing plan differs; preserve it.')
        else:atomic(project/PLAN,plan)
        print('PLAN_RECORDED_COMMIT_BEFORE_REAL_TRAINING');return
    if a.command=='_worker':
        require(a.seed is not None and a.gate is not None,'Missing supervisor gate.')
        worker(a.private_root,a.seed,a.gate)
    elif a.command=='compare':compare(a.private_root)
    elif a.command=='run-all':
        for seed in SEEDS:supervised(a.private_root,seed)
        compare(a.private_root)
    else:
        require(a.seed is not None,'A declared seed is required.')
        supervised(a.private_root,a.seed,a.command=='resume')

if __name__=='__main__':
    try:main()
    except BaseException:
        traceback.print_exc();sys.exit(1)
