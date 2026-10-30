"""Fixed study execution, reusing V2 recovery/locks and schedule lease/job helpers.

Nothing in this module runs on import. Real MRI smoke/run/replay are explicit
user commands; implementation verification uses synthetic fetch/model inputs.
"""
from __future__ import annotations
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid
import torch
from g012 import core, storage as st
import train_convnext_v2 as v2
from train_schedule_only import LeaseBudget, WindowsJob

require = core.require
STOPPED = ('WALL_LIMIT_INCONCLUSIVE','BLOCKED_OOM','BLOCKED_HEADROOM')


class Budget(LeaseBudget):
    """Existing prepaid ledger arithmetic; serialize only its live file I/O."""
    def reserve(self):
        with st.budget_io(self.path.parent): return super().reserve()

    def finish(self,status):
        with st.budget_io(self.path.parent): return super().finish(status)


class Resources:
    def __init__(self, device, path=None):
        self.device = device; self.path = path; self.last = None
        self.value = {'scope':'observed process allocator peaks; CUDA device free-memory samples; not total Windows VRAM',
                      'peak_allocated_gib':0., 'peak_reserved_gib':0., 'min_device_free_gib':None,
                      'device_total_gib':None, 'warning_allocated_above_4_5_gib':False}
        if device.type == 'cuda': v2.reset_memory_peak(device)
        self.observe()

    def observe(self):
        if self.device.type != 'cuda': return
        free, total = torch.cuda.mem_get_info(self.device)
        memory = v2.memory_info(self.device)
        for key in ('peak_allocated_gib','peak_reserved_gib'):
            self.value[key] = max(self.value[key], memory[key])
        self.value['device_total_gib'] = total/2**30
        self.value.setdefault('initial_device_free_gib', free/2**30)
        self.value['min_device_free_gib'] = min(self.value['min_device_free_gib'] or float('inf'), free/2**30)
        self.value['warning_allocated_above_4_5_gib'] = self.value['peak_allocated_gib'] > 4.5
        if self.path is not None and self.value != self.last:
            v2.atomic_json(self.path,self.value); self.last = copy.deepcopy(self.value)
        require(free >= max(2**30, .10*total),
                'HEADROOM_BLOCKER: require at least max(1 GiB, 10% device capacity) free; do not change bag size/precision.')


def identity(plan, variant, manifests):
    return {'plan_sha256':core.digest(plan), 'configuration':core.config(variant),
            'manifest_sha256':manifests, 'architecture':'ConvNeXtTiny', 'model_label_map':{'NC':0,'AD':1},
            'source_sha256':plan['sources']}


def initial_state(model, optimizer, expected, environment, initial):
    return {'format_version':1, 'identity':expected, 'environment':environment,
            'initial_tensor_sha256':initial, 'epoch':0, 'history':[], 'best':None, 'bad_epochs':0,
            'completed':False, 'stop_reason':None, 'model_state':v2.tensor_state(model),
            'optimizer_state':optimizer.state_dict(), 'rng':v2.rng_state(),
            'custom_rng':{'next_epoch':1,'next_scan_position':0}, 'schedule_position':0,
            'budget_snapshot':None, 'epoch_loop_seconds':[], 'order_augmentation_digests':[]}


def advance(state, model, optimizer, train_bags, val_bags, fetch_train, fetch_val, device,
            variant, epoch, resource=None, operations=None):
    """A full epoch transaction. Used unchanged by the synthetic recovery test."""
    core = operations if operations is not None else globals()['core']
    require(epoch == state['epoch']+1, 'Recovery must repeat from the complete epoch boundary.')
    matched = core.schedule(train_bags, epoch)
    # Separate model stochastic-depth stream from order/augmentation streams.
    torch.manual_seed(core.seed_for('model_training',epoch))
    if device.type == 'cuda': torch.cuda.manual_seed_all(core.seed_for('model_training',epoch))
    v2.set_epoch(model,optimizer,core.base_config(str(device)),epoch)
    trained = core.train_epoch(model,train_bags,fetch_train,optimizer,device,variant,matched,resource)
    metrics, predictions = core.predict(model,val_bags,fetch_val,device,variant,resource)
    improved = state['best'] is None or v2.selection_key(metrics) > v2.selection_key(state['best']['metrics'])
    if improved:
        state['best'] = {'epoch':epoch,'metrics':metrics,'model_state':v2.tensor_state(model),
                         'predictions':predictions}
        state['bad_epochs'] = 0
    else: state['bad_epochs'] += 1
    m = metrics['patient']
    row = {'epoch':epoch,'learning_rate':optimizer.param_groups[0]['lr'], **trained,
           'patient_accuracy':m['accuracy'],'patient_macro_f1':m['macro_f1'],
           'patient_auroc':m['auroc'],'patient_log_loss':m['log_loss'],'patient_brier':m['brier'],
           'ad_recall':m['per_class']['AD']['recall'],'nc_recall':m['per_class']['NC']['recall'],
           'scan_accuracy':metrics['scan']['accuracy'],'scan_log_loss':metrics['scan']['log_loss'],
           'slice_accuracy_diagnostic':metrics['slice']['accuracy'],'slice_log_loss_diagnostic':metrics['slice']['log_loss'],
           'selected_as_best':improved}
    if operations is not None:
        row.update({'validation_attention_'+k+'_mean':v['mean'] for k,v in metrics['attention'].items()})
        row['native_scan_CE'] = metrics['native_scan_CE']
    state['history'].append(row); state['epoch'] = epoch
    state['order_augmentation_digests'].append({'epoch':epoch,'order_digest':matched['order_digest'],
                                              'augmentation_digest':matched['augmentation_digest']})
    state['custom_rng'] = matched; state['schedule_position'] = epoch
    state['model_state'] = v2.tensor_state(model); state['optimizer_state'] = optimizer.state_dict()
    state['rng'] = v2.rng_state()
    if epoch >= 30: state['completed'] = True; state['stop_reason'] = 'MAXIMUM_EPOCHS'
    elif epoch >= 12 and state['bad_epochs'] >= 8:
        state['completed'] = True; state['stop_reason'] = 'PATIENCE_AFTER_MINIMUM_12'
    return row


def finish(folder, state):
    expected = state['identity']; best = state['best']
    checkpoint = {'format_version':'G012-1', 'identity':expected, 'epoch':best['epoch'],
                  'initial_tensor_sha256':state['initial_tensor_sha256'], 'model_state':best['model_state']}
    # Idempotent recovery after a completed state but interrupted artifact write.
    path = folder/'PRIVATE_selected.pt'
    if path.exists():
        old = torch.load(path,map_location='cpu',weights_only=True)
        require(old['identity'] == expected and old['epoch'] == best['epoch']
                and core.tensor_digest(old['model_state']) == core.tensor_digest(best['model_state']),
                'Existing selected checkpoint differs.')
    else: v2.save_state(path,checkpoint)
    st.immutable_json(folder/'PRIVATE_selected_predictions.json',best['predictions'])
    summary = {'identity':expected,'training_completed':True,'stop_reason':state['stop_reason'],
        'selected_epoch':best['epoch'],'completed_epoch':state['epoch'],'metrics':best['metrics'],
        'parameters':sum(t.numel() for t in best['model_state'].values()),
        'initial_tensor_sha256':state['initial_tensor_sha256'],
        'committed_epoch_loop_seconds':sum(state['epoch_loop_seconds']),
        'updates_per_complete_epoch':531, 'slice_metrics_are_diagnostic':True,
        'objective_scale_not_comparable_across_losses':True}
    st.immutable_json(folder/'training_summary.json',summary)
    st.write_text_atomic(folder/'history.csv',st.history_csv(state['history']))
    st.immutable_json(folder/'PRIVATE_order_augmentation_digests.json',state['order_augmentation_digests'])
    names = ('PRIVATE_selected.pt','PRIVATE_selected_predictions.json','training_summary.json','history.csv',
             'PRIVATE_order_augmentation_digests.json')
    st.immutable_json(folder/'completion.json',{'identity':expected,'hashes':{n:st.sha(folder/n) for n in names}})


def worker(args):
    # Parent installs the kill-on-close job before the child is allowed to work.
    while not args.gate.is_file(): time.sleep(.05)
    gate = st.read(args.gate)
    require(gate['released'] is True,'Supervisor gate is not released.')
    plan, root, binding, bundle, groups = st.context(args)
    folder = root/args.variant
    device = v2.choose_device('cuda'); v2.seed_everything(42)
    env = v2.environment(device)
    require(env == plan['expected_environment'], 'Pinned runtime/GPU differs from the frozen study environment.')
    resource = Resources(device,folder/('resource_session-'+uuid.uuid4().hex+'.json'))
    model = v2.make_model(core.base_config()).to(device); optimizer = v2.make_optimizer(model,core.base_config())
    expected = identity(plan,args.variant,bundle.manifest_hashes)
    initial = core.tensor_digest(model.state_dict())
    require(initial == st.read(root/'PRIVATE_matched_schedule.json')['initial_tensor_sha256'], 'Initial tensors differ across variants.')
    if gate['fresh']:
        require(not (folder/'PRIVATE_resume.pt').exists() and not (folder/'PRIVATE_previous.pt').exists(),'Fresh state would overwrite recovery evidence.')
        state = initial_state(model,optimizer,expected,env,initial)
        state['budget_snapshot'] = st.read_budget(folder/'active_budget.json')
        v2.save_state(folder/'PRIVATE_resume.pt',state)
    else: state = st.recover_state(folder,expected,model,optimizer,env)
    require(state['initial_tensor_sha256'] == initial,'Recovery initialization differs.')
    if state['completed']:
        finish(folder,state); return
    for epoch in range(state['epoch']+1,31):
        st.check_disk(root); started = time.perf_counter()
        print(f'{args.variant} epoch {epoch}/30: 1061 full scans, two-scan accumulation, seed 42',flush=True)
        row = advance(state,model,optimizer,groups['train'],groups['validation'],
            lambda bag,angles:core.load_scan(bundle.root,bag,angles),
            lambda bag:core.load_scan(bundle.root,bag),device,args.variant,epoch,resource)
        require(row['optimizer_updates'] == 531,'Unexpected epoch update count.')
        expected_schedule = st.read(root/'PRIVATE_matched_schedule.json')['epochs'][epoch-1]
        require(state['custom_rng']['order_digest'] == expected_schedule['order_digest']
                and state['custom_rng']['augmentation_digest'] == expected_schedule['augmentation_digest'], 'Matched schedule drift.')
        state['epoch_loop_seconds'].append(time.perf_counter()-started)
        # The live external ledger is authoritative and never restored backwards.
        state['budget_snapshot'] = st.read_budget(folder/'active_budget.json')
        st.check_disk(root)
        v2.save_state(folder/'PRIVATE_resume.pt',state)
        st.write_text_atomic(folder/'history.csv',st.history_csv(state['history']))
        print(f"COMMITTED_EPOCH {epoch}; patient accuracy={row['patient_accuracy']:.6f}; selected={state['best']['epoch']}",flush=True)
        if state['completed']:
            finish(folder,state); return


def smoke(args):
    plan, root, binding, bundle, groups = st.context(args)
    with v2.run_lock(root):
        folder = root/('smoke-'+args.variant)
        expected = identity(plan,args.variant,bundle.manifest_hashes)
        if (folder/'smoke.json').exists():
            receipt = st.read(folder/'smoke.json')
            require(receipt['identity'] == expected and receipt['passed'] and receipt['checkpoint_sha256'] == st.sha(folder/'PRIVATE_smoke.pt'),
                    'Existing smoke evidence differs.')
            print('ALREADY_COMPLETED_SMOKE_NON_CANDIDATE',args.variant); return
        require(not folder.exists(),'Incomplete smoke directory exists; preserve it and investigate, do not replace it.')
        st.check_disk(root); folder.mkdir()
        try:
            device = v2.choose_device('cuda'); v2.seed_everything(42)
            require(v2.environment(device) == plan['expected_environment'], 'Smoke runtime differs from the frozen environment.')
            resources = Resources(device,folder/'resources.json')
            bags = groups['train']
            indices = sorted([next(i for i,b in enumerate(bags) if b.label == 1)]
                             + [i for i,b in enumerate(bags) if b.label == 0][:2])
            selected = tuple(bags[i] for i in indices)
            require(len(selected) == 3,'Smoke requires three complete training scans.')
            model = v2.make_model(core.base_config()).to(device)
            initial = core.tensor_digest(model.state_dict())
            require(initial == st.read(root/'PRIVATE_matched_schedule.json')['initial_tensor_sha256'],'Smoke initialization differs.')
            optimizer = v2.make_optimizer(model,core.base_config()); v2.set_epoch(model,optimizer,core.base_config(),1)
            matched = core.schedule(selected,1); started = time.perf_counter()
            result = core.train_epoch(model,selected,lambda bag,a:core.load_scan(bundle.root,bag,a),optimizer,device,args.variant,matched,resources)
            require(result['optimizer_updates'] == 2 and result['training_scans'] == 3,'Smoke did not exercise pair and one-scan tail.')
            model.eval(); sample = core.load_scan(bundle.root,selected[0]).to(device)
            with torch.inference_mode(): logits = model(sample).cpu()
            checkpoint = {'format_version':'G012-1','scope':'NON_CANDIDATE_TRAINING_ONLY_SMOKE',
                'identity':expected,'initial_tensor_sha256':initial,'model_state':v2.tensor_state(model),
                'optimizer_state':optimizer.state_dict()}
            v2.save_state(folder/'PRIVATE_smoke.pt',checkpoint)
            del model,optimizer,checkpoint
            value = torch.load(folder/'PRIVATE_smoke.pt',map_location='cpu',weights_only=True)
            require(value['identity'] == expected,'Smoke checkpoint identity differs.')
            clone = v2.make_model(core.base_config()).to(device); core.strict_state(clone,value['model_state']); clone.eval()
            with torch.inference_mode(): actual = clone(sample).cpu()
            require(torch.equal(logits,actual),'Smoke round-trip logits changed.')
            resources.observe()
            receipt = {'identity':expected,'passed':True,'scope':'NON_CANDIDATE_3_TRAINING_SCANS_60_SLICES_2_UPDATES',
                'result':result,'initial_tensor_sha256':initial,'checkpoint_roundtrip_logits_equal':True,
                'checkpoint_sha256':st.sha(folder/'PRIVATE_smoke.pt'),'resources':resources.value,
                'execution_seconds_excluding_process_startup':time.perf_counter()-started,
                'validation_images_decoded':0,'test_images_decoded':0,'accuracy_used_for_selection':False}
            st.immutable_json(folder/'smoke.json',receipt)
            print('G012_SMOKE_PASSED_NON_CANDIDATE',args.variant,json.dumps(resources.value))
        except BaseException as exc:
            v2.atomic_json(folder/'blocker.json',{'status':'BLOCKED_OOM' if isinstance(exc,torch.cuda.OutOfMemoryError) else 'SMOKE_BLOCKED',
                                                 'exception_type':type(exc).__name__})
            raise


def supervised(args, resume=False, replay_only=False, runtime=None):
    # Optional study services reuse this exact OS-job/lease supervisor. Defaults
    # retain G012 behaviour; its historical byte bindings live in its worktree.
    st = runtime.storage if runtime is not None else globals()['st']
    core = runtime.operations if runtime is not None else globals()['core']
    identity = runtime.identity if runtime is not None else globals()['identity']
    verify_replay = runtime.verify_replay if runtime is not None else globals()['verify_replay']
    plan, root, binding, bundle, groups = st.context(args)
    with v2.run_lock(root):
        expected = identity(plan,args.variant,bundle.manifest_hashes)
        for variant in (() if runtime is not None else core.VARIANTS):
            r = st.read(root/('smoke-'+variant)/'smoke.json')
            require(r['passed'] and r['identity'] == identity(plan,variant,bundle.manifest_hashes)
                    and r['checkpoint_sha256'] == st.sha(root/('smoke-'+variant)/'PRIVATE_smoke.pt'),
                    'All three fixed smokes must pass before real training.')
        if runtime is not None:
            runtime.ready(plan,root,bundle,args.variant)
        else:
            historical = st.read(root/'a1_historical_regression.json')
            require(historical['passed'] and historical['plan_sha256'] == core.digest(plan),'Historical regression check required.')
        folder = root/args.variant
        if replay_only and not folder.exists():
            print('REPLAY_NOT_AVAILABLE_PROTOCOL_INCOMPLETE',args.variant); return
        require(not (resume or replay_only) or folder.is_dir(),'No original attempt exists to resume/replay.')
        fresh = not folder.exists()
        if fresh:
            st.check_disk(root)
            folder.mkdir(); st.immutable_json(folder/'attempt.json',{'attempt_id':uuid.uuid4().hex,'identity':expected})
        else: require(st.read(folder/'attempt.json')['identity'] == expected,'Attempt configuration changed.')
        if (folder/'completion.json').exists():
            st.verify_completion(folder,expected)
            require(st.read(folder/'active_budget.json')['charged_seconds'] < st.CAP,'Completed output arrived after budget exhaustion; protocol inconclusive.')
            if not replay_only:
                old_status = st.read(folder/'run_status.json')['status'] if (folder/'run_status.json').exists() else 'INTERRUPTED'
                if old_status not in STOPPED:
                    v2.atomic_json(folder/'run_status.json',{'status':'COMPLETED_TRAINING','replay_required':not (folder/'checkpoint_replay.json').exists()})
                print('ALREADY_COMPLETED_VERIFIED_SKIPPED',args.variant); return
            if (folder/'checkpoint_replay.json').exists():
                verify_replay(folder,expected)
                print('ALREADY_REPLAYED_VERIFIED',args.variant); return
        elif replay_only:
            print('REPLAY_NOT_AVAILABLE_PROTOCOL_INCOMPLETE',args.variant); return
        if (folder/'run_status.json').exists():
            require(st.read(folder/'run_status.json')['status'] not in STOPPED,'Run has a terminal budget/GPU blocker; do not extend/restart it.')
        if not fresh:
            require((folder/'PRIVATE_resume.pt').is_file() or (folder/'PRIVATE_previous.pt').is_file(),
                    'Existing attempt has no recoverable checkpoint; do not start a replacement.')
        st.check_disk(root)
        used = sum(st.read(p)['charged_seconds'] for p in root.glob('G[012]/active_budget.json' if runtime is None else runtime.budget_glob))
        require(used < st.TOTAL_CAP,'Nine-hour study cap exhausted.' if runtime is None else 'G3 three-hour total active cap exhausted.')
        ledger = Budget(folder/'active_budget.json',cap=st.CAP)
        run_deadline = ledger.deadline - (runtime.replay_reserve_seconds if runtime is not None and not replay_only else 0.)
        gate = folder/('gate-'+uuid.uuid4().hex+'.json'); process = None; job = None; outcome = 'INTERRUPTED'
        v2.atomic_json(folder/'run_status.json',{'status':'RUNNING'})
        try:
            with (folder/'execution.log').open('ab',buffering=0) as log:
                command = [sys.executable,'-B',str(st.BASE/'g012'/'run.py' if runtime is None else runtime.entry),'_replay_worker' if replay_only else '_worker',
                    '--data-root',str(args.data_root),'--splits-dir',str(args.splits_dir),
                    '--study-root',str(root),'--variant',args.variant,'--gate',str(gate)]
                if runtime is not None: command.extend(runtime.worker_arguments(args))
                process = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,
                                           creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                job = WindowsJob(process)
                v2.atomic_json(gate,{'released':True,'fresh':fresh})
                while process.poll() is None:
                    if not ledger.reserve() or time.monotonic() >= run_deadline:
                        job.close(); process.wait(timeout=30); outcome = 'WALL_LIMIT_INCONCLUSIVE'; break
                    time.sleep(min(.5,max(0.,run_deadline-time.monotonic())))
                else:
                    if process.returncode == 0:
                        st.verify_completion(folder,expected)
                        if replay_only: verify_replay(folder,expected)
                        outcome = 'COMPLETED_TRAINING'
                    else:
                        status = st.read(folder/'worker_failure.json')['status'] if (folder/'worker_failure.json').exists() else 'FAILED_NEEDS_INVESTIGATION'
                        outcome = status
                if time.monotonic() >= run_deadline:
                    outcome = 'WALL_LIMIT_INCONCLUSIVE'
        finally:
            if job is not None: job.close()
            if process is not None and process.poll() is None: process.kill(); process.wait(timeout=30)
            ledger.finish(outcome)
            v2.atomic_json(folder/'run_status.json',{'status':outcome,'replay_required':not (folder/'checkpoint_replay.json').exists()})
        print('G012_RUN' if runtime is None else 'G3_RUN',args.variant,outcome,flush=True)
        require(outcome == 'COMPLETED_TRAINING','Stopped at a blocker/incomplete run; preserve evidence and report status.')


def verify_replay(folder,expected):
    r = st.read(folder/'checkpoint_replay.json')
    require(r['passed'] and r['identity'] == expected
            and r['checkpoint_sha256'] == st.sha(folder/'PRIVATE_selected.pt')
            and r['predictions_sha256'] == st.sha(folder/'PRIVATE_replay_predictions.json')
            and r['absolute_probability_tolerance'] == 2e-5
            and r['native_aggregation'] == expected['configuration']['scan_aggregation'], 'Replay receipt binding changed.')
    actual = st.read(folder/'PRIVATE_replay_predictions.json')
    checks = core.replay_difference(st.read(folder/'PRIVATE_selected_predictions.json'),actual)
    require(checks == r['checks'], 'Replay measurements differ from the bound cached outputs.')
    metrics = {name:core.scored([x['label'] for x in actual[key]],[x['p_ad'] for x in actual[key]])
               for name,key in [('slice','slices'),('scan','scans'),('patient','patients')]}
    require(metrics == r['metrics'], 'Replay aggregate metrics differ from its cached probabilities.')
    return r


def replay(args):
    return supervised(args,replay_only=True)


def replay_worker(args):
    while not args.gate.is_file(): time.sleep(.05)
    require(st.read(args.gate)['released'] is True,'Supervisor gate is not released.')
    plan, root, binding, bundle, groups = st.context(args)
    # The shared parent supervisor owns the study lock and budget.
    folder = root/args.variant; expected = identity(plan,args.variant,bundle.manifest_hashes)
    if not (folder/'completion.json').exists():
        print('REPLAY_NOT_AVAILABLE_PROTOCOL_INCOMPLETE',args.variant); return
    st.verify_completion(folder,expected)
    checkpoint_hash = st.sha(folder/'PRIVATE_selected.pt')
    if (folder/'checkpoint_replay.json').exists():
        verify_replay(folder,expected)
        print('ALREADY_REPLAYED_VERIFIED',args.variant); return
    st.check_disk(root); device = v2.choose_device('cuda'); v2.seed_everything(42)
    require(v2.environment(device) == plan['expected_environment'], 'Replay runtime differs from the frozen environment.')
    model = v2.make_model(core.base_config()).to(device)
    checkpoint = torch.load(folder/'PRIVATE_selected.pt',map_location='cpu',weights_only=True)
    require(checkpoint['format_version'] == 'G012-1' and checkpoint['identity'] == expected,'Selected checkpoint configuration/aggregation differs.')
    core.strict_state(model,checkpoint['model_state'])
    resources = Resources(device,folder/'replay_resources.json'); started = time.perf_counter()
    metrics, actual = core.predict(model,groups['validation'],lambda b:core.load_scan(bundle.root,b),device,args.variant,resources)
    checks = core.replay_difference(st.read(folder/'PRIVATE_selected_predictions.json'),actual)
    st.immutable_json(folder/'PRIVATE_replay_predictions.json',actual)
    st.immutable_json(folder/'checkpoint_replay.json',{'identity':expected,'passed':True,
        'checkpoint_sha256':checkpoint_hash,'predictions_sha256':st.sha(folder/'PRIVATE_replay_predictions.json'),
        'absolute_probability_tolerance':2e-5,'checks':checks,'native_aggregation':core.NATIVE[args.variant],
        'metrics':metrics,'resources':resources.value,'inference_seconds':time.perf_counter()-started,
        'test_images_decoded':0})
    print('G012_FULL_VALIDATION_REPLAY_PASSED',args.variant)
