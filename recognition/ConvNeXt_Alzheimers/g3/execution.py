"""G3-specific model/output adapters; inherited epoch transaction and supervisor."""
from __future__ import annotations
import copy
import time
import uuid
from types import SimpleNamespace
import torch
import train_convnext_v2 as v2
from g012 import core,engine as inherited
from g3 import model as gm,operations as ops,storage as st
require=core.require


def identity(plan,variant,manifests):
    require(variant=='G3','REUSE_G2 permits only one new G3 seed-42 run; no U3.')
    return {'plan_sha256':core.digest(plan),'configuration':gm.configuration(),'manifest_sha256':manifests,
            'architecture':'ConvNeXtTiny_GatedAttention_128','model_label_map':{'NC':0,'AD':1},
            'control_mode':'REUSE_G2','source_sha256':plan['sources']}


def ready(plan,root,bundle,variant):
    require(variant=='G3' and st.read(root/'control_verification.json')['passed'],'Control verification required.')
    r=st.read(root/'smoke-G3'/'smoke.json')
    require(r['passed'] and r['identity']==identity(plan,variant,bundle.manifest_hashes)
            and r['checkpoint_sha256']==st.sha(root/'smoke-G3'/'PRIVATE_smoke.pt'),'Bound G3 smoke required.')
    require(st.sha(root/'PRIVATE_initial_G3.pt')==st.read(root/'initialization.json')['checkpoint_sha256'],'Immutable epoch zero changed.')
    status=root/'G3'/'run_status.json'
    if status.exists():
        require(st.read(status)['status'] in ('RUNNING','INTERRUPTED','DISK_BLOCKED','COMPLETED_TRAINING'),
                'Terminal wall/GPU/integrity/nonfinite blocker: preserve evidence and investigate; no automatic continuation.')


def runtime():
    return SimpleNamespace(storage=st,operations=ops,identity=identity,verify_replay=verify_replay,
        ready=ready,budget_glob='G3/active_budget.json',entry=st.BASE/'g3'/'run.py',replay_reserve_seconds=300.,
        worker_arguments=lambda a:['--historical-root',str(a.historical_root),'--historical-source',str(a.historical_source)])


def supervised(args,resume=False,replay_only=False):
    return inherited.supervised(args,resume,replay_only,runtime=runtime())


def finish(folder,state):
    best=state['best']; expected=state['identity']
    value={'format_version':'G3-1','identity':expected,'epoch':best['epoch'],
           'initial_tensor_sha256':state['initial_tensor_sha256'],'model_state':best['model_state']}
    path=folder/'PRIVATE_selected.pt'
    if path.exists():
        old=torch.load(path,map_location='cpu',weights_only=True)
        require(old['identity']==expected and old['epoch']==best['epoch']
                and core.tensor_digest(old['model_state'])==core.tensor_digest(value['model_state']),'Existing selected weights differ.')
    else:v2.save_state(path,value)
    st.immutable_json(folder/'PRIVATE_selected_predictions.json',best['predictions'])
    st.immutable_json(folder/'training_summary.json',{'identity':expected,'training_completed':True,
        'stop_reason':state['stop_reason'],'selected_epoch':best['epoch'],'completed_epoch':state['epoch'],
        'metrics':best['metrics'],'parameters':28015586,'initial_tensor_sha256':state['initial_tensor_sha256'],
        'shared_initial_tensor_sha256':st.INITIAL,'committed_epoch_loop_seconds':sum(state['epoch_loop_seconds']),
        'updates_per_complete_epoch':531,'slice_metrics_are_diagnostic':True})
    st.write_text_atomic(folder/'history.csv',st.history_csv(state['history']))
    st.immutable_json(folder/'PRIVATE_order_augmentation_digests.json',state['order_augmentation_digests'])
    names=('PRIVATE_selected.pt','PRIVATE_selected_predictions.json','training_summary.json','history.csv','PRIVATE_order_augmentation_digests.json')
    st.immutable_json(folder/'completion.json',{'identity':expected,'hashes':{n:st.sha(folder/n) for n in names}})


def gate(args):
    while not args.gate.is_file():time.sleep(.05)
    r=st.read(args.gate); require(r['released'] is True,'Supervisor gate closed.'); return r


def initialize(root,device):
    v2.seed_everything(42); net=gm.make_model().to(device)
    require(core.tensor_digest(net.state_dict())==st.read(root/'initialization.json')['full_digest'],'New initial state differs.')
    require(core.tensor_digest({k:v for k,v in net.state_dict().items() if not k.startswith('attention.')})==st.INITIAL,'Shared state differs.')
    return net


def worker(args):
    released=gate(args); plan,root,binding,bundle,groups=st.context(args)
    folder=root/'G3'; device=v2.choose_device('cuda');v2.seed_everything(42);env=v2.environment(device)
    require(env==plan['expected_environment'],'Pinned runtime/GPU differs.')
    resources=inherited.Resources(device,folder/('resource_session-'+uuid.uuid4().hex+'.json'))
    net=initialize(root,device); optimizer=v2.make_optimizer(net,core.base_config())
    expected=identity(plan,'G3',bundle.manifest_hashes); initial=core.tensor_digest(net.state_dict())
    if released['fresh']:
        require(not (folder/'PRIVATE_resume.pt').exists() and not (folder/'PRIVATE_previous.pt').exists(),'Fresh attempt conflicts.')
        state=inherited.initial_state(net,optimizer,expected,env,initial)
        state['budget_snapshot']=st.read_budget(folder/'active_budget.json'); v2.save_state(folder/'PRIVATE_resume.pt',state)
    else:state=st.recover_state(folder,expected,net,optimizer,env)
    require(state['initial_tensor_sha256']==initial,'Recovery initial identity differs.')
    if state['completed']:finish(folder,state); return
    schedules=st.read(root/'PRIVATE_matched_schedule.json')['epochs']
    for epoch in range(state['epoch']+1,31):
        st.check_disk(root); start=time.perf_counter()
        print('G3_EPOCH',epoch,flush=True)
        row=inherited.advance(state,net,optimizer,groups['train'],groups['validation'],
            lambda b,a:core.load_scan(bundle.root,b,a),lambda b:core.load_scan(bundle.root,b),device,'G3',epoch,resources,operations=ops)
        require(row['optimizer_updates']==531,'Unexpected update count.')
        wanted=schedules[epoch-1]; actual=state['custom_rng']
        require(all(actual[k]==wanted[k] for k in ('order_digest','augmentation_digest')),'Schedule drift.')
        state['epoch_loop_seconds'].append(time.perf_counter()-start)
        state['budget_snapshot']=st.read_budget(folder/'active_budget.json'); st.check_disk(root)
        v2.save_state(folder/'PRIVATE_resume.pt',state); st.write_text_atomic(folder/'history.csv',st.history_csv(state['history']))
        print('COMMITTED_EPOCH',epoch,'SELECTED',state['best']['epoch'],flush=True)
        if state['completed']:finish(folder,state);return


def verify_replay(folder,expected):
    r=st.read(folder/'checkpoint_replay.json')
    require(r['passed'] and r['identity']==expected and r['checkpoint_sha256']==st.sha(folder/'PRIVATE_selected.pt')
            and r['predictions_sha256']==st.sha(folder/'PRIVATE_replay_predictions.json')
            and r['absolute_probability_tolerance']==2e-5 and r['attention_absolute_tolerance']==2e-5
            and r['native_aggregation']==gm.configuration()['scan_aggregation'],'G3 replay binding differs.')
    actual=st.read(folder/'PRIVATE_replay_predictions.json')
    selected=st.read(folder/'PRIVATE_selected_predictions.json');ops.verify_native(selected);ops.verify_native(actual)
    require(ops.replay_difference(selected,actual)==r['checks'],'Replay checks changed.')
    for name,key in [('slice','slices'),('scan','scans'),('patient','patients')]:
        require(core.scored([x['label'] for x in actual[key]],[x['p_ad'] for x in actual[key]])==r['metrics'][name],
                'Replay metrics differ from cached predictions.')
    return r


def replay_worker(args):
    gate(args); plan,root,binding,bundle,groups=st.context(args);folder=root/'G3'
    expected=identity(plan,'G3',bundle.manifest_hashes);st.verify_completion(folder,expected)
    device=v2.choose_device('cuda');v2.seed_everything(42)
    require(v2.environment(device)==plan['expected_environment'],'Replay environment differs.')
    net=gm.make_model().to(device)
    gm.strict_checkpoint(net,torch.load(folder/'PRIVATE_selected.pt',map_location='cpu',weights_only=True),expected)
    resources=inherited.Resources(device,folder/'replay_resources.json');start=time.perf_counter()
    metrics,predictions=ops.predict(net,groups['validation'],lambda b:core.load_scan(bundle.root,b),device,'G3',resources)
    checks=ops.replay_difference(st.read(folder/'PRIVATE_selected_predictions.json'),predictions)
    st.immutable_json(folder/'PRIVATE_replay_predictions.json',predictions)
    st.immutable_json(folder/'checkpoint_replay.json',{'identity':expected,'passed':True,'checkpoint_sha256':st.sha(folder/'PRIVATE_selected.pt'),
        'predictions_sha256':st.sha(folder/'PRIVATE_replay_predictions.json'),'absolute_probability_tolerance':2e-5,
        'attention_absolute_tolerance':2e-5,'native_aggregation':gm.configuration()['scan_aggregation'],'checks':checks,
        'metrics':metrics,'resources':resources.value,'inference_seconds':time.perf_counter()-start,'test_images_decoded':0})
    print('G3_FULL_VALIDATION_REPLAY_PASSED')


def smoke(args):
    start=time.perf_counter();plan,root,binding,bundle,groups=st.context(args)
    with v2.run_lock(root):
        folder=root/'smoke-G3';expected=identity(plan,'G3',bundle.manifest_hashes)
        if (folder/'smoke.json').exists():
            r=st.read(folder/'smoke.json');require(r['passed'] and r['identity']==expected
                and r['checkpoint_sha256']==st.sha(folder/'PRIVATE_smoke.pt'),'Smoke receipt differs.')
            print('ALREADY_COMPLETED_SMOKE_NON_CANDIDATE');return
        require(not folder.exists(),'Incomplete smoke exists; preserve and investigate.')
        st.check_disk(root);folder.mkdir()
        try:
            device=v2.choose_device('cuda');v2.seed_everything(42)
            require(v2.environment(device)==plan['expected_environment'],'Smoke environment differs.')
            resources=inherited.Resources(device,folder/'resources.json');net=initialize(root,device)
            optimizer=v2.make_optimizer(net,core.base_config());v2.set_epoch(net,optimizer,core.base_config(),1)
            bags=groups['train'];indices=sorted([next(i for i,b in enumerate(bags) if b.label==1)]+[i for i,b in enumerate(bags) if b.label==0][:2])
            selected=tuple(bags[i] for i in indices);require(len(selected)==3,'Three predetermined training scans required.')
            # Verify initial uniformity on the first training scan in the smoke,
            # not a new validation compatibility inference.
            sample=core.load_scan(bundle.root,selected[0]).to(device)
            net.eval()
            with torch.inference_mode():z,s,a=net(sample)
            require(torch.allclose(a,torch.full_like(a,1/len(a)),atol=1e-7,rtol=0),'Initial attention is not uniform.')
            initial=core.tensor_digest(net.state_dict())
            result=ops.train_epoch(net,selected,lambda b,angles:core.load_scan(bundle.root,b,angles),optimizer,device,'G3',core.schedule(selected,1),resources)
            require(result['optimizer_updates']==2 and core.tensor_digest(net.state_dict())!=initial,'Smoke pair/tail/update failure.')
            original=torch.load(root/'PRIVATE_initial_G3.pt',map_location='cpu',weights_only=True)['model_state']
            current=v2.tensor_state(net)
            changed={category:any(not torch.equal(v,original[k]) for k,v in current.items() if
                     (k.startswith('attention.') if category=='attention' else k.startswith('head.') if category=='head'
                      else not k.startswith(('head.','attention.')))) for category in ('backbone','head','attention')}
            require(all(changed.values()),'Smoke did not update backbone/head/attention.')
            del original,current
            value={'format_version':'G3-1','identity':expected,'epoch':0,'initial_tensor_sha256':initial,'model_state':v2.tensor_state(net)}
            net.eval()
            with torch.inference_mode():before=tuple(x.cpu() for x in net(sample))
            v2.save_state(folder/'PRIVATE_smoke.pt',value);del net,optimizer,value
            clone=gm.make_model().to(device);gm.strict_checkpoint(clone,torch.load(folder/'PRIVATE_smoke.pt',map_location='cpu',weights_only=True),expected);clone.eval()
            with torch.inference_mode():after=tuple(x.cpu() for x in clone(sample))
            require(all(torch.equal(a,b) for a,b in zip(before,after)),'Smoke roundtrip differs.')
            resources.observe()
            st.immutable_json(folder/'smoke.json',{'passed':True,'identity':expected,'scope':'NON_CANDIDATE_3_FIXED_TRAIN_SCANS_PAIR_AND_TAIL',
                'result':result,'finite_loss_and_all_gradient_norms':True,'updated_parameter_groups':changed,
                'initial_uniformity_passed':True,'checkpoint_roundtrip_logits_equal':True,
                'checkpoint_sha256':st.sha(folder/'PRIVATE_smoke.pt'),'resources':resources.value,
                'execution_seconds_excluding_process_startup':time.perf_counter()-start,'test_images_decoded':0,'validation_images_decoded':0})
            print('G3_SMOKE_PASSED_NON_CANDIDATE')
        except BaseException as exc:
            v2.atomic_json(folder/'blocker.json',{'status':'SMOKE_BLOCKED','exception_type':type(exc).__name__});raise
