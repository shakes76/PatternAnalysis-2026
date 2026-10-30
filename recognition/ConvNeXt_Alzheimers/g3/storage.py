"""G3 bindings; reuse immutable I/O, manifest checks and complete-epoch recovery."""
from __future__ import annotations
import csv
import io
import json
from pathlib import Path
import subprocess
import time
import torch
import dataset
import train_convnext_v2 as v2
from g012 import core,storage as old
from g012.storage import (read,sha,immutable_json,output_guard,check_disk,read_budget,
                         history_csv,write_text_atomic,recover_state,verify_completion)
from g3 import model

BASE=Path(__file__).resolve().parents[1]
PLAN=BASE/'g3'/'G3_PLAN.json'
CAP=10800.; TOTAL_CAP=10800.; DISK_GIB=12
INITIAL='0a1fa0bc64eb8d5cceee73d246f1272caccfa0fc98790802516532a94679bcfa'
require=core.require


def implementation(committed=True):
    plan=read(PLAN)
    require(plan['control_mode']=='REUSE_G2' and plan['configuration']==model.configuration(), 'Fixed G3 plan/config differs.')
    require(plan['sources']=={n:sha(BASE/n) for n in plan['sources']},'G3 source bindings changed.')
    require(sha(BASE/'APPROVED_PARTITIONS.json')==plan['approved_protocol_sha256'],'Approved protocol differs.')
    if committed:
        names=['recognition/ConvNeXt_Alzheimers/'+n for n in [*plan['sources'],'g3/G3_PLAN.json','g3/G3_PLAN.md','g3/G3_CONTROL_COMPARABILITY.json']]
        subprocess.run(['git','-C',str(BASE.parents[1]),'ls-files','--error-unmatch','--',*names],check=True,stdout=subprocess.DEVNULL)
        subprocess.run(['git','-C',str(BASE.parents[1]),'diff','--exit-code','HEAD','--',*names],check=True,stdout=subprocess.DEVNULL)
    return plan


def verify_control(root,source,plan):
    """Metadata, saved validation arrays and hashes only; never loads an MRI."""
    frozen=read(root/'frozen_plan.json')
    require(core.digest(frozen)==plan['historical_plan_sha256'],'Historical plan differs.')
    require(frozen['sources']=={n:sha(source/n) for n in frozen['sources']},'Historical G012 source differs.')
    folder=root/'G2'; summary=read(folder/'training_summary.json'); expected=summary['identity']
    require(expected['configuration']==core.config('G2') and expected['source_sha256']==frozen['sources']
            and expected['manifest_sha256']==plan['manifest_sha256'],'G2 shared configuration differs.')
    verify_completion(folder,expected)
    from g012.engine import verify_replay
    replay=verify_replay(folder,expected)
    require(read(folder/'run_status.json')['status']=='COMPLETED_TRAINING' and summary['training_completed']
            and summary['selected_epoch']==20 and summary['completed_epoch']==28
            and summary['stop_reason']=='PATIENCE_AFTER_MINIMUM_12'
            and summary['initial_tensor_sha256']==INITIAL,'G2 completion/initialization differs.')
    require(read(folder/'active_budget.json')['charged_seconds']<10800.,'Historical G2 budget exhausted.')
    history=list(csv.DictReader(io.StringIO((folder/'history.csv').read_text(encoding='utf-8-sig'))))
    best=max(history,key=lambda r:(float(r['patient_accuracy']),float(r['patient_macro_f1']),-float(r['patient_log_loss'])))
    require(len(history)==28 and int(best['epoch'])==20 and all(int(r['optimizer_updates'])==531 for r in history),'G2 selection/update history differs.')
    selected=read(folder/'PRIVATE_selected_predictions.json')
    # Reconstruct native G2 aggregation from saved logits, retaining float32.
    rows=selected['slices']; scans={r['scan_id']:[] for r in rows}
    for row in rows: scans[row['scan_id']].append(row['logits'])
    for row in selected['scans']:
        q=float(torch.softmax(torch.tensor(scans[row['scan_id']],dtype=torch.float32).mean(0),0)[1])
        require(abs(q-row['p_ad'])<=2e-7 and (q>=.5)==(row['p_ad']>=.5),'Historical G2 native aggregation differs.')
    names=plan['historical_files']
    hashes={n:sha(root/n) for n in names}
    require(hashes==plan['historical_files'],'Bound historical evidence changed.')
    return {'mode':'REUSE_G2','passed':True,'historical_replay_reused':True,'new_MRI_inference':False,
            'historical_hashes':hashes,'replay_checks':replay['checks'],
            'initial_tensor_digest':INITIAL,'old_epoch_zero_tensor_file_available':False}


def context(args,committed=True):
    plan=implementation(committed); root=output_guard(args.study_root,args.data_root,args.splits_dir)
    binding=read(root/'PRIVATE_study_binding.json')
    require(binding['plan_sha256']==core.digest(plan) and binding['data_root']==str(args.data_root.resolve())
            and binding['splits_dir']==str(args.splits_dir.resolve()) and binding['historical_root']==str(args.historical_root.resolve())
            and binding['historical_source']==str(args.historical_source.resolve()),'G3 explicit input binding differs.')
    verify_control(args.historical_root,args.historical_source,plan)
    bundle=dataset.load_finalized_splits(args.data_root,args.splits_dir)
    require(bundle.manifest_hashes==plan['manifest_sha256'],'Approved manifest hash differs.')
    groups=core.checked_groups(bundle,read(BASE/'APPROVED_PARTITIONS.json'))
    return plan,root,binding,bundle,groups


def prepare(args):
    start=time.perf_counter(); plan=implementation()
    receipt=verify_control(args.historical_root,args.historical_source,plan)
    bundle=dataset.load_finalized_splits(args.data_root,args.splits_dir)
    groups=core.checked_groups(bundle,read(BASE/'APPROVED_PARTITIONS.json'))
    require(bundle.manifest_hashes==plan['manifest_sha256'],'Manifest drift.')
    expected={'plan_sha256':core.digest(plan),'data_root':str(args.data_root.resolve()),
              'splits_dir':str(args.splits_dir.resolve()),'historical_root':str(args.historical_root.resolve()),
              'historical_source':str(args.historical_source.resolve())}
    root=output_guard(args.study_root,args.data_root,args.splits_dir)
    if root.exists():
        require((root/'PRIVATE_study_binding.json').is_file() and read(root/'PRIVATE_study_binding.json')==expected,
                'Conflicting/incomplete G3 output: preserve and investigate.')
    else: check_disk(root.parent); root.mkdir()
    with v2.run_lock(root):
        check_disk(root); immutable_json(root/'PRIVATE_study_binding.json',expected)
        immutable_json(root/'frozen_plan.json',plan)
        immutable_json(root/'control_verification.json',receipt)
        recorded=read(args.historical_root/'PRIVATE_matched_schedule.json')
        require(recorded['initial_tensor_sha256']==INITIAL,'Historical initial digest differs.')
        schedules=[{k:v for k,v in core.schedule(groups['train'],e).items() if k not in ('order_rng_state','angle_rng_state')} for e in range(1,31)]
        require(schedules==recorded['epochs'],'The 30 order/rotation arrays differ.')
        immutable_json(root/'PRIVATE_matched_schedule.json',recorded)
        # Immutable new epoch zero; unlike the historical rotating E0 checkpoint.
        with core.isolated_rng():
            v2.seed_everything(42); candidate=model.make_model()
            shared={k:v for k,v in candidate.state_dict().items() if not k.startswith('attention.')}
            require(core.tensor_digest(shared)==INITIAL,'Reconstructed shared initialization differs.')
            state={'format_version':'G3-E0','shared_digest':INITIAL,'model_state':v2.tensor_state(candidate),'rng':v2.rng_state()}
            path=root/'PRIVATE_initial_G3.pt'
            if path.exists():
                old_state=torch.load(path,map_location='cpu',weights_only=True)
                require(core.tensor_digest(old_state['model_state'])==core.tensor_digest(state['model_state']),'Immutable E0 differs.')
            else: v2.save_state(path,state)
            initial_digest=core.tensor_digest(state['model_state'])
        immutable_json(root/'initialization.json',{'shared_digest':INITIAL,'full_digest':initial_digest,'checkpoint_sha256':sha(path)})
        # A1 evidence stays bound to its historical preparation, no calibration.
        require(sha(args.historical_root/'PRIVATE_a1_reference.json')==plan['historical_files']['PRIVATE_a1_reference.json'],'A1 reference changed.')
        immutable_json(root/'PRIVATE_a1_reference.json',read(args.historical_root/'PRIVATE_a1_reference.json'))
        if not (root/'preparation.json').exists():
            immutable_json(root/'preparation.json',{'passed':True,'images_decoded':0,'test_manifest_scope':'hashes/grouping/disjointness only',
                'execution_seconds':time.perf_counter()-start,'plan_sha256':core.digest(plan),'initial_checkpoint_sha256':sha(path)})
    print('G3_PREPARED_REUSE_G2_NO_IMAGES_DECODED')
