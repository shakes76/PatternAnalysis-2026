"""Prediction from explicit inputs, without author-only run/report pointers.

Only train/validation inference is exposed here. Historical test evaluation stays
in its separately guarded commands and frozen revision. No fitting is performed.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import sys

BASE=Path(__file__).resolve().parent
def require(ok,message):
    if not ok:raise ValueError(message)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def source_directory(value,explicit=None):
    expected=value.get('source_code_sha256')
    allowed={'dataset.py','modules.py','modules_v2.py','training_utils.py','train_convnext_v2.py',
             'V2_EXPERIMENT_PLAN.md','train.py','train_experiment.py','check_experiment.py','check_baseline.py'}
    require(isinstance(expected,dict) and bool(expected) and set(expected)<=allowed,
            'Unsupported source provenance; do not silently waive missing checks.')
    candidates=[Path(explicit).resolve()] if explicit else [BASE,BASE/'historical_runtime']
    for directory in candidates:
        if all((directory/name).is_file() and sha(directory/name)==digest for name,digest in expected.items()):
            return directory
    raise ValueError('Checkpoint source bytes do not match this checkout or the exact historical runtime. '
                     'Use its verified original source snapshot; never rewrite the checkpoint hashes.')

def strict_model(value,modules):
    import torch
    require(value.get('model_label_map')=={'NC':0,'AD':1},'Unexpected checkpoint labels.')
    architecture=value.get('architecture');config=value.get('configuration',{})
    if architecture=='ConvNeXtTiny':
        require(value.get('format_version')==2 and value.get('pretrained')=={'used':False}
                and config.get('recipe')=='scratch_augmented','Expected scratch A1-family checkpoint.')
        model=modules.ConvNeXtTiny()
    elif architecture=='BaselineCNN-v1':
        require(value.get('model_name')=='baseline' and config.get('model')=='baseline','Unsupported CNN checkpoint.')
        model=modules.BaselineCNN()
    else:raise ValueError('Unsupported architecture; no automatic conversion or pretrained adapter.')
    expected=model.state_dict();actual=value.get('model_state')
    require(isinstance(actual,dict) and set(actual)==set(expected),'Checkpoint state keys differ.')
    for name,target in expected.items():
        tensor=actual[name]
        require(isinstance(tensor,torch.Tensor) and tensor.shape==target.shape and tensor.dtype==target.dtype,
                f'Checkpoint tensor shape/dtype mismatch: {name}')
        require(bool(torch.isfinite(tensor).all()),'Non-finite checkpoint tensor.')
    model.load_state_dict(actual,strict=True);return model.eval()

def frozen_rules(args,value,manifests,data):
    if args.rules is None:
        require(args.policy_rules is None,'Policy rules require explicit base calibration rules.')
        return None,None
    require(args.rules_sha256 is not None and sha(args.rules)==args.rules_sha256,'Frozen rules checksum differs.')
    rules=read(args.rules)
    require(rules.get('manifest_sha256')==manifests and rules.get('preprocessing')==data.preprocessing_spec(0.),
            'Frozen calibration uses different manifests/preprocessing.')
    alias='a1' if value['architecture']=='ConvNeXtTiny' else 'cnn'
    entry=rules['models'][alias]
    require(entry['checkpoint_sha256']==args.checkpoint_sha256 and entry['selected_epoch']==value['epoch']
            and entry['configuration']==value['configuration'] and entry['architecture']==value['architecture']
            and entry['diagnostic_threshold']==.5,'Frozen calibration is bound to a different selected checkpoint.')
    t=entry['temperature'];tau=entry['confidence_threshold']
    require(isinstance(t,(int,float)) and math.isfinite(t) and .5<=t<=10,'Invalid frozen temperature.')
    require(tau is None or (isinstance(tau,(int,float)) and math.isfinite(tau) and .5<=tau<=1),'Invalid frozen cutoff.')
    policy_entries=None
    if args.policy_rules is not None:
        require(args.policy_rules_sha256 is not None and sha(args.policy_rules)==args.policy_rules_sha256,
                'Frozen policy file checksum differs.')
        policies=read(args.policy_rules);p=policies['models'][alias]
        require(policies['primary_classification_cutoff']==.5 and p['checkpoint_sha256']==args.checkpoint_sha256
                and p['selected_epoch']==value['epoch'] and p['temperature']==t,'Policy binding differs.')
        policy_entries=p['policies'];require(set(policy_entries)=={'reference','strict_cn','no_auto_cn'},'Unexpected policy set.')
        for name,rule in policy_entries.items():
            require(rule['original_confidence_threshold']==tau,'Policy changed the original cutoff.')
            require(rule['additional_CN_probability_cap']==(.05 if name=='strict_cn' else None)
                    and rule['automatic_CN_disabled']==(name=='no_auto_cn'),'Policy differs from the frozen definitions.')
    return entry,policy_entries

def action(q,tau,cap=None,disabled=False):
    accepted=tau is not None and max(q,1-q)>=tau
    if accepted and q>=.5:return 'AD_flag'
    if accepted and q<.5 and not disabled and (cap is None or q<=cap):return 'auto_CN'
    return 'refer'

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--checkpoint-sha256',required=True)
    p.add_argument('--data-root',type=Path,required=True);p.add_argument('--splits-dir',type=Path,required=True)
    p.add_argument('--split',choices=('train','validation'),required=True)
    p.add_argument('--input-list',type=Path,help='UTF-8 relative paths; must select complete patients, not partial scan bags.')
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    p.add_argument('--source-dir',type=Path);p.add_argument('--rules',type=Path);p.add_argument('--rules-sha256')
    p.add_argument('--policy-rules',type=Path);p.add_argument('--policy-rules-sha256')
    args=p.parse_args(argv)
    import torch
    require(sha(args.checkpoint)==args.checkpoint_sha256,'Checkpoint bytes differ from the supplied checksum.')
    value=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    source=source_directory(value,args.source_dir)
    # Dedicated CLI starts before importing dataset/model modules. Refuse mixed
    # namespaces rather than claiming old source while executing edited modules.
    for name in ('dataset','modules','modules_v2','training_utils'):
        require(name not in sys.modules,'Run prediction in a fresh process, not a mixed module namespace.')
    sys.path.insert(0,str(source))
    import dataset as data
    import modules
    import training_utils as utils
    for name,module in [('dataset',data),('modules',modules),('training_utils',utils)]:
        require(Path(module.__file__).resolve().parent==source,'Loaded source is outside the verified runtime.')
    bundle=data.load_finalized_splits(args.data_root,args.splits_dir)
    require(value.get('manifest_sha256')==bundle.manifest_hashes,'Checkpoint and input manifest identities differ.')
    records=bundle.records[args.split]
    if args.input_list is not None:
        paths=args.input_list.read_text(encoding='utf-8-sig').splitlines()
        require(bool(paths) and len(set(paths))==len(paths),'Input list is empty or repeats paths.')
        indexed={r.image_path:r for r in records};require(set(paths)<=set(indexed),'Input list is outside the selected partition.')
        subjects={indexed[path].subject_id for path in paths}
        selected=tuple(r for r in records if r.subject_id in subjects)
        require(set(paths)=={r.image_path for r in selected},'Select all slices/scans of each chosen patient for hierarchical aggregation.')
        records=selected
        bundle=replace(bundle,records={**bundle.records,args.split:records})
    out=args.output_dir.resolve()
    require(not out.exists(),'Use a NEW prediction output directory.')
    for protected in (args.data_root.resolve(),args.splits_dir.resolve(),BASE):
        require(out!=protected and protected not in out.parents,'Output is inside protected inputs/source.')
    require(not any((d/'.git').exists() for d in (out,*out.parents)),'Patient predictions must stay outside Git.')
    rules,policies=frozen_rules(args,value,bundle.manifest_hashes,data)
    utils.seed_everything(42);device=utils.choose_device(args.device)
    model=strict_model(value,modules).to(device)
    dataset=data.ADNISliceDataset(bundle,args.split)
    loader=data.make_loader(dataset,batch_size=8,num_workers=0,shuffle=False,pin_memory=device.type=='cuda')
    probabilities=[];paths=[];started=__import__('time').perf_counter()
    with torch.inference_mode():
        for batch in loader:
            logits=model(batch['image'].to(device))
            require(bool(torch.isfinite(logits).all()),'Non-finite prediction logits.')
            probabilities.extend(torch.softmax(logits,1)[:,1].cpu().tolist());paths.extend(batch['image_path'])
    utils.synchronize(device);seconds=__import__('time').perf_counter()-started
    require(paths==[r.image_path for r in records],'Prediction coverage/order differs.')
    metrics,private=utils.aggregate_predictions(records,probabilities)
    if rules is not None:
        confidence_path=source/'confidence_analysis.py'
        expected=read(BASE/'HISTORICAL_RUNTIME.json')['sha256']['confidence_analysis.py']
        require(sha(confidence_path)==expected,'Calibration application source changed.')
        from confidence_analysis import scaled_probabilities
        raw=[r['p_ad'] for r in private['patients']];q=scaled_probabilities(raw,rules['temperature']).tolist()
        require(all((a>=.5)==(b>=.5) for a,b in zip(raw,q)),'Frozen scaling changed class decisions.')
        for row,prob in zip(private['patients'],q):
            row['calibrated_p_ad']=prob
            row['actions']={'reference':action(prob,rules['confidence_threshold'])}
            if policies is not None:
                row['actions']={name:action(prob,v['original_confidence_threshold'],
                    v['additional_CN_probability_cap'],v['automatic_CN_disabled']) for name,v in policies.items()}
    receipt={'scope':args.split,'slice_count':len(records),'patient_count':len(private['patients']),
        'selected_epoch':value['epoch'],'architecture':value['architecture'],'checkpoint_sha256':args.checkpoint_sha256,
        'manifest_sha256':bundle.manifest_hashes,'verified_source_directory':str(source),
        'verified_source_sha256':value['source_code_sha256'],'preprocessing':data.preprocessing_spec(0.),
        'metrics':metrics,'inference_seconds':seconds,'calibration_fitted':False,'threshold_search':False,
        'frozen_rules_applied':rules is not None,'frozen_policy_comparison_applied':policies is not None,
        'test_images_decoded':0,'historical_report_tree_required':False}
    out.mkdir(parents=True)
    (out/'PRIVATE_predictions.json').write_text(json.dumps(private,indent=2)+'\n',encoding='utf-8')
    (out/'prediction_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt,indent=2));return receipt

if __name__=='__main__':main()
