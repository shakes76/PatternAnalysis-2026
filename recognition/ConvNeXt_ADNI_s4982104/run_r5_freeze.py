"""Freeze accepted validation decisions and rehearse with synthetic patients only."""
import argparse
import ast
import copy
import csv
import json
from pathlib import Path
import shutil
import tempfile
from collections import Counter
import numpy as np
from final_evaluation import digest, manifest_records, evaluate, load_protocol
from run_r5_ensemble import MODELS, SEEDS, require, sha

ROOT=Path(__file__).resolve().parent
SOURCES={
'r5_ensemble_validation/ensemble_summary.json':'cd2e5f92fd07b7154b3ba04ffdfe65f3aedca02a02577e5eb5b64c0c9cf2d109',
'r5_temperature_validation/temperature_summary.json':'0c13afa9f7b35b36d5fb04dacb3ff0ba4ad9e8d212fa10075c8adc51e27c7bd9',
'r5_ensemble_thresholds/threshold_summary.json':'e8f29b5319dfe1b97d71a1e7f703fb8ac778ed704fa8e3b419b2cbfa9ecd3dc2',
'r5_referral_validation/referral_summary.json':'60ef5534cbb85cabc309424aeec6981f605f34c2cc4def08c1acdfcfeb8c63f3'}

def code_inventory():
    pending=['final_evaluation.py','run_r5_freeze.py','predict.py'];seen=set()
    while pending:
        name=pending.pop()
        if name in seen:continue
        seen.add(name)
        for node in ast.walk(ast.parse((ROOT/name).read_text())):
            modules=([node.module] if isinstance(node,ast.ImportFrom) else [a.name for a in node.names] if isinstance(node,ast.Import) else [])
            for mod in modules:
                local=(mod or '').split('.')[0]+'.py'
                if (ROOT/local).is_file():pending.append(local)
    return {name:sha(ROOT/name) for name in sorted(seen|{'requirements.txt','r4_frozen_bests.json'})}

def build(evidence):
    evidence=Path(evidence);accepted=[]
    for name,value in SOURCES.items():
        p=evidence/name;require(sha(p)==value,'Accepted summary SHA mismatch: '+name)
        summary=json.loads(p.read_text());require(summary['status']=='passed' and summary['split']=='validation','Unaccepted source')
        for output,expected in summary['output_sha256'].items():
            require(sha(p.parent/output)==expected,'Accepted output SHA mismatch: '+output)
        hashes=summary.get('code_sha256',{})
        if isinstance(hashes,str):hashes={'run_r5_ensemble.py':hashes}
        for name,value in hashes.items():require(sha(ROOT/name)==value,'Accepted code SHA mismatch: '+name)
        accepted.append(summary)
    _,temperature,threshold,referral=accepted
    require(temperature['temperatures']==threshold['temperatures']==referral['temperatures'],'Temperature source mismatch')
    lock_path=ROOT/'r4_frozen_bests.json'
    require(sha(lock_path)=='d781fb6bec3f9e378aaf6ddfa08229aadb2fc3c7361159f01e87b3be0106ebb4','Weight lock changed')
    lock=json.loads(lock_path.read_text())
    for name,value in lock['data_files'].items():require(sha(evidence/name)==value,'Split SHA mismatch')
    with (evidence/'split_seed42_60_20_20/split_samples.csv').open() as f:rows=list(csv.DictReader(f))
    rows=manifest_records(rows);patients={}
    for r in rows:
        identity=(r['split'],r['image_id'],r['label'])
        require(patients.setdefault(r['patient_id'],identity)==identity,'Patient leakage/scan conflict')
    require(Counter(r['patient_id'] for r in rows)==Counter({p:20 for p in patients}),'Wrong slice counts')
    require(Counter((s,y) for s,_,y in patients.values())==Counter({('train',0):275,('train',1):133,('validation',0):92,('validation',1):44,('test',0):92,('test',1):44}),'Wrong split counts')
    test=[r for r in rows if r['split']=='test']
    from dataset import PREPROCESSING,CONVNEXT_PREPROCESSING
    return dict(schema=1,scope='frozen_final_evaluation',models=list(MODELS),seeds=list(SEEDS),
        primary_model='ConvNeXt-F',source_summaries_sha256=SOURCES,split_files_sha256=lock['data_files'],
        checkpoints=lock['runs'],selection_rule='patient_nll',
        temperatures=temperature['temperatures'],supplemental_thresholds=threshold['thresholds'],
        cohort=dict(split='test',patients=136,ad=44,nc=92,slices=2720,manifest_records_sha256=digest(test)),
        aggregation=dict(slices_per_patient=20,seed_weights=[1/3]*3),
        inference=dict(batch_size=4,precision='float32',tf32=False,amp=False,compile=False,num_workers=0,cpu_threads=2,
            cudnn_benchmark=False,deterministic=True,memory_format='contiguous',mode='inference_mode',shuffle=False,drop_last=False,
            preprocessing={'CNN-A':PREPROCESSING,'CNN-B':PREPROCESSING,'ConvNeXt-F':CONVNEXT_PREPROCESSING}),
        evaluation=dict(main_threshold=.5,nll_epsilon=1e-7,ece_bins=10,referral_rates=[0,.05,.1,.2],primary_referral_rate=.1,
            bootstrap_replicates=2000,bootstrap_seed=42,confidence_level=.95,quantile_method='linear'),
        referral=dict(ranking='abs(raw ensemble p - 0.5) ascending then patient_id lexicographic',rounding='ceil(rate*N)',
            selection_inputs=['patient_id','p_ad_raw'],human_correction_assumed=False,zero_denominator=None),
        reporting=dict(primary='full-cohort calibrated ConvNeXt-F ensemble at 0.5',
            metrics=['accuracy','ad_recall','nc_specificity','macro_f1','auroc','nll','brier','confidence_ece_at_0_5'],
            seed_sd='sample SD, separate from patient bootstrap intervals',
            bootstrap='2000 stratified patient draws, PCG64 seed42, NC then AD; shared indices; percentile 95%',
            uncertainty_scope='conditional on fixed models, calibration, thresholds and class counts; ECE/referral descriptive',
            test_policy='apply frozen rules once; no fitting, threshold search, model selection or human-oracle correction'),
        runtime_reference=dict(python='3.11.13',torch='2.14.0+cu130',torchvision='0.29.0',cuda='13.0',gpu='NVIDIA A100-PCIE-40GB'),
        code_sha256=code_inventory(),physical_checkpoint_verification='required on cluster before R6 inference; not claimed locally')

def synthetic(protocol):
    p=copy.deepcopy(protocol)
    manifest=[dict(file_path=f'synthetic/{i:03d}/{j:02d}.jpeg',patient_id=f'synthetic_{i:03d}',image_id=f'fake_{i:03d}',label=int(i<44),split='test') for i in range(136) for j in range(20)]
    p['cohort']=dict(split='test',patients=136,ad=44,nc=92,slices=2720,manifest_records_sha256=digest(manifest_records(manifest)))
    rng=np.random.default_rng(42);runs=[]
    for entry in p['checkpoints']:
        probabilities=rng.uniform(.05,.95,len(manifest))
        slices=[dict(r,p_ad=float(v),prediction=int(v>=.5)) for r,v in zip(manifest,probabilities)]
        runs.append(dict(entry=entry,protocol_sha256=digest(p),inference=p['inference'],slices=slices))
    report=evaluate(p,manifest,runs)
    return dict(evidence_kind='synthetic_only_not_model_performance',real_image_reads=0,
        frozen_protocol_digest=digest(protocol),synthetic_protocol_digest=digest(p),report=report)

def publish(destination,payload):
    destination=Path(destination)
    if destination.exists():raise FileExistsError(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    temp=Path(tempfile.mkdtemp(prefix='.r5e-',dir=destination.parent))
    try:
        (temp/'summary.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
        if 'protocol' in payload:
            config=temp/'r5_final_protocol.json'
            config.write_text(json.dumps(payload['protocol'],indent=2,allow_nan=False)+'\n')
            (temp/'r5_final_protocol.sha256').write_text(sha(config)+'  r5_final_protocol.json\n')
        temp.rename(destination)
    finally:
        if temp.exists():shutil.rmtree(temp)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-root',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--verify-protocol',type=Path)
    parser.add_argument('--expected-sha256')
    args=parser.parse_args()
    require(not args.output.exists(),'Output already exists')
    if args.verify_protocol:
        require(bool(args.expected_sha256),'Explicit expected SHA required')
        protocol=load_protocol(args.verify_protocol,args.expected_sha256,ROOT)
    else:
        require(args.evidence_root is not None,'Evidence root required')
        protocol=build(args.evidence_root)
    payload=synthetic(protocol)
    require(payload==synthetic(protocol),'Synthetic repeat mismatch')
    payload['status']='passed';payload['protocol']=protocol
    publish(args.output,payload)
    print('R5e synthetic rehearsal passed; no real images read:',args.output)

if __name__=='__main__':main()
