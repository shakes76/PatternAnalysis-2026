"""Frozen post-prediction evaluation: no training, fitting, model or image IO.

Patient bootstrap indices are shared across every model/view and paired contrast.
All calibration/threshold/referral decisions come from a hash-verified protocol.
"""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import numpy as np
from run_r5_ensemble import MODELS, SEEDS, require, sha
from run_r5_temperature import scale, reliability
from run_r5_referral import select_referrals, confusion, divide

METRICS=('accuracy','ad_recall','nc_specificity','macro_f1','auroc','nll','brier')


def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def manifest_records(records):
    result=[]
    for row in records:
        y={'NC':0,'AD':1}.get(row['label'],row['label'])
        require(type(y) is int and y in (0,1),'Invalid manifest label')
        result.append({**{k:row[k] for k in ('file_path','patient_id','image_id','split')},'label':y})
    return sorted(result,key=lambda r:r['file_path'])


def load_protocol(path,expected_sha256,code_root):
    path=Path(path)
    require(sha(path)==expected_sha256,'Protocol SHA mismatch')
    protocol=json.loads(path.read_text())
    for name,value in protocol['code_sha256'].items():
        require(Path(name).name==name,'Unsafe code identity')
        require(sha(Path(code_root)/name)==value,'Code SHA mismatch: '+name)
    return protocol


def aggregate_slices(rows,expected,slices_per_patient):
    expected=manifest_records(expected);mapping={r['file_path']:r for r in expected}
    require(expected and len(mapping)==len(expected),'Empty/duplicate manifest')
    meta={};scans={};counts=Counter()
    for r in expected:
        require(all(isinstance(r[k],str) and r[k] for k in ('file_path','patient_id','image_id')),'Empty manifest identity')
        pid=r['patient_id'];m=(r['image_id'],r['label'])
        require(meta.setdefault(pid,m)==m,'Manifest scan/label conflict')
        require(scans.setdefault(r['image_id'],pid)==pid,'Shared scan identity')
        counts[pid]+=1
    require(all(n==slices_per_patient for n in counts.values()),'Manifest slice count mismatch')
    groups=defaultdict(list);seen=set()
    for r in rows:
        path=r['file_path'];require(path in mapping and path not in seen,'Unexpected/duplicate prediction slice')
        seen.add(path);ref=mapping[path]
        require(r.get('split',ref['split'])==ref['split'],'Prediction split mismatch')
        require(type(r['label']) is int and all(r[k]==ref[k] for k in ('patient_id','image_id','label')),'Prediction metadata mismatch')
        p=r['p_ad'];require(type(p) in (int,float) and math.isfinite(p) and 0<=p<=1,'Invalid probability')
        require(r['prediction']==int(p>=.5),'Prediction/class mismatch')
        groups[r['patient_id']].append((path,p))
    require(seen==set(mapping),'Missing prediction slices/patients')
    result=[]
    for pid in sorted(meta):
        p=math.fsum(v for _,v in sorted(groups[pid]))/slices_per_patient
        result.append(dict(patient_id=pid,image_id=meta[pid][0],label=meta[pid][1],slice_count=slices_per_patient,p_ad=p,prediction=int(p>=.5)))
    return result


def bootstrap_indices(labels,replicates=2000,seed=42):
    labels=np.asarray(labels);require(set(labels.tolist())=={0,1},'Bootstrap requires both classes')
    require(type(replicates) is int and replicates>0,'Invalid bootstrap count')
    rng=np.random.Generator(np.random.PCG64(seed))
    # NC then AD; every replicate preserves both class counts and the full N.
    return np.concatenate([rng.choice(np.flatnonzero(labels==y),size=(replicates,int((labels==y).sum())),replace=True) for y in (0,1)],axis=1)


def bootstrap_metrics(labels,probabilities,indices,threshold,nll_terms=None):
    y=np.asarray(labels,dtype=int);p=np.asarray(probabilities,dtype=float)
    require(len(y)==len(p) and set(y.tolist())=={0,1},'Invalid metric cohort')
    require(np.isfinite(p).all() and ((p>=0)&(p<=1)).all(),'Invalid metric probability')
    q=p[indices];truth=y[indices];pred=q>=threshold
    tp=((truth==1)&pred).sum(axis=1);fn=((truth==1)&~pred).sum(axis=1)
    tn=((truth==0)&~pred).sum(axis=1);fp=((truth==0)&pred).sum(axis=1)
    require(np.all(tp+fn==tp[0]+fn[0]) and np.all(tn+fp==tn[0]+fp[0]),'Indices not stratified')
    pos=q[truth==1].reshape(len(indices),-1);neg=q[truth==0].reshape(len(indices),-1)
    auc=((pos[:,:,None]>neg[:,None,:]).mean(axis=(1,2))+.5*(pos[:,:,None]==neg[:,None,:]).mean(axis=(1,2)))
    if nll_terms is None:
        clipped=np.clip(p,1e-7,1-1e-7)
        nll_terms=np.where(y==1,-np.log(clipped),-np.log1p(-clipped))
    return dict(accuracy=(tp+tn)/len(y),ad_recall=tp/(tp+fn),nc_specificity=tn/(tn+fp),
        macro_f1=(2*tp/(2*tp+fp+fn)+2*tn/(2*tn+fp+fn))/2,auroc=auc,
        nll=np.asarray(nll_terms)[indices].mean(axis=1),brier=((q-truth)**2).mean(axis=1))


def interval(values):
    return np.quantile(values,[.025,.975],method='linear').tolist()


def referral_metrics(rows,rate):
    ids=set(select_referrals([dict(patient_id=r['patient_id'],p_ad_raw=r['p_ad_raw']) for r in rows],rate))
    result={'requested_referral_rate':rate}
    for name,group in [('full',rows),('referred',[r for r in rows if r['patient_id'] in ids]),('retained',[r for r in rows if r['patient_id'] not in ids])]:
        cm=confusion(group);n=len(group);err=cm['fn']+cm['fp']
        result.update({name+'_'+k:v for k,v in dict(n=n,**cm,errors=err,error_rate=divide(err,n),
            ad_n=cm['tp']+cm['fn'],nc_n=cm['tn']+cm['fp'],
            ad_recall=divide(cm['tp'],cm['tp']+cm['fn']),nc_specificity=divide(cm['tn'],cm['tn']+cm['fp'])).items()})
    result.update(coverage=result['retained_n']/len(rows),actual_referral_rate=len(ids)/len(rows),
        error_capture_rate=divide(result['referred_errors'],result['full_errors']),
        ad_miss_capture_rate=divide(result['referred_fn'],result['full_fn']))
    return result


def evaluate(protocol,expected,runs):
    """Evaluate a nine-run in-memory bundle after caller verifies protocol bytes/code.

Per-run protocol digest and checkpoint identities bind producer outputs. The R6
GPU producer must independently hash the actual checkpoint bytes before inference.
"""
    require(protocol['scope']=='frozen_final_evaluation' and protocol['schema']==1,'Wrong protocol schema')
    require(protocol['models']==list(MODELS) and protocol['seeds']==list(SEEDS),'Wrong models/seeds')
    cfg=protocol['evaluation']
    require(cfg==dict(main_threshold=.5,nll_epsilon=1e-7,ece_bins=10,referral_rates=[0,.05,.1,.2],primary_referral_rate=.1,
        bootstrap_replicates=2000,bootstrap_seed=42,confidence_level=.95,quantile_method='linear'),'Unapproved evaluation rules')
    require(protocol['aggregation']==dict(slices_per_patient=20,seed_weights=[1/3]*3),'Wrong aggregation')
    expected=manifest_records(expected);cohort=protocol['cohort']
    require(cohort['split']=='test' and all(r['split']=='test' for r in expected),'Final evaluation requires test cohort')
    require(digest(expected)==cohort['manifest_records_sha256'],'Frozen manifest identity mismatch')
    require(len(expected)==cohort['slices'],'Wrong slice count')
    entries={(r['model'],r['seed']):r for r in protocol['checkpoints']}
    required={(m,s) for m in MODELS for s in SEEDS}
    require(len(protocol['checkpoints'])==9 and set(entries)==required,'Wrong checkpoint inventory')
    require(len(runs)==9 and {(r['entry']['model'],r['entry']['seed']) for r in runs}==required,'Wrong nine prediction runs')
    require(set(protocol['temperatures'])==set(MODELS) and set(protocol['supplemental_thresholds'])==set(MODELS),'Wrong calibration inventory')
    singles={};identity=None
    for run in runs:
        entry=run['entry'];key=(entry['model'],entry['seed'])
        require(entry==entries[key],'Checkpoint identity mismatch')
        require(run['protocol_sha256']==digest(protocol),'Prediction protocol mismatch')
        require(run['inference']==protocol['inference'],'Prediction inference mismatch')
        rows=aggregate_slices(run['slices'],expected,20)
        current=[(r['patient_id'],r['image_id'],r['label']) for r in rows]
        require(identity is None or identity==current,'Different patient cohort')
        identity=current;singles[key]=rows
    y=np.array([item[2] for item in identity]);n=len(y)
    require(n==cohort['patients'] and int(y.sum())==cohort['ad'] and int((y==0).sum())==cohort['nc'],'Wrong patient/class counts')
    indices=bootstrap_indices(y,cfg['bootstrap_replicates'],cfg['bootstrap_seed'])
    views={};patient_tables={};reliability_bins=[];referrals=[]
    for model in MODELS:
        for seed in SEEDS:
            tag=f'{model}/seed{seed}';rows=singles[model,seed]
            views[tag]=(np.array([r['p_ad'] for r in rows]),.5,None)
            patient_tables[tag]=rows
        raw=np.array([math.fsum(singles[model,s][i]['p_ad'] for s in SEEDS)/3 for i in range(n)])
        t=protocol['temperatures'][model];threshold=protocol['supplemental_thresholds'][model]
        require(type(t) in (int,float) and math.isfinite(t) and .05<t<20,'Invalid frozen T')
        require(type(threshold) in (int,float) and math.isfinite(threshold) and 0<=threshold<=1,'Invalid frozen threshold')
        cal=np.array([scale(p,t) for p in raw]);require(np.array_equal(raw>=.5,cal>=.5),'Numerical class change')
        z=np.log(np.clip(raw,1e-7,1-1e-7))-np.log1p(-np.clip(raw,1e-7,1-1e-7))
        terms=np.logaddexp(0,(1-2*y)*z/t)
        views[f'{model}/ensemble_raw']=(raw,.5,None)
        views[f'{model}/ensemble_calibrated']=(cal,.5,terms)
        views[f'{model}/supplement_ad80']=(cal,threshold,terms)
        rows=[dict(patient_id=identity[i][0],image_id=identity[i][1],label=int(y[i]),slice_count=20,
                   p_ad_raw=float(raw[i]),p_ad=float(cal[i]),prediction=int(cal[i]>=.5),
                   prediction_ad80=int(cal[i]>=threshold),confidence=float(max(cal[i],1-cal[i]))) for i in range(n)]
        patient_tables[f'{model}/ensemble']=rows
        for stage,ps in [('raw',raw),('calibrated',cal)]:
            bins,ece=reliability([dict(label=int(a),p_ad=float(b)) for a,b in zip(y,ps)])
            reliability_bins.extend(dict(model=model,stage=stage,**r) for r in bins)
        for rate in cfg['referral_rates']:referrals.append(dict(model=model,**referral_metrics(rows,rate)))
        for row in rows:
            for rate in cfg['referral_rates']:
                ids=select_referrals([dict(patient_id=r['patient_id'],p_ad_raw=r['p_ad_raw']) for r in rows],rate)
                row[f'refer_{int(rate*100)}_percent']=int(row['patient_id'] in ids)
    results=[];draws={};estimates={}
    for tag,(p,threshold,terms) in views.items():
        draws[tag]=bootstrap_metrics(y,p,indices,threshold,terms)
        estimates[tag]={k:float(v[0]) for k,v in bootstrap_metrics(y,p,np.arange(n)[None,:],threshold,terms).items()}
        cm=confusion([dict(label=int(a),p_ad=float(b)) for a,b in zip(y,p)]) if threshold==.5 else {
            'tn':int(((y==0)&(p<threshold)).sum()),'fp':int(((y==0)&(p>=threshold)).sum()),
            'fn':int(((y==1)&(p<threshold)).sum()),'tp':int(((y==1)&(p>=threshold)).sum())}
        _,ece=reliability([dict(label=int(a),p_ad=float(b)) for a,b in zip(y,p)])
        results.append(dict(view=tag,threshold=threshold,n=n,**cm,**estimates[tag],confidence_ece_at_0_5=ece,
                            intervals={k:interval(draws[tag][k]) for k in METRICS}))
    seed_summary=[]
    for model in MODELS:
        seed_summary.append(dict(model=model,**{k:dict(mean=statistics.mean(estimates[f'{model}/seed{s}'][k] for s in SEEDS),
               sample_sd=statistics.stdev(estimates[f'{model}/seed{s}'][k] for s in SEEDS)) for k in METRICS}))
    pairs=[('ConvNeXt-F/ensemble_calibrated','CNN-A/ensemble_calibrated'),
           ('ConvNeXt-F/ensemble_calibrated','CNN-B/ensemble_calibrated'),
           ('CNN-B/ensemble_calibrated','CNN-A/ensemble_calibrated')]
    paired=[]
    for first,second in pairs+[(f'{m}/ensemble_calibrated',f'{m}/ensemble_raw') for m in MODELS]:
        keys=METRICS if first.split('/')[0]!=second.split('/')[0] else ('nll','brier')
        paired.append(dict(first=first,second=second,direction='first minus second',metrics={k:dict(
            estimate=estimates[first][k]-estimates[second][k],interval=interval(draws[first][k]-draws[second][k])) for k in keys}))
    return dict(patient_count=n,metrics=results,seed_summary=seed_summary,paired_differences=paired,
        referrals=referrals,reliability_bins=reliability_bins,patient_tables=patient_tables,
        bootstrap=dict(replicates=2000,seed=42,method='stratified patient percentile 95%, linear quantile',
            generator='numpy PCG64; NC then AD',indices_sha256=hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest(),
            paired_indices_shared=True,scope='full-cohort scalar metrics only; ECE/referral descriptive',
            conditional_on='fixed trained models, T, thresholds and class counts; no refitting'))
