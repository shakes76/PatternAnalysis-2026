"""Compare predeclared variants using cached validation predictions only."""
import argparse,csv,hashlib,json,sys
from collections import Counter
from pathlib import Path
import numpy as np
S=Path(__file__).resolve().parent; R=S.parents[1]
sys.path.insert(0,str(S/'code'))
from evaluation import score_records,select_threshold,iou,gt_record
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,obj):p.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def errors(records,t):
    c=Counter()
    for r in records:
        if r['class_id']!=1:continue
        ps=[p for p in r['predictions'] if p['confidence']>t]
        if any(p['class_id']==1 and iou(r['box'],p['box'])>=.5 for p in ps):k='TP'
        elif any(p['class_id']==0 and iou(r['box'],p['box'])>=.5 for p in ps):k='localized_wrong_class'
        elif any(p['class_id']==1 for p in ps):k='malignant_box_below_IoU_0_5'
        elif ps:k='wrong_class_and_low_IoU'
        else:k='no_retained_prediction'
        c[k]+=1
    assert sum(c.values())==41
    return dict(c)

def row_metrics(records,t):
    out={}
    for r in records:
        ps=[p for p in r['predictions'] if p['confidence']>t]
        good=max((iou(r['box'],p['box']) for p in ps if p['class_id']==r['class_id']),default=0.)
        out[r['image_id']]=[int(r['class_id']==1),int(r['class_id']==1 and good>=.5),
             good if r['class_id']==1 else 0.,int(r['class_id']==0),
             int(r['class_id']==0 and any(p['class_id']==1 for p in ps))]
    return out

def cluster_ci(baseline,candidate,rows,draws=5000):
    # Paired resampling of connected known-patient/known-lesion components.
    ids=[r['image_id'] for r in rows]; parent={i:i for i in ids}
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]]; i=parent[i]
        return i
    owner={}
    for r in rows:
        for field in ('patient_id','lesion_id'):
            if not r[field]:continue
            key=(field,r[field]); i=r['image_id']
            if key in owner:parent[find(i)]=find(owner[key])
            owner[key]=i
    components={}
    for i in ids:components.setdefault(find(i),[]).append(i)
    a=np.array([np.sum([baseline[i] for i in group],axis=0) for group in components.values()])
    b=np.array([np.sum([candidate[i] for i in group],axis=0) for group in components.values()])
    rng=np.random.default_rng(20261021)
    index=rng.integers(0,len(a),size=(draws,len(a)))
    aa=a[index].sum(axis=1); bb=b[index].sum(axis=1)
    valid=(aa[:,0]>0)&(aa[:,3]>0)
    result={}
    for name,num,den in [('malignant_recall',1,0),('malignant_all_GT_correct_IoU',2,0),('benign_false_alert_rate',4,3)]:
        delta=bb[valid,num]/bb[valid,den]-aa[valid,num]/aa[valid,den]
        result[name]={'candidate_minus_baseline_mean':float(b[:,num].sum()/b[:,den].sum()-a[:,num].sum()/a[:,den].sum()),
                      'paired_percentile_95_interval':np.quantile(delta,[.025,.975]).tolist()}
    return {'known_identity_components':len(a),'draws':draws,'valid_draws':int(valid.sum()),
       'conditioning':'Fixed fitted models and validation-selected thresholds; no correction for tuning/selection; missing patient links remain unknown',
       'differences':result}

def compare(runmap,allow_partial=False):
    plan=read(S/'plan.json'); policy=read(S/'n_cls1.json')['evaluation']
    with (R/'data/known_lesion_v1/val.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    expected={r['image_id']:gt_record(r) for r in rows}
    sources={'baseline_n':R/plan['baseline_run']/'val_n','baseline_s':R/plan['baseline_run']/'val_s'}
    for name,run in runmap.items():sources[name]=R/run/name/'val_n'
    reports={};preds={}; provenance={}
    for name,path in sources.items():
        if not (path/'evaluation.json').is_file():
            if allow_partial:continue
            raise FileNotFoundError(path)
        ev=read(path/'evaluation.json'); records=read(path/'predictions.json')
        assert ev['split']=='val' and ev['manifest_sha256']['val']==plan_manifest()['val']
        assert {r['image_id'] for r in records}==set(expected) and len(records)==165
        for r in records:
            g=expected[r['image_id']]
            assert r['class_id']==g['class_id'] and np.allclose(r['box'],g['box'],rtol=0,atol=1e-12)
        selected=select_threshold(records,policy)
        assert selected==ev['operating_point'],'Cached operating-point results changed'
        run=read(path.parent/('train_s' if name=='baseline_s' else 'train_n')/'run.json')
        assert run['status']=='completed' and ev['checkpoint_sha256']==run['best_sha256']
        if name in plan['config_sha256']:assert ev['config_sha256']==plan['config_sha256'][name]
        point=selected['selected']; t=point['threshold'] if point else None
        reports[name]={'selected':point,'grid':selected['grid'],'mAP50':ev['mAP50'],'mAP50_95':ev['mAP50_95'],'ap':ev['ap'],
            'fixed_thresholds':{str(t):score_records(records,t) for t in plan['fixed_threshold_comparisons']},
            'strict_IoU_075':score_records(records,t,.75) if t is not None else None,
            'malignant_failure_categories_at_selected':errors(records,t) if t is not None else None,
            'score_floor_diagnosis':errors(records,policy['score_floor']),
            'resources':{k:run[k] for k in ('elapsed_seconds','optimizer_steps','parameters','peak_cuda_allocated_bytes','checkpoint_selection')},
            'checkpoint_sha256':ev['checkpoint_sha256']}
        provenance[name]={'evaluation_path':str(path.relative_to(R)),'evaluation_sha256':sha(path/'evaluation.json'),
                         'predictions_sha256':sha(path/'predictions.json'),'training_record_sha256':sha(path.parent/('train_s' if name=='baseline_s' else 'train_n')/'run.json')}
        preds[name]=records
    result={'status':'partial' if set(plan['variant_training_changes'])-set(reports) else 'completed',
            'scope':'validation_only','new_reassessment_inference':False,'plan_sha256':sha(S/'plan.json'),
            'counts':{'Benign':124,'Malignant':41},'reports':reports,'provenance':provenance,
            'teacher_guidance':plan['teacher_guidance'],'patient_independence_proven':False}
    base=reports['baseline_n']['selected']; baseline=row_metrics(preds['baseline_n'],base['threshold'])
    result['conditional_bootstrap']={}
    for name,r in reports.items():
        if name=='baseline_n' or r['selected'] is None:continue
        result['conditional_bootstrap'][name]=cluster_ci(baseline,row_metrics(preds[name],r['selected']['threshold']),rows)
    candidates=[n for n in reports if n!='baseline_s' and reports[n]['selected'] is not None]
    def rank(n):
        r=reports[n];p=r['selected'];m=p['classes']['Malignant']
        return (-m['recall'],-m['iou']['all_gt_class_correct']['mean'],-r['ap']['1']['AP50_95'],p['image_alerts']['benign_fp'],n!='baseline_n',n)
    best=min(candidates,key=rank)
    win=reports[best]['selected']['classes']['Malignant']; old=base['classes']['Malignant']
    passes=(win['tp']-old['tp']>=3 and win['iou']['all_gt_class_correct']['mean']>old['iou']['all_gt_class_correct']['mean'])
    result['decision']={'highest_ranked':best,'meets_predeclared_material_improvement_rule':passes,
                        'confirmation_required':passes and result['status']=='completed',
                        'recommendation':'pending_seed_confirmation' if passes else 'retain_baseline_n',
                        'not_a_fresh_test_result':True}
    write(S/'results.json',result)
    lines=['model,threshold,M_TP,M_FP,M_FN,M_recall,B_false_alerts,M_IoU_all_GT,M_IoU_TP_only,M_AP50,M_AP50_95,macro_AP50_95']
    for n,r in reports.items():
        p=r['selected'];m=p['classes']['Malignant']
        lines.append(','.join(map(str,[n,p['threshold'],m['tp'],m['fp'],m['fn'],m['recall'],p['image_alerts']['benign_fp'],m['iou']['all_gt_class_correct']['mean'],m['iou']['tp_only']['mean'],r['ap']['1']['AP50'],r['ap']['1']['AP50_95'],r['mAP50_95']])))
    (S/'summary.csv').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines));print(json.dumps(result['decision']))
    return result

def plan_manifest():return read(S/'n_cls1.json')['manifest_sha256']
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runmap',type=Path);p.add_argument('--allow-partial',action='store_true');a=p.parse_args()
    compare(read(a.runmap) if a.runmap else {},a.allow_partial)
