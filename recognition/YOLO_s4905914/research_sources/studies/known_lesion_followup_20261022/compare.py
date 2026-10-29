"""Score locked follow-up runs; only validation records are read."""
import csv,hashlib,json,sys
from pathlib import Path
import numpy as np
S=Path(__file__).resolve().parent;R=S.parents[1]
sys.path.insert(0,str(S/'code'))
from evaluation import score_records,select_threshold,gt_record,iou
from runtime import write_json
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())

def errors(records,t):
    from collections import Counter
    counts=Counter()
    for r in records:
        if r['class_id']!=1:continue
        p=[x for x in r['predictions'] if x['confidence']>t]
        if any(x['class_id']==1 and iou(r['box'],x['box'])>=.5 for x in p):name='TP'
        elif any(x['class_id']==0 and iou(r['box'],x['box'])>=.5 for x in p):name='localized_wrong_class'
        elif any(x['class_id']==1 for x in p):name='malignant_box_low_IoU'
        elif p:name='wrong_class_and_low_IoU'
        else:name='no_retained_prediction'
        counts[name]+=1
    assert sum(counts.values())==41
    return dict(counts)

def report(name,path,model,cfg,expected,plan):
    ev=read(path/f'val_{model}/evaluation.json');records=read(path/f'val_{model}/predictions.json')
    run=read(path/f'train_{model}/run.json');weight=path/f'train_{model}/fit/weights/best.pt'
    assert run['status']=='completed' and ev['split']=='val'
    assert ev['manifest_sha256']==plan['manifest_sha256']==run['manifest_sha256']
    assert ev['checkpoint_sha256']==run['best_sha256']==sha(weight)
    assert ev['config_sha256']==run['config_sha256']==sha(cfg)
    assert ev['training_run_sha256']==sha(path/f'train_{model}/run.json')
    assert len(records)==165 and {r['image_id'] for r in records}==set(expected)
    for r in records:
        g=expected[r['image_id']]
        assert r['class_id']==g['class_id'] and np.allclose(r['box'],g['box'],rtol=0,atol=1e-12)
    c=read(cfg);policy=c['evaluation'];selected=select_threshold(records,policy)
    assert selected==ev['operating_point']
    assert run['optimizer_steps']==2880 and run['checkpoint_selection']['completed_epochs']==60
    assert run['initial_weights_sha256']==c['models'][model]['sha256']
    stage=read(path/'inputs/staged/audit.json')
    reference=read(R/'outputs/known_lesion_optimization_20261021_681452/n_geometry/inputs/staged/audit.json')
    assert stage['counts']=={'train':768,'val':165}
    for k,v in stage['files'].items():
        for field in ('file_sha256','pixel_sha256','size'):assert v[field]==reference['files'][k][field]
    assert not (path/'inputs/staged/reassessment.yaml').exists()
    if name!='reference_geometry':
        loss=read(path/f'train_{model}/loss_configuration.json')
        assert loss['positive_weights']==c['project_loss']['positive_weights'] and loss['training_only']
        assert run['source_sha256']=={k:v for k,v in plan['source_sha256'].items() if k.endswith('.py')}
    point=selected['selected'];threshold=point['threshold'] if point else None
    return {'selected':point,'ap':ev['ap'],'mAP50':ev['mAP50'],'mAP50_95':ev['mAP50_95'],
        'grid':selected['grid'],'fixed_thresholds':{str(t):score_records(records,t) for t in (.05,.4)},
        'strict_matching':{str(t):score_records(records,threshold,t) for t in (.75,.8)} if point else {},
        'relaxed_budgets_diagnostic_only':{str(b):select_threshold(records,{**policy,'benign_false_alert_budget':b})['selected'] for b in (.2,.3)},
        'failure_categories':errors(records,threshold) if point else {},
        'score_floor':score_records(records,policy['score_floor']),
        'resources':{k:run[k] for k in ('elapsed_seconds','optimizer_steps','parameters','peak_cuda_allocated_bytes','checkpoint_selection')},
        'evaluation_path':str((path/f'val_{model}').relative_to(R)),
        'checkpoint_path':str(weight.relative_to(R)),'config_path':str(cfg.relative_to(R)),
        'checkpoint_sha256':sha(weight),'evaluation_sha256':sha(path/f'val_{model}/evaluation.json'),
        'prediction_sha256':sha(path/f'val_{model}/predictions.json')}

def qualifies(r,b):
    if r['selected'] is None:return False
    a=r['selected']['classes'];z=b['selected']['classes']
    return (a['Malignant']['tp']>=z['Malignant']['tp']+3
        and a['Malignant']['iou']['all_gt_class_correct']['mean']>z['Malignant']['iou']['all_gt_class_correct']['mean']
        and a['Benign']['iou']['all_gt_class_correct']['mean']>=z['Benign']['iou']['all_gt_class_correct']['mean']-.03)

def rank(item):
    name,r=item;p=r['selected'];m=p['classes']['Malignant']
    return (-m['tp'],-m['iou']['all_gt_class_correct']['mean'],-r['ap']['1']['AP50_95'],p['image_alerts']['benign_fp'],name)

def main():
    p=read(S/'plan.json'); runmap=read(S/'runmap.json')
    rows=list(csv.DictReader((R/'data/known_lesion_v1/val.csv').open()))
    expected={r['image_id']:gt_record(r) for r in rows}
    old=R/p['reference_study'];d=read(old/'decision.json')
    reports={'reference_geometry':report('reference_geometry',(R/d['evaluation_path']).parent,'n',R/d['config_path'],expected,p)}
    for name,change in p['variants'].items():
        if name not in runmap:continue
        path=R/runmap[name]/name;model=change['model']
        if not (path/f'val_{model}/evaluation.json').exists():continue
        reports[name]=report(name,path,model,S/f'{name}.json',expected,p)
    complete=all(n in reports for n in p['variants'])
    eligible={n:r for n,r in reports.items() if n!='reference_geometry' and qualifies(r,reports['reference_geometry'])}
    winner=min(eligible.items(),key=rank)[0] if eligible else None
    result={'status':'completed_initial' if complete else 'partial','scope':'validation only',
        'new_reassessment_inference':False,'plan_sha256':sha(S/'plan.json'),'reports':reports,
        'eligible':list(eligible),'confirmation_candidate':winner if complete else None,
        'decision':'confirm_candidate' if complete and winner else 'retain_reference' if complete else 'pending',
        'protected_files_unchanged':all(sha(R/name)==d for name,d in p['protected_sha256'].items())}
    assert result['protected_files_unchanged']
    write_json(S/'results.json',result)
    for name,r in reports.items():
        x=r['selected'];m=x['classes']['Malignant']
        print(name,'threshold',x['threshold'],'MTP',m['tp'],'Balerts',x['image_alerts']['benign_fp'],
            'M_IoU',round(m['iou']['all_gt_class_correct']['mean'],4))
    print(result['status'],result['decision'],result['confirmation_candidate'])
if __name__=='__main__':main()
