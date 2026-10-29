"""Paired validation changes, retaining recovered and newly missed cases."""
import json,sys
from pathlib import Path
S=Path(__file__).resolve().parent;R=S.parents[1]
sys.path.insert(0,str(S/'code'))
from evaluation import iou
read=lambda p:json.loads(p.read_text())
x=read(S/'results.json');assert x['status']=='completed_initial'
def outcomes(r):
    records=read(R/r['evaluation_path']/'predictions.json');t=r['selected']['threshold'];out={}
    for row in records:
        ps=[p for p in row['predictions'] if p['confidence']>t]
        correct=max((iou(row['box'],p['box']) for p in ps if p['class_id']==row['class_id']),default=0.)
        out[row['image_id']]={'class_id':row['class_id'],'tp':correct>=.5,
            'correct_class_best_iou':correct,'malignant_alert':any(p['class_id']==1 for p in ps),
            'retained_predictions':ps}
    return out
base=outcomes(x['reports']['reference_geometry']);result={}
for name,r in x['reports'].items():
    if name=='reference_geometry':continue
    now=outcomes(r)
    recovered=[i for i,a in now.items() if a['class_id']==1 and a['tp'] and not base[i]['tp']]
    lost=[i for i,a in now.items() if a['class_id']==1 and not a['tp'] and base[i]['tp']]
    new_alert=[i for i,a in now.items() if a['class_id']==0 and a['malignant_alert'] and not base[i]['malignant_alert']]
    removed_alert=[i for i,a in now.items() if a['class_id']==0 and not a['malignant_alert'] and base[i]['malignant_alert']]
    assert len(recovered)-len(lost)==r['selected']['classes']['Malignant']['tp']-x['reports']['reference_geometry']['selected']['classes']['Malignant']['tp']
    changed=set(recovered+lost+new_alert+removed_alert)
    result[name]={'recovered_malignant':sorted(recovered),'newly_missed_malignant':sorted(lost),
        'new_benign_alerts':sorted(new_alert),'removed_benign_alerts':sorted(removed_alert),
        'cases':{i:{'reference':base[i],'candidate':now[i]} for i in sorted(changed)}}
(S/'paired_case_changes.json').write_text(json.dumps(result,indent=2)+'\n')
print({n:{k:len(v) for k,v in x.items() if k!='cases'} for n,x in result.items()})
