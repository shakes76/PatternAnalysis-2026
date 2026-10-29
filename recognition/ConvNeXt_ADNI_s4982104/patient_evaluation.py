"""R1/V2 patient evaluation; no image reads or RNG use; V1 callers unchanged.

Average the 20 probabilities FIRST, then score each patient with equal weight.
Binary Brier = mean((p_AD-y)**2), not the sum over two class probabilities.
NLL uses natural logs and clips ONLY the patient mean for log evaluation.
Classification uses unclipped p_AD >=0.5. Undefined AUROC is None; zero-denominator
precision/recall follows the existing zero=0 convention. The expected manifest
is mandatory: group counts alone cannot detect an entirely missing patient.
"""
from collections import defaultdict
import math
from train import classification_metrics

NLL_EPSILON=1e-7
POLICY=dict(version='patient-evaluation-v2-r1',aggregation='arithmetic mean of all selected slice probabilities',
            positive_label='AD=1',negative_label='NC=0',threshold=0.5,nll_epsilon=NLL_EPSILON,
            nll='equal-patient mean negative natural log after probability averaging; clip only for log',
            brier='equal-patient mean (unclipped mean p_AD - label)^2',zero_division=0)


def _require(condition,message):
    if not condition:raise ValueError(message)


def evaluate_patients(rows,expected_records,expected_slices=20):
    _require(type(expected_slices) is int and expected_slices>0,'Invalid expected slice count')
    rows=list(rows);expected_records=list(expected_records)
    _require(bool(rows) and bool(expected_records),'Empty predictions/manifest')
    manifest={};patient_meta={};scan_owner={};counts=defaultdict(int)
    for r in expected_records:
        for k in ('file_path','patient_id','image_id'):
            _require(isinstance(r.get(k),str) and bool(r[k].strip()),'Missing manifest identity: '+k)
        raw=r.get('label');label={'NC':0,'AD':1}.get(raw,raw) if isinstance(raw,str) else raw
        _require(type(label) is int and label in (0,1),'Invalid manifest label')
        path=r['file_path'];patient=r['patient_id'];scan=r['image_id']
        _require(path not in manifest,'Duplicate manifest path')
        _require(patient_meta.setdefault(patient,(scan,label))==(scan,label),'Manifest patient scan/label conflict')
        _require(scan_owner.setdefault(scan,patient)==patient,'Scan belongs to multiple patients')
        counts[patient]+=1;manifest[path]=(patient,scan,label)
    _require(all(n==expected_slices for n in counts.values()),'Manifest patient slice count mismatch')
    seen=set();groups=defaultdict(list)
    for r in rows:
        path=r.get('file_path');_require(isinstance(path,str) and path in manifest,'Prediction path outside manifest')
        _require(path not in seen,'Duplicate prediction path');seen.add(path)
        label=r.get('label');_require(type(label) is int and label in (0,1),'Prediction labels must be integer NC=0/AD=1')
        _require((r.get('patient_id'),r.get('image_id'),label)==manifest[path],'Prediction patient/scan/label mismatch')
        p=r.get('p_ad');_require(type(p) in (float,int) and math.isfinite(p) and 0<=p<=1,'Invalid probability')
        groups[r['patient_id']].append((path,float(p)))
    _require(seen==set(manifest),'Missing predictions, including whole patients')
    patients=[];losses=[];squared=[];clipped=0
    for patient,items in sorted(groups.items()):
        _require(len(items)==expected_slices,'Prediction patient slice count mismatch')
        p=math.fsum(x[1] for x in sorted(items))/len(items);scan,y=patient_meta[patient]
        q=min(1-NLL_EPSILON,max(NLL_EPSILON,p));clipped+=int(q!=p)
        nll=-math.log(q) if y else -math.log1p(-q);brier=(p-y)**2
        losses.append(nll);squared.append(brier)
        patients.append(dict(patient_id=patient,image_id=scan,label=y,slice_count=len(items),p_ad=p,
                             prediction=int(p>=.5),nll=nll,brier=brier))
    metrics=classification_metrics([r['label'] for r in patients],[r['p_ad'] for r in patients])
    metrics.update(nll=math.fsum(losses)/len(patients),brier=math.fsum(squared)/len(patients),
                   balanced_accuracy=(metrics['per_class']['AD']['recall']+metrics['per_class']['NC']['recall'])/2)
    return dict(policy=dict(POLICY),patient_count=len(patients),slice_count=len(rows),expected_slices=expected_slices,
                nll_clipped_patient_count=clipped,metrics=metrics,patients=patients)
