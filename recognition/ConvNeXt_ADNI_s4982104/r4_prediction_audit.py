"""R4 two-stage provenance and numeric drift records (no model loading)."""
import math
import statistics
from pathlib import Path
from check_dataloader import require
from predict import write_csv
from train import classification_metrics


def historical_settings(training):
    cfg=training['config'];model=cfg['model']
    require(model in ('SimpleCNN','convnext_tiny'),'Unknown historical model')
    expected_batch=16 if model=='SimpleCNN' else 4
    require(cfg['batch_size']==expected_batch,'Unexpected historical batch')
    if model=='SimpleCNN':
        flags=training.get('backend_flags',{})
        keys=('cudnn_allow_tf32','matmul_allow_tf32','cudnn_benchmark','cudnn_deterministic')
        require(all(type(flags.get(k)) is bool for k in keys),'Missing historical backend flags')
        flags={k:flags[k] for k in keys}
        provenance='recorded training backend_flags'
    else:
        flags=dict(cudnn_allow_tf32=False,matmul_allow_tf32=False,cudnn_benchmark=False,cudnn_deterministic=True)
        provenance='explicit flags in hash-verified run_r3_convnext.py'
    return dict(batch_size=expected_batch,mode='no_grad',source=provenance,**flags)

def differences(actual,reference,unit):
    require(unit in ('slice','patient'),'Invalid comparison unit')
    key='file_path' if unit=='slice' else 'patient_id'
    require(actual and reference,'Empty predictions')
    a={r[key]:r for r in actual};b={r[key]:r for r in reference}
    require(len(a)==len(actual)==len(b)==len(reference) and a.keys()==b.keys(),'Missing/duplicate/different identities')
    fields=['patient_id','image_id','label']+(['slice_count'] if unit=='patient' else [])
    rows=[]
    for identity,row in a.items():
        ref=b[identity]
        require(all(str(row[k])==str(ref[k]) for k in fields),'Metadata mismatch')
        p,q=float(row['p_ad']),float(ref['p_ad'])
        require(all(math.isfinite(v) and 0<=v<=1 for v in (p,q)),'Invalid probability')
        require(int(row['prediction'])==int(p>=.5) and int(ref['prediction'])==int(q>=.5),'Prediction/threshold inconsistency')
        delta=p-q
        rows.append(dict(identity=identity,patient_id=row['patient_id'],reference_probability=q,
            actual_probability=p,signed_difference=delta,absolute_difference=abs(delta),
            r4_probability_close=math.isclose(p,q,rel_tol=1e-5,abs_tol=1e-6),
            r3_probability_close=abs(delta)<=1e-6+1e-5*abs(q),
            reference_prediction=int(ref['prediction']),actual_prediction=int(row['prediction']),
            class_changed=int(row['prediction'])!=int(ref['prediction'])))
    return rows,dict(count=len(rows),r4_exceed_count=sum(not r['r4_probability_close'] for r in rows),
        r3_exceed_count=sum(not r['r3_probability_close'] for r in rows),
        class_change_count=sum(r['class_changed'] for r in rows),
        max_absolute_difference=max(r['absolute_difference'] for r in rows),
        mean_absolute_difference=statistics.mean(r['absolute_difference'] for r in rows))



def patient_metrics(rows):
    ys=[int(r['label']) for r in rows];ps=[float(r['p_ad']) for r in rows]
    result=classification_metrics(ys,ps)
    result['nll']=math.fsum(-math.log(max(1e-7,min(1-1e-7,p)) if y else 1-max(1e-7,min(1-1e-7,p))) for y,p in zip(ys,ps))/len(ys)
    result['brier']=math.fsum((p-y)**2 for y,p in zip(ys,ps))/len(ys)
    return result


def record_predictions(out,prefix,slices,patients,reference_slices,reference_patients):
    out=Path(out)
    write_csv(out/(prefix+'_slice_predictions.csv'),slices)
    write_csv(out/(prefix+'_patient_predictions.csv'),patients)
    sd,ss=differences(slices,reference_slices,'slice')
    pd,ps=differences(patients,reference_patients,'patient')
    write_csv(out/(prefix+'_slice_differences.csv'),sd)
    write_csv(out/(prefix+'_patient_differences.csv'),pd)
    return dict(slice_drift=ss,patient_drift=ps,patient_metrics=patient_metrics(patients),
                reference_patient_metrics=patient_metrics(reference_patients))
