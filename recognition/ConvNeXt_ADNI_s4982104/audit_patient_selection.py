"""Read-only V1 CSV audit for R1. Writes fresh retrospective results outside V1.

No checkpoint selection or V1 file is modified. CNN missing epoch predictions
remain unavailable. CLI: --base ADNI_MANIFEST_LOCAL --output-dir NEW_DIRECTORY.
"""
import argparse
import csv
from collections import defaultdict
from decimal import Decimal,localcontext
import hashlib
import json
import math
from pathlib import Path
from patient_evaluation import evaluate_patients,POLICY


def independent_scores(rows):
    """Independent Decimal means/NLL/Brier, explicit confusion and pairwise AUC."""
    groups=defaultdict(list)
    for r in rows:groups[r['patient_id']].append(r)
    with localcontext() as ctx:
        ctx.prec=50;eps=Decimal('0.0000001');pairs=[];losses=[];bs=[]
        for group in groups.values():
            p=sum(Decimal(str(r['p_ad'])) for r in group)/Decimal(len(group));y=int(group[0]['label'])
            q=min(1-eps,max(eps,p));losses.append(-(q if y else 1-q).ln());bs.append((p-y)**2);pairs.append((y,float(p)))
        result=dict(nll=float(sum(losses)/len(losses)),brier=float(sum(bs)/len(bs)))
    cm=[[0,0],[0,0]]
    for y,p in pairs:cm[y][int(p>=.5)]+=1
    pc={}
    for i,name in enumerate(('NC','AD')):
        support=sum(cm[i]);predicted=cm[0][i]+cm[1][i]
        precision=cm[i][i]/predicted if predicted else 0.;recall=cm[i][i]/support if support else 0.
        pc[name]=dict(precision=precision,recall=recall,f1=2*precision*recall/(precision+recall) if precision+recall else 0.,support=support)
    positive=[p for y,p in pairs if y];negative=[p for y,p in pairs if not y]
    auc=sum((a>b)+.5*(a==b) for a in positive for b in negative)/(len(positive)*len(negative)) if positive and negative else None
    result.update(confusion_matrix=cm,per_class=pc,accuracy=(cm[0][0]+cm[1][1])/len(pairs),
        macro_f1=(pc['NC']['f1']+pc['AD']['f1'])/2,balanced_accuracy=(pc['NC']['recall']+pc['AD']['recall'])/2,auroc=auc)
    return result


def close(a,b):
    if isinstance(b,dict):return all(k in a and close(a[k],v) for k,v in b.items())
    if isinstance(b,list):return len(a)==len(b) and all(close(x,y) for x,y in zip(a,b))
    if isinstance(b,float):return math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10)
    return a==b


def audit(base,output):
    base=Path(base);out=Path(output);out.mkdir(parents=True,exist_ok=False)
    snapshots={};load=lambda p:json.loads(read(p))
    def read(p):
        p=Path(p);data=p.read_bytes();snapshots[str(p)]=hashlib.sha256(data).hexdigest();return data.decode()
    def csvrows(p):return list(csv.DictReader(read(p).splitlines()))
    manifest=base/'split_seed42_60_20_20/split_samples.csv'
    expected=[r for r in csvrows(manifest) if r['split']=='validation']
    assert len(expected)==2720
    def check(folder,summary):
        folder=Path(folder)
        for n,h in summary['output_sha256'].items():
            read(folder/n);assert snapshots[str(folder/n)]==h,('CSV hash',folder,n)
        rows=csvrows(folder/'slice_predictions.csv')
        for r in rows:r.update(label=int(r['label']),p_ad=float(r['p_ad']))
        result=evaluate_patients(rows,expected);independent=independent_scores(rows)
        assert close(result['metrics'],independent),('Independent score mismatch',folder)
        saved=csvrows(folder/'patient_predictions.csv');assert len(saved)==len(result['patients'])
        lookup={r['patient_id']:r for r in saved};assert len(lookup)==len(saved)
        for r in result['patients']:
            old=lookup[r['patient_id']]
            assert old['image_id']==r['image_id'] and int(old['label'])==r['label'] and int(old['slice_count'])==r['slice_count']
            assert math.isclose(float(old['p_ad']),r['p_ad'],rel_tol=1e-12,abs_tol=1e-12) and int(old['prediction'])==r['prediction']
        assert close(result['metrics'],summary['patient_metrics']),('V1 metric mismatch',folder)
        m=result['metrics']
        return dict(patient_nll=m['nll'],patient_brier=m['brier'],patient_accuracy=m['accuracy'],ad_recall=m['per_class']['AD']['recall'],ad_precision=m['per_class']['AD']['precision'],nc_specificity=m['per_class']['NC']['recall'],balanced_accuracy=m['balanced_accuracy'],macro_f1=m['macro_f1'],auroc=m['auroc'])
    epoch_rows=[];best_rows=[];selection=[]
    try:
        runs=[('ConvNeXt_N4',base/'convnext_stability_lr3e5/run')]+[(f'ConvNeXt_{seed}',base/f'convnext_formal/seed{seed}_{job}/run') for seed,job in [(42,702352),(43,702999),(44,703004)]]
        for name,root in runs:
            t=load(root/'training/training_summary.json');assert t['status']=='passed'
            assert t['input_sha256']['split_samples']==snapshots[str(manifest)]
            this=[]
            for h in t['history']:
                folder=root/f"diagnostics/epoch_{h['epoch']}";d=load(folder/'summary.json');assert d==h['diagnostics']
                row=dict(run=name,epoch=h['epoch'],slice_ce=h['validation']['loss'],**check(folder,d));epoch_rows.append(row);this.append(row)
            old=min(this,key=lambda x:x['slice_ce']);new=min(this,key=lambda x:x['patient_nll'])
            assert old['epoch']==t['best_epoch']
            selection.append(dict(run=name,v1_slice_ce_epoch=old['epoch'],retrospective_patient_nll_epoch=new['epoch'],same_epoch=old['epoch']==new['epoch'],v1_patient_nll=old['patient_nll'],candidate_patient_nll=new['patient_nll'],candidate_weight_available=old['epoch']==new['epoch'],v1_unchanged=True))
        specs=[('ConvNeXt_N3','convnext_pilot/run/prediction')]+[(name,str(root.relative_to(base)/'prediction')) for name,root in runs]
        specs += [('CNN_C3','c3_validation/prediction'),('CNN_E2_A','e2_compatibility/pair/A/prediction'),('CNN_E2_B','e2_compatibility/pair/B/prediction')]
        specs += [(f'CNN_{arm}42_{which}',f'{folder}/run/prediction_{which}') for arm,folder in [('A','e3_A42'),('B','e4_B42')] for which in ['first3','full']]
        specs += [(f'CNN_{arm}{seed}',f'e5_pairs/run/{arm}{seed}/prediction_full') for seed in [43,44] for arm in ['A','B']]
        for name,relative in specs:
            folder=base/relative;summary=load(folder/'prediction_summary.json');assert summary['status']=='passed'
            assert summary['input_sha256']['split_samples']==snapshots[str(manifest)]
            best_rows.append(dict(run=name,source=str(folder),**check(folder,summary)))
        def save_csv(name,rows):
            with (out/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
        save_csv('convnext_epoch_patient_metrics.csv',epoch_rows);save_csv('saved_best_patient_metrics.csv',best_rows);save_csv('selection_comparison.csv',selection)
        for p,h in snapshots.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,('V1 changed',p)
        result=dict(status='passed',policy=POLICY,convnext_epochs_checked=len(epoch_rows),saved_best_runs_checked=len(best_rows),independent_decimal_scores_match=True,saved_patient_csv_and_classification_match=True,v1_files_unchanged=True,mri_reads=dict(train=0,validation=0,test=0),selection_comparison=selection,limitations=['CNN epoch probabilities were not retained: cannot retrospectively select patient-NLL epoch.','N3 has only saved best predictions; no invented epoch2/3 patient results.','Retrospective NLL candidate is diagnostic only; no V1 checkpoint or official score replaced.','If candidate differs from V1 best, its weights are unavailable locally and cannot be reconstructed from CSV.'],input_file_sha256=snapshots)
    except Exception as e:
        result=dict(status='failed',error=f'{type(e).__name__}: {e}',input_file_sha256=snapshots)
        (out/'audit_summary.json').write_text(json.dumps(result,indent=2)+'\n');raise
    (out/'audit_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    result=audit(a.base,a.output_dir);print(json.dumps({k:v for k,v in result.items() if k!='input_file_sha256'},indent=2))
