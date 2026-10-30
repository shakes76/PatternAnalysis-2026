"""Study-specific forward/loss extension; existing epoch transaction stays shared."""
from __future__ import annotations
import math
import json
import statistics
import torch
from g012.core import schedule, seed_for, base_config, digest
from g012 import core
from g3.model import loss

VARIANTS=('G3',)


def diagnostics(weights):
    a=weights.detach().double(); k=len(a)
    h=float(-(a*torch.log(a.clamp_min(torch.finfo(a.dtype).tiny))).sum())
    return {'maximum_weight':float(a.max()),'normalised_entropy':h/math.log(k) if k>1 else 1.,
            'effective_slices':math.exp(h),'K':k}


def distribution(rows):
    return {key:{'mean':statistics.mean(r[key] for r in rows),'minimum':min(r[key] for r in rows),
                 'maximum':max(r[key] for r in rows),'population_sd':statistics.pstdev(r[key] for r in rows)}
            for key in ('maximum_weight','normalised_entropy','effective_slices')}


def train_epoch(model,bags,fetch,optimizer,device,variant,matched,resource=None):
    core.require(variant=='G3','Only fixed G3 is permitted under REUSE_G2.')
    core.require(len(bags)>0 and all(b.split=='train' for b in bags),'Only nonempty training scans may enter optimization.')
    model.train(); optimizer.zero_grad(set_to_none=True)
    norms=[]; objectives=[]; attention=[]; normalization_errors=[]
    for pos,index in enumerate(matched['order']):
        bag=bags[index]; window=min(2,len(bags)-(pos//2)*2)
        x=fetch(bag,matched['angles'][index]).to(device)
        slices,z,a=model(x); value=loss(z,bag.label)
        core.require(bool(torch.isfinite(value)) and bool(torch.isfinite(a).all()),'Nonfinite attention/loss.')
        normalization_errors.append(abs(float(a.detach().sum())-1.))
        core.require(normalization_errors[-1]<=2e-6,'Attention weights do not normalise over the scan.')
        attention.append(diagnostics(a)); objectives.append(float(value.detach()))
        (value/window).backward()
        if resource: resource.observe()
        del x,slices,z,a,value
        if (pos+1)%2==0 or pos==len(bags)-1:
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
            norms.append(float(norm)); optimizer.step(); optimizer.zero_grad(set_to_none=True)
    d=distribution(attention)
    return {'training_objective':math.fsum(objectives)/len(bags),'optimizer_updates':len(norms),
            'training_scans':len(bags),'max_preclip_gradient_norm':max(norms),
            'mean_preclip_gradient_norm':statistics.mean(norms),
            'preclip_gradient_norms_json':json.dumps(norms,separators=(',',':')),
            'clipped_update_fraction':sum(n>5. for n in norms)/len(norms),
            'maximum_attention_normalization_error':max(normalization_errors),
            **{'training_attention_'+k+'_mean':v['mean'] for k,v in d.items()}}


def predict(model,bags,fetch,device,variant,resource=None):
    core.require(variant=='G3' and all(b.split=='validation' for b in bags),'Validation only.')
    logits=[]; native=[]; weights=[]
    with core.isolated_rng(),torch.inference_mode():
        model.eval()
        for bag in bags:
            z,scan,a=model(fetch(bag).to(device))
            logits.append(z.cpu()); native.append(scan.cpu()); weights.append(a.cpu())
            if resource: resource.observe()
        _,predictions=core.aggregate(bags,logits,'G2')
        for bag,row,z,a in zip(bags,predictions['scans'],native,weights):
            row.update(logits=z.tolist(),p_ad=float(torch.softmax(z,0)[1]),
                       attention_weights=a.tolist(),slice_order=[r.image_path for r in bag.rows],
                       attention_diagnostics=diagnostics(a))
        # Rebuild patients after replacing the uniform scan probabilities.
        by_patient={}
        for row in predictions['scans']: by_patient.setdefault(row['subject_id'],[]).append(row['p_ad'])
        for row in predictions['patients']:
            row['p_ad']=math.fsum(by_patient[row['subject_id']])/len(by_patient[row['subject_id']])
        metrics={name:core.scored([r['label'] for r in predictions[key]],[r['p_ad'] for r in predictions[key]])
                 for name,key in [('slice','slices'),('scan','scans'),('patient','patients')]}
        metrics['attention']=distribution([r['attention_diagnostics'] for r in predictions['scans']])
        metrics['native_scan_CE']=math.fsum(float(loss(z,b.label)) for z,b in zip(native,bags))/len(bags)
        return metrics,predictions


def replay_difference(left,right):
    checks=core.replay_difference(left,right)
    original={r['scan_id']:r for r in left['scans']}; delta=0.
    for row in right['scans']:
        old=original[row['scan_id']]
        core.require(row['slice_order']==old['slice_order'] and len(row['attention_weights'])==len(old['attention_weights']),
                     'Attention slice order differs.')
        delta=max(delta,max(abs(x-y) for x,y in zip(row['attention_weights'],old['attention_weights'])))
    core.require(delta<=2e-5,'Attention replay tolerance exceeded.')
    checks['attention']={'absolute_tolerance':2e-5,'max_absolute_difference':delta,'slice_order_unchanged':True}
    return checks


def verify_native(predictions):
    """Saved tensors only: native attention/logit score and patient hierarchy."""
    by_scan={}
    for row in predictions['slices']:by_scan.setdefault(row['scan_id'],[]).append(row)
    patient={}
    for row in predictions['scans']:
        slices=by_scan[row['scan_id']]
        core.require(row['slice_order']==[r['image_path'] for r in slices],'Saved native slice ordering differs.')
        weights=torch.tensor(row['attention_weights'],dtype=torch.float32)
        z=torch.tensor([r['logits'] for r in slices],dtype=torch.float32)
        native=torch.tensor(row['logits'],dtype=torch.float32)
        core.require(len(weights)==len(slices) and bool(torch.isfinite(weights).all()) and bool((weights>=0).all())
                     and abs(float(weights.sum())-1.)<=2e-6,'Invalid saved attention weights.')
        torch.testing.assert_close((weights[:,None]*z).sum(0),native,atol=2e-5,rtol=2e-6)
        q=float(torch.softmax(native,0)[1])
        core.require(abs(q-row['p_ad'])<=2e-5 and (q>=.5)==(row['p_ad']>=.5),'Saved native scan score differs.')
        patient.setdefault(row['subject_id'],[]).append(row['p_ad'])
    for row in predictions['patients']:
        values=patient[row['subject_id']]
        core.require(row['p_ad']==math.fsum(values)/len(values),'Saved patient mean-scan aggregation differs.')
