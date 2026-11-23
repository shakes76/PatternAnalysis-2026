"""Cached-evidence-only comparison and explicit aggregate ZIP allowlist."""
from __future__ import annotations
import csv
import io
import json
import math
from pathlib import Path
import zipfile
import torch
import train_convnext_v2 as v2
from g012 import core,results as old
from g3 import storage as st,execution as ex,model as gm


def history(folder):
    rows=list(csv.DictReader(io.StringIO((folder/'history.csv').read_text(encoding='utf-8-sig'))))
    return [{k:(v=='True' if k=='selected_as_best' else v if k=='preclip_gradient_norms_json' else float(v))
             for k,v in row.items()} for row in rows]


def scored(predictions):
    return {name:core.scored([r['label'] for r in predictions[key]],[r['p_ad'] for r in predictions[key]])
            for name,key in [('slice','slices'),('scan','scans'),('patient','patients')]}


def paired(reference,candidate):
    result=old.paired(candidate,reference)
    a={r['subject_id']:r for r in reference['patients']}; b={r['subject_id']:r for r in candidate['patients']}
    result['by_class']={}
    for label,name in [(0,'NC'),(1,'AD')]:
        gained=lost=0
        for key,x in a.items():
            if x['label']!=label:continue
            before=int(x['p_ad']>=.5)==label; after=int(b[key]['p_ad']>=.5)==label
            gained+=int(after and not before);lost+=int(before and not after)
        result['by_class'][name]={'gained_correct':gained,'lost_correct':lost,'net_correct':gained-lost}
    return result


def privacy(value):
    raw=json.dumps(value,allow_nan=False)
    for forbidden in ('subject_id','scan_id','image_path','slice_order','attention_weights','PRIVATE_',':\\','C:/Users','C:\\\\Users'):
        core.require(forbidden not in raw,'Aggregate export contains a forbidden private field/path.')


def compare(args):
    plan,root,binding,bundle,groups=st.context(args)
    with v2.run_lock(root):
        ref=st.read(root/'PRIVATE_a1_reference.json'); predictions={'A1':ref['predictions']}
        core.require(ref['metrics']['patient']['macro_f1']==.8012151782643586,'Exact A1 macro-F1 differs.')
        rows=[{'name':'A1','evidence_scope':'historical retained; no new inference','selected_epoch':13,
               'configuration':plan['a1_configuration'],
               'completed_epoch':ref['completed_epoch'],'metrics':ref['metrics'],'selected_score':old.score(ref['metrics']),
               'history':ref['history'],'parameters':ref['parameters'],'resources':ref['recorded_resources'],
               'last_five_completed_epochs':old.trajectory(ref['history'])}]
        for name,folder in [('G2',args.historical_root/'G2'),('G3',root/'G3')]:
            historical=name=='G2'
            row={'name':name,'designated_matched_uniform_control':historical,
                 'configuration':core.config('G2') if historical else gm.configuration(),
                 'evidence_scope':'historical; receipt reused, no new inference' if historical else 'new G3 attempt',
                 'protocol_complete':False,'integrity_and_replay_passed':False,
                 'parameters':27818594 if historical else 28015586,'updates_per_complete_epoch':531,
                 'slice_metrics':'diagnostic only; not checkpoint selection',
                 'status':st.read(folder/'run_status.json')['status'] if (folder/'run_status.json').exists() else 'NOT_STARTED',
                 'resources':old.resource_summary(folder),'selected_score':None,'history':[]}
            if (folder/'completion.json').exists():
                summary=st.read(folder/'training_summary.json'); expected=summary['identity'] if historical else ex.identity(plan,name,bundle.manifest_hashes)
                st.verify_completion(folder,expected);hist=history(folder)
                selected=st.read(folder/'PRIVATE_selected_predictions.json');predictions[name]=selected
                metrics=scored(selected)
                core.require(all(metrics[k]==summary['metrics'][k] for k in metrics),'Cached selected metric mismatch.')
                best=max(hist,key=lambda r:(r['patient_accuracy'],r['patient_macro_f1'],-r['patient_log_loss']))
                core.require(int(best['epoch'])==summary['selected_epoch'] and len(hist)==summary['completed_epoch'],'Frozen selection/history mismatch.')
                core.require(summary['completed_epoch']==30 or (summary['completed_epoch']>=12
                    and summary['completed_epoch']-summary['selected_epoch']>=8
                    and summary['stop_reason']=='PATIENCE_AFTER_MINIMUM_12'),'Run stopping rule differs.')
                schedules=st.read(root/'PRIVATE_matched_schedule.json')['epochs']
                prefix=st.read(folder/'PRIVATE_order_augmentation_digests.json')
                core.require(len(prefix)==len(hist) and all(all(p[k]==s[k] for k in ('order_digest','augmentation_digest')) for p,s in zip(prefix,schedules)), 'Completed schedules differ.')
                row.update(selected_epoch=summary['selected_epoch'],completed_epoch=summary['completed_epoch'],metrics=metrics,
                           selected_score=old.score(metrics),history=hist,last_five_completed_epochs=old.trajectory(hist),stop_reason=summary['stop_reason'])
                row['resources']['committed_epoch_loop_seconds']=summary['committed_epoch_loop_seconds']
                row['protocol_complete']=row['status']=='COMPLETED_TRAINING' and row['resources']['active_execution_charged_seconds']<10800
                if (folder/'checkpoint_replay.json').exists():
                    if historical:
                        from g012.engine import verify_replay
                        replay=verify_replay(folder,expected)
                    else:replay=ex.verify_replay(folder,expected)
                    row['integrity_and_replay_passed']=True
                    row['replay']={k:replay[k] for k in ('passed','checks','absolute_probability_tolerance','inference_seconds','resources')}
                if not historical:
                    row['selected_attention']=summary['metrics']['attention']
                    row['native_scan_CE']=summary['metrics']['native_scan_CE']
                    row['initialization']=st.read(root/'initialization.json')
            elif (folder/'PRIVATE_resume.pt').exists():
                state=torch.load(folder/'PRIVATE_resume.pt',map_location='cpu',weights_only=True)
                core.require(state['identity']==ex.identity(plan,name,bundle.manifest_hashes),'Partial state identity differs.')
                row.update(completed_epoch=state['epoch'],history=state['history'],last_five_completed_epochs=old.trajectory(state['history']))
                if state['best']:
                    row.update(selected_epoch=state['best']['epoch'],metrics=state['best']['metrics'],selected_score=old.score(state['best']['metrics']))
                    predictions[name]=state['best']['predictions']
                row['interpretation']='INCONCLUSIVE: incomplete training, no successful protocol claim.'
            if not historical:row['screen']=old.screen(row,ref['metrics']['patient']['macro_f1'])
            else:row['historical_A1_screen']=old.screen(row,ref['metrics']['patient']['macro_f1'])
            row['committed_optimizer_updates']=531*int(row.get('completed_epoch',0))
            if (folder/'active_budget.json').exists():
                ledger=st.read(folder/'active_budget.json')
                row['resources']['sessions']=[{k:v for k,v in s.items() if k in
                    ('started_utc','finished_utc','charged_seconds','observed_elapsed_seconds','status')} for s in ledger['sessions']]
            rows.append(row)
        paired_results={}
        if 'G3' in predictions:
            for name in ('A1','G2'):
                paired_results[name+'_to_G3']=paired(predictions[name],predictions['G3'])
                candidate=rows[-1]['selected_score'];reference=next(r['selected_score'] for r in rows if r['name']==name)
                paired_results[name+'_to_G3']['metric_differences']={k:candidate[k]-reference[k] for k in
                    ('correct','AD_detections','accuracy','macro_f1','auroc','log_loss','brier','NC_recall','AD_recall')}
                paired_results[name+'_to_G3']['NLL_deterioration_over_0_02_warning']=candidate['log_loss']-reference['log_loss']>.02
        value={'study':'G3_Seed42','control_mode':'REUSE_G2','rows':rows,'paired_changes':paired_results,
               'limitations':['one seed','reused development validation patients','prior test exposure; unmet 0.80 test requirement',
                              'attention weights are model weights, not clinical/anatomical explanations','no automatic promotion'],
               'budgets':{'new_G3_cap_seconds':10800,'training_replay_reserve_seconds':300,'historical_G2_recharged':False},
               'preparation':st.read(root/'preparation.json'),
               'non_candidate_smoke':{k:v for k,v in st.read(root/'smoke-G3'/'smoke.json').items() if k not in ('identity',)} if (root/'smoke-G3'/'smoke.json').exists() else None}
        privacy(value)
        destination=root/'aggregate_exports'/core.digest(value)[:20];destination.mkdir(parents=True,exist_ok=True)
        st.immutable_json(destination/'comparison.json',value)
        public_plan={k:v for k,v in plan.items() if k not in ('sources','historical_files')}
        privacy(public_plan);st.immutable_json(destination/'plan.json',public_plan)
        st.immutable_json(destination/'control_comparability.json',st.read(st.BASE/'g3'/'G3_CONTROL_COMPARABILITY.json'))
        for row in rows:
            if row['history']:st.write_text_atomic(destination/(row['name']+'_history.csv'),st.history_csv(row['history']))
        text='# G3 seed-42 cached-evidence comparison\n\nMatched uniform control: historical G2; not newly reproduced.\n\n'
        text+='| Model | Selected/completed | Correct | AD detected | Macro-F1 | NLL | Brier | Status |\n|---|---|---|---|---|---|---|---|\n'
        for row in rows:
            s=row.get('selected_score')
            text+=f"| {row['name']} | {row.get('selected_epoch','?')}/{row.get('completed_epoch',0)} | {s['correct'] if s else '?'} | {s['AD_detections'] if s else '?'} | {s['macro_f1'] if s else '?'} | {s['log_loss'] if s else '?'} | {s['brier'] if s else '?'} | {row.get('status','HISTORICAL')} |\n"
        text+='\nFull native scan/slice diagnostics, both-class precision/recall, confusion counts, complete histories, last-five summaries, paired changes, attention and resources are in comparison.json. NLL >0.02 deterioration is a warning, not a gate. Incomplete runs are inconclusive. No clinical benefit or model promotion follows a screen pass.\n'
        text+='\nG3 initial screen: '+json.dumps(rows[-1]['screen'],sort_keys=True)+'\n'
        for name,change in paired_results.items():
            text+='\n'+name+': '+json.dumps(change,sort_keys=True)+'\n'
        st.write_text_atomic(destination/'RESULTS.md',text)
        plot(destination,value)
        st.immutable_json(root/'latest_aggregate.json',{'directory':destination.name}) if not (root/'latest_aggregate.json').exists() else v2.atomic_json(root/'latest_aggregate.json',{'directory':destination.name})
        print('G3_AGGREGATE_COMPARISON_EXPORTED',destination)


def plot(destination,value):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for row in value['rows']:
        h=row.get('history',[])
        for ax,key in zip(axes,('patient_accuracy','patient_log_loss')):
            if h:ax.plot([r['epoch'] for r in h],[r[key] for r in h],label=row['name'])
            ax.set(xlabel='Completed epoch',ylabel=key);ax.legend()
    fig.tight_layout();fig.savefig(destination/'trajectories.png',dpi=150);plt.close(fig)


ALLOWLIST=('comparison.json','plan.json','control_comparability.json','A1_history.csv','G2_history.csv','G3_history.csv','RESULTS.md','trajectories.png')


def bundle(args):
    plan,root,binding,bundle,groups=st.context(args)
    with v2.run_lock(root):
        name=st.read(root/'latest_aggregate.json')['directory']
        core.require(len(name)==20 and all(c in '0123456789abcdef' for c in name),'Unsafe export directory.')
        folder=root/'aggregate_exports'/name
        for filename in ALLOWLIST:
            path=folder/filename
            if path.exists() and path.suffix=='.json':privacy(st.read(path))
            elif path.exists() and path.suffix in ('.csv','.md'):privacy(path.read_text(encoding='utf-8'))
        target=folder/'COMP3710_G3_Seed42_Results.zip'
        if target.exists():
            with zipfile.ZipFile(target) as archive:
                core.require(set(archive.namelist())=={n for n in ALLOWLIST if (folder/n).is_file()}
                    and all(archive.read(n)==(folder/n).read_bytes() for n in archive.namelist()),'Existing ZIP differs.')
        else:
            temporary=target.with_suffix('.zip.tmp')
            with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as archive:
                for filename in ALLOWLIST:
                    if (folder/filename).is_file():archive.write(folder/filename,filename)
            temporary.replace(target)
        print('G3_RESULTS_BUNDLE',target,'SHA256',st.sha(target))
