"""Aggregate development summaries/export. No inference, fitting or promotion."""
from __future__ import annotations
import csv
import io
import json
import math
from pathlib import Path
import statistics
import zipfile
import torch
import train_convnext_v2 as v2
from g012 import core, storage as st
from g012.engine import identity, verify_replay


def score(metrics):
    m = metrics['patient']; cm = m['confusion_matrix_rows_true_columns_predicted_NC_AD']
    return {'correct':cm[0][0]+cm[1][1], 'AD_detections':cm[1][1],
            **{k:m[k] for k in ('accuracy','macro_f1','auroc','log_loss','brier')},
            'NC_precision':m['per_class']['NC']['precision'],'NC_recall':m['per_class']['NC']['recall'],
            'AD_precision':m['per_class']['AD']['precision'],'AD_recall':m['per_class']['AD']['recall']}


def trajectory(history):
    tail = history[-5:]
    names = ('patient_accuracy','patient_macro_f1','patient_log_loss','patient_brier','ad_recall','nc_recall')
    result = {'epochs':[int(r['epoch']) for r in tail], 'scope':'last up to five COMPLETED epochs; population SD'}
    for key in names:
        values = [float(r[key]) for r in tail if key in r]
        if values:
            result[key] = {'values':values,'mean':statistics.mean(values),'population_sd':statistics.pstdev(values),
                           'minimum':min(values),'maximum':max(values)}
    return result


def paired(a, b):
    left = {r['subject_id']:r for r in a['patients']}; right = {r['subject_id']:r for r in b['patients']}
    core.require(left.keys() == right.keys() and len(left) == len(a['patients']) == len(right) == len(b['patients'])
                 and all(left[k]['label'] == right[k]['label'] for k in left), 'Paired validation membership differs.')
    counts = {'both_correct':0,'candidate_only_correct':0,'reference_only_correct':0,'both_wrong':0}
    for key in left:
        x = int(left[key]['p_ad'] >= .5) == left[key]['label']; y = int(right[key]['p_ad'] >= .5) == right[key]['label']
        name = 'both_correct' if x and y else 'candidate_only_correct' if x else 'reference_only_correct' if y else 'both_wrong'
        counts[name] += 1
    return counts


def screen(row, a1_f1):
    if row.get('selected_score') is None:
        return {'assessable':False,'passes_development_screen':False,'reason':'No completed-epoch selection.'}
    s = row['selected_score']
    gates = {'at_least_86_correct':s['correct'] >= 86,'at_least_22_AD':s['AD_detections'] >= 22,
             'macro_f1_at_least_exact_A1':s['macro_f1'] >= a1_f1,
             'completed_integrity_and_replay':row['protocol_complete'] and row['integrity_and_replay_passed']}
    return {'assessable':row['protocol_complete'],'passes_development_screen':all(gates.values()),'gates':gates,
            'interpretation':'Screen for further confirmation only; no test-performance/clinical claim or adoption.'}


def resource_summary(folder):
    ledger = st.read(folder/'active_budget.json') if (folder/'active_budget.json').exists() else None
    readings = [st.read(path) for path in folder.glob('resource_session-*.json')]
    result = {'active_execution_charged_seconds':ledger['charged_seconds'] if ledger else 0.,
              'invocations':len(ledger['sessions']) if ledger else 0,
              'budget_scope':'supervised training AND selected-replay worker wall time incl setup/validation/I/O/repeated work; five-second prepaid crash reservations; excludes non-candidate smoke, preparation and downtime',
              'measurement_scope':'max recorded process allocator peaks, including persisted partial-session samples; interrupted unflushed observations may be unavailable'}
    if readings:
        for key in ('peak_allocated_gib','peak_reserved_gib'):
            result[key] = max(r[key] for r in readings)
        result['min_sampled_device_free_gib'] = min(r['min_device_free_gib'] for r in readings)
    return result


def compare(args):
    plan, root, binding, bundle, groups = st.context(args)
    with v2.run_lock(root):
        reference = st.read(root/'PRIVATE_a1_reference.json')
        a1 = {'name':'A1','status':'HISTORICAL_RETAINED','selected_epoch':13,'completed_epoch':reference['completed_epoch'],
              'metrics':reference['metrics'],'selected_score':score(reference['metrics']),
              'parameters':reference['parameters'],'history':reference['history'],
              'last_five_completed_epochs':trajectory(reference['history']),
              'updates_per_complete_epoch':664,'recorded_resources':reference['recorded_resources'],
              'resource_scope':'historical summary scopes; active interruption-inclusive budget not available as a matched G012 measurement'}
        rows = []; private = {'A1':reference['predictions']}
        expected_schedules = st.read(root/'PRIVATE_matched_schedule.json')
        for variant in core.VARIANTS:
            folder = root/variant
            row = {'name':variant,'status':'NOT_STARTED','selected_epoch':None,'completed_epoch':0,
                   'metrics':None,'selected_score':None,'history':[],'parameters':27818594,
                   'protocol_complete':False,'integrity_and_replay_passed':False,
                   'updates_per_complete_epoch':531,'native_scan_aggregation':core.NATIVE[variant],
                   'slice_metrics':'diagnostic per-slice softmax, not the checkpoint selection unit',
                   'training_objective':core.LOSSES[variant],'resources':resource_summary(folder)}
            smoke_folder = root/('smoke-'+variant)
            if (smoke_folder/'smoke.json').exists():
                smoke = st.read(smoke_folder/'smoke.json')
                row['non_candidate_smoke'] = {k:smoke[k] for k in ('passed','scope','result','resources','execution_seconds_excluding_process_startup','checkpoint_roundtrip_logits_equal')}
            elif (smoke_folder/'blocker.json').exists():
                row['non_candidate_smoke_blocker'] = st.read(smoke_folder/'blocker.json')
            if (folder/'run_status.json').exists(): row['status'] = st.read(folder/'run_status.json')['status']
            state = None
            if (folder/'completion.json').exists():
                expected = identity(plan,variant,bundle.manifest_hashes); st.verify_completion(folder,expected)
                summary = st.read(folder/'training_summary.json')
                with (folder/'history.csv').open(encoding='utf-8-sig',newline='') as f:
                    hist = [{k:(v == 'True' if k == 'selected_as_best' else float(v)) for k,v in r.items()} for r in csv.DictReader(f)]
                row.update(selected_epoch=summary['selected_epoch'],completed_epoch=summary['completed_epoch'],
                           metrics=summary['metrics'],history=hist,stop_reason=summary['stop_reason'],parameters=summary['parameters'])
                row['resources']['committed_epoch_loop_seconds'] = summary['committed_epoch_loop_seconds']
                private[variant] = st.read(folder/'PRIVATE_selected_predictions.json')
                actual_schedule = st.read(folder/'PRIVATE_order_augmentation_digests.json')
                schedule_ok = len(actual_schedule) == row['completed_epoch'] and all(
                    actual['order_digest'] == expected['order_digest'] and actual['augmentation_digest'] == expected['augmentation_digest']
                    for actual,expected in zip(actual_schedule,expected_schedules['epochs']))
                core.require(schedule_ok and summary['initial_tensor_sha256'] == expected_schedules['initial_tensor_sha256'], 'Matched initialization/order/augmentation integrity failed.')
                row['protocol_complete'] = row['status'] == 'COMPLETED_TRAINING' and row['resources']['active_execution_charged_seconds'] < st.CAP
                if (folder/'checkpoint_replay.json').exists():
                    replay = verify_replay(folder,expected)
                    row['replay'] = {k:replay[k] for k in ('passed','absolute_probability_tolerance','checks','native_aggregation','resources','inference_seconds')}
                    row['integrity_and_replay_passed'] = True
            elif (folder/'PRIVATE_resume.pt').exists() or (folder/'PRIVATE_previous.pt').exists():
                for path in (folder/'PRIVATE_resume.pt',folder/'PRIVATE_previous.pt'):
                    if not path.exists(): continue
                    try: state = torch.load(path,map_location='cpu',weights_only=True); break
                    except Exception: continue
                core.require(state is not None,'No readable partial-run epoch evidence; preserve and investigate.')
                core.require(state['identity'] == identity(plan,variant,bundle.manifest_hashes),'Partial run identity differs.')
                row.update(completed_epoch=state['epoch'],history=state['history'],stop_reason='PROTOCOL_INCOMPLETE')
                if state['best']:
                    row.update(selected_epoch=state['best']['epoch'],metrics=state['best']['metrics'])
                    private[variant] = state['best']['predictions']
                del state
            if row['metrics']:
                row['selected_score'] = score(row['metrics'])
                row['last_five_completed_epochs'] = trajectory(row['history'])
            row['development_screen'] = screen(row,plan['a1_metrics']['patient']['macro_f1'])
            rows.append(row)
        all_rows = {'A1':a1,**{r['name']:r for r in rows}}
        for row in rows:
            if row['selected_score'] is None: continue
            for name in ('A1','G0'):
                ref = all_rows[name]
                if ref['selected_score'] is None: continue
                delta = {k:row['selected_score'][k]-ref['selected_score'][k] for k in row['selected_score']}
                row['versus_'+name] = {'selected_score_differences':delta,
                    'paired_correctness_counts':paired(private[row['name']],private[name]),
                    'ordinary_NLL_plus_0_02_review_warning':delta['log_loss'] >= .02,
                    'warning_is_not_a_hard_gate':True,
                    'higher_selected_patient_accuracy':delta['correct'] > 0}
        historical = st.read(root/'a1_historical_regression.json') if (root/'a1_historical_regression.json').exists() else {'passed':False}
        for row in rows:
            row['integrity_and_replay_passed'] = row['integrity_and_replay_passed'] and historical['passed'] and historical.get('plan_sha256') == core.digest(plan)
            row['development_screen'] = screen(row,plan['a1_metrics']['patient']['macro_f1'])
        result = {'study':'G012_Seed42','plan_sha256':core.digest(plan),'A1':a1,'variants':rows,
            'historical_regression':historical,'total_active_execution_charged_seconds':sum(r['resources']['active_execution_charged_seconds'] for r in rows),
            'all_three_protocols_completed':all(r['protocol_complete'] and r['integrity_and_replay_passed'] for r in rows),
            'independent_validation_patients':102,'repeated_use_of_same_patients':True,
            'automatic_adoption':False,'test_inference_performed':False,
            'attribution':'G1 vs G0 isolates loss in matched grouped batching; G2 also changes scan aggregation; G0 vs historical A1 changes batching/update count. One seed does not establish repeatability.'}
        identifier = core.digest(result)[:20]; destination = root/'aggregate_exports'/identifier
        destination.mkdir(parents=True,exist_ok=True)
        st.immutable_json(destination/'G012_STUDY_SUMMARY.json',result)
        st.immutable_json(destination/'frozen_plan.json',plan)
        st.immutable_json(destination/'configurations.json',plan['configurations'])
        for name,row in all_rows.items():
            path = destination/(name+'_history.csv')
            value = st.history_csv(row['history'])
            if path.exists(): core.require(path.read_text(encoding='utf-8') == value,'Aggregate history changed.')
            else: path.write_text(value,encoding='utf-8',newline='\n')
        st.immutable_json(root/'aggregate_export_pointer.json',{'relative_directory':destination.relative_to(root).as_posix()}) if not (root/'aggregate_export_pointer.json').exists() else v2.atomic_json(root/'aggregate_export_pointer.json',{'relative_directory':destination.relative_to(root).as_posix()})
        plots(destination,result)
        print('G012_COMPARISON_WRITTEN',destination/'G012_STUDY_SUMMARY.json')
        print('ALL_THREE_COMPLETE_AND_REPLAYED',result['all_three_protocols_completed'])
        return destination,result


def plots(destination,result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    path = destination/'development_trajectories.png'
    if not path.exists():
        fig,axes = plt.subplots(1,3,figsize=(15,4))
        for row in [result['A1'],*result['variants']]:
            h = row['history']
            for ax,key in zip(axes,('patient_accuracy','patient_macro_f1','patient_log_loss')):
                if h: ax.plot([r['epoch'] for r in h],[r[key] for r in h],label=row.get('name','A1'))
                ax.set(xlabel='Completed epoch',ylabel=key)
        for ax in axes: ax.legend()
        fig.suptitle('Repeated development cohort; no test inference');fig.tight_layout();fig.savefig(path,dpi=130);plt.close(fig)
    path = destination/'separate_training_objectives.png'
    if not path.exists():
        fig,axes = plt.subplots(1,3,figsize=(15,4))
        for ax,row in zip(axes,result['variants']):
            h = row['history']
            if h: ax.plot([r['epoch'] for r in h],[r['training_objective'] for r in h])
            ax.set(title=row['name'],xlabel='Completed epoch',ylabel=row['training_objective'])
        fig.suptitle('Different objective scales; lower raw objective is not comparative performance');fig.tight_layout();fig.savefig(path,dpi=130);plt.close(fig)


ALLOWLIST = ('frozen_plan.json','configurations.json','G012_STUDY_SUMMARY.json',
             'A1_history.csv','G0_history.csv','G1_history.csv','G2_history.csv',
             'development_trajectories.png','separate_training_objectives.png')


def validate_public(value):
    """Export is built from aggregates, never by recursively copying a study dir."""
    if isinstance(value,dict):
        core.require(not (set(value) & {'subject_id','scan_id','image_path','logits','model_state','optimizer_state','predictions','data_root','splits_dir'}),
                     'Private field reached aggregate export.')
        for item in value.values(): validate_public(item)
    elif isinstance(value,list):
        for item in value: validate_public(item)
    elif isinstance(value,str):
        import re
        core.require(not re.search(r'\b\d{3}_S_\d{4}\b|[A-Za-z]:[\\/]|^/',value), 'Identifier/absolute path reached export.')


def bundle(args):
    destination,result = compare(args)
    # Reacquire the study lock: export immutable aggregate files only.
    root = args.study_root.resolve()
    with v2.run_lock(root):
        for name in ('frozen_plan.json','configurations.json','G012_STUDY_SUMMARY.json'):
            validate_public(st.read(destination/name))
        path = destination/'COMP3710_G012_Seed42_Results.zip'
        if path.exists():
            with zipfile.ZipFile(path) as archive:
                core.require(tuple(archive.namelist()) == ALLOWLIST and all(archive.read(n) == (destination/n).read_bytes() for n in ALLOWLIST), 'Existing ZIP differs; do not overwrite it.')
        else:
            temporary = path.with_suffix('.zip.tmp')
            with zipfile.ZipFile(temporary,'x',compression=zipfile.ZIP_DEFLATED) as archive:
                for name in ALLOWLIST: archive.write(destination/name,name)
            temporary.replace(path)
        print('AGGREGATE_RESULTS_ZIP',path)
        return path
