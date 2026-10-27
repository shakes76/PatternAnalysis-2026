"""Synthetic metadata-only fixtures, never actual experiment results.

The fixture has the full cohort sizes but intentionally artificial confusion
counts, checkpoint bytes and surface distances. It opens no MRI/prediction array
and makes no contour or tensor-identity claim. Unequal visits expose accidental
case pooling; a missed synthetic prostate exposes unmatched HD95 comparisons.
"""
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import summarize_v2 as summary

OLD_SHAPE = [64, 128, 128]
NEW_SHAPE = [96, 192, 192]
MODELS = ('unet2d', 'improved3d')
V2_SCHEDULE = [1, 5, 10, 15, 20, 25, 30]


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False, indent=1) + '\n')


def counts(shape, n):
    return [math.prod(shape) - 5*n, n, n, n, n, n]


def make_case(record, shape, n, error, surface=False, missed=False):
    gt = counts(shape, n)
    matrix = np.diag(gt).astype(np.int64)
    if missed:
        matrix[5, 5] = 0
        matrix[5, 1] = n
    else:
        matrix[4, 4] -= error
        matrix[5, 5] -= error
        matrix[4, 5] = error
        matrix[5, 4] = error
    per_class = []
    for c in range(6):
        t, p, tp = int(matrix[c].sum()), int(matrix[:, c].sum()), int(matrix[c, c])
        per_class.append(dict(class_index=c, target_voxels=t, predicted_voxels=p,
                              intersection=tp, union=t+p-tp,
                              dice=2*tp/(t+p) if t+p else None,
                              iou=tp/(t+p-tp) if t+p-tp else None))
    case = dict(patient=record['patient'], visit=record['visit'], per_class=per_class,
                foreground_mean_dice=sum(x['dice'] for x in per_class[1:])/5,
                foreground_mean_iou=sum(x['iou'] for x in per_class[1:])/5,
                shape_dhw=shape, spacing_dhw=record['spacing_dhw_mm'],
                inference_ms=10.+int(record['visit'][4:]), peak_allocated_cuda_bytes=100000)
    if surface:
        case.pop('inference_ms'); case.pop('peak_allocated_cuda_bytes')
        case['prostate_surface'] = dict(empty_status='missed' if missed else 'both_present',
            hd95_mm=None if missed else 1.+.5*error+.1*int(record['visit'][4:]))
    return case


def aggregate(cases, surface=False):
    groups = defaultdict(list)
    for c in cases:
        groups[c['patient']].append(c)
    patients = []
    for name, visits in sorted(groups.items()):
        row = dict(patient=name, num_cases=len(visits),
            per_class=[dict(class_index=c, **{m: sum(v['per_class'][c][m] for v in visits)/len(visits)
                                             for m in ('dice', 'iou')}) for c in range(6)],
            **{f'foreground_mean_{m}': sum(v[f'foreground_mean_{m}'] for v in visits)/len(visits)
               for m in ('dice', 'iou')})
        if surface:
            values = [v['prostate_surface']['hd95_mm'] for v in visits if v['prostate_surface']['hd95_mm'] is not None]
            row.update(prostate_hd95_mm=sum(values)/len(values) if values else None,
                       hd95_defined_cases=len(values))
        patients.append(row)
    result = dict(num_cases=len(cases), num_patients=len(patients), patients=patients,
        per_class=[dict(class_index=c, **{m: sum(p['per_class'][c][m] for p in patients)/len(patients)
                                         for m in ('dice', 'iou')}) for c in range(6)],
        **{f'foreground_mean_{m}': sum(p[f'foreground_mean_{m}'] for p in patients)/len(patients)
           for m in ('dice', 'iou')})
    if surface:
        defined = [p['prostate_hd95_mm'] for p in patients if p['prostate_hd95_mm'] is not None]
        result.update(prostate_hd95_mm_conditional=sum(defined)/len(defined) if defined else None,
                      prostate_empty_status_counts=dict(Counter(c['prostate_surface']['empty_status'] for c in cases)),
                      hd95_defined_cases=sum(p['hd95_defined_cases'] for p in patients),
                      hd95_defined_patients=len(defined))
    return result


def pair_summary(before, after):
    by = {(x['patient'], x['visit']): x for x in after}
    groups = defaultdict(list)
    for b in before:
        groups[b['patient']].append((b, by[b['patient'], b['visit']]))
    patients, hd_patients = [], []
    for name, pairs in sorted(groups.items()):
        row = dict(patient=name, num_cases=len(pairs))
        for label, accessor in [('prostate', lambda c:c['per_class'][5]['dice']),
                                ('foreground', lambda c:c['foreground_mean_dice'])]:
            b, a = (sum(accessor(p[i]) for p in pairs)/len(pairs) for i in (0, 1))
            row.update({f'v1_{label}_dice': b, f'v2_{label}_dice': a, f'{label}_delta': a-b})
        patients.append(row)
        supported = [p for p in pairs if all(x['prostate_surface']['hd95_mm'] is not None for x in p)]
        if supported:
            b, a = (sum(p[i]['prostate_surface']['hd95_mm'] for p in supported)/len(supported) for i in (0, 1))
            hd_patients.append(dict(patient=name, num_common_cases=len(supported), v1_hd95_mm=b,
                                    v2_hd95_mm=a, delta_hd95_mm=a-b))
    supported_count = sum(p['num_common_cases'] for p in hd_patients)
    return dict(patients=patients, paired_hd95=dict(eligible_case_pairs=supported_count,
        excluded_case_pairs=len(before)-supported_count, num_patients=len(hd_patients), patients=hd_patients,
        **{field: sum(x[field] for x in hd_patients)/len(hd_patients) if hd_patients else None
           for field in ('v1_hd95_mm', 'v2_hd95_mm', 'delta_hd95_mm')}))


class SyntheticStudy:
    """A complete artificial directory layout with no pixel/checkpoint tensors."""
    def __init__(self, root):
        self.root = Path(root)
        self.review, self.project, self.artifacts = (self.root/n for n in ('review', 'v1_project', 'artifacts'))
        self.review.mkdir(); self.project.mkdir(); self.artifacts.mkdir()
        for p in [self.project/'modules.py', self.review/'v2_pipeline.py', self.review/'formal_v2_gpu.slurm']:
            p.write_text('SYNTHETIC FIXTURE ONLY; never executable model code\n')
        self.old, self.new = self.manifests()
        write(self.project/'artifacts/cache64/manifest.json', self.old)
        self.old_manifest = self.project/'artifacts/cache64/manifest.json'
        identity = dict(manifest_sha256=summary.sha(self.old_manifest), records={})
        for r in self.old['records']:
            identity['records'][r['cache_path']] = dict(arrays={
                name:dict(sha256=str(index+1)*64) for index,name in enumerate(('mask','spacing'))})
        write(self.project/'artifacts/cache64/cache_identity.json', identity)
        self.identity = identity
        self.old_digests = {}
        for model in MODELS:
            path = self.project/f'artifacts/full_seed42/{model}/best.pt'
            path.parent.mkdir(parents=True)
            path.write_bytes(('SYNTHETIC v1 selected bytes '+model).encode())
            self.old_digests[model] = summary.sha(path)
        cfg = dict(epochs=30, train_slices_per_scan=32, train_patches_per_scan=2,
                   train_batch_size_2d=16, train_batch_size_3d=1, validation_epochs=V2_SCHEDULE,
                   seed=3711,amp=True)
        originals = {str(p.relative_to(self.project)):summary.sha(p) for p in [
            self.project/'modules.py',self.old_manifest,self.project/'artifacts/cache64/cache_identity.json',
            *(self.project/f'artifacts/full_seed42/{m}/best.pt' for m in MODELS)]}
        self.protocol = dict(configuration=cfg, required_original_sha256=originals,
            required_review_sha256={p.name:summary.sha(p) for p in [self.review/'v2_pipeline.py',self.review/'formal_v2_gpu.slurm']},
            adoption=dict(minimum_prostate_dice_gain=.01,minimum_foreground_dice_difference=-.01))
        write(self.review/'protocol_v2.json',self.protocol)
        self.protocol_path = self.review/'protocol_v2.json'
        self.new.update(original_manifest_sha256=summary.sha(self.old_manifest),protocol_sha256=summary.sha(self.protocol_path))
        write(self.artifacts/'cache_v2/manifest.json',self.new)
        self.new_manifest = self.artifacts/'cache_v2/manifest.json'
        self.identities = {m:dict(model=m,manifest_sha256=summary.sha(self.new_manifest),
            protocol_sha256=summary.sha(self.protocol_path),configuration=cfg,
            source_hashes=originals,v2_pipeline_sha256=summary.sha(self.review/'v2_pipeline.py')) for m in MODELS}
        self.new_digests, selected = {}, {}
        for model in MODELS:
            self.make_run('v1',model)
            selected[model] = self.make_run('v2',model)
        self.freeze = dict(status='frozen',manifest_sha256=summary.sha(self.new_manifest),
                           protocol_sha256=summary.sha(self.protocol_path),models=selected)
        write(self.artifacts/'selection_v2.json',self.freeze)
        self.freeze_path = self.artifacts/'selection_v2.json'
        completion = dict(status='completed',manifest_bytes_unchanged=True,
            manifest_sha256=summary.sha(self.new_manifest),protocol_sha256=summary.sha(self.protocol_path),
            selection_freeze_sha256=summary.sha(self.freeze_path),
            records=[dict(patient=r['patient'],visit=r['visit'],class_voxel_counts=counts(NEW_SHAPE,20))
                     for r in self.new['records'] if r['split']=='test'])
        write(self.artifacts/'cache_v2/test_completion_receipt.json',completion)
        self.raw_paths={}
        for split in ('validation','test'):
            for model in MODELS:
                for version in ('v1','v2'):
                    evaluation=self.raw_evaluation(version,model,split)
                    path=(self.project/f'artifacts/{"validation_seed42" if split=="validation" else "test_final"}/{model}/evaluation.json'
                          if version=='v1' else self.artifacts/f'{"validation_v2" if split=="validation" else "test_v2"}/{model}/evaluation.json')
                    write(path,evaluation);self.raw_paths[version,model,split]=path
            self.make_common(split)

    def manifests(self):
        splits = dict(train=[f'T{i:02}' for i in range(26)],
                      validation=[f'V{i:02}' for i in range(6)],test=[f'Q{i:02}' for i in range(6)])
        visits = dict(train=[5]*13+[6]*13,validation=[3,5,6,8,8,8],test=[1,2,3,8,8,8])
        old, new = [], []
        for split,names in splits.items():
            for pi,(name,num) in enumerate(zip(names,visits[split])):
                a=np.diag([2.,2.,2.5,1.]);a[:3,3]=[pi*3.,-20.,10.]
                t=np.eye(4);t[:3,:3]*=2/3;t[:3,3]=-.5/3
                for vi in range(num):
                    common=dict(patient=name,visit=f'Week{vi}',split=split,cache_path=f'volumes/{name}_Week{vi}.npz')
                    old.append(dict(common,class_voxel_counts=counts(OLD_SHAPE,10),
                        resampled_affine_xyz=a.tolist(),spacing_dhw_mm=[2.5,2.,2.]))
                    row=dict(common,resampled_affine_xyz=(a@t).tolist(),spacing_dhw_mm=[5/3,4/3,4/3])
                    if split!='test':row['class_voxel_counts']=counts(NEW_SHAPE,20)
                    new.append(row)
        return (dict(patient_splits=splits,label_ids=list(range(6)),target_shape_dhw=OLD_SHAPE,records=old),
                dict(patient_splits=deepcopy(splits),label_ids=list(range(6)),target_shape_dhw=NEW_SHAPE,records=new))

    def raw_evaluation(self, version,model,split,epoch=None):
        if epoch is None:epoch=26 if version=='v1' else 25
        manifest=self.old if version=='v1' else self.new
        base = (3 if model=='unet2d' else 1) if version=='v1' else (2 if model=='unet2d' else 8)
        if version=='v1' and epoch!=26:base=5 if epoch<26 else base+1
        if version=='v2':base+=2*(5-min(V2_SCHEDULE.index(epoch),5))
        cases=[]
        for r in manifest['records']:
            if r['split']!=split:continue
            variant=int(r['patient'][1:])%2
            error=base+variant*(1 if version=='v1' else 2)
            missed=version=='v2' and model=='improved3d' and split=='test' and r['patient']=='Q00'
            cases.append(make_case(r,OLD_SHAPE if version=='v1' else NEW_SHAPE,
                                   10 if version=='v1' else 20,error,missed=missed))
        result=dict(cases=cases,aggregate=aggregate(cases),split=split,
            checkpoint_epoch=26 if version=='v1' else 25,
            checkpoint_sha256=(self.old_digests if version=='v1' else self.new_digests).get(model,'0'*64),
            mean_inference_ms=sum(c['inference_ms'] for c in cases)/len(cases),
            peak_allocated_cuda_bytes=100000,latency_protocol='SYNTHETIC transfer/forward/argmax timing fixture')
        if version=='v1':
            result.update(model_config=dict(model=model),manifest_sha256=summary.sha(self.old_manifest),
                          gpu_name='SYNTHETIC_GPU',device='cuda',python_version='3.fixture',torch_version='fixture',cuda_version='fixture')
        else:
            result['identity']=deepcopy(self.identities[model])
            if split=='test':result['selection_freeze_sha256']=summary.sha(self.freeze_path)
        return result

    def make_run(self,version,model):
        root=self.project/f'artifacts/full_seed42/{model}' if version=='v1' else self.artifacts/f'formal_v2/{model}'
        root.mkdir(parents=True,exist_ok=True)
        if version=='v2':
            (root/'best.pt').write_bytes(('SYNTHETIC v2 selected bytes '+model).encode())
            self.new_digests[model]=summary.sha(root/'best.pt')
        cfg=self.protocol['configuration']
        examples=143*(64 if model=='unet2d' else 1) if version=='v1' else 143*(32 if model=='unet2d' else 2)
        batch=16 if model=='unet2d' else 1
        batches=math.ceil(examples/batch)
        hist=[]
        for epoch in range(1,31):
            scheduled=version=='v1' or epoch in V2_SCHEDULE
            ev=self.raw_evaluation(version,model,'validation',epoch) if scheduled else None
            row=dict(epoch=epoch,train_loss=1./epoch,training_examples=examples,training_batches=batches,
                optimizer_steps=batches-(epoch==1),amp_overflow_skipped_optimizer_steps=int(epoch==1),
                train_seconds=2.,peak_allocated_cuda_bytes_training=100000,validation=ev)
            if version=='v1':row['validation_foreground_dice']=ev['aggregate']['foreground_mean_dice']
            else:
                requested=examples//2
                cs=[0]+[requested//5]*5;cs[5]+=requested-sum(cs)
                row.update(validation_prostate_dice=ev['aggregate']['per_class'][5]['dice'] if ev else None,
                    sampling=dict(foreground_requested=requested,foreground_missing_fallback=0,
                                  selected_class_counts=cs,hw_flip_count=examples))
            hist.append(row)
        run=dict(status='completed',completed_epochs=30,best_epoch=26 if version=='v1' else 25,
                 parameters=481830 if model=='unet2d' else 1627634,gpu_name='SYNTHETIC_GPU',
                 device='cuda',python_version='3.fixture',torch_version='fixture',cuda_version='fixture')
        if version=='v1':
            run['config']=dict(batch_size=batch,model=model,manifest_sha256=summary.sha(self.old_manifest))
        else:
            expected=np.array(counts(NEW_SHAPE,20))*143
            weights=1/np.sqrt(expected/expected.sum());weights/=weights.mean();weights=np.clip(weights,.05,10);weights/=weights.mean()
            run.update(identity=deepcopy(self.identities[model]),best_validation_prostate_dice=hist[24]['validation_prostate_dice'],
                       best_checkpoint_sha256=self.new_digests[model],training_voxel_counts=expected.tolist(),
                       class_weights=weights.tolist(),numpy_version='fixture')
        write(root/'history.json',hist);write(root/'run.json',run)
        if version=='v2':return dict(completed_epochs=30,checkpoint_epoch=25,
            validation_prostate_dice=run['best_validation_prostate_dice'],checkpoint_sha256=self.new_digests[model],
            history_sha256=summary.sha(root/'history.json'),run_sha256=summary.sha(root/'run.json'))

    def make_common(self,split):
        report=dict(split=split,retrospective_test_acknowledged=split=='test',models={},common_gt_inputs=[],
            input_files={k:dict(sha256=summary.sha(path)) for k,path in [
                ('v1_manifest',self.old_manifest),('v2_manifest',self.new_manifest),
                ('v1_cache_identity',self.project/'artifacts/cache64/cache_identity.json'),
                ('selection_freeze',self.freeze_path)]})
        report['input_files']['evaluations']={}
        for r in self.old['records']:
            if r['split']==split:
                arrays=self.identity['records'][r['cache_path']]['arrays']
                report['common_gt_inputs'].append(dict(patient=r['patient'],visit=r['visit'],
                    decoded_mask_sha256=arrays['mask']['sha256'],decoded_spacing_sha256=arrays['spacing']['sha256']))
        for model in MODELS:
            entries={}
            for version in ('v1','v2'):
                path=self.raw_paths[version,model,split];raw=summary.load(path)
                report['input_files']['evaluations'][f'{version}_{model}']=dict(sha256=summary.sha(path),checkpoint_sha256=raw['checkpoint_sha256'])
                new_index={(r['patient'],r['visit']):r for r in self.new['records']}
                cases=[]
                for r in self.old['records']:
                    if r['split']!=split:continue
                    variant=int(r['patient'][1:])%2
                    error=(3 if model=='unet2d' else 1) if version=='v1' else (1 if model=='unet2d' else 4)
                    missed=version=='v2' and model=='improved3d' and split=='test' and r['patient']=='Q00'
                    row=make_case(r,OLD_SHAPE,10,error+variant,surface=True,missed=missed)
                    if version=='v2':
                        rev=np.eye(4)[[2,1,0,3]]
                        row['target_common_dhw_to_source_v2_dhw']=(rev@np.linalg.inv(new_index[r['patient'],r['visit']]['resampled_affine_xyz'])@np.asarray(r['resampled_affine_xyz'])@rev).tolist()
                    cases.append(row)
                entries[version]=dict(cases=cases,aggregate=aggregate(cases,True))
            b,a=(entries[v]['aggregate'] for v in ('v1','v2'))
            pd=a['per_class'][5]['dice']-b['per_class'][5]['dice'];fd=a['foreground_mean_dice']-b['foreground_mean_dice']
            passed=pd>=.01-1e-12 and fd>=-.01-1e-12
            entries.update(paired=pair_summary(entries['v1']['cases'],entries['v2']['cases']),
                gate=dict(prostate_dice_delta=pd,foreground_dice_delta=fd,minimum_prostate_delta=.01,
                    minimum_foreground_delta=-.01,floating_point_tolerance=1e-12,numeric_conditions_pass=passed,
                    replacement_supported_by_validation_gate=passed if split=='validation' else None))
            report['models'][model]=entries
        write(self.artifacts/f'common_{split}/common_grid_comparison.json',report)

    def args(self):
        return SimpleNamespace(review_root=self.review,v1_project=self.project,artifacts_root=self.artifacts,
            protocol=None,selection_freeze=None,common_validation=None,common_test=None,
            output_dir=self.root/'final_output')

    def mutate(self,path,callback):
        obj=summary.load(path);callback(obj);write(path,obj)

    def add_runtime_receipts(self):
        adapter=self.review/'evaluate_v2_deterministic.py'
        adapter.write_text('SYNTHETIC runtime adapter bytes, never executable\n')
        amendment=self.review/'engineering_execution_amendment.json'
        write(amendment,dict(fixture_only=True,protocol_sha256=summary.sha(self.protocol_path),
                             adapter_sha256=summary.sha(adapter)))
        for split in ('validation','test'):
            for model in MODELS:
                path=self.raw_paths['v2',model,split]
                evaluation=summary.load(path)
                metrics=path.with_name('metrics.json');write(metrics,evaluation)
                receipt=dict(status='completed',evaluation_sha256=summary.sha(path),metrics_sha256=summary.sha(metrics),
                    checkpoint_sha256=evaluation['checkpoint_sha256'],protocol_sha256=summary.sha(self.protocol_path),
                    manifest_sha256=summary.sha(self.new_manifest),adapter_sha256=summary.sha(adapter),
                    execution_amendment_file=amendment.name,execution_amendment_sha256=summary.sha(amendment),
                    device='cuda',gpu_name='SYNTHETIC_EVALUATION_GPU',python_version='3.eval_fixture',
                    torch_version='eval_fixture',cuda_version='eval_fixture',numpy_version='eval_fixture',
                    scipy_version='eval_fixture',nibabel_version='eval_fixture',amp=True,seed=3711,
                    cudnn_deterministic=True,cudnn_benchmark=False,deterministic_algorithms_enabled=True,
                    deterministic_algorithms_warn_only=True,slurm_job_id='12345',slurm_node_list='SYNTHETIC_NODE')
                write(path.with_name('runtime_receipt.json'),receipt)


class SummaryFixtures(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.study=SyntheticStudy(self.temp.name)

    def test_complete_pair_produces_all_results_and_retains_failed_arm(self):
        result=summary.build_summary(self.study.args())
        self.assertEqual(result['status'],'audit_passed_complete_pair')
        self.assertEqual(result['validation_adoption']['eligible_models'],['unet2d'])
        self.assertEqual(set(result['common_grid_results']['test']),{'v1_unet2d','v2_unet2d','v1_improved3d','v2_improved3d'})
        val=result['common_grid_results']['validation']
        self.assertAlmostEqual(val['v1_unet2d']['aggregate']['per_class'][5]['dice'],.65)
        self.assertAlmostEqual(val['v2_unet2d']['aggregate']['per_class'][5]['dice'],.85)
        self.assertAlmostEqual(val['v2_improved3d']['aggregate']['per_class'][5]['dice'],.55)
        self.assertGreater(len(result['common_grid_results']['test']['v2_improved3d']['failed_prostate_dice_below_point7_cases']),0)
        text=summary.render_markdown(result)
        self.assertIn('improved3d',text);self.assertIn('不是本实验验证过的临床安全阈值',text)
        argv=['--review-root',str(self.study.review),'--v1-project',str(self.study.project),
              '--artifacts-root',str(self.study.artifacts),'--output-dir',str(self.study.args().output_dir)]
        with redirect_stdout(StringIO()):summary.main(argv)
        self.assertTrue((self.study.args().output_dir/'summary.json').exists())
        self.assertTrue((self.study.args().output_dir/'RESULTS.md').exists())

    def test_corrupt_dice_and_case_weighted_aggregation_are_rejected(self):
        path=self.study.raw_paths['v1','unet2d','validation']
        self.study.mutate(path,lambda e:e['cases'][0]['per_class'][5].update(dice=.123))
        with self.assertRaisesRegex(ValueError,'Numerical mismatch'):summary.build_summary(self.study.args())

    def test_case_weighting_cannot_replace_patient_weighting(self):
        path=self.study.raw_paths['v1','unet2d','validation']
        def mutate(e):e['aggregate']['per_class'][5]['dice']=sum(c['per_class'][5]['dice'] for c in e['cases'])/len(e['cases'])
        self.study.mutate(path,mutate)
        with self.assertRaisesRegex(ValueError,'aggregate/5/dice'):summary.build_summary(self.study.args())

    def test_wrong_selected_epoch_and_weight_bytes_are_rejected(self):
        self.study.mutate(self.study.freeze_path,lambda f:f['models']['unet2d'].update(checkpoint_epoch=30))
        with self.assertRaisesRegex(ValueError,'Freeze epoch'):summary.build_summary(self.study.args())

    def test_weight_byte_tampering_is_rejected(self):
        path=self.study.artifacts/'formal_v2/improved3d/best.pt'
        path.write_bytes(path.read_bytes()+b'changed')
        with self.assertRaisesRegex(ValueError,'weight byte'):summary.build_summary(self.study.args())

    def test_incomplete_source_coverage_is_rejected_even_with_updated_run_hash(self):
        path=self.study.artifacts/'formal_v2/improved3d/run.json'
        self.study.mutate(path,lambda r:r['identity']['source_hashes'].pop('modules.py'))
        self.study.mutate(self.study.freeze_path,lambda f:f['models']['improved3d'].update(run_sha256=summary.sha(path)))
        with self.assertRaisesRegex(ValueError,'source coverage'):summary.build_summary(self.study.args())

    def test_unfinished_pair_reads_no_histories_and_creates_no_final_report(self):
        path=self.study.artifacts/'formal_v2/improved3d/run.json'
        self.study.mutate(path,lambda r:r.update(status='running',completed_epochs=20))
        original=summary.load;opened=[]
        def monitored(p):opened.append(Path(p).name);return original(p)
        args=self.study.args()
        argv=['--review-root',str(args.review_root),'--v1-project',str(args.v1_project),
              '--artifacts-root',str(args.artifacts_root),'--output-dir',str(args.output_dir)]
        with patch.object(summary,'load',side_effect=monitored),self.assertRaisesRegex(ValueError,'Incomplete formal pair'):
            summary.main(argv)
        self.assertNotIn('history.json',opened)
        self.assertFalse(args.output_dir.exists())

    def test_hd95_uses_common_cases_and_common_patients(self):
        result=summary.build_summary(self.study.args())
        pair=result['comparisons']['test']['improved3d']['paired']['paired_hd95']
        self.assertEqual((pair['eligible_case_pairs'],pair['excluded_case_pairs'],pair['num_patients']),(29,1,5))
        self.assertNotIn('Q00',{p['patient'] for p in pair['patients']})
        path=self.study.artifacts/'common_test/common_grid_comparison.json'
        self.study.mutate(path,lambda x:x['models']['improved3d']['paired']['paired_hd95'].update(eligible_case_pairs=30,excluded_case_pairs=0))
        with self.assertRaisesRegex(ValueError,'HD95 common support'):summary.build_summary(self.study.args())

    def test_runtime_receipts_never_fabricate_direct_evaluation_metadata(self):
        args=self.study.args();key='v2_improved3d'
        initial=summary.build_summary(args)['resources'][key]
        self.assertEqual(initial['inference_runtime_evidence'],'inferred_from_same_job_runner')
        self.assertFalse(initial['inference_runtime_fully_recorded'])
        self.assertIsNone(initial['inference_runtime_metadata']['gpu_name'])
        p=self.study.artifacts/'formal_execution_receipt.json'
        receipt=dict(runner_sha256=summary.sha(self.study.review/'formal_v2_gpu.slurm'))
        write(p,receipt)
        partial=summary.build_summary(args)['resources'][key]
        self.assertEqual(partial['inference_runtime_evidence'],'runner_digest_bound_receipt_only')
        receipt.update(job_id=12345,state='COMPLETED',same_allocation=True,node_list='SYNTHETIC_NODE',provenance_source='slurm_sacct_and_frozen_runner')
        write(p,receipt)
        strong=summary.build_summary(args)['resources'][key]
        self.assertEqual(strong['inference_runtime_evidence'],'same_allocation_runner_provenance_with_execution_receipt')
        self.assertFalse(strong['inference_runtime_fully_recorded'])

    def test_actual_runtime_sidecars_are_bound_to_both_evaluation_and_metrics(self):
        self.study.add_runtime_receipts()
        result=summary.build_summary(self.study.args())
        resource=result['resources']['v2_improved3d']
        self.assertEqual(resource['inference_runtime_evidence'],'direct_runtime_receipt_bound_to_evaluation')
        self.assertEqual(resource['inference_runtime_metadata']['gpu_name'],'SYNTHETIC_EVALUATION_GPU')
        self.assertFalse(resource['inference_runtime_directly_in_evaluation_json'])
        self.assertTrue(resource['inference_runtime_fully_recorded'])
        self.assertFalse(resource['inference_matches_training_runtime_on_recorded_fields'])
        path=self.study.raw_paths['v2','unet2d','validation'].with_name('runtime_receipt.json')
        self.study.mutate(path,lambda r:r.update(evaluation_sha256='f'*64))
        with self.assertRaisesRegex(ValueError,'runtime receipt hash'):summary.build_summary(self.study.args())

    def test_changed_execution_flags_are_rejected_without_touching_scores(self):
        self.study.add_runtime_receipts()
        path=self.study.raw_paths['v2','improved3d','test'].with_name('runtime_receipt.json')
        self.study.mutate(path,lambda r:r.update(cudnn_deterministic=False))
        with self.assertRaisesRegex(ValueError,'deterministic execution'):summary.build_summary(self.study.args())

    def test_declared_execution_amendment_requires_every_runtime_receipt(self):
        self.study.add_runtime_receipts()
        self.study.raw_paths['v2','unet2d','validation'].with_name('runtime_receipt.json').unlink()
        with self.assertRaisesRegex(ValueError,'Missing required evaluation runtime receipt'):
            summary.build_summary(self.study.args())

    def test_metrics_content_and_execution_amendment_cannot_be_substituted(self):
        self.study.add_runtime_receipts()
        evaluation=self.study.raw_paths['v2','unet2d','test']
        path=evaluation.with_name('metrics.json')
        self.study.mutate(path,lambda m:m.update(unexpected='coherent hash but wrong JSON'))
        self.study.mutate(evaluation.with_name('runtime_receipt.json'),lambda r:r.update(metrics_sha256=summary.sha(path)))
        with self.assertRaisesRegex(ValueError,'metrics/evaluation JSONs differ'):summary.build_summary(self.study.args())
        self.study.add_runtime_receipts()
        amendment=self.study.review/'engineering_execution_amendment.json'
        self.study.mutate(amendment,lambda a:a.update(changed_after_execution=True))
        with self.assertRaisesRegex(ValueError,'amendment digest'):summary.build_summary(self.study.args())

    def test_execution_receipt_binds_actual_new_driver_and_refuses_escape(self):
        driver=self.study.review/'formal_v2_deterministic_gpu.slurm'
        driver.write_text('SYNTHETIC actual driver identity\n')
        path=self.study.artifacts/'formal_execution_receipt.json'
        receipt=dict(runner_file=driver.name,runner_sha256=summary.sha(driver),job_id=12345,state='COMPLETED',
                     node_list='SYNTHETIC_NODE',same_allocation=True,provenance_source='slurm_sacct_and_frozen_runner')
        write(path,receipt)
        resource=summary.build_summary(self.study.args())['resources']['v2_improved3d']
        self.assertEqual(resource['same_job_runner_sha256'],summary.sha(driver))
        self.assertEqual(resource['inference_runtime_evidence'],'same_allocation_runner_provenance_with_execution_receipt')
        receipt['runner_file']='../escape.slurm';write(path,receipt)
        with self.assertRaisesRegex(ValueError,'safe path'):summary.build_summary(self.study.args())

    def test_fixed_gate_cannot_be_relaxed_to_accept_failed_arm(self):
        path=self.study.artifacts/'common_validation/common_grid_comparison.json'
        self.study.mutate(path,lambda x:x['models']['improved3d']['gate'].update(minimum_prostate_delta=-1.,numeric_conditions_pass=True,replacement_supported_by_validation_gate=True))
        with self.assertRaisesRegex(ValueError,'Adoption threshold'):summary.build_summary(self.study.args())

    def test_wrong_world_mapping_is_rejected(self):
        path=self.study.artifacts/'common_validation/common_grid_comparison.json'
        self.study.mutate(path,lambda x:x['models']['unet2d']['v2']['cases'][0].update(target_common_dhw_to_source_v2_dhw=np.eye(4).tolist()))
        with self.assertRaisesRegex(ValueError,'common-world mapping'):summary.build_summary(self.study.args())

    def test_duplicate_patient_in_paired_summary_is_rejected(self):
        path=self.study.artifacts/'common_validation/common_grid_comparison.json'
        self.study.mutate(path,lambda x:x['models']['unet2d']['paired']['patients'].append(deepcopy(x['models']['unet2d']['paired']['patients'][0])))
        with self.assertRaisesRegex(ValueError,'Paired patient coverage'):summary.build_summary(self.study.args())

    def test_multiclass_voxel_totals_must_match_even_with_coherent_class_arithmetic(self):
        path=self.study.raw_paths['v1','unet2d','validation']
        def mutate(e):
            c=e['cases'][0];r=c['per_class'][1]
            r['predicted_voxels']+=1;r['union']+=1
            r['dice']=2*r['intersection']/(r['target_voxels']+r['predicted_voxels']);r['iou']=r['intersection']/r['union']
        self.study.mutate(path,mutate)
        with self.assertRaisesRegex(ValueError,'class totals differ'):summary.build_summary(self.study.args())


if __name__=='__main__':unittest.main()
