"""Clearly synthetic full-cohort metadata and pixel comparison fixtures.

No real MRI, new experimental score, or real TEST prediction is read. The one
end-to-end fixture uses artificial sparse arrays and meaningless checkpoint
bytes in a temporary tree, with explicit fixture-only pin substitutions.
"""
from contextlib import ExitStack, redirect_stdout
from collections import Counter
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import compare_hard_followup as h
import compare_common_grid as c
import test_summarize_v2 as fixture


def write(path, obj):
    fixture.write(path, obj)


def sparse_target(shape, n):
    x = np.zeros(shape, dtype=np.uint8)
    for ci in range(1, 6):
        x.reshape(-1)[-n*(6-ci):-n*(5-ci) if ci < 5 else None] = ci
    return x


def prediction_for_case(target, row):
    prediction = target.copy()
    if row['per_class'][5]['predicted_voxels'] == 0:
        prediction[target == 5] = 1
    else:
        error = row['per_class'][5]['target_voxels'] - row['per_class'][5]['intersection']
        prediction.reshape(-1)[np.flatnonzero(target.reshape(-1) == 4)[:error]] = 5
        prediction.reshape(-1)[np.flatnonzero(target.reshape(-1) == 5)[:error]] = 4
    actual = c.class_metrics(prediction, target)
    assert actual['per_class'] == row['per_class']
    return prediction


class HardSyntheticStudy:
    def __init__(self, root, pixels=False):
        s = fixture.SyntheticStudy(root)
        self.s, self.review, self.project, self.artifacts = s, s.review, s.project, s.artifacts
        for name in ['train_hard_followup.py', 'hard_followup_gpu.slurm']:
            (self.review / name).write_text('SYNTHETIC nonexecutable fixture source bytes ' + name)
        old_target = sparse_target(fixture.OLD_SHAPE, 10)
        self.spacing = np.asarray([2.5, 2., 2.], dtype=np.float32)
        for r in s.old['records']:
            s.identity['records'][r['cache_path']]['arrays'] = {
                'mask': {'sha256': c.decoded_array_sha256(old_target)},
                'spacing': {'sha256': c.decoded_array_sha256(self.spacing)}}
        write(self.project / 'artifacts/cache64/cache_identity.json', s.identity)
        s.protocol.update(experiment_id='independent_resolution_sampling_v2',
            qa=dict(fit_case=['B006','Week0'],fit_steps=400,fit_seed=3711,
                    minimum_fit_prostate_dice=.9,minimum_fit_foreground_dice=.9,
                    maximum_final_initial_loss_ratio=.25,fit_amp=False))
        s.protocol['required_original_sha256']['artifacts/cache64/cache_identity.json'] = h.sha(self.project / 'artifacts/cache64/cache_identity.json')
        write(s.protocol_path, s.protocol)
        self.base_sha = h.sha(s.protocol_path)
        s.new['protocol_sha256'] = self.base_sha
        write(s.new_manifest, s.new)
        for model in s.identities:
            s.identities[model].update(protocol_sha256=self.base_sha, manifest_sha256=h.sha(s.new_manifest),
                                       source_hashes=s.protocol['required_original_sha256'])
        # Regenerate only artificial metadata for the retained Hard candidate.
        selected = s.make_run('v2', 'improved3d')
        original_new_root = self.artifacts / 'formal_v2/improved3d'
        hard_root = self.artifacts / 'hard_followup/improved3d'
        hard_root.parent.mkdir()
        shutil.copytree(original_new_root, hard_root)
        self.run_root = hard_root
        self.qa = {'status':'failed_fit_gate','protocol_sha256':self.base_sha,
                   'manifest_sha256':h.sha(s.new_manifest), 'models':{
                       'unet2d': {'passed':False},
                       'improved3d': {'passed':True, **{f:True for f in
                           ('full_image_only_inference_passed','native_geometry_serialisation_passed',
                            'independent_metric_check_passed','sampler_batch_backward_passed')}}}}
        first = {'step':0,'loss':3.,'prostate_dice':0.,'foreground_dice':0.}
        self.fit = dict(status='failed_fit_gate',protocol_sha256=self.base_sha,
            manifest_sha256=h.sha(s.new_manifest),case=['B006','Week0'],steps=400,seed=3711,amp=False,
            models={'unet2d':{'passed':False,'history':[first,dict(step=400,loss=.4,prostate_dice=.82,foreground_dice=.86)]},
                    'improved3d':{'passed':True,'history':[first,dict(step=400,loss=.06,prostate_dice=.923,foreground_dice=.957)]}})
        write(self.artifacts / 'qa_v2/fit/fit_receipt.json', self.fit)
        self.qa['fit_receipt_sha256'] = h.sha(self.artifacts / 'qa_v2/fit/fit_receipt.json')
        write(self.artifacts / 'qa_v2/qa_receipt.json', self.qa)
        self.scope = dict(experiment_id='hard_only_after_stopped_v2_pair',status='frozen_synthetic_fixture',base_protocol_sha256=self.base_sha,
            stopped_pair_status='stopped_after_QA_failure',scope_change_is_scientific_plan_revision_not_engineering_fix=True,
            formal_models=['improved3d'],configuration=s.protocol['configuration'],adoption=s.protocol['adoption'],
            configuration_exactly_inherited=True,seed=3711,epochs=30,max_gpu_walltime_minutes=60,
            reused_reference_models=['v1_unet2d','v1_improved3d'],no_new_parameters_or_candidates=True,
            manifest_sha256=h.sha(s.new_manifest),base_manifest_sha256=h.sha(s.new_manifest),
            qa_receipt_sha256=h.sha(self.artifacts/'qa_v2/qa_receipt.json'),
            fit_receipt_sha256=h.sha(self.artifacts/'qa_v2/fit/fit_receipt.json'),
            required_scope_sha256={name:h.sha(self.review/name) for name in ['train_hard_followup.py','hard_followup_gpu.slurm']})
        self.scope_path = self.review / 'hard_followup_protocol.json'
        write(self.scope_path, self.scope)
        self.scope_sha = h.sha(self.scope_path)
        execution = dict(status='completed',scope_plan_sha256=self.scope_sha,
            training_executor_sha256=h.sha(self.review/'train_hard_followup.py'),
            scope_driver_sha256=h.sha(self.review/'hard_followup_gpu.slurm'),
            best_checkpoint_sha256=s.new_digests['improved3d'])
        self.execution_path = self.artifacts/'hard_followup/scope_execution.json'
        write(self.execution_path, execution)
        self.freeze = dict(status='frozen',experiment_id=self.scope['experiment_id'],
            base_protocol_sha256=self.base_sha,protocol_sha256=self.base_sha,
            manifest_sha256=h.sha(s.new_manifest),scope_plan_sha256=self.scope_sha,
            scope_execution_sha256=h.sha(self.execution_path),models={'improved3d':selected})
        self.freeze_path = self.artifacts/'hard_selection.json'
        write(self.freeze_path,self.freeze)
        self.eval_paths = {}
        old_records = {(r['patient'],r['visit']):r for r in s.old['records']}
        new_records = {(r['patient'],r['visit']):r for r in s.new['records']}
        for split in ('validation','test'):
            for key,version,model in [('v1_unet2d','v1','unet2d'),('v1_improved3d','v1','improved3d'),('hard_improved3d','v2','improved3d')]:
                ev = s.raw_evaluation(version,model,split)
                if version == 'v2':
                    path = self.artifacts/f'hard_{split}/improved3d/evaluation.json'
                    ev.update(scope_plan_sha256=self.scope_sha,actual_scope_id=self.scope['experiment_id'])
                    if split=='test':ev['selection_freeze_sha256']=h.sha(self.freeze_path)
                else:
                    path = s.raw_paths[version,model,split]
                for row in ev['cases']:
                    case = row['patient'],row['visit']
                    record = new_records[case] if version=='v2' else old_records[case]
                    row['prediction_file'] = f'predictions/{case[0]}_{case[1]}.npz'
                    if pixels:
                        target = sparse_target(fixture.NEW_SHAPE,20) if version=='v2' else old_target
                        prediction = prediction_for_case(target,row)
                        dest = path.parent/row['prediction_file']; dest.parent.mkdir(parents=True,exist_ok=True)
                        np.savez_compressed(dest,prediction=prediction,spacing_dhw=record['spacing_dhw_mm'],
                                            label_ids=np.arange(6),affine_xyz=record['resampled_affine_xyz'])
                        cache = self.project/'artifacts/cache64'/old_records[case]['cache_path']
                        if not cache.exists():
                            cache.parent.mkdir(parents=True,exist_ok=True)
                            np.savez_compressed(cache,mask=old_target,spacing=self.spacing)
                write(path,ev)
                self.eval_paths[key,split] = path
                if version=='v2':
                    write(path.with_name('metrics.json'),ev)
                    runtime = dict(status='completed',scope_plan_sha256=self.scope_sha,
                        base_protocol_sha256=self.base_sha,protocol_sha256=self.base_sha,
                        manifest_sha256=h.sha(s.new_manifest),checkpoint_sha256=s.new_digests['improved3d'],
                        evaluation_sha256=h.sha(path),metrics_sha256=h.sha(path.with_name('metrics.json')),
                        adapter_file='train_hard_followup.py',adapter_sha256=h.sha(self.review/'train_hard_followup.py'),
                        scope_driver_sha256=h.sha(self.review/'hard_followup_gpu.slurm'),device='cuda',gpu_name='SYNTHETIC_GPU',amp=True,
                        seed=3711,cudnn_deterministic=True,cudnn_benchmark=False,deterministic_algorithms_enabled=True,
                        deterministic_algorithms_warn_only=True,**{k:'SYNTHETIC' for k in
                        ('python_version','torch_version','cuda_version','numpy_version','scipy_version','nibabel_version','slurm_job_id','slurm_node_list')})
                    write(path.with_name('runtime_receipt.json'),runtime)
        completion = json.loads((self.artifacts/'cache_v2/test_completion_receipt.json').read_text())
        completion.update(protocol_sha256=self.base_sha,manifest_sha256=h.sha(s.new_manifest),
                          scope_plan_sha256=self.scope_sha,selection_freeze_sha256=h.sha(self.freeze_path))
        write(self.artifacts/'cache_v2/test_completion_receipt.json',completion)

    def patches(self):
        stack=ExitStack()
        stack.enter_context(patch.object(h,'BASE_PROTOCOL_SHA256',self.base_sha))
        stack.enter_context(patch.object(h,'V1_MANIFEST_SHA256',h.sha(self.s.old_manifest)))
        stack.enter_context(patch.object(c,'FROZEN_V1_CACHE_IDENTITY_SHA256',h.sha(self.project/'artifacts/cache64/cache_identity.json')))
        stack.enter_context(patch.object(c,'FROZEN_V1_CHECKPOINTS',self.s.old_digests))
        return stack

    def args(self,split='test'):
        return SimpleNamespace(review_root=self.review,v1_project=self.project,artifacts_root=self.artifacts,
            scope_plan=self.scope_path,split=split,retrospective_test_ack=split=='test',output_dir=self.s.root/'result')


class HardComparisonTests(unittest.TestCase):
    def test_full_three_model_pixel_pipeline_retains_failure_empty_and_all_patients(self):
        with TemporaryDirectory() as d:
            s=HardSyntheticStudy(d,pixels=True)
            with s.patches(): result=h.compare(s.args())
            self.assertEqual(set(result['models']),set(h.MODEL_KEYS))
            self.assertEqual(result['models']['hard_improved3d']['aggregate']['num_cases'],30)
            self.assertEqual(result['models']['hard_improved3d']['aggregate']['num_patients'],6)
            self.assertEqual(result['models']['hard_improved3d']['aggregate']['prostate_empty_status_counts'].get('missed'),1)
            self.assertEqual(result['paired']['v1_improved3d']['paired_hd95']['eligible_case_pairs'],29)
            self.assertEqual(result['paired']['v1_improved3d']['paired_hd95']['num_patients'],5)
            self.assertIsNone(result['adoption_gate']['replacement_supported_by_validation_gate'])
            text=h.render_markdown(result)
            self.assertIn('原双模型方案',text);self.assertIn('Q00/Week0',text)
            self.assertNotIn('v2_unet2d',json.dumps(result))

    def test_changed_gate_or_failed_pair_relabelling_is_refused(self):
        with TemporaryDirectory() as d:
            s=HardSyntheticStudy(d)
            with s.patches():
                changed=deepcopy(s.scope);changed['adoption']['minimum_prostate_dice_gain']=0.
                write(s.scope_path,changed)
                with self.assertRaisesRegex(ValueError,'adoption'):h.verify_scope(s.scope_path,s.review)
                write(s.scope_path,s.scope)
                qa=deepcopy(s.qa);qa['models']['unet2d']['passed']=True
                write(s.artifacts/'qa_v2/qa_receipt.json',qa)
                with self.assertRaisesRegex(ValueError,'failed new 2D'):h.verify_failed_pair_qa(s.artifacts,s.s.protocol)

    def test_fixed_hard_final_step_not_best_and_no_budget_extension(self):
        with TemporaryDirectory() as d:
            s=HardSyntheticStudy(d)
            for change in ('step','prostate_dice'):
                fit=deepcopy(s.fit)
                fit['models']['improved3d']['history'][-1][change]=300 if change=='step' else .899
                write(s.artifacts/'qa_v2/fit/fit_receipt.json',fit)
                qa=deepcopy(s.qa);qa['fit_receipt_sha256']=h.sha(s.artifacts/'qa_v2/fit/fit_receipt.json')
                write(s.artifacts/'qa_v2/qa_receipt.json',qa)
                with s.patches(),self.assertRaises(ValueError):h.verify_failed_pair_qa(s.artifacts,s.s.protocol)

    def test_fake_missing_second_model_and_wrong_scope_freeze_are_refused(self):
        with TemporaryDirectory() as d:
            s=HardSyntheticStudy(d)
            for field,value in [('scope_plan_sha256','0'*64),('models',dict(s.freeze['models'],unet2d=s.freeze['models']['improved3d']))]:
                changed=deepcopy(s.freeze);changed[field]=value
                with s.patches(),self.assertRaises(ValueError):h.verify_selection(changed,s.scope,s.scope_sha,h.sha(s.s.new_manifest),s.run_root)

    def test_runtime_wrong_seed_and_rewritten_metric_are_not_certified(self):
        with TemporaryDirectory() as d:
            s=HardSyntheticStudy(d)
            path=s.eval_paths['hard_improved3d','test'];runtime=path.with_name('runtime_receipt.json')
            original=h.load(runtime)
            changed=dict(original,seed=42);write(runtime,changed)
            with s.patches(),self.assertRaisesRegex(ValueError,'settings differ'):
                h.verify_runtime(runtime,path,s.scope_sha,h.sha(s.s.new_manifest),s.s.new_digests['improved3d'],s.review,Counter())
            write(runtime,original);metrics=h.load(path.with_name('metrics.json'));metrics['cases'][0]['per_class'][5]['dice']=.999
            write(path.with_name('metrics.json'),metrics)
            with s.patches(),self.assertRaisesRegex(ValueError,'Metrics and evaluation'):
                h.verify_runtime(runtime,path,s.scope_sha,h.sha(s.s.new_manifest),s.s.new_digests['improved3d'],s.review,Counter())

    def test_unfinished_run_is_not_read_or_scored(self):
        with TemporaryDirectory() as d:
            s=HardSyntheticStudy(d);run=h.load(s.run_root/'run.json');run['status']='running';write(s.run_root/'run.json',run)
            (s.run_root/'history.json').unlink()
            with s.patches(),patch.object(np,'load',side_effect=AssertionError('No pixel may be opened')):
                with self.assertRaisesRegex(ValueError,'not complete'):h.compare(s.args())

    def test_original_gt_cannot_move_while_counts_stay_identical(self):
        with TemporaryDirectory() as d:
            s=HardSyntheticStudy(d,pixels=True)
            r=next(r for r in s.s.old['records'] if r['split']=='test')
            path=s.project/'artifacts/cache64'/r['cache_path']
            with np.load(path,allow_pickle=False) as z:target=z['mask'].copy();spacing=z['spacing'].copy()
            flat=target.reshape(-1);p=int(np.flatnonzero(flat==5)[0]);bg=int(np.flatnonzero(flat==0)[0])
            flat[p],flat[bg]=flat[bg],flat[p]
            self.assertEqual(np.bincount(target.reshape(-1),minlength=6).tolist(),r['class_voxel_counts'])
            np.savez_compressed(path,mask=target,spacing=spacing)
            with s.patches(),self.assertRaisesRegex(ValueError,'decoded identity'):h.compare(s.args())

    def test_wrong_same_family_baseline_evaluation_is_refused_before_pixels(self):
        with TemporaryDirectory() as d:
            s=HardSyntheticStudy(d);path=s.eval_paths['v1_improved3d','test'];ev=h.load(path)
            ev['checkpoint_sha256']='a'*64;write(path,ev)
            with s.patches(),patch.object(np,'load',side_effect=AssertionError('No pixel may be opened')):
                with self.assertRaisesRegex(ValueError,'Wrong selected checkpoint'):h.compare(s.args())

    def test_known_test_requires_ack_before_missing_inputs_are_read(self):
        args=['--v1-project','/missing','--artifacts-root','/missing','--scope-plan','/missing',
              '--split','test','--output-dir','/tmp/known_missing_hard_fixture_output']
        with self.assertRaisesRegex(ValueError,'acknowledgement'):h.main(args)


if __name__=='__main__':unittest.main()
