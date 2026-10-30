"""Synthetic S1/S2 checks. No downloads, patient data, or GPU required.

The integration fixture has training/validation JPEGs and no test image files.
Small networks test recovery, not medical performance; a separate test checks
full ConvNeXt-Tiny parameter count and smoothed-loss backward.
"""
from __future__ import annotations
import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_smoothing as s
import train_convnext_v2 as v2
from dataset import SliceRecord, SplitBundle, ADNISliceDataset


torch.set_num_threads(1)


class TinyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Linear(1, 8), nn.GELU(), nn.Dropout(.2))
        self.head = nn.Linear(8, 2)
    def forward(self, x):
        return self.head(self.features(x.mean((2, 3))))


def tiny_model(config):
    return TinyNetwork()


def fixture(base, count=6):
    root = base/'data'; root.mkdir()
    records = {}
    for split in ('train', 'validation', 'test'):
        records[split] = []
        for i in range(count):
            label = i % 2
            subject = f'{split}-person-{i}'
            path = f'{split}/{i}.jpeg'
            records[split].append(SliceRecord(path, subject, subject+'-scan', i,
                'train', 'AD' if label else 'NC', label*2, label, split))
            if split != 'test':
                dest = root/path; dest.parent.mkdir(exist_ok=True)
                image = np.full((240, 256), 40+label*150+i, dtype=np.uint8)
                Image.fromarray(image).save(dest)
        records[split] = tuple(records[split])
    return SplitBundle(root, records, {}, {x+'.csv': 'synthetic-'+x for x in records})


def configs(experiment='S1'):
    c = s.configuration(asdict(v2.Config('scratch_augmented')), experiment)
    return replace(c, epochs=3, decay_end_epoch=3, min_epochs=3, warmup_epochs=1,
                   patience=4, device='cpu', batch_size=2, accumulation_steps=2)


def tensor_tree_equal(case, first, second):
    if isinstance(first, torch.Tensor):
        case.assertTrue(torch.equal(first, second))
    elif isinstance(first, dict):
        case.assertEqual(first.keys(), second.keys())
        for k in first: tensor_tree_equal(case, first[k], second[k])
    elif isinstance(first, (tuple, list)):
        case.assertEqual(len(first), len(second))
        for a, b in zip(first, second): tensor_tree_equal(case, a, b)
    else:
        case.assertEqual(first, second)


class LossAndScheduleTests(unittest.TestCase):
    def test_s1_changes_only_declared_fields(self):
        ref = asdict(v2.Config('scratch_augmented'))
        config = asdict(s.configuration(ref, 'S1'))
        self.assertEqual({k: config[k] for k in ref}, ref)
        self.assertEqual(config['label_smoothing'], .1)
        self.assertEqual(config['decay_end_epoch'], 30)

    def test_s2_changes_only_schedule_beyond_s1(self):
        ref = asdict(v2.Config('scratch_augmented'))
        a, b = (asdict(s.configuration(ref, x)) for x in ('S1', 'S2'))
        self.assertEqual({k for k in a if a[k] != b[k]}, {'experiment', 'learning_rate', 'decay_end_epoch'})
        self.assertEqual(b['learning_rate'], 1e-4)
        self.assertEqual(b['decay_end_epoch'], 12)

    def test_no_extra_recipe_or_reference_changes(self):
        ref = asdict(v2.Config('scratch_augmented'))
        with self.assertRaises(ValueError): s.configuration(ref, 'S3')
        ref['seed'] = 43
        with self.assertRaises(ValueError): s.configuration(ref, 'S1')

    def test_smoothing_matches_torch_ce(self):
        logits = torch.tensor([[2., -1.], [-2., 1.], [.3, .4]], dtype=torch.float64, requires_grad=True)
        y = torch.tensor([0, 1, 0])
        smooth, hard = s.per_example_losses(logits, y, .1)
        torch.testing.assert_close(smooth, nn.functional.cross_entropy(logits, y, label_smoothing=.1, reduction='none'))
        torch.testing.assert_close(hard, nn.functional.cross_entropy(logits, y, reduction='none'))

    def test_binary_targets_are_95_05(self):
        logits = torch.tensor([[1., 3.], [1., 3.]])
        smooth, _ = s.per_example_losses(logits, torch.tensor([0, 1]), .1)
        logp = logits.log_softmax(1)
        expected = -(torch.tensor([[.95, .05], [.05, .95]])*logp).sum(1)
        torch.testing.assert_close(smooth, expected)

    def test_zero_smoothing_is_hard_ce(self):
        logits = torch.tensor([[1., -1.]], requires_grad=True)
        a, b = s.per_example_losses(logits, torch.tensor([0]), 0.)
        self.assertTrue(torch.equal(a, b))

    def test_losses_do_not_mutate_labels(self):
        labels = torch.tensor([0, 1]); original = labels.clone()
        s.per_example_losses(torch.randn(2, 2), labels, .1)
        self.assertTrue(torch.equal(labels, original))

    def test_gradient_matches_pytorch(self):
        a = torch.randn(5, 2, dtype=torch.float64, requires_grad=True)
        b = a.detach().clone().requires_grad_(); y = torch.tensor([0, 1, 0, 1, 1])
        s.per_example_losses(a, y, .1)[0].mean().backward()
        nn.functional.cross_entropy(b, y, label_smoothing=.1).backward()
        torch.testing.assert_close(a.grad, b.grad)

    def test_invalid_loss_inputs(self):
        for x, y, epsilon in ((torch.zeros(2, 3), torch.tensor([0, 1]), .1),
                              (torch.zeros(2, 2), torch.tensor([0., 1.]), .1),
                              (torch.zeros(2, 2), torch.tensor([0, 2]), .1),
                              (torch.zeros(2, 2), torch.tensor([0, 1]), 1.),
                              (torch.full((2, 2), float('nan')), torch.tensor([0, 1]), .1)):
            with self.assertRaises(ValueError): s.per_example_losses(x, y, epsilon)

    def test_s1_schedule_matches_original_each_epoch(self):
        old = v2.Config('scratch_augmented'); new = s.configuration(asdict(old), 'S1')
        model = tiny_model(old); optimizer = v2.make_optimizer(model, old)
        for epoch in range(1, 31):
            v2.set_epoch(model, optimizer, old, epoch)
            self.assertEqual(s.learning_rate(epoch, new), optimizer.param_groups[0]['lr'])

    def test_s2_warmup_peak_floor_and_hold(self):
        c = s.configuration(asdict(v2.Config('scratch_augmented')), 'S2')
        self.assertAlmostEqual(s.learning_rate(1, c), 1e-4/3)
        self.assertAlmostEqual(s.learning_rate(3, c), 1e-4)
        self.assertAlmostEqual(s.learning_rate(4, c), 1e-4)
        self.assertAlmostEqual(s.learning_rate(12, c), 1e-6)
        for e in range(13, 31): self.assertEqual(s.learning_rate(e, c), 1e-6)
        self.assertTrue(all(s.learning_rate(e+1, c) <= s.learning_rate(e, c) for e in range(4, 30)))

    def test_all_parameter_groups_receive_lr(self):
        c = configs(); model = tiny_model(c); opt = v2.make_optimizer(model, c)
        model.features[0].weight.requires_grad_(False)
        s.set_epoch(model, opt, c, 2)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        self.assertTrue(all(g['lr'] == s.learning_rate(2, c) for g in opt.param_groups))

    def test_invalid_configs(self):
        for changes in ({'recipe': 'transfer_patient'}, {'seed': 43}, {'label_smoothing': -.1},
                        {'decay_end_epoch': 1}, {'min_lr': 1.}, {'experiment': 'bad'}):
            with self.assertRaises((ValueError, RuntimeError)):
                replace(configs(), **changes).validate()

    def test_full_tiny_smoothed_backward(self):
        c = configs(); model = v2.make_model(c)
        self.assertEqual(v2.parameter_count(model), 27818594)
        loss, _ = s.per_example_losses(model(torch.zeros(1, 1, 224, 224)), torch.tensor([1]), .1)
        loss.mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_no_test_command(self):
        with self.assertRaises(SystemExit): s.main(['evaluate', '--private-root', 'dummy'])

    def test_no_unplanned_seed_argument(self):
        with self.assertRaises(SystemExit): s.main(['run', '--private-root', 'dummy', '--experiment', 'S1', '--seed', '43'])

    def test_requires_experiment_only_for_run(self):
        with self.assertRaises(ValueError): s.main(['run', '--private-root', 'dummy'])
        with self.assertRaises(ValueError): s.main(['compare', '--private-root', 'dummy', '--experiment', 'S1'])

    def test_promotion_requires_accuracy_and_recall(self):
        ref = {'patients': 102, 'correct_patients': 85, 'AD_missed': 11}
        self.assertTrue(s.promotion_screen(ref, {'patients': 102, 'correct_patients': 86, 'AD_missed': 10}))
        self.assertFalse(s.promotion_screen(ref, {'patients': 102, 'correct_patients': 87, 'AD_missed': 12}))
        self.assertFalse(s.promotion_screen(ref, {'patients': 102, 'correct_patients': 84, 'AD_missed': 10}))

    def test_confidence_counts_aggregate_only(self):
        predictions = {'patients': [{'subject_id':'secret1','label':0,'p_ad':.95},
                                   {'subject_id':'secret2','label':1,'p_ad':.6}]}
        result = s.confidence_counts(predictions)
        self.assertEqual(result['incorrect_patients'], 1)
        self.assertEqual(result['incorrect_at_raw_confidence_ge_0_90'], 1)
        self.assertNotIn('secret', json.dumps(result))


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.bundle = fixture(self.base)
        self.c = configs()

    def run_training(self, name, resume=False, limit=None, config=None):
        folder = self.base/name; folder.mkdir(exist_ok=True)
        return s.train_run(folder, self.bundle, config or self.c, resume=resume,
                           model_factory=tiny_model, max_new_epochs=limit)

    def test_complete_without_test_images(self):
        with patch('torch.hub.load_state_dict_from_url', side_effect=AssertionError('No downloads')):
            state = self.run_training('completed')
        self.assertTrue(state['completed'])
        self.assertFalse((self.bundle.root/'test').exists())
        self.assertFalse(s.summary_for(state)['new_final_test_model_evaluation_performed'])
        s.verify_completed(self.base/'completed', self.c)

    def test_s1_resume_matches_uninterrupted_full_state(self):
        whole = self.run_training('whole')
        partial = self.run_training('part', limit=1)
        self.assertFalse(partial['completed'])
        resumed = self.run_training('part', resume=True)
        for key in ('model_state', 'optimizer_state', 'best', 'rng', 'schedule_state', 'loader_state'):
            tensor_tree_equal(self, whole[key], resumed[key])
        self.assertEqual(resumed['resume_count'], 1)

    def test_s2_resume_matches_uninterrupted(self):
        c = configs('S2')
        a = self.run_training('s2_whole', config=c)
        self.run_training('s2_part', config=c, limit=2)
        b = self.run_training('s2_part', config=c, resume=True)
        for key in ('model_state', 'optimizer_state', 'best', 'rng'):
            tensor_tree_equal(self, a[key], b[key])

    def test_previous_checkpoint_fallback(self):
        expected = self.run_training('whole')
        self.run_training('broken', limit=2)
        (self.base/'broken/PRIVATE_resume.pt').write_bytes(b'not a torch checkpoint')
        actual = self.run_training('broken', resume=True)
        tensor_tree_equal(self, expected['model_state'], actual['model_state'])
        self.assertEqual(len(list((self.base/'broken').glob('PRIVATE_unreadable-*.pt'))), 1)

    def test_no_readable_checkpoint_refuses_restart(self):
        p = self.base/'empty'; p.mkdir()
        with self.assertRaises(ValueError): self.run_training('empty', resume=True)

    def test_changed_source_refuses_resume(self):
        self.run_training('source', limit=1)
        with patch.object(s, 'source_hashes', return_value={'changed':'hash'}):
            with self.assertRaises(ValueError): self.run_training('source', resume=True)

    def test_changed_configuration_refuses_resume(self):
        self.run_training('config', limit=1)
        with self.assertRaises(ValueError):
            self.run_training('config', resume=True, config=replace(self.c, label_smoothing=.2))

    def test_changed_manifests_refuse_resume(self):
        self.run_training('manifest', limit=1)
        self.bundle.manifest_hashes['train.csv'] = 'changed'
        with self.assertRaises(ValueError): self.run_training('manifest', resume=True)

    def test_changed_environment_refuses_resume(self):
        self.run_training('env', limit=1)
        with patch.object(v2, 'environment', return_value={'changed':'environment'}):
            with self.assertRaises(ValueError): self.run_training('env', resume=True)

    def test_final_artifact_recovery_does_not_train(self):
        original = self.run_training('final')
        (self.base/'final/PRIVATE_best_model.pt').unlink()
        with patch.object(s, 'train_epoch', side_effect=AssertionError('Must not train')):
            recovered = self.run_training('final', resume=True)
        self.assertTrue((self.base/'final/PRIVATE_best_model.pt').is_file())
        tensor_tree_equal(self, original['model_state'], recovered['model_state'])

    def test_saved_weights_reproduce_selected_patient_metrics(self):
        self.run_training('weights')
        state = torch.load(self.base/'weights/PRIVATE_best_model.pt', weights_only=True)
        model = tiny_model(self.c); model.load_state_dict(state['model_state'])
        loader = DataLoader(ADNISliceDataset(self.bundle, 'validation'), batch_size=2, shuffle=False)
        p, _ = s.predict_validation(model, loader, torch.device('cpu'))
        metrics, _ = s.aggregate_predictions(self.bundle.records['validation'], p)
        report = json.loads((self.base/'weights/smoothing_summary.json').read_text())
        self.assertEqual(metrics, report['selected_validation_metrics'])

    def test_zero_smoothing_equivalent_to_v2_trajectory(self):
        c = replace(self.c, label_smoothing=0.)
        original_config = v2.Config(**{k:v for k,v in asdict(c).items() if k in asdict(v2.Config('scratch_augmented'))})
        folder = self.base/'old'; folder.mkdir()
        old = v2.train_run(folder, self.bundle, original_config, None, self.base, model_factory=tiny_model)
        new = self.run_training('new', config=c)
        tensor_tree_equal(self, old['model_state'], new['model_state'])
        self.assertEqual(old['best']['metrics'], new['best']['metrics'])

    def test_smoothed_and_hard_losses_both_recorded(self):
        state = self.run_training('losses')
        self.assertNotEqual(state['history'][0]['train_smoothed_loss'], state['history'][0]['train_hard_log_loss'])
        self.assertFalse(s.summary_for(state)['validation_targets_smoothed'])

    def test_training_weighting_rejected(self):
        dataset = v2.WeightedDataset(self.bundle, self.c); dataset.weights *= 2
        loader = DataLoader(dataset, batch_size=2)
        model = tiny_model(self.c); opt = v2.make_optimizer(model, self.c)
        with self.assertRaises(ValueError): s.train_epoch(model, loader, opt, torch.device('cpu'), self.c)

    def test_validation_not_usable_for_training(self):
        loader = DataLoader(ADNISliceDataset(self.bundle, 'validation'), batch_size=2)
        model = tiny_model(self.c); opt = v2.make_optimizer(model, self.c)
        with self.assertRaises(ValueError): s.train_epoch(model, loader, opt, torch.device('cpu'), self.c)

    def test_partial_accumulation_matches_larger_batch(self):
        c = replace(self.c, rotation_degrees=0., batch_size=2, accumulation_steps=2)
        data = v2.WeightedDataset(self.bundle, c)
        class Linear(nn.Module):
            def __init__(self):
                super().__init__(); self.head=nn.Linear(1,2)
            def forward(self,x): return self.head(x.mean((2,3)))
        a=Linear(); b=copy.deepcopy(a)
        oa=v2.make_optimizer(a,c); ob=v2.make_optimizer(b,c)
        s.train_epoch(a, DataLoader(data,batch_size=2,shuffle=False), oa, torch.device('cpu'), c)
        large=replace(c,batch_size=4,accumulation_steps=1)
        s.train_epoch(b, DataLoader(data,batch_size=4,shuffle=False), ob, torch.device('cpu'), large)
        for key in a.state_dict(): torch.testing.assert_close(a.state_dict()[key], b.state_dict()[key], rtol=1e-6, atol=1e-7)

    def test_metrics_not_artificially_smoothed(self):
        state = self.run_training('hard')
        predictions = state['best']['private_predictions']['patients']
        exact = s.binary_metrics([r['label'] for r in predictions], [r['p_ad'] for r in predictions])
        self.assertEqual(exact, state['best']['metrics']['patient'])

    def test_modified_learning_rate_history_rejected(self):
        self.run_training('history')
        p=self.base/'history/history.csv'; text=p.read_text()
        lines=text.splitlines(); headings=lines[0].split(','); values=lines[1].split(',')
        values[headings.index('learning_rate')]='0.123'
        lines[1]=','.join(values);p.write_text('\n'.join(lines)+'\n')
        with self.assertRaises(ValueError): s.verify_completed(self.base/'history',self.c)

    def test_lock_released_after_failure(self):
        with self.assertRaises(RuntimeError):
            with v2.run_lock(self.base): raise RuntimeError('synthetic failure')
        with v2.run_lock(self.base): pass




class OrchestrationTests(unittest.TestCase):
    """Short synthetic runs exercise the actual check/run/compare/publish paths."""
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.private=self.root/'private files'; self.private.mkdir()
        self.project=self.root/'project files'; self.project.mkdir()
        self.bundle=fixture(self.private)
        self.reference=self.private/'A1'; self.reference.mkdir()
        old= v2.Config('scratch_augmented',epochs=3,min_epochs=3,patience=4,
                       warmup_epochs=1,batch_size=2,accumulation_steps=2,device='cpu')
        v2.train_run(self.reference,self.bundle,old,None,self.private,model_factory=tiny_model)
        report=json.loads((self.reference/'v2_summary.json').read_text())
        hashes=s.prior.file_hashes(self.reference,s.prior.REFERENCE_FILES)
        self.ctx=(self.private,self.project,self.bundle,None,self.reference,report,hashes)
        (self.project/'README.md').write_bytes(b'# Existing report\r\nPreserve this text.\r\n')
        (self.project/'SMOOTHING_PLAN.md').write_text('Synthetic protocol for orchestration test.\n')
        self.original_training=s.train_run
        self.fixed_configs={name:configs(name) for name in s.EXPERIMENTS}
        self.patches=[patch.object(s,'context',return_value=self.ctx),
                      patch.object(s,'configuration',side_effect=lambda reference,name:self.fixed_configs[name]),
                      patch.object(s,'require_committed'),
                      patch.object(v2,'make_model',side_effect=tiny_model),
                      patch.object(s,'train_run',side_effect=lambda *a,**kw:self.original_training(*a,**kw,model_factory=tiny_model))]
        for item in self.patches:
            item.start();self.addCleanup(item.stop)
        s.check(self.private);s.smoke(self.private)

    def test_end_to_end_bundle_excludes_private_outputs_and_preserves_reference(self):
        before=s.prior.file_hashes(self.reference,s.prior.REFERENCE_FILES)
        s.run_experiment(self.private,'S1');s.run_experiment(self.private,'S2')
        s.compare(self.private)
        with zipfile.ZipFile(self.private/'COMP3710_Smoothing_Results.zip') as z:
            self.assertIn('smoothing_comparison.json',z.namelist())
            self.assertTrue(all('PRIVATE' not in name and not name.endswith('.pt') for name in z.namelist()))
            for name in z.namelist():
                if name.endswith(('.json','.md','.csv')):
                    self.assertNotIn(b'validation-person-',z.read(name))
        self.assertEqual(before,s.prior.file_hashes(self.reference,s.prior.REFERENCE_FILES))
        summary=json.loads((self.private/s.STUDY/'comparison/smoothing_comparison.json').read_text())
        self.assertTrue(summary['comparison_completed'])
        self.assertFalse(summary['candidate_changed_automatically'])
        self.assertFalse(summary['test_accuracy_requirement_demonstrated'])

    def test_completed_run_skips_training_and_publication_is_idempotent(self):
        s.run_experiment(self.private,'S1');s.run_experiment(self.private,'S2')
        with patch.object(s,'train_run',side_effect=AssertionError('Must not repeat completed training')):
            s.run_experiment(self.private,'S1')
        s.compare(self.private)
        original=(self.project/'README.md').read_bytes()
        s.publish(self.private)
        published=(self.project/'README.md').read_bytes()
        self.assertTrue(published.startswith(original))
        s.publish(self.private)
        self.assertEqual(published,(self.project/'README.md').read_bytes())
        self.assertEqual(published.count(s.MARKER.encode()),1)

    def test_changed_study_or_missing_smoke_refuses_run(self):
        path=self.private/s.STUDY/'smoke_summary.json';path.unlink()
        with self.assertRaises(Exception):s.run_experiment(self.private,'S1')
        s.smoke(self.private)
        study_path=self.private/s.STUDY/'study.json'
        study=json.loads(study_path.read_text());study['classification_cutoff']=.6
        study_path.write_text(json.dumps(study))
        with self.assertRaises(ValueError):s.run_experiment(self.private,'S1')

    def test_compare_requires_both_experiments_completed(self):
        s.run_experiment(self.private,'S1')
        with self.assertRaises(Exception):s.compare(self.private)
        self.assertFalse((self.private/'COMP3710_Smoothing_Results.zip').exists())

    def test_incomplete_run_resumes_through_public_command(self):
        def only_one(*args,**kwargs):
            return self.original_training(*args,**kwargs,model_factory=tiny_model,max_new_epochs=1)
        with patch.object(s,'train_run',side_effect=only_one):
            with self.assertRaises(ValueError):s.run_experiment(self.private,'S1')
        s.run_experiment(self.private,'S1')
        summary=json.loads((self.private/s.STUDY/'S1/smoothing_summary.json').read_text())
        self.assertTrue(summary['experiment_completed'])
        self.assertEqual(summary['resume_count'],1)


if __name__ == '__main__': unittest.main()
