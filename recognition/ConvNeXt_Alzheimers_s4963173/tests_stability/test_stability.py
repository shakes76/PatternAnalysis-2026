"""Synthetic-data checks for the additive stability runner; no downloads or GPU.

The integration fixture uses a small network and deliberately has no test images.
It checks execution/recovery/provenance, not ConvNeXt's medical performance.
"""
from __future__ import annotations
from dataclasses import asdict, replace
import copy
import csv
import io
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stability_a1 as s
import train_convnext_v2 as v2
from dataset import SliceRecord, SplitBundle
from training_utils import sha256_file

torch.set_num_threads(1)
BASE = Path(s.__file__).resolve().parent
ORIGINAL_TRAIN = v2.train_run


class TinyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Linear(1, 8), nn.GELU(), nn.Dropout(.2))
        self.head = nn.Linear(8, 2)
    def forward(self, x):
        return self.head(self.features(x.mean((2, 3))))


def tiny_model(config):
    return TinyNetwork()


def fixture(root):
    data = root/'data'; data.mkdir()
    recs = {}
    for split in ('train','validation','test'):
        recs[split] = []
        for label in (0,1):
            for scan in range(2):
                subject = f'synthetic-{split}-{label}-{scan}'
                for i in range(2):
                    path = f'{split}/{label}-{scan}-{i}.jpeg'
                    recs[split].append(SliceRecord(path, subject, subject+'-scan',i,
                        'train', 'AD' if label else 'NC', 2*label,label,split))
                    if split != 'test':
                        dest=data/path;dest.parent.mkdir(exist_ok=True)
                        a=np.full((240,256),30+label*170+scan*3+i,dtype=np.uint8)
                        Image.fromarray(a).save(dest)
        recs[split] = tuple(recs[split])
    return SplitBundle(data,recs,{}, {n+'.csv':'synthetic-'+n for n in recs})


class PureChecks(unittest.TestCase):
    def test_only_43_44_new_runs(self):
        study={'reference_configuration':asdict(v2.Config('scratch_augmented'))}
        for seed in (43,44): self.assertEqual(s.seeded_config(study,seed).seed,seed)
        for seed in (42,45,0,-1):
            with self.assertRaises(ValueError):s.seeded_config(study,seed)

    def test_configuration_changes_only_seed(self):
        config=v2.Config('scratch_augmented')
        study={'reference_configuration':asdict(config)}
        updated=asdict(s.seeded_config(study,43))
        self.assertEqual({k for k in updated if updated[k]!=asdict(config)[k]}, {'seed'})

    def test_no_test_command(self):
        with self.assertRaises(SystemExit):s.main(['evaluate','--private-root','dummy'])

    def test_run_requires_seed(self):
        with self.assertRaises(ValueError):s.main(['run','--private-root','dummy'])

    def test_seed_not_valid_for_compare(self):
        with self.assertRaises(ValueError):s.main(['compare','--private-root','dummy','--seed','43'])

    def test_descriptive_sd_is_sample_sd(self):
        rows=[{k:v for k in ('patient_accuracy','macro_f1','auroc','ad_recall','nc_recall')}
              for v in (.7,.8,.9)]
        a=s.describe(rows)['patient_accuracy']
        self.assertAlmostEqual(a['mean'],.8);self.assertAlmostEqual(a['sample_sd'],.1)
        self.assertEqual(a['number_of_runs'],3)

    def test_atomic_text_has_no_bom(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'pointer.txt';s.atomic_text(p,'hello\n')
            self.assertEqual(p.read_bytes(),b'hello\n')
            self.assertFalse((Path(d)/'pointer.txt.tmp').exists())

    def test_private_output_rejects_outside(self):
        with tempfile.TemporaryDirectory() as d:
            private=Path(d)/'private';private.mkdir()
            with self.assertRaises(ValueError):s.private_path(Path(d)/'outside',private)
            self.assertEqual(s.private_path(private/'ok',private),private/'ok')

    def test_private_output_rejects_git_tree(self):
        with tempfile.TemporaryDirectory() as d:
            private=Path(d);(private/'.git').mkdir()
            with self.assertRaises(ValueError):s.private_path(private/'run',private)

    def test_json_digest_stable_key_order(self):
        self.assertEqual(s.digest_json({'a':1,'b':2}),s.digest_json({'b':2,'a':1}))


class IntegrationChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory()
        cls.template=Path(cls.temp.name)
        cls.bundle=fixture(cls.template)
        cls.config=v2.Config('scratch_augmented',epochs=3,min_epochs=3,patience=3,
                             batch_size=2,accumulation_steps=2,device='cpu',seed=42)
        ref=cls.template/'reference';ref.mkdir()
        with patch('sys.stdout',new=io.StringIO()):
            ORIGINAL_TRAIN(ref,cls.bundle,cls.config,None,cls.template,model_factory=tiny_model)
        cls.ref=ref
    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.private=Path(self.tmp.name)
        import shutil
        self.reference=self.private/'reference'
        shutil.copytree(self.ref,self.reference)
        self.b=self.bundle
        (self.private/'latest_v2_scratch_augmented.txt').write_text(str(self.reference))
        (self.private/'latest_v2_scratch_patient.txt').write_text('UNCHANGED_A2_POINTER')
        self.old_ptr={name:(self.private/name).read_bytes() for name in
                      ('latest_v2_scratch_augmented.txt','latest_v2_scratch_patient.txt')}
        self.old_files=s.file_hashes(self.reference,s.REFERENCE_FILES)
        smoke=self.private/'smoke';smoke.mkdir()
        v2.atomic_json(smoke/'v2_smoke_summary.json',{'passed':True,'source_hashes':v2.source_hashes(),
                        'manifest_sha256':self.b.manifest_hashes})
        (self.private/'latest_v2_smoke.txt').write_text(str(smoke))
        self.ctx=patch.object(v2,'local_context',return_value=(self.private,BASE,self.b,None))
        self.ctx.start();self.addCleanup(self.ctx.stop)
        self.committed=patch.object(s,'require_committed')
        self.committed.start();self.addCleanup(self.committed.stop)
        self.out=patch('sys.stdout',new=io.StringIO())
        self.out.start();self.addCleanup(self.out.stop)
        s.prepare(self.private)
        self.study=s.read_json(self.private/s.STUDY_NAME/'study.json')

    def train(self, seed, max_epochs=None):
        def small(*args,**kwargs):
            kwargs['model_factory']=tiny_model
            if max_epochs is not None:kwargs['max_new_epochs']=max_epochs
            return ORIGINAL_TRAIN(*args,**kwargs)
        with patch.object(v2,'train_run',side_effect=small):s.run_seed(self.private,seed)

    def complete_pair(self):
        self.train(43);self.train(44)

    def test_check_is_idempotent(self):
        before=(self.private/s.STUDY_NAME/'study.json').read_bytes()
        s.prepare(self.private)
        self.assertEqual(before,(self.private/s.STUDY_NAME/'study.json').read_bytes())

    def test_reference_and_a2_unchanged_after_runs(self):
        self.complete_pair()
        self.assertEqual(self.old_files,s.file_hashes(self.reference,s.REFERENCE_FILES))
        for name,data in self.old_ptr.items():self.assertEqual((self.private/name).read_bytes(),data)

    def test_completed_run_not_retrained(self):
        self.train(43)
        folder=self.private/s.STUDY_NAME/'seed-43'
        before=s.file_hashes(folder,s.REFERENCE_FILES)
        with patch.object(v2,'train_run',side_effect=AssertionError('Must not run')):
            s.run_seed(self.private,43)
        self.assertEqual(before,s.file_hashes(folder,s.REFERENCE_FILES))

    def test_interruption_resume_identical_to_uninterrupted(self):
        with self.assertRaisesRegex(ValueError,'without completion'):self.train(43,max_epochs=1)
        self.train(43)
        run=self.private/s.STUDY_NAME/'seed-43'
        resumed=torch.load(run/'PRIVATE_resume.pt',weights_only=True)
        whole=self.private/'uninterrupted';whole.mkdir()
        expected=ORIGINAL_TRAIN(whole,self.b,replace(self.config,seed=43),None,self.private,model_factory=tiny_model)
        for key,value in expected['model_state'].items():self.assertTrue(torch.equal(value,resumed['model_state'][key]),key)
        self.assertEqual(expected['best']['metrics'],resumed['best']['metrics'])
        self.assertEqual(expected['best']['epoch'],resumed['best']['epoch'])
        for key,values in expected['optimizer_state']['state'].items():
            for name,value in values.items():
                actual=resumed['optimizer_state']['state'][key][name]
                if isinstance(value,torch.Tensor):self.assertTrue(torch.equal(value,actual))
                else:self.assertEqual(value,actual)

    def test_corrupt_last_checkpoint_falls_back(self):
        with self.assertRaises(ValueError):self.train(44,max_epochs=2)
        folder=self.private/s.STUDY_NAME/'seed-44'
        (folder/'PRIVATE_resume.pt').write_bytes(b'broken')
        self.train(44)
        self.assertTrue((folder/'stability_run_complete.json').exists())
        self.assertTrue(list(folder.glob('PRIVATE_unreadable-*.pt')))

    def test_old_reference_modification_refused(self):
        (self.reference/'history.csv').write_text((self.reference/'history.csv').read_text()+'\n')
        with self.assertRaisesRegex(ValueError,'evidence changed'):self.train(43)

    def test_extension_change_refused(self):
        with patch.object(s,'extension_hashes',return_value={'changed':'hash'}):
            with self.assertRaisesRegex(ValueError,'Code or recipe changed'):self.train(43)

    def test_core_source_change_refused(self):
        with patch.object(v2,'source_hashes',return_value={'changed':'hash'}):
            with self.assertRaisesRegex(ValueError,'Historical V2 source changed'):self.train(43)

    def test_environment_change_refused(self):
        with patch.object(v2,'environment',return_value={'changed':'environment'}):
            with self.assertRaisesRegex(ValueError,'Software/GPU differs'):self.train(43)

    def test_plan_seed_change_refused(self):
        path=self.private/s.STUDY_NAME/'study.json'
        plan=s.read_json(path);plan['additional_seeds']=[43,45];v2.atomic_json(path,plan)
        with self.assertRaisesRegex(ValueError,'Unexpected stability plan'):self.train(43)

    def test_compare_needs_both_new_seeds(self):
        self.train(43)
        with self.assertRaisesRegex(ValueError,'Seed 44 has not completed'):s.compare(self.private)
        self.assertFalse((self.private/'COMP3710_A1_Stability_Results.zip').exists())

    def test_completed_report_edit_refused(self):
        self.train(43)
        path=self.private/s.STUDY_NAME/'seed-43/v2_summary.json'
        data=s.read_json(path);data['resume_count']+=1;v2.atomic_json(path,data)
        with self.assertRaisesRegex(ValueError,'Completed evidence changed'):s.run_seed(self.private,43)

    def test_compare_all_seeds_no_test_files_bundle_private_excluded(self):
        self.complete_pair();s.compare(self.private)
        summary=s.read_json(self.private/s.STUDY_NAME/'comparison/stability_summary.json')
        self.assertEqual(summary['seeds'],[42,43,44])
        self.assertFalse(summary['candidate_changed_automatically'])
        self.assertEqual(summary['provisional_candidate_seed'],42)
        self.assertFalse(summary['new_final_test_model_evaluation_performed'])
        self.assertEqual(summary['unique_validation_patients_per_run'],4)
        self.assertEqual(summary['two_new_replications_descriptive_statistics']['patient_accuracy']['number_of_runs'],2)
        with zipfile.ZipFile(self.private/'COMP3710_A1_Stability_Results.zip') as z:
            self.assertEqual(len(z.namelist()),12)
            for name in z.namelist():
                self.assertNotIn('PRIVATE',name);self.assertNotIn('study.json',name)
                self.assertNotIn(str(self.private),z.read(name).decode('utf-8'))
        for rec in self.b.records['test']:self.assertFalse((self.b.root/rec.image_path).exists())

    def test_selected_epoch_inconsistent_rejected(self):
        path=self.reference/'v2_summary.json';data=s.read_json(path)
        data['selected_epoch']=100;v2.atomic_json(path,data)
        with self.assertRaisesRegex(ValueError,'Selected epoch'):s.verify_completed(self.reference)

    def test_configuration_edit_rejected(self):
        self.train(43)
        path=self.private/s.STUDY_NAME/'seed-43/config.json'
        data=s.read_json(path);data['learning_rate']*=2;v2.atomic_json(path,data)
        with self.assertRaisesRegex(ValueError,'configuration differs'):self.train(43)

    def test_publication_preserves_readme_and_is_idempotent(self):
        self.complete_pair();s.compare(self.private)
        project=self.private/'fake_project';project.mkdir()
        initial=b'# Existing report\r\nPrior evidence.\r\n'
        (project/'README.md').write_bytes(initial)
        actual_ctx=s.context(self.private)
        changed=(actual_ctx[0],project,*actual_ctx[2:])
        def fake_git(p,*args):return 'topic-recognition' if args==('branch','--show-current') else str(project)
        with patch.object(s,'context',return_value=changed),patch.object(s,'git',side_effect=fake_git):
            s.publish(self.private);first=(project/'README.md').read_bytes();s.publish(self.private)
        self.assertTrue(first.startswith(initial))
        self.assertEqual(first,(project/'README.md').read_bytes())
        self.assertEqual(first.count(s.MARKER.encode()),1)
        backups=list((self.private/s.STUDY_NAME).glob('README-before-stability-*.md'))
        self.assertEqual(len(backups),1);self.assertEqual(backups[0].read_bytes(),initial)

    def test_missing_resume_files_not_silently_reset(self):
        with self.assertRaises(ValueError):self.train(43,max_epochs=1)
        run=self.private/s.STUDY_NAME/'seed-43'
        for name in ('PRIVATE_resume.pt','PRIVATE_previous.pt'):
            p=run/name
            if p.exists():p.unlink()
        with self.assertRaisesRegex(ValueError,'no resumable checkpoint'):self.train(43)

    def test_committed_plan_check_called(self):
        self.train(43)
        self.assertTrue(s.require_committed.called)


if __name__=='__main__':unittest.main()
