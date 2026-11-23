"""Packaging tests with artificial metadata/tensors; no real held-out inference."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from torch import nn
import submission_prepare as prep
import submission_prediction as pred
import train_convnext_v2 as v2
from dataset import DataError,load_finalized_splits

class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.base=Path(self.tmp.name);self.root=self.base/'data';self.root.mkdir()
        metadata={};rows=[]
        for label,name in ((0,'NC'),(1,'AD')):
            folder=self.root/'AD_NC'/'train'/name;folder.mkdir(parents=True)
            for j in range(10):
                subject=f'{label:03d}_S_{j:04d}';scan=f'scan{label}_{j}'
                metadata[scan]={f:f'/synthetic/{subject}/image' for f in prep.reviewed.FIELDS}
                metadata[scan]['label']=2*label
                for index in range(20):
                    path=folder/f'{scan}_{index}.jpeg';path.write_bytes(b'not decoded by partition reproduction')
                    rows.append(dict(image_path=path.relative_to(self.root).as_posix(),subject_id=subject,
                        scan_id=scan,slice_index=index,source_split='train',source_class=name,metadata_label=2*label,label=label))
        for path in ('test/AD','test/NC'):(self.root/'AD_NC'/path).mkdir(parents=True)
        self.meta=self.root/'meta_data_with_label.json';self.meta.write_text(json.dumps(metadata))
        assigned=prep.reviewed.assign_split(rows)
        self.encoded={f'{s}.csv':prep.reviewed.csv_bytes([r for r in assigned if r['split']==s]) for s in prep.reviewed.PARTITIONS}
        self.protocol={'metadata_sha256':prep.sha(self.meta),'mapping_rule':'synthetic fixed grouping',
            'manifest_sha256':{n:hashlib.sha256(v).hexdigest() for n,v in self.encoded.items()},
            'counts':prep.reviewed.split_counts(assigned),'approval_provenance':{'basis':'synthetic test only'}}
        (self.base/'APPROVED_PARTITIONS.json').write_text(json.dumps(self.protocol))
        helper=self.base/'preparation_helpers';helper.mkdir()
        (helper/'prepare_patient_split.py').write_bytes(Path(prep.reviewed.__file__).read_bytes())
    def test_exact_bytes_counts_and_approval_provenance(self):
        reference=self.base/'original';reference.mkdir()
        for n,b in self.encoded.items():(reference/n).write_bytes(b)
        with patch.object(prep,'BASE',self.base):receipt=prep.reproduce(self.root,self.base/'reproduced',reference)
        self.assertEqual(receipt['images_decoded'],0)
        bundle=load_finalized_splits(self.root,self.base/'reproduced')
        self.assertEqual(bundle.manifest_hashes,self.protocol['manifest_sha256'])
        summary=json.loads((self.base/'reproduced'/'split_summary.json').read_text())
        self.assertFalse(summary['new_approval_granted']);self.assertEqual(summary['approval_provenance'],self.protocol['approval_provenance'])
        for n,b in self.encoded.items():self.assertEqual((reference/n).read_bytes(),b)
        with patch.object(prep,'BASE',self.base),self.assertRaises(DataError):prep.reproduce(self.root,self.base/'reproduced')
    def test_metadata_drift_rejected_without_output(self):
        self.meta.write_text('{}')
        with patch.object(prep,'BASE',self.base),self.assertRaises(DataError):prep.reproduce(self.root,self.base/'bad')
        self.assertFalse((self.base/'bad').exists())
    def test_reference_mismatch_rejected_without_output(self):
        reference=self.base/'original';reference.mkdir()
        for n,b in self.encoded.items():(reference/n).write_bytes(b+b'changed')
        with patch.object(prep,'BASE',self.base),self.assertRaises(DataError):prep.reproduce(self.root,self.base/'bad',reference)
        self.assertFalse((self.base/'bad').exists())
    def test_approval_hashes_cannot_be_ignored(self):
        self.protocol['manifest_sha256']['train.csv']='0'*64
        (self.base/'APPROVED_PARTITIONS.json').write_text(json.dumps(self.protocol))
        with patch.object(prep,'BASE',self.base),self.assertRaises(DataError):prep.reproduce(self.root,self.base/'bad')

class StrictLoadingTests(unittest.TestCase):
    class Toy(nn.Module):
        def __init__(self):super().__init__();self.head=nn.Linear(1,2);self.scalar=nn.Parameter(torch.tensor(1.))
        def forward(self,x):return self.head(x)*self.scalar
    def setUp(self):
        self.fake=SimpleNamespace(ConvNeXtTiny=self.Toy,BaselineCNN=self.Toy)
        self.value={'format_version':2,'architecture':'ConvNeXtTiny','pretrained':{'used':False},
            'configuration':{'recipe':'scratch_augmented'},'model_label_map':{'NC':0,'AD':1},
            'model_state':self.Toy().state_dict()}
    def test_valid_state_loads(self):pred.strict_model(self.value,self.fake)
    def test_exact_shape_scalar_conversion_and_dtype(self):
        for tensor in (torch.ones(1),torch.tensor(1.,dtype=torch.float64)):
            value=deepcopy(self.value);value['model_state']['scalar']=tensor
            with self.assertRaises(ValueError):pred.strict_model(value,self.fake)
    def test_missing_extra_and_nonfinite_keys(self):
        for kind in ('missing','extra','nan'):
            value=deepcopy(self.value)
            if kind=='missing':value['model_state'].pop('scalar')
            elif kind=='extra':value['model_state']['unknown']=torch.ones(1)
            else:value['model_state']['scalar']=torch.tensor(float('nan'))
            with self.assertRaises(ValueError):pred.strict_model(value,self.fake)
    def test_pretraining_and_labels_rejected(self):
        for changes in ({'pretrained':{'used':True}},{'model_label_map':{'AD':0,'NC':1}}):
            with self.assertRaises(ValueError):pred.strict_model({**self.value,**changes},self.fake)
    def test_source_hashes_are_checked_not_rewritten(self):
        base=Path(__file__).resolve().parent
        profile=json.loads((base/'HISTORICAL_RUNTIME.json').read_text())
        value={'source_code_sha256':{n:profile['sha256'][n] for n in v2.SOURCE_FILES}}
        self.assertEqual(pred.source_directory(value),base/'historical_runtime')
        value['source_code_sha256']['dataset.py']='0'*64
        with self.assertRaises(ValueError):pred.source_directory(value)
    def test_frozen_bytes_match_retained_selection_not_only_self_manifest(self):
        base=Path(__file__).resolve().parent
        selected=json.loads((base/'A1_SELECTED_MODEL.json').read_text())
        expected=selected['source_code_sha256']
        for name in v2.SOURCE_FILES:
            self.assertEqual(pred.sha(base/'historical_runtime'/name),expected[name],name)

class DispatchAndPolicyTests(unittest.TestCase):
    def test_a1_recipe_is_original_not_schedule_experiment(self):
        c=v2.Config('scratch_augmented');self.assertEqual(c.learning_rate,3e-4)
        self.assertEqual(c.epochs,30);self.assertEqual(c.warmup_epochs,3)
        self.assertNotIn('decay_end_epoch',asdict(c))
    def test_policy_application_matches_existing_function(self):
        from negative_policy import actions
        q=[0,.01,.05,.1,.49,.5,.75,.9,1]
        for name in ('reference','strict_cn','no_auto_cn'):
            expected=actions(q,.75,name)
            actual=[pred.action(p,.75,.05 if name=='strict_cn' else None,name=='no_auto_cn') for p in q]
            self.assertEqual(actual,expected)
    def test_full_training_requires_explicit_flag(self):
        p=Path(__file__).resolve().parent/'train.py'
        result=subprocess.run([sys.executable,'-B',str(p),'a1','--data-root','missing','--splits-dir','missing',
            '--output-dir','not-created'],capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0);self.assertIn('--full-training',result.stderr)
    def test_inference_has_no_test_partition_option(self):
        p=Path(__file__).resolve().parent/'predict.py'
        result=subprocess.run([sys.executable,'-B',str(p),'infer','--help'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0);self.assertIn('{train,validation}',result.stdout)
        self.assertNotIn('private-root',result.stdout)
    def test_legacy_test_confirmation_guard_still_blocks(self):
        p=Path(__file__).resolve().parent/'predict.py'
        result=subprocess.run([sys.executable,'-B',str(p),'evaluate','--private-root','missing'],capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0);self.assertIn('--confirm-final-test',result.stderr)

if __name__=='__main__':unittest.main(verbosity=2)
