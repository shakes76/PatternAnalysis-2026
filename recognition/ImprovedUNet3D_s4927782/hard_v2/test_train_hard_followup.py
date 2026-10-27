"""Hard-only policy and static algorithm checks; no image/GPU/model execution.

Actual saved QA metadata is copied to temporary fixtures; these tests never open
MRI/label/prediction arrays and do not claim checkpoint tensor execution.
"""
import ast
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import train_hard_followup as hard
import v2_pipeline as v2

HERE=Path(__file__).resolve().parent


def fixture(root):
    root=Path(root);art=root/'artifacts';qa=art/'qa_v2';(qa/'fit').mkdir(parents=True)
    paths={
      'qa':('artifacts/qa_v2/qa_receipt.json',qa/'qa_receipt.json'),
      'fit':('artifacts/qa_v2/fit/fit_receipt.json',qa/'fit/fit_receipt.json'),
      'through':('artifacts/through_704494/pipeline_receipt.json',art/'through_704494/pipeline_receipt.json'),
      'manifest':('artifacts/cache_v2/manifest.json',art/'cache_v2/manifest.json')}
    for relative,target in paths.values():
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((HERE/relative).read_bytes())
    return {k:p for k,(_,p) in paths.items()}


class HardPolicyFixtures(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.paths=fixture(self.temp.name);self.base=v2.load_protocol(HERE/'protocol_v2.json')

    def save(self,key,value):
        self.paths[key].write_text(json.dumps(value,indent=2)+'\n')

    def audit(self):return hard.validate_hard_qa_receipt(self.paths['qa'],self.paths['manifest'],self.base)

    def test_real_saved_qa_keeps_pair_failed_while_hard_passes_unchanged_gates(self):
        measured=self.audit()
        self.assertEqual(measured['status'],'failed_fit_gate')
        self.assertFalse(measured['models']['unet2d']['passed'])
        self.assertTrue(measured['models']['improved3d']['passed'])

    def test_cannot_relabel_original_pair_or_failed_new2d(self):
        original=v2.read_json(self.paths['qa'])
        for change in ['overall','new2d']:
            data=copy.deepcopy(original)
            if change=='overall':data['status']='passed'
            else:data['models']['unet2d']['passed']=True
            self.save('qa',data)
            with self.assertRaises(ValueError):self.audit()

    def test_rejects_wrong_fit_bytes_and_wrong_manifest_identity(self):
        original=v2.read_json(self.paths['qa'])
        original['fit_receipt_sha256']='a'*64;self.save('qa',original)
        with self.assertRaises(ValueError):self.audit()
        self.paths['manifest'].write_text('{}')
        with self.assertRaises(ValueError):self.audit()

    def test_hard_threshold_or_budget_failure_cannot_be_certified(self):
        original_fit=v2.read_json(self.paths['fit']);original_qa=v2.read_json(self.paths['qa'])
        for changed in ['dice','steps','gradient']:
            fit=copy.deepcopy(original_fit);qa=copy.deepcopy(original_qa)
            if changed=='dice':
                fit['models']['improved3d']['history'][-1]['prostate_dice']=.899
                qa['models']['improved3d']['final_fit_prostate_dice']=.899
            elif changed=='steps':fit['steps']=401
            else:qa['models']['improved3d']['real_sampler_batch']['auxiliary_weight_gradient_absolute_sums']=[1.,0.]
            self.save('fit',fit);qa['fit_receipt_sha256']=v2.sha256(self.paths['fit']);self.save('qa',qa)
            with self.subTest(changed=changed),self.assertRaises(ValueError):self.audit()

    def test_static_training_algorithm_is_identical_except_named_entry_validator(self):
        original=next(n for n in ast.parse((HERE/'v2_pipeline.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='train')
        actual=next(n for n in ast.parse((HERE/'train_hard_followup.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='train_hard_body')
        actual.name='train'
        for n in ast.walk(actual):
            if isinstance(n,ast.Name) and n.id=='validate_hard_qa_receipt':n.id='validate_qa_receipt'
        self.assertEqual(ast.dump(actual,include_attributes=False),ast.dump(original,include_attributes=False))
        # The imported frozen module's original policy is not monkeypatched.
        self.assertEqual(v2.validate_qa_receipt.__module__,'v2_pipeline')

    def test_test_pixels_cannot_be_loaded_before_a_real_hard_freeze(self):
        with patch.object(hard,'load_new_manifest',side_effect=AssertionError('No pixel/data access')) as load:
            with self.assertRaises(ValueError):
                hard.prepare_test_hard(self.paths['manifest'],Path('missing_raw'),self.base,None,
                    {'experiment_id':hard.SCOPE_ID,'_sha256':'b'*64},None)
            load.assert_not_called()

    def test_hard_freeze_rejects_fabricated_missing_new2d(self):
        scope={'experiment_id':hard.SCOPE_ID,'_sha256':'b'*64}
        value={'status':'frozen','experiment_id':hard.SCOPE_ID,'scope_plan_sha256':'b'*64,
               'manifest_sha256':v2.sha256(self.paths['manifest']),'base_protocol_sha256':self.base['_sha256'],
               'protocol_sha256':self.base['_sha256'],'models':{'improved3d':{'completed_epochs':30,'checkpoint_epoch':5,'checkpoint_sha256':'c'*64}}}
        path=Path(self.temp.name)/'selected.json';path.write_text(json.dumps(value))
        self.assertEqual(hard.require_hard_selection_freeze(path,self.paths['manifest'],self.base,scope,'c'*64),v2.sha256(path))
        value['models']['unet2d']={'completed_epochs':30,'checkpoint_sha256':'d'*64};path.write_text(json.dumps(value))
        with self.assertRaises(ValueError):hard.require_hard_selection_freeze(path,self.paths['manifest'],self.base,scope,'c'*64)

    def test_wrong_actual_hard_checkpoint_is_rejected(self):
        scope={'experiment_id':hard.SCOPE_ID,'_sha256':'b'*64}
        value={'status':'frozen','experiment_id':hard.SCOPE_ID,'scope_plan_sha256':'b'*64,
               'manifest_sha256':v2.sha256(self.paths['manifest']),'base_protocol_sha256':self.base['_sha256'],
               'protocol_sha256':self.base['_sha256'],'models':{'improved3d':{'completed_epochs':30,'checkpoint_epoch':5,'checkpoint_sha256':'c'*64}}}
        path=Path(self.temp.name)/'selected.json';path.write_text(json.dumps(value))
        with self.assertRaises(ValueError):hard.require_hard_selection_freeze(path,self.paths['manifest'],self.base,scope,'e'*64)


if __name__=='__main__':unittest.main()
