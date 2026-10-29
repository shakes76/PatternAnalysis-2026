"""Frozen evaluation guards and matched model comparison on synthetic test scans."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import nibabel as nib
import numpy as np
import torch

from can3d import CAN3D
from dataset import build_inventory, make_patient_split
from experiment import file_sha256, save_checkpoint
from final_evaluate import main
from modules import UNet2D


class FrozenEvaluationTests(unittest.TestCase):
    def test_explicit_test_gate(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(['--model','can3d','--output','unused'])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(['--model','unet2d','--output','unused','--final-test'])

    def test_matched_models_read_only_test_files_and_keep_weights_fixed(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            torch.set_num_threads(1)
            torch.manual_seed(42)
            root = Path(folder)
            for name in ['semantic_MRs','semantic_labels_only']:
                (root/name).mkdir()
            pixels = np.random.default_rng(42).normal(size=(16,16,4)).astype('float32')
            mask = np.zeros((16,16,4),dtype='int16')
            mask[4:12,4:12,1:3] = 5
            for patient in range(5):
                stem = f'A{patient:03}_Week0'
                nib.save(nib.Nifti1Image(pixels,np.eye(4)),root/'semantic_MRs'/f'{stem}_LFOV.nii.gz')
                nib.save(nib.Nifti1Image(mask,np.eye(4)),root/'semantic_labels_only'/f'{stem}_SEMANTIC.nii.gz')
            split = make_patient_split(build_inventory(root),seed=42,val_patients=1,test_patients=3)
            manifest = root/'split.json'
            manifest.write_text(json.dumps(split))
            for name in ['train','validation']:
                for scan in split['splits'][name]['scans']:
                    Path(scan['image_path']).unlink()
                    Path(scan['mask_path']).unlink()
            sources = ['final_evaluate.py','experiment.py','dataset.py','modules.py',
                       'can3d.py','predict.py','validate_best.py']
            plan = dict(plan_version=1,seed=42,shape_dhw=[4,16,16],
                        code_sha256={n:file_sha256(Path(__file__).with_name(n)) for n in sources},
                        manifest_sha256=file_sha256(manifest),test_case_count=3,test_patient_count=3,
                        protocol=dict(warmups_per_case=1,repeats_per_case=2,slice_batch_size=2,example_count=2),models={})
            for name, model in [('can3d',CAN3D()),('unet2d',UNet2D(base_channels=2))]:
                checkpoint = root/f'{name}.pt'
                config = dict(model=name,base_channels=2,mode='fit',seed=42,
                              manifest_sha256=file_sha256(manifest),shape_dhw=[4,16,16])
                save_checkpoint(checkpoint,model,config,epoch=1)
                plan['models'][name] = dict(checkpoint=str(checkpoint),checkpoint_sha256=file_sha256(checkpoint),epoch=1)
            plan_path = root/'plan.json'
            plan_path.write_text(json.dumps(plan))
            common = ['--plan',str(plan_path),'--manifest',str(manifest),'--data-root',str(root),'--device','cpu','--final-test']
            can_output, unet_output = root/'can',root/'unet'
            can_args = common+['--model','can3d','--output',str(can_output)]
            self.assertEqual(main(can_args),0)
            comparison = can_output/'test_metrics.json'
            self.assertEqual(main(common+['--model','unet2d','--output',str(unet_output),
                                          '--comparison-results',str(comparison)]),0)
            reports = [json.loads((out/'test_metrics.json').read_text()) for out in [can_output,unet_output]]
            expected = {s['case_id'] for s in split['splits']['test']['scans']}
            self.assertEqual([r['case_id'] for r in reports[0]['selected_examples']],
                             [r['case_id'] for r in reports[1]['selected_examples']])
            self.assertEqual(len({r['patient_id'] for r in reports[0]['selected_examples']}),2)
            for report,out in zip(reports,[can_output,unet_output]):
                self.assertEqual({r['case_id'] for r in report['cases']},expected)
                self.assertEqual(len(list((out/'examples').glob('*.png'))),2)
                self.assertEqual(len(list((out/'predictions').glob('*.nii.gz'))),3)
                self.assertIsNone(report['peak_allocated_mib'])
                for row in report['cases']:
                    self.assertTrue(row['repeated_predictions_identical'])
                    self.assertEqual(len(row['inference_repeat_seconds']),2)
                    self.assertGreater(row['inference_seconds'],0)
                for path in (out/'predictions').glob('*.nii.gz'):
                    saved = nib.load(path)
                    self.assertEqual(saved.shape,(16,16,4))
                    self.assertEqual(nib.aff2axcodes(saved.affine),('R','A','S'))
                spec = plan['models'][report['model']]
                self.assertEqual(file_sha256(spec['checkpoint']),spec['checkpoint_sha256'])
            with self.assertRaises(FileExistsError):
                main(can_args)
            plan['models']['can3d']['checkpoint_sha256'] = '0'*64
            plan_path.write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError,'checkpoint changed'):
                main(can_args)
            plan['manifest_sha256'] = '0'*64
            plan_path.write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError,'patient split changed'):
                main(can_args)
            plan['code_sha256']['experiment.py'] = '0'*64
            plan_path.write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError,'code differs'):
                main(can_args)


if __name__ == '__main__':
    unittest.main()
