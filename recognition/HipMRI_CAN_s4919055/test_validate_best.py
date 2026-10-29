"""Validation-only checkpoint recheck with synthetic fixtures and absent test data."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import nibabel as nib
import numpy as np

from dataset import build_inventory, make_patient_split
from experiment import file_sha256
from train import main as train_main
from validate_best import low_scoring_patients, main as validation_main


class BestValidationTests(unittest.TestCase):
    def test_examples_do_not_repeat_a_patient_or_include_missing_scores(self):
        cases = [dict(case_id=name, patient_id=patient, dice=[None]*5+[score])
                 for name, patient, score in [('A_1','A',.1), ('A_2','A',.2),
                                              ('B_1','B',.3), ('C_1','C',.4), ('D_1','D',None)]]
        self.assertEqual([x['case_id'] for x in low_scoring_patients(cases)],
                         ['A_1','B_1','C_1'])

    def test_fixed_checkpoint_validation_never_needs_test_files(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder)
            for name in ['semantic_MRs', 'semantic_labels_only']:
                (root/name).mkdir()
            pixels = np.random.default_rng(42).normal(size=(16,16,4)).astype('float32')
            mask = np.zeros((16,16,4),dtype='int16')
            mask[4:12,4:12,1:3] = 5
            for patient in ['A001','A002','A003']:
                nib.save(nib.Nifti1Image(pixels,np.eye(4)),root/'semantic_MRs'/f'{patient}_Week0_LFOV.nii.gz')
                nib.save(nib.Nifti1Image(mask,np.eye(4)),root/'semantic_labels_only'/f'{patient}_Week0_SEMANTIC.nii.gz')
            manifest = make_patient_split(build_inventory(root), seed=42,
                                          val_patients=1, test_patients=1)
            path = root/'split.json'
            path.write_text(json.dumps(manifest))
            for scan in manifest['splits']['test']['scans']:
                Path(scan['image_path']).unlink()
                Path(scan['mask_path']).unlink()
            common = ['--device','cpu','--manifest',str(path),'--data-root',str(root)]
            fit = root/'fit'
            self.assertEqual(train_main(common+['--mode','fit','--epochs','1','--shape','4','16','16',
                 '--base-channels','2','--batch-size','2','--loss','dice-ce','--output',str(fit)]),0)
            ckpt = fit/'best.pt'
            sha = file_sha256(ckpt)
            target = root/'checked'
            argv = common+['--checkpoint',str(ckpt),'--expected-sha256',sha,'--output',str(target)]
            self.assertEqual(validation_main(argv),0)
            report = json.loads((target/'validation_best.json').read_text())
            self.assertEqual(report['checkpoint_epoch'],1)
            self.assertEqual(report['checkpoint_sha256'],sha)
            self.assertTrue(report['matches_recorded_validation_score'])
            self.assertEqual({x['case_id'] for x in report['cases']},
                             {x['case_id'] for x in manifest['splits']['validation']['scans']})
            self.assertEqual(len(list(target.glob('*_prostate.png'))),1)
            self.assertEqual(file_sha256(ckpt),sha)
            with self.assertRaises(FileExistsError):
                validation_main(argv)
            argv[argv.index('--expected-sha256')+1] = '0'*64
            with self.assertRaisesRegex(ValueError,'differs'):
                validation_main(argv)


if __name__ == '__main__':
    unittest.main()
