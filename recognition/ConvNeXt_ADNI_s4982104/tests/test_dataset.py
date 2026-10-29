import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset import ADNISliceDataset, IMAGE_HEIGHT, IMAGE_WIDTH

FIELDS = ['file_path','original_split','image_id','patient_id','scan_date','label','split']

class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.rows = []
        for i, (split, label) in enumerate((('train','NC'),('train','AD'),
                                            ('validation','AD'),('test','NC'))):
            p = self.root / f'{100+i}_0.png'
            Image.new('L', (20+i, 16+i), 0 if label == 'NC' else 255).save(p)
            self.rows.append(dict(zip(FIELDS, (str(p), 'test' if split == 'train' else 'train',
                                               str(100+i), f'001_S_{i:04}', '', label, split))))
        self.write()

    def tearDown(self):
        self.tmp.cleanup()

    def write(self):
        self.manifest = self.root / 'split_samples.csv'
        with self.manifest.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader(); w.writerows(self.rows)
        (self.root / 'split_summary.json').write_text(json.dumps({
            'status':'passed', 'output_sha256':{'split_samples':hashlib.sha256(self.manifest.read_bytes()).hexdigest()}}))

    def test_split_filter_uses_new_split_only(self):
        for split, count in [('train',2),('validation',1),('test',1)]:
            ds = ADNISliceDataset(self.manifest, split)
            self.assertEqual(len(ds), count)
            self.assertTrue(all(r['split'] == split for r in ds.records))
        self.assertTrue(all(r['original_split'] == 'test' for r in ADNISliceDataset(self.manifest,'train').records))

    def test_label_mapping_shape_dtype_and_metadata(self):
        ds = ADNISliceDataset(self.manifest,'train')
        for i, expected in enumerate([0,1]):
            r = ds[i]
            self.assertEqual(r['label'].item(), expected)
            self.assertEqual(r['label'].dtype, torch.long)
            self.assertEqual(r['image'].shape, (1, IMAGE_HEIGHT, IMAGE_WIDTH))
            self.assertEqual(r['image'].dtype, torch.float32)
            self.assertTrue(torch.isfinite(r['image']).all())
            self.assertEqual(r['image'].min().item(), expected)
            self.assertEqual(r['image'].max().item(), expected)
            self.assertEqual(r['patient_id'], self.rows[i]['patient_id'])
            self.assertEqual(r['image_id'], self.rows[i]['image_id'])

    def test_rgb_conversion_scaling_and_repeatability(self):
        Image.new('RGB',(35,47),(128,128,128)).save(self.rows[2]['file_path'])
        ds = ADNISliceDataset(self.manifest,'validation')
        first, second = ds[0]['image'], ds[0]['image']
        self.assertTrue(torch.equal(first,second))
        self.assertTrue(torch.allclose(first,torch.full_like(first,128/255)))

    def test_missing_corrupt_and_truncated_images_fail_with_path(self):
        p = Path(self.rows[0]['file_path'])
        for kind in ('missing','corrupt','truncated'):
            with self.subTest(kind=kind):
                if kind == 'missing': p.unlink()
                elif kind == 'corrupt': p.write_bytes(b'not an image')
                else:
                    Image.new('L',(256,240),128).save(p,format='JPEG')
                    p.write_bytes(p.read_bytes()[:-100])
                with self.assertRaisesRegex(RuntimeError, str(p)):
                    ADNISliceDataset(self.manifest,'train')[0]

    def test_hash_mismatch_fails(self):
        with self.manifest.open('a') as f: f.write('\n')
        with self.assertRaisesRegex(ValueError,'hash'):
            ADNISliceDataset(self.manifest,'train')

    def test_unknown_split_or_label_fails(self):
        with self.assertRaises(ValueError): ADNISliceDataset(self.manifest,'validation_typo')
        self.rows[0]['label']='2'; self.write()
        with self.assertRaisesRegex(ValueError,'label'):
            ADNISliceDataset(self.manifest,'train')

    def test_cross_split_patient_and_duplicate_path_fail(self):
        self.rows[2]['patient_id']=self.rows[0]['patient_id']; self.write()
        with self.assertRaisesRegex(ValueError,'patient'):
            ADNISliceDataset(self.manifest,'train')
        self.rows[2]['patient_id']='001_S_0002'
        self.rows[1]['file_path']=self.rows[0]['file_path']; self.write()
        with self.assertRaisesRegex(ValueError,'duplicate'):
            ADNISliceDataset(self.manifest,'train')

if __name__ == '__main__': unittest.main()
