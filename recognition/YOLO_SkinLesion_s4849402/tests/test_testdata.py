import csv
import io
import json
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset import export
from modules import CLASS_NAMES
from prepare_test import prepare_test
from test_pipeline import fixture


class TestDataPreparationTests(unittest.TestCase):
    def make_data(self, root, conflict=False):
        manifest = fixture(root)
        export(manifest, root/'training', list(CLASS_NAMES), 'official-isic', 'Synthetic test', root)
        archive = root/'archives'
        archive.mkdir()
        ids = ['ISIC_9990000', 'ISIC_9990001', 'ISIC_9990002']
        with (archive/'ISIC-2017_Test_v2_Part3_GroundTruth.csv').open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['image_id','melanoma','seborrheic_keratosis'])
            writer.writerow([ids[0],0 if conflict else 1,1 if conflict else 0])
            writer.writerow([ids[1],0,0])
            writer.writerow([ids[2],0,0])
        unique = io.BytesIO()
        Image.fromarray(np.full((16,24,3), 190, dtype=np.uint8)).save(unique, format='JPEG')
        with zipfile.ZipFile(archive/'ISIC-2017_Test_v2_Data.zip','w') as z:
            z.writestr('folder/'+ids[0]+'.jpg', (root/'sample0.jpg').read_bytes())
            for name in ids[1:]:
                z.writestr('folder/'+name+'.jpg', unique.getvalue())
            z.writestr('unneeded_superpixel.png', b'not needed')
        with zipfile.ZipFile(archive/'ISIC-2017_Test_v2_Part1_GroundTruth.zip','w') as z:
            for name in ids:
                z.writestr('masks/'+name+'_segmentation.png', (root/'sample0_segmentation.png').read_bytes())
        return archive, ids

    def test_overlap_quarantine_and_hardlinks(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'runs') as folder:
            root = Path(folder)
            archive, ids = self.make_data(root)
            report = prepare_test(archive, root/'raw', root/'training', root/'output', expected_count=3)
            self.assertEqual(report['retained_test_images'],1)
            self.assertEqual(len(report['excluded_test_images']),2)
            self.assertEqual(report['image_ids'],[ids[1]])
            a = root/'raw'/(ids[1]+'.jpg')
            b = root/'output/images/test'/(ids[1]+'.jpg')
            self.assertEqual(os.stat(a).st_ino, os.stat(b).st_ino)
            self.assertEqual(len(list((root/'training/images/train').glob('*.jpg'))),1)
            self.assertFalse((root/'raw/unneeded_superpixel.png').exists())

    def test_conflicting_diagnosis_aborts(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'runs') as folder:
            root = Path(folder)
            archive, _ = self.make_data(root, conflict=True)
            with self.assertRaisesRegex(ValueError,'Conflicting diagnosis'):
                prepare_test(archive, root/'raw', root/'training', root/'output', expected_count=3)
            self.assertFalse((root/'output/data.yaml').exists())

    def test_wrong_sample_count_aborts(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'runs') as folder:
            root = Path(folder)
            archive, _ = self.make_data(root)
            with self.assertRaisesRegex(ValueError,'Expected 600'):
                prepare_test(archive, root/'raw', root/'training', root/'output')


if __name__ == '__main__':
    unittest.main()
