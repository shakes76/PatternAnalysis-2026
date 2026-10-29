import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from PIL import Image

class CaseReviewTests(unittest.TestCase):
 def test_deterministic_unique_case_selection_and_export(self):
  from run_r6_case_review import select_cases, export_cases
  rows=[]
  for i,(label,p) in enumerate([(1,.1),(1,.2),(1,.3),(0,.9),(0,.8),(0,.7),(0,.49),(1,.51),(0,.48)]):
   rows.append(dict(patient_id=f'p{i}',image_id=f'i{i}',label=label,p_ad=p,confidence=max(p,1-p),prediction=int(p>=.5),p_ad_raw=p,prediction_ad80=int(p>=.3),refer_10_percent=int(abs(p-.5)<.03)))
  chosen=select_cases(rows)
  self.assertEqual(len(chosen),9);self.assertEqual(len({r['patient_id'] for r in chosen}),9)
  with TemporaryDirectory() as d:
   root=Path(d);manifest=[]
   for r in chosen:
    for j in range(20):
     p=root/f"{r['patient_id']}_{j}.jpeg";Image.new('L',(8,8),j).save(p)
     manifest.append(dict(patient_id=r['patient_id'],image_id=r['image_id'],label='AD' if r['label'] else 'NC',split='test',file_path=str(p)))
   export_cases(chosen,manifest,root/'out')
   self.assertEqual(len(list((root/'out').glob('*.png'))),9)
   self.assertTrue((root/'out/case_index.csv').exists())
 def test_reject_missing_slice_before_output(self):
  from run_r6_case_review import export_cases
  with TemporaryDirectory() as d:
   out=Path(d)/'out'
   with self.assertRaises(ValueError):export_cases([dict(patient_id='p',image_id='i',label=1)],[],out)
   self.assertFalse(out.exists())
