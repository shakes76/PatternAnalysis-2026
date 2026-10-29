import csv,hashlib,json
from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import test_check_dataloader
from run_e3 import run_baseline
from run_e4 import run_weighted, compare_arms

class E4Tests(unittest.TestCase):
    def setUp(self):
        self.f=test_check_dataloader.LoaderTests();self.f.setUp();self.addCleanup(self.f.tearDown)
        with self.f.manifest.open() as f:rows=list(csv.DictReader(f))
        for i,r in enumerate(rows[:5]):r.update(patient_id=f'002_S_{i:04d}',image_id=str(200+i),label='NC' if i<3 else 'AD')
        with self.f.manifest.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        s=json.loads(self.f.summary.read_text());s['output_sha256']['split_samples']=hashlib.sha256(self.f.manifest.read_bytes()).hexdigest();self.f.summary.write_text(json.dumps(s))
    def a(self):
        return run_baseline(self.f.manifest,self.f.summary,self.f.root/'A',device='cpu',epochs=3,
            expected_lengths=None,expected_slices=5,expected_patients=1)
    def b(self):
        return run_weighted(self.f.manifest,self.f.summary,self.f.root/'A',self.f.root/'B',device='cpu',epochs=3,
            expected_lengths=None,expected_slices=5,expected_patients=1)
    def test_paired_end_to_end_and_A_unchanged(self):
        self.a(); before={str(p):p.read_bytes() for p in (self.f.root/'A').rglob('*') if p.is_file()}
        r=self.b();self.assertEqual(r['status'],'passed');self.assertEqual(r['total_read_counts'],dict(train=15,validation=35,test=0))
        self.assertTrue(r['paired_checks']['all_epoch_orders_equal'])
        self.assertTrue((self.f.root/'B/comparison_A42_B42.csv').exists())
        self.assertTrue((self.f.root/'B/training/first3/best.pt').exists())
        self.assertTrue(all(Path(p).read_bytes()==b for p,b in before.items()))
        with self.assertRaises(FileExistsError):self.b()
    def test_missing_reference_fails_before_training(self):
        with self.assertRaises(FileNotFoundError):self.b()
        self.assertFalse((self.f.root/'B/training').exists())
        self.assertEqual(json.loads((self.f.root/'B/e4_summary.json').read_text())['status'],'failed')
    def test_reference_csv_tamper_fails_before_training(self):
        self.a();p=self.f.root/'A/prediction_full/patient_predictions.csv';p.write_text(p.read_text()+'\n')
        with self.assertRaisesRegex(ValueError,'hash'):self.b()
        self.assertFalse((self.f.root/'B/training').exists())
    def test_reject_order_mismatch(self):
        self.a();self.b();p=self.f.root/'B/training/training_summary.json';s=json.loads(p.read_text());s['history'][2]['train']['order_sha256']='bad';p.write_text(json.dumps(s))
        with self.assertRaises(ValueError):compare_arms(self.f.root/'A',self.f.root/'B',3)
