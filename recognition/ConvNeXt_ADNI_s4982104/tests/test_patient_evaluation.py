import importlib.util
import math
import unittest
from copy import deepcopy


def fixture():
    rows=[]
    for patient,image,label,probs in [('001_S_0001','100',0,[.1,.3]),('001_S_0002','200',1,[.6,.8])]:
        for i in range(20):
            rows.append(dict(file_path=f'/data/{image}_{i}.jpeg',patient_id=patient,image_id=image,label=label,p_ad=probs[i%2]))
    expected=[dict(r,split='validation') for r in rows]
    return rows,expected


class PatientEvaluationTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('patient_evaluation'),'R1 shared evaluator missing')
        import patient_evaluation
        return patient_evaluation

    def test_mean_probability_then_nll_and_binary_brier(self):
        api=self.api();rows,expected=fixture();r=api.evaluate_patients(rows,expected)
        self.assertAlmostEqual(r['metrics']['nll'],(-math.log(.8)-math.log(.7))/2,places=14)
        self.assertAlmostEqual(r['metrics']['brier'],(.2**2+.3**2)/2,places=14)
        self.assertEqual(r['metrics']['confusion_matrix'],[[1,0],[0,1]])
        self.assertEqual(r['metrics']['auroc'],1)
        wrong=sum(-math.log(1-x['p_ad'] if x['label']==0 else x['p_ad']) for x in rows)/40
        self.assertNotAlmostEqual(r['metrics']['nll'],wrong,places=6)
        self.assertEqual(r,api.evaluate_patients(list(reversed(rows)),list(reversed(expected))))

    def test_boundary_clipping_only_affects_nll(self):
        api=self.api();rows,expected=fixture()
        for r in rows:r['p_ad']=1-r['label']
        out=api.evaluate_patients(rows,expected)
        self.assertAlmostEqual(out['metrics']['nll'],-math.log(1e-7),places=8)
        self.assertEqual(out['metrics']['brier'],1)
        self.assertEqual(out['metrics']['confusion_matrix'],[[0,1],[1,0]])
        self.assertEqual(out['nll_clipped_patient_count'],2)

    def test_missing_patient_slice_duplicate_and_conflicting_metadata_rejected(self):
        api=self.api();rows,expected=fixture()
        cases=[rows[:-1],rows[:20],rows+[rows[0]],[]]
        for field,value in [('label',1),('image_id','999'),('patient_id','001_S_9999'),('p_ad',float('nan')),('p_ad',1.1)]:
            bad=deepcopy(rows);bad[0][field]=value;cases.append(bad)
        for i,bad in enumerate(cases):
            with self.subTest(case=i),self.assertRaises(ValueError):api.evaluate_patients(bad,expected)

    def test_manifest_invalid_scan_owner_and_wrong_count_rejected(self):
        api=self.api();rows,expected=fixture()
        bad=deepcopy(expected);bad[-1]['image_id']='100'
        with self.assertRaises(ValueError):api.evaluate_patients(rows,bad)
        with self.assertRaises(ValueError):api.evaluate_patients(rows,expected+[expected[0]])
        with self.assertRaises(ValueError):api.evaluate_patients(rows[:-1],expected[:-1])

    def test_labels_and_threshold_and_single_class_auc(self):
        api=self.api();rows,expected=fixture()
        for r in rows:r['p_ad']=.5
        r=api.evaluate_patients(rows,expected)
        self.assertEqual(r['metrics']['confusion_matrix'],[[0,1],[0,1]])
        one=api.evaluate_patients(rows[:20],expected[:20]);self.assertIsNone(one['metrics']['auroc'])
        for invalid in [2,True,'AD']:
            bad=deepcopy(rows);bad[0]['label']=invalid
            with self.assertRaises(ValueError):api.evaluate_patients(bad,expected)
