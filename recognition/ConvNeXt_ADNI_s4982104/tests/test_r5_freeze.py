import copy
import tempfile
import unittest
from pathlib import Path

class FreezeTests(unittest.TestCase):
    def test_synthetic_is_reproducible_and_separate(self):
        from run_r5_freeze import synthetic
        from tests.test_final_evaluation import fixture
        p,_,_=fixture();original=copy.deepcopy(p)
        a=synthetic(p);b=synthetic(p)
        self.assertEqual(a,b);self.assertEqual(p,original)
        self.assertEqual(a['evidence_kind'],'synthetic_only_not_model_performance')
        self.assertEqual(a['report']['patient_count'],136)
        self.assertEqual(len(a['report']['metrics']),18)
    def test_publish_refuses_existing_destination(self):
        from run_r5_freeze import publish
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'result';publish(p,{'a':1})
            with self.assertRaises(FileExistsError):publish(p,{'a':2})
            self.assertIn('1',(p/'summary.json').read_text())
    def test_missing_sources_fail(self):
        from run_r5_freeze import build
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises((FileNotFoundError,ValueError)):build(Path(d))

    def test_published_protocol_can_be_verified(self):
        from run_r5_freeze import publish
        from final_evaluation import load_protocol
        from run_r5_ensemble import sha
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);code=root/'unit.py';code.write_text('pass')
            protocol={'code_sha256':{'unit.py':sha(code)}}
            publish(root/'out',{'protocol':protocol})
            config=root/'out/r5_final_protocol.json'
            expected=(root/'out/r5_final_protocol.sha256').read_text().split()[0]
            self.assertEqual(load_protocol(config,expected,root),protocol)
    def test_accepted_helper_drift_is_rejected(self):
        import json
        from unittest.mock import patch
        import run_r5_freeze as f
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'helper.py').write_text('changed')
            summary=root/'source.json'
            summary.write_text(json.dumps(dict(status='passed',split='validation',output_sha256={},code_sha256={'helper.py':'wrong'})))
            with patch.object(f,'ROOT',root),patch.object(f,'SOURCES',{'source.json':f.sha(summary)}):
                with self.assertRaisesRegex(ValueError,'Accepted code SHA mismatch'):f.build(root)
