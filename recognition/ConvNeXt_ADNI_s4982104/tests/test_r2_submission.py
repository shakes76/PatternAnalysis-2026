"""Execute the real batch wrapper with a stand-in process; never request a GPU."""
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest

SCRIPT=Path(__file__).resolve().parents[1]/'run_r2.slurm'

class R2SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.project=self.root/'project';self.project.mkdir()
        (self.project/'tests').mkdir();(self.project/'tests/test_stub.py').write_text('# fixture')
        (self.project/'run_r2.py').write_text('# fixture')
        self.data=self.root/'data';self.weights=self.root/'weights';self.weights.mkdir()
        split=self.data/'split_seed42_60_20_20';split.mkdir(parents=True)
        for n in ('split_samples.csv','split_summary.json'):(split/n).write_text('fixture')
        for n in ('convnext_tiny-983f1562.pth','convnext_tiny_weights_summary.json'):(self.weights/n).write_text('fixture')
        self.ref=self.root/'n4.json';self.ref.write_text('{}')
        self.python=self.root/'python';self.python.write_text('''#!/usr/bin/env python3
import os,sys,json
from pathlib import Path
with open(os.environ['CALL_LOG'],'a') as f:f.write(json.dumps(sys.argv[1:])+'\\n')
if sys.argv[1:3]==['-m','unittest'] and os.environ.get('FAIL_TESTS')=='1':sys.exit(9)
if 'run_r2.py' in sys.argv:
 out=Path(sys.argv[sys.argv.index('--output-dir')+1]);out.mkdir()
 (out/'r2_summary.json').write_text('{"status":"passed"}')
 (out/'private.pt').write_text('do not return')
''');self.python.chmod(0o755)
        self.log=self.root/'calls.jsonl'
        self.env=dict(os.environ,SLURM_SUBMIT_DIR=str(self.project),SLURM_JOB_ID='123',R2_DATA_ROOT=str(self.data),R2_WEIGHTS_DIR=str(self.weights),CALL_LOG=str(self.log))
    def run_script(self,python=None):
        self.assertTrue(SCRIPT.exists(),'R2 batch wrapper missing')
        return subprocess.run(['bash',str(SCRIPT),str(python or self.python),str(self.ref)],env=self.env,capture_output=True,text=True)
    def test_missing_input_stops_before_python(self):
        self.ref.unlink();r=self.run_script()
        self.assertNotEqual(r.returncode,0);self.assertFalse(self.log.exists())
    def test_failed_tests_stop_training_and_preserve_exit_code(self):
        self.env['FAIL_TESTS']='1';r=self.run_script()
        self.assertEqual(r.returncode,9,r.stderr)
        calls=[json.loads(x) for x in self.log.read_text().splitlines()]
        self.assertTrue(any(c[:2]==['-m','unittest'] for c in calls))
        self.assertFalse(any('run_r2.py' in c for c in calls))
        self.assertEqual((self.data/'r2_diagnostic_123/exit_code.txt').read_text().strip(),'9')
    def test_success_passes_reference_and_archives_evidence_without_weights(self):
        r=self.run_script();self.assertEqual(r.returncode,0,r.stderr)
        calls=[json.loads(x) for x in self.log.read_text().splitlines()]
        command=next(c for c in calls if 'run_r2.py' in c)
        self.assertLess(next(i for i,c in enumerate(calls) if c[:2]==['-m','unittest']), calls.index(command))
        self.assertEqual(command[command.index('--n4-reference')+1],str(self.ref))
        self.assertEqual(command[command.index('--device')+1],'cuda')
        with tarfile.open(self.data/'r2_diagnostic_123_evidence.tar.gz') as archive:
            names=archive.getnames()
        self.assertTrue(any(n.endswith('r2_summary.json') for n in names))
        self.assertFalse(any(n.endswith('.pt') for n in names))
        second=self.run_script();self.assertNotEqual(second.returncode,0)
