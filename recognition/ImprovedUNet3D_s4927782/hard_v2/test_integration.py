"""Metadata-only integration checks; never run a model or open medical arrays."""
import copy
from pathlib import Path
import tempfile
import unittest

from predict import check_selection
from verify_package import audit_comparison, digest, load

HERE = Path(__file__).resolve().parent

class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.selection = load(HERE / 'artifacts/hard_selection.json')
        self.scope_digest = digest(HERE / 'hard_followup_protocol.json')
        self.weight_digest = self.selection['models']['improved3d']['checkpoint_sha256']

    def test_hard_selection_without_fake_second_model(self):
        self.assertEqual(check_selection(self.selection, self.scope_digest, self.weight_digest)['checkpoint_epoch'], 30)

    def test_wrong_checkpoint_is_rejected_before_tensor_import(self):
        with self.assertRaisesRegex(ValueError, 'Weights differ'):
            check_selection(self.selection, self.scope_digest, '0' * 64)

    def test_pair_or_incomplete_selection_is_rejected(self):
        bad = copy.deepcopy(self.selection)
        bad['models']['unet2d'] = {}
        with self.assertRaisesRegex(ValueError, 'Hard-only'):
            check_selection(bad, self.scope_digest, self.weight_digest)
        bad = copy.deepcopy(self.selection)
        bad['models']['improved3d']['completed_epochs'] = 29
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            check_selection(bad, self.scope_digest, self.weight_digest)

    def test_complete_saved_test_count_aggregation_and_common_targets(self):
        result = audit_comparison(HERE / 'artifacts/hard_common_test/comparison.json', 30)
        self.assertAlmostEqual(result['models']['hard_improved3d']['prostate_dice'], 0.7817839094346838)
        self.assertEqual(result['class_rows_recomputed'], 540)

if __name__ == '__main__':
    unittest.main()
