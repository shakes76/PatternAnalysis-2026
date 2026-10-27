"""The preceding real v1 pipeline receipt must substantiate its passed flag."""
import copy
import importlib.util
import unittest

HAS_TORCH = importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import qa_v2


@unittest.skipUnless(HAS_TORCH, 'QA module requires the actual Torch execution environment')
class OriginalThroughGuardTests(unittest.TestCase):
    def fixture(self):
        required = {'artifacts/cache64/manifest.json': 'a' * 64,
                    'modules.py': 'b' * 64, 'utils.py': 'c' * 64,
                    **{f'artifacts/full_seed42/{n}/best.pt': str(i) * 64
                       for i, n in enumerate(('unet2d', 'improved3d'), 1)}}
        review = {'infer_native.py': 'd' * 64, 'real_pipeline_check.py': 'e' * 64}
        keys = [['B006', 'Week0'], ['B038', 'Week0']]
        plan = dict(required_original_sha256=required, required_review_sha256=review,
                    qa=dict(through_cases=keys))
        old = dict(status='passed', selected_case_keys=keys, manifest_sha256='a' * 64,
                   original_modules_sha256='b' * 64, original_utils_sha256='c' * 64,
                   adapter_sha256='d' * 64, verifier_sha256='e' * 64,
                   scope=dict(raw_images=2, cached_train_masks=2, models_per_case=2,
                              test_arrays_read=0, raw_masks_read=0), generalisation_evidence=False,
                   cases=[dict(patient=p, visit=v, split='train', preprocessing_elementwise_equal=True,
                               models={n: dict(checkpoint_sha256=required[f'artifacts/full_seed42/{n}/best.pt'],
                                              cache_prediction_elementwise_equal=True,
                                              native_affine_matches_image=True)
                                       for n in ('unet2d', 'improved3d')}) for p, v in keys])
        return old, plan

    def test_complete_matching_receipt_is_accepted(self):
        old, plan = self.fixture()
        qa_v2.validate_original_through(old, plan)

    def test_passed_flag_cannot_hide_wrong_source_or_test_scope(self):
        old, plan = self.fixture()
        for field in ('manifest_sha256', 'adapter_sha256', 'verifier_sha256'):
            modified = copy.deepcopy(old); modified[field] = 'f' * 64
            with self.subTest(field=field), self.assertRaises(ValueError):
                qa_v2.validate_original_through(modified, plan)
        modified = copy.deepcopy(old); modified['scope']['test_arrays_read'] = 1
        with self.assertRaises(ValueError):
            qa_v2.validate_original_through(modified, plan)

    def test_missing_parity_wrong_checkpoint_and_wrong_case_are_rejected(self):
        old, plan = self.fixture()
        for field, value in [('checkpoint_sha256', 'f' * 64),
                             ('cache_prediction_elementwise_equal', False),
                             ('native_affine_matches_image', False)]:
            modified = copy.deepcopy(old); modified['cases'][0]['models']['improved3d'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                qa_v2.validate_original_through(modified, plan)
        modified = copy.deepcopy(old); modified['cases'][0]['patient'] = 'unrelated_patient'
        with self.assertRaises(ValueError):
            qa_v2.validate_original_through(modified, plan)


if __name__ == '__main__':
    unittest.main()
