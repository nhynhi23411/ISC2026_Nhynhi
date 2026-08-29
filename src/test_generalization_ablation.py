import unittest
import numpy as np
from project_a_generalization_ablation import feature_vector

class GeneralizationAblationTests(unittest.TestCase):
    def test_feature_components_have_expected_dimensions(self):
        vals = np.arange(40, dtype=float); seen = np.ones(40, dtype=bool); seen[10] = False; vals[10] = np.nan
        dims = {c: len(feature_vector(vals, seen, 20, "reporting_aware", c)) for c in ("base", "seen_lags", "missing_run", "mask_full")}
        self.assertEqual(dims, {"base": 8, "seen_lags": 13, "missing_run": 9, "mask_full": 14})
    def test_no_future_feature_dependency(self):
        vals = np.arange(40, dtype=float); seen = np.ones(40, dtype=bool)
        x = feature_vector(vals, seen, 20, "reporting_aware", "mask_full")
        vals[21:] = 9999
        np.testing.assert_array_equal(x, feature_vector(vals, seen, 20, "reporting_aware", "mask_full"))

if __name__ == "__main__": unittest.main(verbosity=2)
