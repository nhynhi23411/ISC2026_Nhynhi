import unittest

import numpy as np

from project_a_missingness_stress import inject_missingness, last_forecasts


class MissingnessStressTests(unittest.TestCase):
    def setUp(self):
        self.values = np.arange(1, 21, dtype=float)
        self.seen = np.ones(20, dtype=bool)
        self.eligible = np.arange(15)

    def test_injection_is_exact_and_never_changes_test_mask(self):
        for mechanism in ("mcar", "block", "value_dependent"):
            result = inject_missingness(
                self.seen, self.values, self.eligible, 0.40, mechanism,
                np.random.default_rng(123),
            )
            self.assertEqual(int(np.sum(self.seen[:15] & ~result[:15])), 6)
            self.assertTrue(result[15:].all())
            self.assertTrue(self.seen.all())

    def test_reporting_aware_uses_only_history_up_to_cutoff(self):
        injected = self.seen.copy()
        injected[7:10] = False
        zero, aware = last_forecasts(self.values, injected, cutoff=9)
        self.assertEqual(zero, 0.0)
        self.assertEqual(aware, 7.0)
        self.values[10:] = 9999.0
        self.assertEqual(last_forecasts(self.values, injected, cutoff=9), (zero, aware))


if __name__ == "__main__":
    unittest.main(verbosity=2)
