import unittest
from unittest.mock import patch

import numpy as np

from msffat.data import load_awf_time_refresh_and_heldout, split_time_indices
from msffat.deployment import detector_step
from msffat.maintenance import sequence_jsd, sequence_symbol_distribution, drift_decision


class MaintenanceTests(unittest.TestCase):
    def test_budget_and_holdout(self):
        x = np.arange(120, dtype=np.float32).reshape(30, 4)
        y = np.repeat([0, 1, 2], 10)
        with patch("msffat.data.load_awf_time_eval", return_value=(x, y)):
            train_x, train_y, valid_x, valid_y, test_x, test_y = load_awf_time_refresh_and_heldout(traces=2)
        self.assertEqual((len(train_x), len(valid_x), len(test_x)), (6, 0, 24))
        self.assertEqual(len(valid_y), 0)
        np.testing.assert_array_equal(np.bincount(train_y), [2, 2, 2])
        self.assertFalse(set(train_x[:, 0]) & set(test_x[:, 0]))
        parts = split_time_indices(y, n_probe=5, traces=2, seed=2)
        self.assertEqual({k: len(v) for k, v in parts.items()}, {"probe": 15, "refresh": 6, "remainder": 9})
        self.assertEqual(len(np.unique(np.concatenate(list(parts.values())))), 30)
        with self.assertRaises(ValueError):
            split_time_indices(np.repeat([0, 1], 6), n_probe=5, traces=2)

    def test_js_definition_and_wrapper_decision_agree(self):
        p = np.array([[1., 0, 0], [0, 1, 0]])
        q = np.array([[0., 0, 1], [0, 1, 0]])
        self.assertAlmostEqual(sequence_jsd(p, q), .5)
        self.assertEqual(sequence_jsd(p, p), 0)
        self.assertEqual(sequence_jsd(q, p), sequence_jsd(p, q))
        x = np.array([[-1, 0], [1, 0]])
        distribution = sequence_symbol_distribution(x)
        class Model:
            def predict(self, x, verbose):
                raise AssertionError("No prediction is allowed when JS does not exceed its threshold")
        result = drift_decision(distribution, x, np.array([0, 1]), Model(), a_base=1)
        expected = detector_step(jsd=0, probe_accuracy=.5, reference_accuracy=1, q=.04, w_pp=2, e=.005)
        self.assertEqual((result.adapt, result.new_tau), (expected.trigger, expected.q_after))
        self.assertIsNone(result.probe_accuracy)
        with self.assertRaises(ValueError):
            sequence_symbol_distribution(np.array([[2, 0]]))


if __name__ == "__main__":
    unittest.main()
