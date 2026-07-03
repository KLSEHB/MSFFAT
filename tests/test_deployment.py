import unittest

import numpy as np

from msffat.deployment import (
    detector_step,
    find_matching_k,
    probe_only_detector_step,
    stratified_indices,
)


class DeploymentTests(unittest.TestCase):
    def test_stratified_indices_are_balanced_and_disjoint(self):
        y = np.repeat(np.arange(3), 20)
        parts = stratified_indices(y, [0, 1, 2], {"probe": 5, "refresh": 5, "remainder": 10}, seed=7)
        self.assertEqual(
            {name: len(idx) for name, idx in parts.items()},
            {"probe": 15, "refresh": 15, "remainder": 30},
        )
        self.assertEqual(len(np.unique(np.concatenate(list(parts.values())))), 60)
        for idx in parts.values():
            np.testing.assert_array_equal(np.bincount(y[idx], minlength=3), np.repeat(len(idx) // 3, 3))

    def test_detector_is_jsd_gated_and_updates_q_in_requested_direction(self):
        low_jsd = detector_step(jsd=0.04, probe_accuracy=0.80, reference_accuracy=0.90, q=0.05, w_pp=3, e=0.01)
        self.assertFalse(low_jsd.trigger)
        self.assertEqual(low_jsd.q_after, 0.05)
        false_alarm = detector_step(jsd=0.06, probe_accuracy=0.88, reference_accuracy=0.90, q=0.05, w_pp=3, e=0.01)
        self.assertFalse(false_alarm.trigger)
        self.assertAlmostEqual(false_alarm.q_after, 0.06)
        trigger = detector_step(jsd=0.06, probe_accuracy=0.86, reference_accuracy=0.90, q=0.05, w_pp=3, e=0.01)
        self.assertTrue(trigger.trigger)
        self.assertAlmostEqual(trigger.q_after, 0.04)

    def test_probe_only_detector_uses_strict_percentage_point_threshold(self):
        stable = probe_only_detector_step(
            probe_accuracy=0.98, reference_accuracy=1.0, threshold_pp=2.0
        )
        self.assertFalse(stable.trigger)
        self.assertAlmostEqual(stable.probe_drop_pp, 2.0)
        shifted = probe_only_detector_step(
            probe_accuracy=0.979, reference_accuracy=1.0, threshold_pp=2.0
        )
        self.assertTrue(shifted.trigger)
        self.assertEqual(shifted.reason, "probe_accuracy_drop")

    def test_probe_only_detector_rejects_negative_threshold(self):
        with self.assertRaises(ValueError):
            probe_only_detector_step(
                probe_accuracy=0.9, reference_accuracy=1.0, threshold_pp=-1.0
            )

    def test_k_search_uses_strict_drop_threshold_and_rejects_degenerate_labels(self):
        k, oracle = find_matching_k([0.0, 1.0, 2.0, 3.0, 4.0], [False, False, False, True, True], step_pp=0.1)
        self.assertEqual(k, 2.0)
        self.assertEqual(oracle, [False, False, False, True, True])
        self.assertEqual(find_matching_k([1, 2, 3], [True, True, True], step_pp=0.1), (None, None))


if __name__ == "__main__":
    unittest.main()
