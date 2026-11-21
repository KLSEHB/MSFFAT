import unittest

import numpy as np

from msffat.deployment import (
    detector_step,
    oracle_requires_update,
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
        self.assertAlmostEqual(low_jsd.q_after, 0.045)
        self.assertIsNone(low_jsd.probe_drop_pp)
        self.assertFalse(low_jsd.accuracy_evaluated)
        false_alarm = detector_step(jsd=0.06, probe_accuracy=0.88, reference_accuracy=0.90, q=0.05, w_pp=3, e=0.01)
        self.assertFalse(false_alarm.trigger)
        self.assertAlmostEqual(false_alarm.q_after, 0.06)
        trigger = detector_step(jsd=0.06, probe_accuracy=0.86, reference_accuracy=0.90, q=0.05, w_pp=3, e=0.01)
        self.assertTrue(trigger.trigger)
        self.assertAlmostEqual(trigger.q_after, 0.05)

    def test_lazy_accuracy_boost_cap_decay_and_preservation_on_trigger(self):
        from unittest.mock import Mock
        forbidden = Mock(side_effect=AssertionError("Probe prediction must stay behind JS gate"))
        out = detector_step(jsd=.04,probe_accuracy=forbidden,reference_accuracy=1,q=.04,w_pp=2,e=.005)
        forbidden.assert_not_called()
        self.assertEqual(out.q_after,.04)
        threshold=.04
        stable=Mock(return_value=1.)
        for _ in range(8):
            out=detector_step(jsd=.1,probe_accuracy=stable,reference_accuracy=1,q=threshold,w_pp=2,e=.005)
            threshold=out.q_after
        self.assertAlmostEqual(threshold,.06)
        update=detector_step(jsd=.1,probe_accuracy=lambda:.9,reference_accuracy=1,q=threshold,w_pp=2,e=.005)
        self.assertTrue(update.trigger)
        self.assertAlmostEqual(update.boost_after,.02)
        self.assertAlmostEqual(update.q_after,threshold)
        for _ in range(8):
            out=detector_step(jsd=0,probe_accuracy=forbidden,reference_accuracy=1,q=threshold,w_pp=2,e=.005)
            threshold=out.q_after
        self.assertAlmostEqual(threshold,.04)
        forbidden.assert_not_called()

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

    def test_four_policies_and_strict_boundaries(self):
        for jsd, accuracy, expected in [(0.03, .99, (False, False)),
                                         (.06, .99, (True, False)),
                                         (.03, .90, (False, True)),
                                         (.06, .90, (True, True))]:
            for mode in ("joint_dynamic", "joint_fixed", "accuracy_only", "js_only"):
                with self.subTest(mode=mode, jsd=jsd, accuracy=accuracy):
                    result = detector_step(jsd=jsd, probe_accuracy=accuracy, reference_accuracy=1,
                                           q=.04, w_pp=2, e=.005, mode=mode)
                    shifted, degraded = expected
                    trigger = degraded if mode == "accuracy_only" else shifted if mode == "js_only" else shifted and degraded
                    self.assertEqual(result.trigger, trigger)
                    if mode != "joint_dynamic":
                        self.assertEqual(result.q_after, .04)
        at_boundary = detector_step(jsd=.04, probe_accuracy=.98, reference_accuracy=1,
                                    q=.04, w_pp=2, e=.005)
        self.assertFalse(at_boundary.trigger)
        self.assertEqual(at_boundary.q_after, .04)

    def test_oracle_is_fixed_and_independent_of_triggers(self):
        self.assertFalse(oracle_requires_update(2.0, 2.0))
        self.assertTrue(oracle_requires_update(2.01, 2.0))
        self.assertFalse(oracle_requires_update(-1.0, 2.0))


if __name__ == "__main__":
    unittest.main()
