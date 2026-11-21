"""Protocol regression checks; fixture numbers are never research results."""
import unittest
import numpy as np
from msffat.deployment import paired_time_indices
from msffat.replay import replay_trajectories
from msffat.results import aggregate_paired


class ProtocolTests(unittest.TestCase):
    def test_max_f1_threshold_matches_enumeration_and_uses_largest_exact_tie(self):
        from msffat.operational import max_f1_threshold
        rng = np.random.default_rng(8)
        for n in (5,31,100):
            y = rng.integers(0,2,n)
            scores = rng.choice([0.,.1,.3,.7,1.],n)
            candidates = np.r_[np.unique(scores),np.nextafter(1.,2.)]
            f1 = []
            for t in candidates:
                pred = scores>=t
                denominator = pred.sum()+y.sum()
                f1.append(2*np.sum(pred&y.astype(bool))/denominator if denominator else 0.)
            expected = max(t for t,v in zip(candidates,f1) if v==max(f1))
            self.assertEqual(max_f1_threshold(y,scores)["threshold"],expected)
        self.assertEqual(max_f1_threshold([0,0],[.2,.8])["threshold"],np.nextafter(1.,2.))
        with self.assertRaises(ValueError):
            max_f1_threshold([0,1],[float("nan"),.5])

    def test_native_input_conversion_and_padding_reject_nonfinite_payloads(self):
        from msffat.data import pad_or_truncate,add_channel
        x = pad_or_truncate([[512,-128],[1,-1,1,-1,1]],maxlen=4)
        np.testing.assert_array_equal(x,[[1,-1,0,0],[1,-1,1,-1]])
        self.assertEqual(add_channel(x).shape,(2,4,1))
        with self.assertRaises(ValueError):
            add_channel(np.array([[512.]]))
        with self.assertRaises(ValueError):
            pad_or_truncate([[float("nan")]])

    def test_budget_prefix_keeps_probe_pool_and_test_exactly_fixed(self):
        y = np.repeat(np.arange(3),100)
        variants = [paired_time_indices(y,[0,1,2],refresh=m,seed=17) for m in (1,2,5,10)]
        for parts in variants:
            for name in ("probe","refresh_pool","remainder"):
                np.testing.assert_array_equal(parts[name],variants[0][name])
            self.assertEqual(len(parts["remainder"]),255)
            self.assertFalse(set(parts["refresh"])&set(parts["remainder"]))
        for lower,upper in zip(variants,variants[1:]):
            self.assertTrue(set(lower["refresh"]) <= set(upper["refresh"]))

    def test_replay_uses_fixed_oracle_resets_q_and_censors_unseen_delay(self):
        events = [{"checkpoint":d,"jsd":js,"probe_accuracy":acc,"reference_probe_accuracy":1.,"remainder_drop_pp":drop}
                  for d,js,acc,drop in [(3,.02,.90,5),(10,.06,.90,5),(14,.06,.90,5),(28,.07,1.,0),(42,.01,.9,5)]]
        data = [{"id":"a","events":events},{"id":"b","events":events}]
        output = replay_trajectories(data)
        dynamic = output["joint_dynamic"]
        self.assertEqual((dynamic["tp"],dynamic["tn"],dynamic["fp"],dynamic["fn"]),(4,2,0,4))
        self.assertEqual(len(dynamic["episodes"]),4)
        self.assertEqual(dynamic["episodes"][0]["checkpoint_delay"],1)
        self.assertTrue(dynamic["episodes"][1]["censored"])
        self.assertEqual(dynamic["events"][5]["q_before"],.04)
        for value in output.values():
            self.assertEqual(value["positive_events"],8)
            self.assertEqual(value["negative_events"],2)
        self.assertEqual(output["accuracy_only"]["fn"],0)
        self.assertIsNone(replay_trajectories([{"id":"positive-only","events":events[:1]}])["joint_dynamic"]["false_alarm_rate"])

    def test_aggregation_rejects_targets_missing_seeds_and_duplicate_runs(self):
        def fixture(seed):
            return {"method_version":"MSFFAT-v3","result_kind":"measured_run","status":"complete","experiment":"paired_adaptation",
                    "model_identity":{"variant":"full","fusion":"proposed"},"seed":seed,"scope":"attention",
                    "policy":"fixed","refresh_per_site_per_update":2,"day0_remainder_accuracy":.9,
                    "rows":[{"day":d,"pre_accuracy":.8,"post_accuracy":.81+.01*seed,
                             "update_executed":d>=14,"runtime_seconds":1. if d>=14 else None} for d in (3,10,14,28,42)]}
        records = [fixture(0),fixture(1)]
        summary = aggregate_paired(records,expected_seeds=(0,1))[0]
        self.assertAlmostEqual(summary["days"][-1]["post_accuracy"]["mean"],.815)
        self.assertAlmostEqual(summary["days"][-1]["post_accuracy"]["std"],np.sqrt(.00005))
        with self.assertRaises(ValueError):
            aggregate_paired(records,expected_seeds=(0,1,2))
        with self.assertRaises(ValueError):
            aggregate_paired(records+[records[0]],expected_seeds=(0,1))
        with self.assertRaises(ValueError):
            aggregate_paired([{"status":"illustrative_target_not_measured"}])
        legacy = [dict(row,method_version="MSFFAT-v2") for row in records]
        with self.assertRaises(ValueError):
            aggregate_paired(legacy,expected_seeds=(0,1))


if __name__ == "__main__":
    unittest.main()
