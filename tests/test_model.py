"""Engineering checks, including small fixtures; never publication measurements."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


@unittest.skipUnless(importlib.util.find_spec("tensorflow"), "TensorFlow not installed")
class ModelTests(unittest.TestCase):
    def test_retrained_ablation_graphs_execute_and_cannot_be_mislabeled_full(self):
        import tensorflow as tf
        from msffat.model import build_msffat, validate_aligned_model
        for variant,fusion in (("ltf_only","proposed"),("msf_only","proposed"),("full","independent"),("full","standard"),("full","concat")):
            with self.subTest(variant=variant,fusion=fusion):
                tf.keras.backend.clear_session()
                model = build_msffat(num_classes=3,variant=variant,fusion=fusion)
                identity = validate_aligned_model(model,allow_ablation=True)
                self.assertEqual((identity["variant"],identity["fusion"]),(variant,fusion))
                values = model(np.zeros((1,5000,1),dtype="float32"),training=False).numpy()
                np.testing.assert_allclose(values.sum(axis=1),1.,atol=1e-6)
                with self.assertRaises(ValueError):
                    validate_aligned_model(model)
        tf.keras.backend.clear_session()
        incompatible = build_msffat(num_classes=3,msf_dropout=.15)
        with self.assertRaises(ValueError):
            validate_aligned_model(incompatible)
        del incompatible,model
        tf.keras.backend.clear_session()

    def test_stateful_entry_records_costs_and_updates_reference_without_test_feedback(self):
        import importlib.util
        import json
        import sys
        from unittest.mock import patch
        import tensorflow as tf
        from msffat.model import channel_attention
        from msffat.maintenance import sequence_symbol_distribution
        script = Path(__file__).resolve().parents[1] / "scripts/simulate_awf_time_deployment.py"
        spec = importlib.util.spec_from_file_location("deployment_entry_test", script)
        entry = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = entry
        spec.loader.exec_module(entry)
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(17)
        inputs = tf.keras.Input((20, 1), name="cells")
        features = tf.keras.layers.Conv1D(8, 3, padding="same", name="sfed_conv")(inputs)
        lt = channel_attention(features, name="ltf_attention")
        ms = channel_attention(features, name="msf_attention")
        fused = tf.keras.layers.Concatenate(name="feature_fusion_concat")([lt, ms])
        head = tf.keras.layers.Flatten(name="head_flatten")(fused)
        outputs = tf.keras.layers.Dense(2, activation="softmax", name="single_label_output")(head)
        model = tf.keras.Model(inputs, outputs, name="MSFFAT_v2_fixture")
        day_data = {}
        remainder_arrays = []
        for day, suffix in entry.DAYS:
            probe = np.ones((10, 20), dtype=np.float32)
            refresh = np.ones((4, 20), dtype=np.float32)
            remainder = np.ones((6, 20), dtype=np.float32)
            remainder_arrays.append(remainder)
            day_data[day] = {"probe_x": probe, "probe_y": np.tile([0, 1], 5),
                             "refresh_x": refresh, "refresh_y": np.tile([0, 1], 2),
                             "remainder_x": remainder, "remainder_y": np.tile([0, 1], 3)}
        def fake_train(args, run_root, model_root, *rest):
            model_path = model_root / "day0.keras"
            model.save(model_path)
            return entry.State((), model_path, sequence_symbol_distribution(-np.ones((10, 20))), 1.0, 1.0)
        def fake_accuracy(model, x, y):
            if mode == "js_only" and any(x is item["probe_x"] for item in day_data.values()):
                raise AssertionError("JS-only policy must never request probe accuracy")
            # Evaluation values intentionally cannot alter the detector thresholds.
            return 0.1 if any(x is rem for rem in remainder_arrays) else 0.99 if x is day_data[3]["probe_x"] else 0.8
        def check_refresh(model, x, y, **kwargs):
            self.assertEqual(len(x), 4)
            self.assertTrue(any(x is data["refresh_x"] for data in day_data.values()))
            return {"scope": "attention", "frozen_digest_unchanged": True}
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            data_root = folder / "raw"
            data_root.mkdir()
            for name in ["tor_200w_2500tr_new.npz"] + [f"tor_200w_100tr_time_test{s}.npz" for _, s in entry.DAYS]:
                (data_root / name).write_bytes(b"engineering fixture; not research data")
            for policy,mode,updates in (("detector","joint_dynamic",1),("fixed","joint_dynamic",3),("never","joint_dynamic",0),("detector","js_only",1)):
                with self.subTest(policy=policy):
                    output = folder / (policy+"_"+mode)
                    argv = [str(script), "--data-root", str(data_root), "--output-root", str(output), "--length", "20", "--update-policy",policy,"--detector-mode",mode]
                    with patch.object(sys, "argv", argv), \
                         patch.object(entry, "prepare_data", return_value=(["a", "b"], None, None, None, day_data)), \
                         patch.object(entry, "train_day0", side_effect=fake_train), \
                         patch.object(entry, "load_model", return_value=model), \
                         patch.object(entry, "model_identity", return_value={"engineering_fixture": True}), \
                         patch.object(entry, "accuracy", side_effect=fake_accuracy), \
                         patch.object(entry, "fit_attention", side_effect=check_refresh):
                        entry.main()
                    result = json.loads((output / "deployment_result.json").read_text())
                    self.assertEqual(result["triggers"],updates)
                    self.assertEqual(result["maintenance_labels_per_site"],sum(r["probe_labels_per_site"] for r in result["rows"])+2*updates)
                    for row in result["rows"]:
                        if not row["probe_accuracy_evaluated"]:
                            self.assertIsNone(row["probe_accuracy"])
                            self.assertIsNone(row["probe_drop_pp"])
                    self.assertEqual(result["reserved_refresh_traces_per_site"],50)
                    self.assertEqual(result["result_kind"],"engineering_check")
                    if policy == "detector" and mode == "joint_dynamic":
                        self.assertEqual(result["maintenance_labels_per_site"],12)
                        self.assertEqual(result["probe_accuracy_checks"],2)
                        self.assertEqual(result["rows"][2]["reference_day"],10)
                        self.assertAlmostEqual(result["rows"][1]["q_before"],.045)
                        self.assertAlmostEqual(result["rows"][1]["q_after"],.045)
                        self.assertAlmostEqual(result["rows"][-1]["q_after"],.04)
                    if mode == "js_only":
                        self.assertEqual(result["maintenance_labels_per_site"],2)
                        self.assertEqual(result["probe_accuracy_checks"],0)
                    if policy == "fixed":
                        self.assertTrue(result["rows"][1]["detector_trigger"])
                        self.assertFalse(result["rows"][0]["update_executed"])
                        self.assertEqual([r["day"] for r in result["rows"] if r["update_executed"]],[14,28,42])
                    self.assertEqual(json.loads((output / "run_manifest.json").read_text())["status"],"complete")
                    self.assertIn("| Day |",(output/"deployment_results.md").read_text())

    def test_full_topology_forward_and_serialization(self):
        import tensorflow as tf
        from msffat.model import build_msffat, validate_aligned_model
        tf.keras.backend.clear_session()
        model = build_msffat(num_classes=3)
        validate_aligned_model(model)
        self.assertEqual(model.get_layer("ltf_attention_statistics").output.shape[-1], 1024)
        self.assertEqual(model.get_layer("msf_attention_statistics").output.shape[-1], 1024)
        self.assertEqual(len(model.get_layer("shared_attention_dense_1")._inbound_nodes),2)
        self.assertEqual(len(model.get_layer("shared_attention_dense_2")._inbound_nodes),2)
        self.assertNotIn("ltf_attention_shared_dense_1",{layer.name for layer in model.layers})
        self.assertEqual(tuple(model.get_layer("feature_fusion_concat").output.shape[1:]), (18, 1024))
        self.assertEqual(model.get_layer("feature_fusion_concat").input[0]._keras_history.operation.name, "ltf_attention_multiply")
        x = np.zeros((1, 5000, 1), dtype=np.float32)
        before = model(x, training=False).numpy()
        self.assertEqual(before.shape, (1, 3))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.keras"
            model.save(path)
            loaded = tf.keras.models.load_model(path, compile=False)
            validate_aligned_model(loaded)
            np.testing.assert_allclose(loaded(x, training=False), before, atol=1e-6)
            # An adapted checkpoint may have no compile state; the public
            # evaluator must initialize its own metric/loss configuration.
            import sys
            from unittest.mock import patch
            script = Path(__file__).resolve().parents[1] / "scripts/evaluate_single.py"
            spec = importlib.util.spec_from_file_location("evaluate_entry_test", script)
            evaluate = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(evaluate)
            with patch.object(sys, "argv", [str(script), "--data-root", "unused", "--model", str(path),
                                            "--dataset", "awf-time", "--classes", "3"]), \
                 patch.object(evaluate, "load_awf_time_eval", return_value=(x[:, :, 0], np.array([0]))):
                evaluate.main()
        from msffat.maintenance import attention_transfer_fit
        traces = np.random.default_rng(3).choice([-1., 1.], size=(2, 5000)).astype("float32")
        fit = attention_transfer_fit(model, traces, np.array([0, 1]), epochs=1, batch_size=2)
        self.assertEqual(fit["trainable_parameters"], 49696)
        self.assertGreater(fit["trainable_weight_delta"], 0)
        self.assertTrue(fit["frozen_digest_unchanged"])
        print("Full model parameters:", model.count_params())
        del loaded, model
        tf.keras.backend.clear_session()

    def test_each_scope_updates_only_allowed_weights(self):
        import tensorflow as tf
        from msffat.model import channel_attention, set_adaptation_scope, validate_aligned_model
        from msffat.maintenance import attention_transfer_fit, frozen_weights_digest
        for scope in ("attention", "head", "attention_head", "full"):
            with self.subTest(scope=scope):
                tf.keras.backend.clear_session()
                tf.keras.utils.set_random_seed(17)
                inputs = tf.keras.Input((20, 4), name="cells")
                features = tf.keras.layers.Conv1D(8, 3, padding="same", name="sfed_conv")(inputs)
                features = tf.keras.layers.BatchNormalization(name="sfed_bn")(features)
                features = tf.keras.layers.Dropout(.5, name="sfed_dropout")(features)
                lt = channel_attention(features, name="ltf_attention")
                ms = channel_attention(features, name="msf_attention")
                fused = tf.keras.layers.Concatenate(name="feature_fusion_concat")([lt, ms])
                head = tf.keras.layers.Flatten(name="head_flatten")(fused)
                head = tf.keras.layers.Dense(8, name="head_dense1")(head)
                head = tf.keras.layers.BatchNormalization(name="head_bn1")(head)
                output = tf.keras.layers.Dense(2, activation="softmax", name="single_label_output")(head)
                model = tf.keras.Model(inputs, output, name="MSFFAT_v2_fixture")
                from unittest.mock import patch
                # This tiny graph exercises freezing only; it must never pass
                # the release's full-architecture acceptance check.
                with self.assertRaises(ValueError):
                    validate_aligned_model(model)
                with patch("msffat.model.validate_aligned_model"):
                    set_adaptation_scope(model, scope)
                digest = frozen_weights_digest(model)
                bn_before = [(layer.moving_mean.numpy().copy(), layer.moving_variance.numpy().copy())
                             for layer in model.layers if isinstance(layer, tf.keras.layers.BatchNormalization)]
                x = np.random.default_rng(7).normal(size=(6, 20)).astype("float32")
                # Real input contract has one channel; the fixture replicates it to four.
                x4 = np.repeat(x[..., None], 4, axis=-1)
                from unittest.mock import patch
                with patch("msffat.maintenance.add_channel", return_value=x4), patch("msffat.model.validate_aligned_model"):
                    fit = attention_transfer_fit(model, x, np.array([0, 1, 0, 1, 0, 1]),
                                                 epochs=2, batch_size=3, scope=scope)
                self.assertGreater(fit["trainable_weight_delta"], 0)
                self.assertEqual(frozen_weights_digest(model), digest)
                if scope != "full":
                    bn_after = [(layer.moving_mean.numpy(), layer.moving_variance.numpy())
                                for layer in model.layers if isinstance(layer, tf.keras.layers.BatchNormalization)]
                    for old, new in zip(bn_before, bn_after):
                        np.testing.assert_array_equal(old[0], new[0])
                        np.testing.assert_array_equal(old[1], new[1])


if __name__ == "__main__":
    unittest.main()
