"""Engineering fixtures for persistence, metrics and external optimizer I/O."""
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
import numpy as np


@unittest.skipUnless(importlib.util.find_spec("tensorflow"),"TensorFlow not installed")
class TrainingTests(unittest.TestCase):
    def test_validation_selected_weights_survive_worse_last_epoch(self):
        import tensorflow as tf
        from msffat.training import initial_callbacks,restore_selection
        tf.keras.backend.clear_session()
        model = tf.keras.Sequential([tf.keras.Input((1,)),tf.keras.layers.Dense(1,use_bias=False)])
        with tempfile.TemporaryDirectory() as folder:
            selected,callbacks = initial_callbacks(Path(folder)/"model.keras")
            callback = callbacks[-1]
            callback.set_model(model)
            model.set_weights([np.array([[1.]],dtype="float32")])
            callback.on_epoch_end(0,{"val_accuracy":.9,"val_loss":.8})
            model.set_weights([np.array([[2.]],dtype="float32")])
            callback.on_epoch_end(1,{"val_accuracy":.8,"val_loss":.1})
            selection = restore_selection(model,selected,types.SimpleNamespace(history={"val_accuracy":[.9,.8]}))
            self.assertEqual(selection["best_epoch"],1)
            model.save(Path(folder)/"model.keras")
            loaded = tf.keras.models.load_model(Path(folder)/"model.keras",compile=False)
            np.testing.assert_array_equal(loaded.get_weights()[0],[[1.]])

    def test_ranking_metrics_are_per_sample_and_serializable(self):
        import tensorflow as tf
        from msffat.ranking_metrics import PrecisionAtK,AveragePrecisionAtK
        y = np.array([[1,0,1],[0,1,1],[0,0,0]],dtype="float32")
        scores = np.array([[.9,.8,.7],[.9,.8,.7],[.9,.8,.7]],dtype="float32")
        for cls,expected in ((PrecisionAtK,1/3),(AveragePrecisionAtK,.25)):
            metric = cls(2)
            metric.update_state(y[:2],scores[:2])
            metric.update_state(y[2:],scores[2:])
            self.assertAlmostEqual(float(metric.result()),expected,places=6)
            restored = tf.keras.metrics.deserialize(tf.keras.metrics.serialize(metric))
            self.assertIsInstance(restored,cls)
            restored.update_state(y,scores,sample_weight=np.array([1,2,0]))
            expected_weighted = .5 if cls is PrecisionAtK else 1/3
            self.assertAlmostEqual(float(restored.result()),expected_weighted,places=6)

    def test_external_complete_multi_input_checkpoint_uses_refresh_only_engine(self):
        import tensorflow as tf
        from msffat.baseline_adapters import KerasArchiveBackend
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(12)
        a,b = tf.keras.Input((4,1)),tf.keras.Input((4,1))
        features = tf.keras.layers.Concatenate()([tf.keras.layers.Flatten()(a),tf.keras.layers.Flatten()(b)])
        features = tf.keras.layers.Dense(4,name="feature_dense")(features)
        features = tf.keras.layers.BatchNormalization(name="feature_bn")(features)
        output = tf.keras.layers.Dense(2,activation="softmax",name="original_head")(features)
        model = tf.keras.Model([a,b],output)
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            model.save(folder/"complete.keras")
            spec = {"checkpoint":"complete.keras","head_layers":["original_head"]}
            backend = KerasArchiveBackend(spec,folder,2,4)
            backend.preprocess = lambda x:[x[...,None],x[...,None]]
            x = np.random.default_rng(1).normal(size=(6,4)).astype("float32")
            y = np.array([0,1,0,1,0,1])
            before = backend.model.get_layer("feature_dense").get_weights()
            fit = backend.fit_refresh(x,y,scope="head",epochs=2,batch_size=4,learning_rate=.001,seed=2)
            self.assertTrue(fit["bn_moments_frozen"])
            self.assertGreater(fit["trainable_weight_delta"],0)
            for old,new in zip(before,backend.model.get_layer("feature_dense").get_weights()):
                np.testing.assert_array_equal(old,new)
            self.assertEqual(backend.predict(x).shape,(6,2))
            backend.save(folder/"updated.keras")

    def test_external_class_order_and_original_split_must_match(self):
        from msffat.baseline_adapters import verify_initial_split,load_external_backend
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ("original","common"):
                sub = root/name
                sub.mkdir()
                (sub/"cohort.json").write_text(json.dumps({"sites":["a","b"]}),encoding="utf-8")
                (sub/"run_manifest.json").write_text(json.dumps({"data_sha256":{"raw":"abc"}}),encoding="utf-8")
                np.savez(sub/"day0_indices.npz",train=np.array([0,1]),valid=np.array([2,3]))
            spec = {"class_order":["a","b"],"initial_split_root":"original"}
            verify_initial_split(spec,root,root/"common",["a","b"])
            np.savez(root/"original/day0_indices.npz",train=np.array([0,2]),valid=np.array([1,3]))
            with self.assertRaises(ValueError):
                verify_initial_split(spec,root,root/"common",["a","b"])
            (root/"spec.json").write_text(json.dumps({"engineering_fixture":True}),encoding="utf-8")
            with self.assertRaises(ValueError):
                load_external_backend(root/"spec.json",classes=2,length=4,cohort=["a","b"],shared_root=root/"common")


if __name__ == "__main__":
    unittest.main()
