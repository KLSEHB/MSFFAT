"""Load complete external models without substituting proxy architectures.

The release does not redistribute third-party implementations/checkpoints.
Specs bind a model to an implementation revision, preprocessing, class order and
the Day0 split it was trained on. Keras full-model archives work directly; other
frameworks can supply the same backend protocol through backend_factory.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

from .data import add_channel


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()


def import_callable(reference, root):
    filename, symbol = reference.rsplit(":", 1)
    path = (root / filename).resolve()
    spec = importlib.util.spec_from_file_location("external_" + sha256(path)[:16], path)
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return getattr(module, symbol), path


def verify_initial_split(spec, root, shared_root, cohort):
    if spec["class_order"] != cohort:
        raise ValueError("External model class order differs from the common cohort")
    initial_root = (root / spec["initial_split_root"]).resolve()
    for filename in ("cohort.json", "day0_indices.npz"):
        if not (initial_root / filename).exists():
            raise ValueError("External model requires its original Day0 training split")
    if json.loads((initial_root / "cohort.json").read_text(encoding="utf-8"))["sites"] != cohort:
        raise ValueError("Original checkpoint cohort differs from its declared class order")
    with np.load(initial_root / "day0_indices.npz") as actual, np.load(shared_root / "day0_indices.npz") as expected:
        for key in ("train", "valid"):
            if not np.array_equal(actual[key], expected[key]):
                raise ValueError("External initial training/validation indices differ from the common experiment")
    model_manifest = json.loads((initial_root / "run_manifest.json").read_text(encoding="utf-8"))
    shared_manifest = json.loads((shared_root / "run_manifest.json").read_text(encoding="utf-8"))
    if model_manifest.get("data_sha256") != shared_manifest.get("data_sha256"):
        raise ValueError("External checkpoint was trained on different dataset content")


class KerasArchiveBackend:
    protocol = "refresh-ce-v1"
    framework = "tensorflow"

    def __init__(self, spec, root, classes, length):
        import tensorflow as tf
        # Importing the original implementation registers its custom layers.
        self.source_paths = []
        if spec.get("registration_module"):
            _, path = import_callable(spec["registration_module"], root)
            self.source_paths.append(path)
        checkpoint = (root / spec["checkpoint"]).resolve()
        self.model = tf.keras.models.load_model(checkpoint, compile=False)
        if self.model.output_shape[-1] != classes:
            raise ValueError("External checkpoint has a different output class count")
        self.spec = spec
        self.preprocess = add_channel
        if spec.get("preprocess"):
            self.preprocess, path = import_callable(spec["preprocess"], root)
            self.source_paths.append(path)
        self.from_logits = bool(spec.get("from_logits", False))
        self.source_paths.append(checkpoint)

    def predict(self, x):
        import tensorflow as tf
        values = np.asarray(self.model.predict(self.preprocess(x), batch_size=256, verbose=0))
        if self.from_logits:
            values = tf.nn.softmax(values, axis=-1).numpy()
        if values.ndim != 2 or not np.isfinite(values).all() or np.any(values < 0) or not np.allclose(values.sum(axis=1), 1, atol=1e-4):
            raise ValueError("External backend must return finite multiclass probabilities")
        return values

    def configure(self, model, scope):
        from .maintenance import iter_layers
        if scope not in {"head", "published_partial", "full"}:
            raise ValueError("External scope must be head, published_partial or full")
        names = self.spec.get("head_layers" if scope == "head" else "partial_layers", [])
        if scope != "full" and not names:
            raise ValueError("Exact original head/partial layer names are required")
        layers = list(iter_layers(model))
        known = {layer.name for layer in layers}
        if scope != "full" and not set(names) <= known:
            raise ValueError("Requested external trainable layers do not exist")
        model.trainable = True
        for layer in layers:
            if not getattr(layer, "layers", ()):
                layer.trainable = scope == "full" or layer.name in names
        if not model.trainable_variables:
            raise ValueError("No external trainable parameters selected")

    def fit_refresh(self, x, y, *, scope, epochs, batch_size, learning_rate, seed):
        from .maintenance import fit_refresh_model
        return fit_refresh_model(self.model, self.preprocess(x), y, configure_trainable=self.configure,
                                 scope=scope, epochs=epochs, batch_size=batch_size,
                                 learning_rate=learning_rate, seed=seed, from_logits=self.from_logits)

    def save(self, path):
        path = Path(path).with_suffix(".keras")
        self.model.save(path)
        return path


def load_external_backend(spec_path, *, classes, length, cohort, shared_root, allow_engineering=False):
    spec_path = Path(spec_path).resolve()
    root = spec_path.parent
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    required = {"method", "implementation_url", "revision", "full_configuration", "checkpoint", "initial_split_root", "class_order"}
    if not required <= spec.keys() or any(not spec[k] for k in required):
        raise ValueError("External spec lacks full-method provenance/configuration")
    if spec.get("engineering_fixture", False) and not allow_engineering:
        raise ValueError("Engineering fixture cannot enter a paper comparison")
    verify_initial_split(spec, root, Path(shared_root), cohort)
    if spec.get("backend_factory"):
        factory, source = import_callable(spec["backend_factory"], root)
        backend = factory(spec=spec, root=root, classes=classes, length=length)
        paths = [source, root / spec["checkpoint"]]
    else:
        backend = KerasArchiveBackend(spec, root, classes, length)
        paths = backend.source_paths
    for method in ("predict", "fit_refresh", "save"):
        if not callable(getattr(backend, method, None)):
            raise ValueError(f"External backend lacks {method}")
    if getattr(backend, "protocol", None) != "refresh-ce-v1":
        raise ValueError("External backend does not declare the frozen refresh-only optimization contract")
    return backend, {"spec": spec, "spec_sha256": sha256(spec_path),
                     "source_sha256": {Path(p).name: sha256(p) for p in paths}}
