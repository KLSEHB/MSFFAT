"""One explicit validation-selection rule for all initial training entries."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import tensorflow as tf


def initial_callbacks(output, *, monitor="val_accuracy", start_epoch=0):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite trained model: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    selected = output.with_suffix(".selected.weights.h5")
    if selected.exists():
        raise FileExistsError(f"Refusing to reuse a previous selection checkpoint: {selected}")
    return selected, [
        tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor, mode="max", factor=np.sqrt(.1), patience=5, min_lr=1e-5),
        tf.keras.callbacks.EarlyStopping(monitor=monitor, mode="max", patience=10,
                                        restore_best_weights=True, start_from_epoch=start_epoch),
        tf.keras.callbacks.ModelCheckpoint(selected, monitor=monitor, mode="max",
                                           save_best_only=True, save_weights_only=True),
    ]


def restore_selection(model, checkpoint, history, *, monitor="val_accuracy"):
    values = np.asarray(history.history[monitor], dtype=float)
    if not len(values) or not np.isfinite(values).all() or not Path(checkpoint).exists():
        raise ValueError("No valid validation-selected checkpoint")
    model.load_weights(checkpoint)
    epoch_numbers = getattr(history,"epoch",list(range(len(values))))
    return {"selection_metric": monitor, "selection_mode": "max",
            "best_epoch": int(epoch_numbers[int(np.argmax(values))]) + 1, "best_validation_value": float(values.max())}


def validate_class_labels(splits, classes):
    for y in splits:
        y = np.asarray(y)
        if y.ndim != 1 or not len(y) or not np.isfinite(y).all() or not np.equal(y, np.floor(y)).all():
            raise ValueError("Expected nonempty integer class labels")
        if y.min() < 0 or y.max() >= classes:
            raise ValueError("Dataset labels and model class count disagree")
    if not np.array_equal(np.unique(splits[0]),np.arange(classes)):
        raise ValueError("The training split is missing one or more declared classes")


def write_training_record(output, *, config, selection, splits, metrics, model):
    from .model import model_identity
    record = {"method_version": "MSFFAT-v3", "model_identity": model_identity(model),
              "config": vars(config), "selection": selection, "metrics": metrics,
              "split_sizes": dict(zip(("train", "validation", "test"), [len(splits[i]) for i in (0,2,4)])),
              "tensorflow": tf.__version__, "model_sha256": file_digest(output),
              "split_array_sha256": {name: array_digest(value) for name,value in zip(
                  ("train_x","train_y","valid_x","valid_y","test_x","test_y"),splits)}}
    prepared = getattr(config,"prepared_root",None) or getattr(config,"data_root",None)
    if prepared and (Path(prepared)/"metadata.json").exists():
        record["prepared_data_metadata"] = json.loads((Path(prepared)/"metadata.json").read_text(encoding="utf-8"))
    path = Path(output).with_suffix(".training.json")
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_digest(array):
    array = np.asarray(array)
    digest = hashlib.sha256((str(array.shape)+str(array.dtype)).encode())
    for start in range(0,len(array),256):
        digest.update(np.ascontiguousarray(array[start:start+256]).tobytes())
    return digest.hexdigest()
