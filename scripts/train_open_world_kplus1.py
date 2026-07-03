"""Train and evaluate MSFFAT as a 95+1 open-world classifier."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from msffat.data import add_channel, load_prepared_wfl_ow
from msffat.metrics import kplus1_classification_metrics
from msffat.model import build_msffat


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-root", required=True)
    parser.add_argument("--samples", type=int, required=True, choices=[5, 10, 20, 50])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-stopping-start-epoch", type=int, default=10)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--expected-gpu", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    gpus = tf.config.list_physical_devices("GPU")
    if args.require_gpu and not gpus:
        raise RuntimeError("TensorFlow found no GPU")
    gpu_name = tf.config.experimental.get_device_details(gpus[0]).get("device_name") if gpus else None
    if args.expected_gpu and args.expected_gpu.lower() not in (gpu_name or "").lower():
        raise RuntimeError(f"Expected GPU containing {args.expected_gpu!r}, got {gpu_name!r}")
    x_train, y_train, x_valid, y_valid, x_test, y_test = load_prepared_wfl_ow(args.prepared_root, args.samples)
    counts = np.bincount(np.asarray(y_train), minlength=96)
    if np.any(counts == 0):
        raise ValueError(f"Every K+1 class must have training samples, got counts={counts.tolist()}")
    class_weight = {i: float(len(y_train) / (96 * count)) for i, count in enumerate(counts)}
    model = build_msffat((5000, 1), 96, "single")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"], jit_compile=False)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    callbacks = [
        ReduceLROnPlateau(monitor="val_accuracy", factor=np.sqrt(0.1), patience=5, min_lr=1e-5, verbose=1),
        EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True, start_from_epoch=args.early_stopping_start_epoch),
        ModelCheckpoint(output_dir / "model.best_val_loss.weights.h5", monitor="val_loss", save_best_only=True, save_weights_only=True, mode="min", verbose=1),
    ]
    started = time.time()
    history = model.fit(
        add_channel(x_train), y_train, batch_size=args.batch_size, epochs=args.epochs,
        validation_data=(add_channel(x_valid), y_valid), callbacks=callbacks,
        class_weight=class_weight, verbose=1,
    )
    probabilities = model.predict(add_channel(x_test), batch_size=args.batch_size, verbose=1)
    predictions = np.argmax(probabilities, axis=1)
    metrics, confusion = kplus1_classification_metrics(y_test, predictions, 95)
    model_path = output_dir / "model.keras"
    model.save(model_path)
    np.save(output_dir / "confusion_matrix.npy", confusion)
    keys = list(history.history)
    with (output_dir / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", *keys]); writer.writeheader()
        for epoch in range(len(history.history[keys[0]])):
            writer.writerow({"epoch": epoch + 1, **{key: history.history[key][epoch] for key in keys}})
    result = {
        "status": "complete", "samples_per_monitored_class": args.samples, "seed": args.seed,
        "samples_in_unmonitored_class": int(counts[95]),
        "train_samples": int(len(y_train)),
        "classes": 96, "unmonitored_label": 95, "balanced_class_weights": True,
        "gpu": gpu_name, "best_epoch": int(np.argmax(history.history["val_accuracy"]) + 1),
        "epochs_ran": len(history.history["loss"]), "runtime_seconds": float(time.time() - started),
        "model": str(model_path), **metrics,
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
