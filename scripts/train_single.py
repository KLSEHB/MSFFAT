"""Train MSFFAT in the single-label setting."""

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
from tensorflow.keras.utils import to_categorical

from msffat.data import add_channel, load_awf_cw, load_df_closed_world, load_prepared_wfl_cw
from msffat.model import build_msffat


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--dataset", choices=["df-cw", "awf-cw", "df-defense", "wfl-cw"], required=True)
    parser.add_argument("--defense", default="NoDef", help="For df-defense: WTFPAD, WalkieTalkie, etc.")
    parser.add_argument("--samples", type=int, default=None, help="Training traces per site for DF-style data.")
    parser.add_argument("--awf-part", default="200", choices=["100", "200", "500", "900"])
    parser.add_argument("--awf-traces", type=int, default=100)
    parser.add_argument("--length", type=int, default=5000)
    parser.add_argument("--classes", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", default="models/msffat_single.hdf5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metrics-output", default=None)
    parser.add_argument("--history-output", default=None)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--early-stopping-start-epoch", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    if args.require_gpu and not tf.config.list_physical_devices("GPU"):
        raise RuntimeError("--require-gpu was set, but TensorFlow found no GPU")

    if args.dataset == "df-cw":
        splits = load_df_closed_world(args.data_root, defense="NoDef", samples=args.samples)
    elif args.dataset == "df-defense":
        splits = load_df_closed_world(args.data_root, defense=args.defense, samples=args.samples)
    elif args.dataset == "awf-cw":
        splits = load_awf_cw(args.data_root, part=args.awf_part, traces=args.awf_traces, maxlen=args.length)
    else:
        if args.samples is None:
            raise ValueError("--samples is required for --dataset wfl-cw")
        splits = load_prepared_wfl_cw(args.data_root, samples=args.samples)

    x_train, y_train, x_valid, y_valid, x_test, y_test = splits
    y_train = to_categorical(y_train, args.classes)
    y_valid = to_categorical(y_valid, args.classes)
    y_test_cat = to_categorical(y_test, args.classes)

    model = build_msffat(input_shape=(args.length, 1), num_classes=args.classes, mode="single")
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
        jit_compile=False,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path.with_suffix(".best_val_loss.weights.h5")
    callbacks = [
        ReduceLROnPlateau(monitor="val_accuracy", factor=np.sqrt(0.1), patience=5, min_lr=1e-5, verbose=1),
        EarlyStopping(
            monitor="val_accuracy",
            patience=10,
            restore_best_weights=True,
            start_from_epoch=args.early_stopping_start_epoch,
        ),
        ModelCheckpoint(
            checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            mode="min",
            verbose=1,
        ),
    ]
    started = time.time()
    history = model.fit(
        add_channel(x_train),
        y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_data=(add_channel(x_valid), y_valid),
        callbacks=callbacks,
        verbose=1,
    )
    score = model.evaluate(add_channel(x_test), y_test_cat, verbose=1)
    # EarlyStopping restores the best validation-accuracy weights. Persist that
    # exact in-memory model rather than leaving a val-loss checkpoint behind.
    model.save(args.output)
    best_epoch = int(np.argmax(history.history["val_accuracy"]) + 1)
    result = {
        "status": "complete",
        "dataset": args.dataset,
        "samples_per_class": args.samples,
        "seed": args.seed,
        "classes": args.classes,
        "test_loss": float(score[0]),
        "test_accuracy": float(score[1]),
        "best_epoch": best_epoch,
        "epochs_ran": len(history.history["loss"]),
        "runtime_seconds": float(time.time() - started),
        "model": str(Path(args.output).resolve()),
    }
    if args.history_output:
        history_path = Path(args.history_output)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        keys = list(history.history)
        with history_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["epoch", *keys])
            writer.writeheader()
            for epoch in range(len(history.history[keys[0]])):
                writer.writerow({"epoch": epoch + 1, **{key: history.history[key][epoch] for key in keys}})
    if args.metrics_output:
        metrics_path = Path(args.metrics_output)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(result, indent=2) + "\n")
    print(result)


if __name__ == "__main__":
    main()
