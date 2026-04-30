"""Train/evaluate MSFFAT on the recovered DF open-world split.

This script assumes the open-world labels in the recovered dataset are already
encoded consistently across training, validation, monitored test, and
unmonitored test files.  It reports aggregate accuracy over the concatenated
monitored/unmonitored test split.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

from msffat.data import add_channel, load_df_open_world_evaluation, load_df_open_world_training, sample_per_label
from msffat.model import build_msffat


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--classes", type=int, required=True, help="Number of output classes including any unmonitored class if present.")
    parser.add_argument("--samples", type=int, default=None, help="Optional training samples per label.")
    parser.add_argument("--length", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", default="models/msffat_df_ow.hdf5")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    x_train, y_train, x_valid, y_valid = load_df_open_world_training(args.data_root)
    x_train, y_train = sample_per_label(x_train, y_train, args.samples, seed=args.seed)
    x_mon, y_mon, x_unmon, y_unmon = load_df_open_world_evaluation(args.data_root)
    x_test = np.concatenate([x_mon, x_unmon], axis=0)
    y_test = np.concatenate([y_mon, y_unmon], axis=0)

    model = build_msffat(input_shape=(args.length, 1), num_classes=args.classes, mode="single")
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        ReduceLROnPlateau(monitor="val_accuracy", factor=np.sqrt(0.1), patience=5, min_lr=1e-5, verbose=1),
        EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True),
        ModelCheckpoint(args.output, monitor="val_loss", save_best_only=True, mode="min", verbose=1),
    ]
    model.fit(
        add_channel(x_train),
        to_categorical(y_train, args.classes),
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_data=(add_channel(x_valid), to_categorical(y_valid, args.classes)),
        callbacks=callbacks,
        verbose=1,
    )
    score = model.evaluate(add_channel(x_test), to_categorical(y_test, args.classes), verbose=1)
    print({"test_loss": float(score[0]), "open_world_accuracy": float(score[1])})


if __name__ == "__main__":
    main()
