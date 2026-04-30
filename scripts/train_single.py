"""Train MSFFAT in the single-label setting."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

from msffat.data import add_channel, load_awf_cw, load_df_closed_world
from msffat.model import build_msffat


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--dataset", choices=["df-cw", "awf-cw", "df-defense"], required=True)
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
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    if args.dataset == "df-cw":
        splits = load_df_closed_world(args.data_root, defense="NoDef", samples=args.samples)
    elif args.dataset == "df-defense":
        splits = load_df_closed_world(args.data_root, defense=args.defense, samples=args.samples)
    else:
        splits = load_awf_cw(args.data_root, part=args.awf_part, traces=args.awf_traces, maxlen=args.length)

    x_train, y_train, x_valid, y_valid, x_test, y_test = splits
    y_train = to_categorical(y_train, args.classes)
    y_valid = to_categorical(y_valid, args.classes)
    y_test_cat = to_categorical(y_test, args.classes)

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
        y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_data=(add_channel(x_valid), y_valid),
        callbacks=callbacks,
        verbose=1,
    )
    score = model.evaluate(add_channel(x_test), y_test_cat, verbose=1)
    print({"test_loss": float(score[0]), "test_accuracy": float(score[1])})


if __name__ == "__main__":
    main()
