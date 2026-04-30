"""Train MSFFAT with the sigmoid multi-label head on ARES k-tab data."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from msffat.data import add_channel, load_ares_ktab
from msffat.metrics import average_precision_at_k, precision_at_k
from msffat.model import build_msffat


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--tabs", type=int, required=True, choices=[2, 3, 4, 5])
    parser.add_argument("--classes", type=int, default=100)
    parser.add_argument("--length", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    x_train, y_train, x_valid, y_valid, x_test, y_test = load_ares_ktab(args.data_root, k=args.tabs)
    model = build_msffat(input_shape=(args.length, 1), num_classes=args.classes, mode="multi", msf_dropout=0.15, head_dropout=0.1)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[precision_at_k(args.tabs), average_precision_at_k(args.tabs), tf.keras.metrics.AUC(multi_label=True, name="auc")],
    )
    output = args.output or f"models/msffat_{args.tabs}tab.hdf5"
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        ReduceLROnPlateau(monitor=f"val_ap@{args.tabs}", factor=np.sqrt(0.1), patience=5, min_lr=1e-5, verbose=1),
        EarlyStopping(monitor=f"val_ap@{args.tabs}", patience=10, restore_best_weights=True),
        ModelCheckpoint(output, monitor="val_loss", save_best_only=True, mode="min", verbose=1),
    ]
    model.fit(
        add_channel(x_train),
        y_train.astype("float32"),
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_data=(add_channel(x_valid), y_valid.astype("float32")),
        callbacks=callbacks,
        verbose=1,
    )
    score = model.evaluate(add_channel(x_test), y_test.astype("float32"), verbose=1)
    print({f"p@{args.tabs}": float(score[1]), f"ap@{args.tabs}": float(score[2]), "auc": float(score[3])})


if __name__ == "__main__":
    main()
