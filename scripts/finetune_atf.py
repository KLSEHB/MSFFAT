"""Attention-transfer fine-tuning for AWF-Time drift intervals."""

from __future__ import annotations

import argparse
import random
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

from msffat.data import add_channel, load_awf_time_refresh_and_heldout
from msffat.maintenance import attention_transfer_fit
from msffat.model import TemporalCropToMatch, set_attention_only_trainable


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--suffix", required=True, choices=["3d", "10d", "2w", "4w", "6w"])
    parser.add_argument("--traces", type=int, default=2, help="Refresh traces per site.")
    parser.add_argument("--classes", type=int, default=200)
    parser.add_argument("--length", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    model = tf.keras.models.load_model(args.model, custom_objects={"TemporalCropToMatch": TemporalCropToMatch})
    set_attention_only_trainable(model)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    x_train, y_train, x_valid, y_valid, x_test, y_test = load_awf_time_refresh_and_heldout(
        args.data_root, suffix=args.suffix, traces=args.traces, maxlen=args.length
    )
    y_train = to_categorical(y_train, args.classes)
    y_valid = to_categorical(y_valid, args.classes)
    callbacks = [
        ReduceLROnPlateau(monitor="val_accuracy", factor=np.sqrt(0.1), patience=5, min_lr=1e-6, verbose=1),
        EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True),
    ]
    start = time.time()
    attention_transfer_fit(model, x_train, y_train, x_valid, y_valid, batch_size=args.batch_size, epochs=args.epochs, callbacks=callbacks)
    print("ATF time seconds:", round(time.time() - start, 2))

    score = model.evaluate(add_channel(x_test), to_categorical(y_test, args.classes), verbose=1)
    print({"test_loss": float(score[0]), "test_accuracy": float(score[1])})
    if args.output:
        model.save(args.output)


if __name__ == "__main__":
    main()
