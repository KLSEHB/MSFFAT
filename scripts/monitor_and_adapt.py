"""Calibration-based drift monitoring followed by optional ATF.

This script implements the manuscript's deployment-oriented maintenance logic.
It computes sequence-level JSD and aggregate probe accuracy on attacker-collected
calibration traces, decides whether to adapt, and runs attention-transfer
fine-tuning if triggered.
"""

from __future__ import annotations

import argparse
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

from msffat.data import add_channel, load_awf_cw, load_awf_time_eval, load_awf_time_refresh
from msffat.maintenance import attention_transfer_fit, drift_decision, sequence_symbol_distribution
from msffat.model import TemporalCropToMatch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--suffix", required=True, choices=["3d", "10d", "2w", "4w", "6w"])
    parser.add_argument("--classes", type=int, default=200)
    parser.add_argument("--length", type=int, default=5000)
    parser.add_argument("--refresh-traces", type=int, default=2)
    parser.add_argument("--a-base", type=float, required=True, help="Historical/baseline aggregate accuracy.")
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--delta-tau", type=float, default=0.005)
    parser.add_argument("--tau-min", type=float, default=0.0)
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
    train_x, _, _, _, _, _ = load_awf_cw(args.data_root, part="200", traces=2500, maxlen=args.length)
    probe_x, probe_y = load_awf_time_eval(args.data_root, suffix=args.suffix, maxlen=args.length)
    q = sequence_symbol_distribution(train_x)
    decision = drift_decision(
        q,
        probe_x,
        probe_y,
        model,
        a_base=args.a_base,
        tau=args.tau,
        epsilon=args.epsilon,
        delta_tau=args.delta_tau,
        tau_min=args.tau_min,
    )
    print(decision)

    if decision.adapt:
        x_train, y_train, x_valid, y_valid = load_awf_time_refresh(
            args.data_root, suffix=args.suffix, traces=args.refresh_traces, maxlen=args.length
        )
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        callbacks = [
            ReduceLROnPlateau(monitor="val_accuracy", factor=np.sqrt(0.1), patience=5, min_lr=1e-6, verbose=1),
            EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True),
        ]
        attention_transfer_fit(
            model,
            x_train,
            to_categorical(y_train, args.classes),
            x_valid,
            to_categorical(y_valid, args.classes),
            batch_size=args.batch_size,
            epochs=args.epochs,
            callbacks=callbacks,
        )
        if args.output:
            model.save(args.output)

    score = model.evaluate(add_channel(probe_x), to_categorical(probe_y, args.classes), verbose=1)
    print({"post_decision_accuracy": float(score[1]), "new_tau": decision.new_tau})


if __name__ == "__main__":
    main()
