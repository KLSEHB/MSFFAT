"""Evaluate a saved single-label MSFFAT model."""

from __future__ import annotations

import argparse

import tensorflow as tf
from tensorflow.keras.utils import to_categorical

from msffat.data import add_channel, load_awf_time_eval, load_df_closed_world
from msffat.model import TemporalCropToMatch, validate_aligned_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", choices=["df-cw", "awf-time"], required=True)
    parser.add_argument("--suffix", default="6w")
    parser.add_argument("--classes", type=int, required=True)
    parser.add_argument("--length", type=int, default=5000)
    parser.add_argument("--allow-legacy", action="store_true", help="Explicit historical-model evaluation; not a v2 result")
    parser.add_argument("--allow-ablation",action="store_true",help="Evaluate a named retrained branch/fusion control")
    return parser.parse_args()


def main():
    args = parse_args()
    model = tf.keras.models.load_model(args.model, custom_objects={"TemporalCropToMatch": TemporalCropToMatch}, compile=False)
    if not args.allow_legacy:
        identity = validate_aligned_model(model,allow_ablation=args.allow_ablation)
    if model.output_shape[-1] != args.classes:
        raise ValueError("--classes does not match model outputs")
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    if args.dataset == "df-cw":
        _, _, _, _, x_test, y_test = load_df_closed_world(args.data_root,maxlen=args.length)
    else:
        x_test, y_test = load_awf_time_eval(args.data_root, suffix=args.suffix, maxlen=args.length)
    score = model.evaluate(add_channel(x_test), to_categorical(y_test, args.classes), verbose=1)
    print({"method_version": "historical" if args.allow_legacy else "MSFFAT-v3",
           "model_identity":None if args.allow_legacy else identity,
           "test_loss": float(score[0]), "test_accuracy": float(score[1])})


if __name__ == "__main__":
    main()
