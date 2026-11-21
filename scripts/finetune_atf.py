"""Independent-interval adaptation; not the sequential deployment protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from msffat.data import add_channel, load_awf_time_eval, split_time_indices
from msffat.maintenance import attention_transfer_fit
from msffat.model import TemporalCropToMatch, validate_aligned_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--suffix", required=True, choices=["3d", "10d", "2w", "4w", "6w"])
    parser.add_argument("--traces", type=int, default=2, help="TOTAL new labeled refresh traces per site")
    parser.add_argument("--classes", type=int, default=200)
    parser.add_argument("--length", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--scope", choices=["attention", "head", "attention_head", "full"], default="attention")
    parser.add_argument("--output", required=True, help="New .keras checkpoint path")
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    output = Path(args.output)
    metadata = output.with_suffix(".json")
    split_path = output.with_suffix(".indices.npz")
    if output.suffix != ".keras" or any(p.exists() for p in (output, metadata, split_path)):
        parser.error("Use a new .keras output path with no existing checkpoint/metadata/split")
    model = tf.keras.models.load_model(args.model, custom_objects={"TemporalCropToMatch": TemporalCropToMatch}, compile=False)
    validate_aligned_model(model)
    if model.output_shape[-1] != args.classes:
        parser.error("--classes does not match checkpoint; use the sequential entry for complete-cohort models")
    x, y = load_awf_time_eval(args.data_root, suffix=args.suffix, maxlen=args.length)
    if not np.array_equal(np.unique(y), np.arange(args.classes)):
        parser.error("Dataset label mapping does not match --classes; use the saved cohort with the sequential entry")
    parts = split_time_indices(y, traces=args.traces, n_probe=0, seed=args.seed)
    refresh, heldout = parts["refresh"], parts["remainder"]
    before = model.predict(add_channel(x[heldout]), verbose=0).argmax(axis=1)
    started = time.perf_counter()
    fit = attention_transfer_fit(model, x[refresh], y[refresh], epochs=args.epochs,
                                 batch_size=args.batch_size, learning_rate=args.learning_rate,
                                 seed=args.seed, scope=args.scope)
    seconds = time.perf_counter() - started
    after = model.predict(add_channel(x[heldout]), verbose=0).argmax(axis=1)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(output)
    np.savez_compressed(split_path, **parts)
    from msffat.baseline_adapters import sha256
    input_hash = sha256(args.model)
    result = {"method_version": "MSFFAT-v3", "protocol": "independent_interval",
              "config": vars(args), "refresh_labels_per_site": args.traces,
              "new_validation_labels_per_site": 0, "fit": fit, "update_seconds": seconds,
              "pre_accuracy": float(np.mean(before == y[heldout])),
              "post_accuracy": float(np.mean(after == y[heldout])),
              "input_checkpoint_sha256": input_hash}
    metadata.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
