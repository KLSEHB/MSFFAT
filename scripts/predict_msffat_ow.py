"""Export full OW valid/test probabilities from a trained MSFFAT K+1 model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from msffat.data import add_channel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prepared-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--require-gpu", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    gpus = tf.config.list_physical_devices("GPU")
    if args.require_gpu and not gpus:
        raise RuntimeError("TensorFlow found no GPU")
    gpu_name = tf.config.experimental.get_device_details(gpus[0]).get("device_name") if gpus else None
    root = Path(args.prepared_root)
    x_valid = np.load(root / "valid_X.npy", mmap_mode="r")
    y_valid = np.load(root / "valid_y.npy")
    x_test = np.load(root / "test_X.npy", mmap_mode="r")
    y_test = np.load(root / "test_y.npy")
    model = tf.keras.models.load_model(args.model, compile=False)
    valid_probs = model.predict(add_channel(x_valid), batch_size=args.batch_size, verbose=1).astype("float32")
    valid_preds = np.argmax(valid_probs, axis=1).astype("int64")
    probs = model.predict(add_channel(x_test), batch_size=args.batch_size, verbose=1).astype("float32")
    preds = np.argmax(probs, axis=1).astype("int64")
    out = Path(args.output_root).resolve() / "MSFFAT"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "valid_probs.npy", valid_probs)
    np.save(out / "valid_y.npy", y_valid.astype("int64"))
    np.save(out / "valid_pred.npy", valid_preds)
    np.save(out / "test_probs.npy", probs)
    np.save(out / "test_y.npy", y_test.astype("int64"))
    np.save(out / "test_pred.npy", preds)
    metadata = {
        "method": "MSFFAT",
        "model": str(Path(args.model).resolve()),
        "prepared_root": str(root.resolve()),
        "valid_size": int(len(y_valid)),
        "test_size": int(len(y_test)),
        "num_classes": int(probs.shape[1]),
        "gpu": gpu_name,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
