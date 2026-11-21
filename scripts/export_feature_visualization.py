"""Export branch activations, sigmoid gates and fused features without training.

Inputs must already be canonical signed 5,000-position traces. Sample indices
are explicit and recorded; this tool never selects examples by model outcomes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

LAYERS = {
    "raw_ltf": "ltf_s3_pool",
    "raw_msf": "msf_dropout3",
    "gate_ltf": "ltf_attention_weights",
    "gate_msf": "msf_attention_weights",
    "weighted_ltf": "ltf_attention_multiply",
    "weighted_msf": "msf_attention_multiply",
    "fused": "feature_fusion_concat",
}


def extract_features(model, inputs):
    import tensorflow as tf
    from msffat.model import validate_aligned_model

    validate_aligned_model(model)
    x = np.asarray(inputs, dtype="float32")
    if x.ndim == 2:
        x = x[..., None]
    if x.ndim != 3 or x.shape[1:] != (5000, 1) or not len(x):
        raise ValueError("Expected nonempty inputs of shape (N,5000) or (N,5000,1)")
    if not np.isfinite(x).all() or not np.isin(x, [-1, 0, 1]).all():
        raise ValueError("Inputs must contain canonical {-1,0,+1} direction symbols")
    probe = tf.keras.Model(model.inputs, [model.get_layer(name).output for name in LAYERS.values()])
    values = probe([tf.convert_to_tensor(x)], training=False)
    result = {key: value.numpy() for key, value in zip(LAYERS, values)}
    for branch in ("ltf", "msf"):
        assert result["raw_"+branch].shape == (len(x), 18, 512)
        assert result["gate_"+branch].shape == (len(x), 512)
        expected = result["raw_"+branch] * result["gate_"+branch][:, None, :]
        np.testing.assert_allclose(result["weighted_"+branch], expected, rtol=2e-6, atol=1e-6)
    np.testing.assert_allclose(result["fused"], np.concatenate(
        [result["weighted_ltf"], result["weighted_msf"]], axis=-1), rtol=0, atol=0)
    return result


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True, help="NPZ with prepared input arrays")
    parser.add_argument("--x-key", default="x")
    parser.add_argument("--y-key", default="y")
    parser.add_argument("--indices", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata = args.output.with_suffix(".json")
    if args.output.suffix != ".npz" or args.output.exists() or metadata.exists():
        parser.error("Use a new .npz output path; existing arrays or metadata are never overwritten")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import tensorflow as tf
    from msffat.model import model_identity

    model = tf.keras.models.load_model(args.model, compile=False)
    with np.load(args.inputs, allow_pickle=False) as archive:
        x = archive[args.x_key]
        indices = np.asarray(args.indices, dtype="int64")
        if (indices < 0).any() or (indices >= len(x)).any() or len(np.unique(indices)) != len(indices):
            parser.error("--indices must be distinct valid sample positions")
        result = extract_features(model, x[indices])
        result["indices"] = indices
        if args.y_key in archive:
            result["labels"] = archive[args.y_key][indices]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **result)
    record = {"result_kind": "checkpoint_forward_features", "status": "complete",
              "method_version": "MSFFAT-v3", "model_identity": model_identity(model),
              "model_sha256": sha256(args.model), "input_sha256": sha256(args.inputs),
              "output_sha256": sha256(args.output), "selected_indices": indices.tolist(),
              "inference_mode": True, "layer_map": LAYERS,
              "shapes": {key: list(value.shape) for key,value in result.items()}}
    metadata.write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
