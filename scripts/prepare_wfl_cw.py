"""Prepare deterministic few-shot caches from WFL's CW NPZ splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SHOTS = (5, 10, 20, 50)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--length", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-size", type=int, default=2048)
    return parser.parse_args()


def write_direction_array(source, destination: Path, length: int, chunk_size: int, indices=None):
    if indices is not None:
        source = source[indices]
    out = np.lib.format.open_memmap(
        destination, mode="w+", dtype="float32", shape=(source.shape[0], length)
    )
    for start in range(0, source.shape[0], chunk_size):
        stop = min(start + chunk_size, source.shape[0])
        out[start:stop] = np.sign(source[start:stop, :length]).astype("float32", copy=False)
    out.flush()
    del out


def main():
    args = parse_args()
    source_root = Path(args.dataset_dir).resolve()
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    train = np.load(source_root / "train.npz")
    x_train = train["X"]
    y_train = np.asarray(train["y"], dtype="int64")
    labels = np.unique(y_train)
    if not np.array_equal(labels, np.arange(len(labels))):
        raise ValueError(f"CW labels must be contiguous from zero, got {labels}")

    rng = np.random.default_rng(args.seed)
    ordered = {}
    for label in labels:
        indices = np.flatnonzero(y_train == label)
        rng.shuffle(indices)
        if len(indices) < max(SHOTS):
            raise ValueError(f"Class {label} has only {len(indices)} training samples")
        ordered[int(label)] = indices

    for shots in SHOTS:
        selected = np.concatenate([ordered[int(label)][:shots] for label in labels])
        # Keep class-major ordering, matching the legacy per-label sampler.
        write_direction_array(
            x_train, output_root / f"train_{shots}_X.npy", args.length, args.chunk_size, selected
        )
        np.save(output_root / f"train_{shots}_y.npy", y_train[selected])
    del x_train
    train.close()

    split_sizes = {}
    for split in ("valid", "test"):
        archive = np.load(source_root / f"{split}.npz")
        x = archive["X"]
        y = np.asarray(archive["y"], dtype="int64")
        write_direction_array(
            x, output_root / f"{split}_X.npy", args.length, args.chunk_size
        )
        np.save(output_root / f"{split}_y.npy", y)
        split_sizes[split] = int(len(y))
        del x
        archive.close()

    metadata = {
        "source": str(source_root),
        "length": args.length,
        "seed": args.seed,
        "classes": int(len(labels)),
        "shots": list(SHOTS),
        "train_sizes": {str(shots): int(shots * len(labels)) for shots in SHOTS},
        **{f"{name}_size": size for name, size in split_sizes.items()},
    }
    (output_root / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
