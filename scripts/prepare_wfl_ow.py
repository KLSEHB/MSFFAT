"""Prepare deterministic K+1 few-shot caches from WFL's OW splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from msffat.data import pad_or_truncate
from msffat.baseline_adapters import sha256

DEFAULT_SHOTS = (5, 10, 20, 50)
UNMONITORED_LABEL = 95


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--length", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--shots", type=int, nargs="+", default=list(DEFAULT_SHOTS))
    parser.add_argument(
        "--unmonitored-shots",
        type=int,
        default=None,
        help="If set, sample this many unmonitored training traces for the K+1 class. Use -1 to match the current monitored shot count. Defaults to all unmonitored traces.",
    )
    return parser.parse_args()


def write_direction_array(source, destination, length, chunk_size, indices=None):
    if indices is not None:
        source = source[indices]
    out = np.lib.format.open_memmap(destination, mode="w+", dtype="float32", shape=(len(source), length))
    for start in range(0, len(source), chunk_size):
        stop = min(start + chunk_size, len(source))
        out[start:stop] = pad_or_truncate(source[start:stop],length)
    out.flush()
    del out


def main():
    args = parse_args()
    source_root = Path(args.dataset_dir).resolve()
    output_root = Path(args.output_dir).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("Use a new empty cache directory")
    output_root.mkdir(parents=True, exist_ok=True)
    if args.length < 1 or args.chunk_size < 1:
        raise ValueError("length/chunk-size must be positive")
    train = np.load(source_root / "train.npz")
    x_train = train["X"]
    y_train = np.asarray(train["y"], dtype="int64")
    labels = np.unique(y_train)
    if not np.array_equal(labels, np.arange(UNMONITORED_LABEL + 1)):
        raise ValueError(f"Expected labels 0..{UNMONITORED_LABEL}, got {labels}")
    rng = np.random.default_rng(args.seed)
    ordered = {}
    for label in range(UNMONITORED_LABEL):
        indices = np.flatnonzero(y_train == label)
        rng.shuffle(indices)
        ordered[label] = indices
    unmonitored = np.flatnonzero(y_train == UNMONITORED_LABEL)
    rng.shuffle(unmonitored)
    if any(shot < 1 for shot in args.shots) or args.unmonitored_shots is not None and args.unmonitored_shots < -1:
        raise ValueError("Invalid shot budget")
    if args.unmonitored_shots == 0:
        raise ValueError("K+1 training requires an unmonitored class")
    for shots in args.shots:
        if any(len(ordered[label]) < shots for label in ordered):
            raise ValueError("Insufficient monitored examples for requested budget")
        monitored = np.concatenate([ordered[label][:shots] for label in range(UNMONITORED_LABEL)])
        unmonitored_count = shots if args.unmonitored_shots == -1 else args.unmonitored_shots
        unmonitored_selected = unmonitored if unmonitored_count is None else unmonitored[:unmonitored_count]
        if unmonitored_count is not None and len(unmonitored_selected) != unmonitored_count:
            raise ValueError("Insufficient unmonitored examples for requested budget")
        selected = np.concatenate([monitored, unmonitored_selected])
        write_direction_array(x_train, output_root / f"train_{shots}_X.npy", args.length, args.chunk_size, selected)
        np.save(output_root / f"train_{shots}_y.npy", y_train[selected])
        np.save(output_root / f"train_{shots}_indices.npy",selected)
    del x_train
    train.close()
    sizes = {}
    for split in ("valid", "test"):
        archive = np.load(source_root / f"{split}.npz")
        x = archive["X"]
        y = np.asarray(archive["y"], dtype="int64")
        write_direction_array(x, output_root / f"{split}_X.npy", args.length, args.chunk_size)
        np.save(output_root / f"{split}_y.npy", y)
        sizes[split] = int(len(y))
        del x
        archive.close()
    metadata = {
        "source": str(source_root), "length": args.length, "seed": args.seed,
        "monitored_classes": UNMONITORED_LABEL, "unmonitored_label": UNMONITORED_LABEL,
        "unmonitored_train_samples_total": int(len(unmonitored)),
        "unmonitored_train_samples_selected": None if args.unmonitored_shots is None else int(args.unmonitored_shots),
        "shots": list(args.shots),
        "train_sizes": {
            str(k): int(k * UNMONITORED_LABEL + (len(unmonitored) if args.unmonitored_shots is None else (k if args.unmonitored_shots == -1 else args.unmonitored_shots)))
            for k in args.shots
        },
        "valid_size": sizes["valid"], "test_size": sizes["test"],
    }
    metadata["source_sha256"] = {f"{part}.npz":sha256(source_root/f"{part}.npz") for part in ("train","valid","test")}
    metadata["cache_sha256"] = {p.name:sha256(p) for p in output_root.glob("*.npy")}
    (output_root / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
