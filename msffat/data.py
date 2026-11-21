"""Configurable data loaders for MSFFAT experiments.

Expected raw data are not bundled in this repository.  Set ``MSFFAT_DATA_ROOT``
or pass ``--data-root`` to the scripts.  The loaders mirror the legacy path
layout but avoid hard-coded user-specific directories.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .label_maps import dict_dict


ArraySplit = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def resolve_data_root(data_root: Optional[str] = None) -> Path:
    root = data_root or os.environ.get("MSFFAT_DATA_ROOT")
    if not root:
        raise ValueError("Provide --data-root or set MSFFAT_DATA_ROOT.")
    return Path(root).expanduser().resolve()


def pad_or_truncate(sequences, maxlen: int = 5000, dtype="float32") -> np.ndarray:
    """Convert signed traffic values to directions; keep the first maxlen cells."""
    if maxlen < 1:
        raise ValueError("maxlen must be positive")
    out = np.zeros((len(sequences), maxlen), dtype=dtype)
    for i, seq in enumerate(sequences):
        arr = np.asarray(seq, dtype=dtype)[:maxlen]
        if arr.ndim != 1 or not np.isfinite(arr).all():
            raise ValueError("Expected finite one-dimensional signed cell sequences")
        out[i, : len(arr)] = np.sign(arr)
    return out


def add_channel(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype="float32")
    if not np.isfinite(x).all() or not np.isin(x,[-1,0,1]).all():
        raise ValueError("Native model input must contain only finite direction symbols -1,0,+1")
    if x.ndim == 2:
        return x[..., np.newaxis]
    if x.ndim == 3 and x.shape[-1] == 1:
        return x
    raise ValueError("Expected (N,T) or (N,T,1) direction sequences")


def sample_per_label(x: np.ndarray, y: np.ndarray, samples: Optional[int], seed: int = 42):
    if samples is None:
        return x, y
    if samples < 1:
        raise ValueError("samples must be positive")
    rng = np.random.default_rng(seed)
    selected = []
    for label in np.unique(y):
        idx = np.flatnonzero(y == label)
        if len(idx) < samples:
            raise ValueError(f"Class {label} has fewer than {samples} samples")
        idx = rng.choice(idx, size=samples, replace=False)
        selected.append(idx)
    selected = np.concatenate(selected)
    selected.sort()
    return x[selected], y[selected]


def split_dataset(x, y, val_split=0.05, test_split=0.05, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = np.arange(x.shape[0])
    rng.shuffle(idx)
    x = x[idx]
    y = y[idx]
    test_start = int(len(x) * (1 - test_split))
    val_start = int(len(x) * (1 - val_split - test_split))
    return x[:val_start], y[:val_start], x[val_start:test_start], y[val_start:test_start], x[test_start:], y[test_start:]


def _load_pickle(path: Path):
    with path.open("rb") as handle:
        return np.asarray(pickle.load(handle, encoding="bytes"))


def load_df_closed_world(
    data_root: Optional[str] = None,
    defense: str = "NoDef",
    samples: Optional[int] = None,
    seed: int = 42,
    maxlen: int = 5000,
) -> ArraySplit:
    root = resolve_data_root(data_root) / "df" / "ClosedWorld" / defense
    if defense == "NoDef":
        suffix = "NoDef"
    elif defense.upper() == "WTFPAD":
        suffix = "WTFPAD"
    elif defense == "WalkieTalkie":
        suffix = "WalkieTalkie"
    else:
        suffix = defense

    x_train = _load_pickle(root / f"X_train_{suffix}.pkl")
    y_train = _load_pickle(root / f"y_train_{suffix}.pkl")
    x_train, y_train = sample_per_label(x_train, y_train, samples, seed=seed)
    x_valid = _load_pickle(root / f"X_valid_{suffix}.pkl")
    y_valid = _load_pickle(root / f"y_valid_{suffix}.pkl")
    x_test = _load_pickle(root / f"X_test_{suffix}.pkl")
    y_test = _load_pickle(root / f"y_test_{suffix}.pkl")
    return pad_or_truncate(x_train, maxlen), y_train, pad_or_truncate(x_valid, maxlen), y_valid, pad_or_truncate(x_test, maxlen), y_test


def load_df_open_world_training(data_root: Optional[str] = None, samples: Optional[int] = None, seed: int = 42, maxlen: int = 5000):
    root = resolve_data_root(data_root) / "df" / "OpenWorld" / "NoDef"
    x_train = _load_pickle(root / "X_train_NoDef.pkl")
    y_train = _load_pickle(root / "y_train_NoDef.pkl")
    x_train, y_train = sample_per_label(x_train, y_train, samples, seed=seed)
    x_valid = _load_pickle(root / "X_valid_NoDef.pkl")
    y_valid = _load_pickle(root / "y_valid_NoDef.pkl")
    return pad_or_truncate(x_train,maxlen), y_train, pad_or_truncate(x_valid,maxlen), y_valid


def load_df_open_world_evaluation(data_root: Optional[str] = None):
    root = resolve_data_root(data_root) / "df" / "OpenWorld" / "NoDef"
    return (
        _load_pickle(root / "X_test_Mon_NoDef.pkl"),
        _load_pickle(root / "y_test_Mon_NoDef.pkl"),
        _load_pickle(root / "X_test_Unmon_NoDef.pkl"),
        _load_pickle(root / "y_test_Unmon_NoDef.pkl"),
    )


def _label_map(part: str):
    return dict_dict[part]


def load_awf_cw(
    data_root: Optional[str] = None,
    part: str = "200",
    traces: Optional[int] = 100,
    maxlen: int = 5000,
    seed: int = 42,
) -> ArraySplit:
    path = resolve_data_root(data_root) / "dlwf" / f"tor_{part}w_2500tr.npz"
    npz = np.load(path, allow_pickle=True)
    data = npz["data"]
    labels = npz["labels"]
    npz.close()

    label_map = _label_map(part)
    filtered_x = []
    filtered_y = []
    counts = {}
    for x, y in zip(data, labels):
        if y not in label_map:
            continue
        count = counts.get(y, 0)
        if traces is not None and count >= traces:
            continue
        filtered_x.append(x)
        filtered_y.append(label_map[y])
        counts[y] = count + 1

    x = pad_or_truncate(filtered_x, maxlen=maxlen)
    y = np.asarray(filtered_y, dtype="int64")
    return split_dataset(x, y, seed=seed)


def load_awf_time_eval(data_root: Optional[str] = None, suffix: str = "6w", maxlen: int = 5000):
    path = resolve_data_root(data_root) / "dlwf" / f"tor_200w_100tr_time_test{suffix}.npz"
    npz = np.load(path, allow_pickle=True)
    data = npz["data"]
    labels = npz["labels"]
    npz.close()

    label_map = _label_map("200")
    pairs = [(x, label_map[y]) for x, y in zip(data, labels) if y in label_map]
    x = pad_or_truncate([p[0] for p in pairs], maxlen=maxlen)
    y = np.asarray([p[1] for p in pairs], dtype="int64")
    return x, y


def load_awf_time_refresh(
    data_root: Optional[str] = None,
    suffix: str = "6w",
    traces: int = 2,
    maxlen: int = 5000,
    seed: int = 42,
):
    """Exactly traces refresh samples/site; empty validation arrays for compatibility."""
    x_train, y_train, x_valid, y_valid, _, _ = load_awf_time_refresh_and_heldout(
        data_root, suffix, traces, maxlen, seed)
    return x_train, y_train, x_valid, y_valid


def load_awf_time_refresh_and_heldout(
    data_root: Optional[str] = None,
    suffix: str = "6w",
    traces: int = 2,
    maxlen: int = 5000,
    seed: int = 42,
):
    """Exactly traces refresh samples/site; no additional validation labels."""
    x, y = load_awf_time_eval(data_root=data_root, suffix=suffix, maxlen=maxlen)
    parts = split_time_indices(y, traces=traces, n_probe=0, seed=seed)
    idx, test = parts["refresh"], parts["remainder"]
    return x[idx], y[idx], x[:0], y[:0], x[test], y[test]


def split_time_indices(y, *, traces: int = 2, n_probe: int = 5, seed: int = 2024):
    """Balanced, disjoint probe/refresh partitions; all other traces are held out."""
    from .deployment import stratified_indices
    if traces < 1 or n_probe < 0:
        raise ValueError("Require traces >= 1 and n_probe >= 0")
    labels, counts = np.unique(y, return_counts=True)
    if not len(labels) or counts.min() <= traces + n_probe:
        raise ValueError("Each site needs probe + refresh + at least one test trace")
    parts = stratified_indices(np.asarray(y), labels.tolist(),
                               {"probe": n_probe, "refresh": traces}, seed=seed)
    used = np.concatenate(list(parts.values()))
    parts["remainder"] = np.setdiff1d(np.arange(len(y)), used)
    return parts


def load_ares_ktab(data_root: Optional[str] = None, k: int = 2, maxlen: int = 5000) -> ArraySplit:
    root = resolve_data_root(data_root) / f"{k}tab"
    train = np.load(root / "train.npz", allow_pickle=True)
    valid = np.load(root / "valid.npz", allow_pickle=True)
    test = np.load(root / "test.npz", allow_pickle=True)
    out = (pad_or_truncate(train["X"], maxlen), train["y"], pad_or_truncate(valid["X"], maxlen), valid["y"], pad_or_truncate(test["X"], maxlen), test["y"])
    train.close()
    valid.close()
    test.close()
    return out


def load_prepared_wfl_cw(data_root: Optional[str] = None, samples: int = 5) -> ArraySplit:
    """Load a prepared Website-Fingerprinting-Library CW few-shot split.

    The cache is produced by ``scripts/prepare_wfl_cw.py``. Arrays are opened
    as memory maps so validation and test data do not need to be duplicated in
    host memory before Keras consumes them.
    """
    root = resolve_data_root(data_root)
    required = {
        "x_train": root / f"train_{samples}_X.npy",
        "y_train": root / f"train_{samples}_y.npy",
        "x_valid": root / "valid_X.npy",
        "y_valid": root / "valid_y.npy",
        "x_test": root / "test_X.npy",
        "y_test": root / "test_y.npy",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing prepared WFL-CW files: " + ", ".join(missing))
    return tuple(np.load(required[name], mmap_mode="r") for name in required)  # type: ignore[return-value]


def load_prepared_wfl_ow(data_root: Optional[str] = None, samples: int = 5) -> ArraySplit:
    """Load a prepared WFL K+1 open-world few-shot split."""
    root = resolve_data_root(data_root)
    required = {
        "x_train": root / f"train_{samples}_X.npy",
        "y_train": root / f"train_{samples}_y.npy",
        "x_valid": root / "valid_X.npy",
        "y_valid": root / "valid_y.npy",
        "x_test": root / "test_X.npy",
        "y_test": root / "test_y.npy",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing prepared WFL-OW files: " + ", ".join(missing))
    return tuple(np.load(required[name], mmap_mode="r") for name in required)  # type: ignore[return-value]
