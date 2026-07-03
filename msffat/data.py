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
    out = np.zeros((len(sequences), maxlen), dtype=dtype)
    for i, seq in enumerate(sequences):
        arr = np.asarray(seq, dtype=dtype)[:maxlen]
        out[i, : len(arr)] = arr
    return out


def add_channel(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype="float32")[:, :, np.newaxis]


def sample_per_label(x: np.ndarray, y: np.ndarray, samples: Optional[int], seed: int = 42):
    if samples is None:
        return x, y
    rng = np.random.default_rng(seed)
    selected = []
    for label in np.unique(y):
        idx = np.flatnonzero(y == label)
        if len(idx) > samples:
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
    x_train, y_train = sample_per_label(x_train, y_train, samples)
    x_valid = _load_pickle(root / f"X_valid_{suffix}.pkl")
    y_valid = _load_pickle(root / f"y_valid_{suffix}.pkl")
    x_test = _load_pickle(root / f"X_test_{suffix}.pkl")
    y_test = _load_pickle(root / f"y_test_{suffix}.pkl")
    return x_train, y_train, x_valid, y_valid, x_test, y_test


def load_df_open_world_training(data_root: Optional[str] = None, samples: Optional[int] = None):
    root = resolve_data_root(data_root) / "df" / "OpenWorld" / "NoDef"
    x_train = _load_pickle(root / "X_train_NoDef.pkl")
    y_train = _load_pickle(root / "y_train_NoDef.pkl")
    x_train, y_train = sample_per_label(x_train, y_train, samples)
    x_valid = _load_pickle(root / "X_valid_NoDef.pkl")
    y_valid = _load_pickle(root / "y_valid_NoDef.pkl")
    return x_train, y_train, x_valid, y_valid


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
    return split_dataset(x, y)


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
    x, y = load_awf_time_eval(data_root=data_root, suffix=suffix, maxlen=maxlen)
    df = pd.DataFrame(x)
    df["label"] = y
    train_x, train_y, val_x, val_y = [], [], [], []
    rng = np.random.default_rng(seed)
    for label in sorted(np.unique(y)):
        group = df[df["label"] == label]
        n = min(len(group), traces * 2)
        sampled = group.iloc[rng.choice(len(group), size=n, replace=False)]
        values = sampled.drop(columns=["label"]).to_numpy(dtype="float32")
        labels = sampled["label"].to_numpy(dtype="int64")
        train_x.append(values[::2])
        train_y.append(labels[::2])
        val_x.append(values[1::2])
        val_y.append(labels[1::2])
    return (
        np.concatenate(train_x),
        np.concatenate(train_y),
        np.concatenate(val_x),
        np.concatenate(val_y),
    )


def load_awf_time_refresh_and_heldout(
    data_root: Optional[str] = None,
    suffix: str = "6w",
    traces: int = 2,
    maxlen: int = 5000,
    seed: int = 42,
):
    """Split an AWF-Time interval into refresh and held-out test traces.

    For each monitored site, up to ``2 * traces`` samples are selected for the
    refresh batch: alternating samples form train and validation sets.  All
    non-selected traces remain in the held-out test set.
    """
    x, y = load_awf_time_eval(data_root=data_root, suffix=suffix, maxlen=maxlen)
    rng = np.random.default_rng(seed)
    train_idx, val_idx, refresh_idx = [], [], []
    all_idx = np.arange(len(y))
    for label in sorted(np.unique(y)):
        idx = np.flatnonzero(y == label)
        n = min(len(idx), traces * 2)
        chosen = rng.choice(idx, size=n, replace=False)
        chosen.sort()
        train_idx.extend(chosen[::2])
        val_idx.extend(chosen[1::2])
        refresh_idx.extend(chosen)
    refresh_idx = np.asarray(refresh_idx, dtype=int)
    heldout_idx = np.setdiff1d(all_idx, refresh_idx, assume_unique=False)
    train_idx = np.asarray(train_idx, dtype=int)
    val_idx = np.asarray(val_idx, dtype=int)
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx], x[heldout_idx], y[heldout_idx]


def load_ares_ktab(data_root: Optional[str] = None, k: int = 2) -> ArraySplit:
    root = resolve_data_root(data_root) / f"{k}tab"
    train = np.load(root / "train.npz", allow_pickle=True)
    valid = np.load(root / "valid.npz", allow_pickle=True)
    test = np.load(root / "test.npz", allow_pickle=True)
    out = (train["X"], train["y"], valid["X"], valid["y"], test["X"], test["y"])
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
