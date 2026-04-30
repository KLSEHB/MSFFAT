"""Check whether expected dataset files are present."""

from __future__ import annotations

import argparse
from pathlib import Path


CORE_FILES = [
    "df/ClosedWorld/NoDef/X_train_NoDef.pkl",
    "df/ClosedWorld/NoDef/y_train_NoDef.pkl",
    "df/ClosedWorld/NoDef/X_valid_NoDef.pkl",
    "df/ClosedWorld/NoDef/y_valid_NoDef.pkl",
    "df/ClosedWorld/NoDef/X_test_NoDef.pkl",
    "df/ClosedWorld/NoDef/y_test_NoDef.pkl",
    "dlwf/tor_200w_2500tr.npz",
    "dlwf/tor_200w_100tr_time_test3d.npz",
    "dlwf/tor_200w_100tr_time_test10d.npz",
    "dlwf/tor_200w_100tr_time_test2w.npz",
    "dlwf/tor_200w_100tr_time_test4w.npz",
    "dlwf/tor_200w_100tr_time_test6w.npz",
]

FULL_FILES = CORE_FILES + [
    "df/OpenWorld/NoDef/X_train_NoDef.pkl",
    "df/OpenWorld/NoDef/y_train_NoDef.pkl",
    "df/OpenWorld/NoDef/X_valid_NoDef.pkl",
    "df/OpenWorld/NoDef/y_valid_NoDef.pkl",
    "df/OpenWorld/NoDef/X_test_Mon_NoDef.pkl",
    "df/OpenWorld/NoDef/y_test_Mon_NoDef.pkl",
    "df/OpenWorld/NoDef/X_test_Unmon_NoDef.pkl",
    "df/OpenWorld/NoDef/y_test_Unmon_NoDef.pkl",
    "dlwf/tor_100w_2500tr.npz",
    "dlwf/tor_500w_2500tr.npz",
    "dlwf/tor_900w_2500tr.npz",
    "2tab/train.npz",
    "2tab/valid.npz",
    "2tab/test.npz",
    "3tab/train.npz",
    "3tab/valid.npz",
    "3tab/test.npz",
    "4tab/train.npz",
    "4tab/valid.npz",
    "4tab/test.npz",
    "5tab/train.npz",
    "5tab/valid.npz",
    "5tab/test.npz",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--profile", choices=["core", "full"], default="core")
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.data_root)
    files = CORE_FILES if args.profile == "core" else FULL_FILES
    missing = [path for path in files if not (root / path).exists()]
    present = len(files) - len(missing)
    print(f"Dataset root: {root}")
    print(f"Profile: {args.profile}")
    print(f"Present: {present}/{len(files)}")
    if missing:
        print("Missing files:")
        for path in missing:
            print(f"  - {path}")
        raise SystemExit(1)
    print("All expected files are present.")


if __name__ == "__main__":
    main()
