"""Print MSFFAT model summary and parameter count."""

from __future__ import annotations

import argparse

from msffat.model import build_msffat


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes", type=int, required=True)
    parser.add_argument("--length", type=int, default=5000)
    parser.add_argument("--mode", choices=["single", "multi"], default="single")
    return parser.parse_args()


def main():
    args = parse_args()
    model = build_msffat(input_shape=(args.length, 1), num_classes=args.classes, mode=args.mode)
    model.summary()
    print({"params": int(model.count_params())})


if __name__ == "__main__":
    main()
