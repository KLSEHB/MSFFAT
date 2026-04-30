"""Compute sequence-level JS divergence for AWF-Time intervals."""

from __future__ import annotations

import argparse

import numpy as np

from msffat.data import load_awf_cw, load_awf_time_eval
from msffat.maintenance import sequence_jsd, sequence_symbol_distribution


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--suffix", required=True, choices=["3d", "10d", "2w", "4w", "6w"])
    parser.add_argument("--length", type=int, default=5000)
    parser.add_argument("--awf-traces", type=int, default=2500)
    return parser.parse_args()


def main():
    args = parse_args()
    x_train, _, x_valid, _, x_test, _ = load_awf_cw(args.data_root, part="200", traces=args.awf_traces, maxlen=args.length)
    x_train = np.concatenate([x_train, x_valid, x_test], axis=0)
    x_probe, _ = load_awf_time_eval(args.data_root, suffix=args.suffix, maxlen=args.length)
    q = sequence_symbol_distribution(x_train)
    p = sequence_symbol_distribution(x_probe)
    print({"suffix": args.suffix, "d_seq": sequence_jsd(p, q)})


if __name__ == "__main__":
    main()
