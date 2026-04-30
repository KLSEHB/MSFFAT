"""Print or execute common MSFFAT experiment commands."""

from __future__ import annotations

import argparse
import shlex
import subprocess


def command_to_string(parts):
    return " ".join(shlex.quote(str(part)) for part in parts)


def build_commands(data_root: str):
    commands = []
    for samples in [5, 10, 20, 50]:
        commands.append([
            "python", "scripts/train_single.py",
            "--data-root", data_root,
            "--dataset", "df-cw",
            "--samples", samples,
            "--classes", 95,
            "--output", f"models/msffat_df_cw_{samples}shot.hdf5",
        ])

    for part in [100, 200, 500, 900]:
        commands.append([
            "python", "scripts/train_single.py",
            "--data-root", data_root,
            "--dataset", "awf-cw",
            "--awf-part", part,
            "--awf-traces", 2500,
            "--classes", part,
            "--output", f"models/msffat_awf_cw{part}.hdf5",
        ])

    for suffix in ["3d", "10d", "2w", "4w", "6w"]:
        commands.append([
            "python", "scripts/evaluate_single.py",
            "--data-root", data_root,
            "--model", "models/msffat_awf_cw200.hdf5",
            "--dataset", "awf-time",
            "--suffix", suffix,
            "--classes", 200,
        ])
        commands.append([
            "python", "scripts/finetune_atf.py",
            "--data-root", data_root,
            "--model", "models/msffat_awf_cw200.hdf5",
            "--suffix", suffix,
            "--traces", 2,
            "--classes", 200,
            "--output", f"models/msffat_awf_cw200_{suffix}_atf.hdf5",
        ])

    for tabs in [2, 3, 4, 5]:
        commands.append([
            "python", "scripts/train_multitab.py",
            "--data-root", data_root,
            "--tabs", tabs,
            "--classes", 100,
            "--output", f"models/msffat_{tabs}tab.hdf5",
        ])

    return commands


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--execute", action="store_true", help="Run commands instead of printing them.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands. This is the default.")
    return parser.parse_args()


def main():
    args = parse_args()
    commands = build_commands(args.data_root)
    for command in commands:
        text = command_to_string(command)
        print(text)
        if args.execute:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
