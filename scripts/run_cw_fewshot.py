"""Run and aggregate the four WFL-CW few-shot experiments."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


SHOTS = (5, 10, 20, 50)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--early-stopping-start-epoch", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).with_name("train_single.py")
    rows = []
    for shots in SHOTS:
        run_dir = output_root / f"{shots}shot"
        run_dir.mkdir(parents=True, exist_ok=True)
        metrics = run_dir / "metrics.json"
        if metrics.exists():
            existing = json.loads(metrics.read_text())
            if existing.get("status") == "complete":
                print(f"Skipping completed {shots}-shot run")
                rows.append(existing)
                continue
        command = [
            sys.executable,
            str(script),
            "--data-root", str(Path(args.prepared_root).resolve()),
            "--dataset", "wfl-cw",
            "--samples", str(shots),
            "--classes", "95",
            "--length", "5000",
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--seed", str(args.seed),
            "--early-stopping-start-epoch", str(args.early_stopping_start_epoch),
            "--output", str(run_dir / "model.keras"),
            "--metrics-output", str(metrics),
            "--history-output", str(run_dir / "history.csv"),
        ]
        if args.require_gpu:
            command.append("--require-gpu")
        with (run_dir / "train.log").open("w") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        rows.append(json.loads(metrics.read_text()))

    fields = [
        "samples_per_class", "seed", "test_loss", "test_accuracy", "best_epoch",
        "epochs_ran", "runtime_seconds", "model",
    ]
    with (output_root / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["samples_per_class"]))
    print((output_root / "summary.csv").read_text())


if __name__ == "__main__":
    main()
