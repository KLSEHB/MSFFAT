"""Aggregate and verify K+1 open-world results with shot-limited unmonitored training."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import tensorflow as tf


ROOT = Path("runs/open_world_kplus1_unmon_shot/results")
SHOTS = (5, 10)
FIELDS = [
    "samples_per_monitored_class", "samples_in_unmonitored_class", "train_samples",
    "accuracy", "macro_precision", "macro_recall", "macro_f1",
    "weighted_precision", "weighted_recall", "weighted_f1", "monitored_site_accuracy",
    "unmonitored_accuracy", "detection_precision", "detection_tpr", "detection_fpr",
    "detection_f1", "best_epoch", "epochs_ran", "runtime_seconds", "gpu", "model",
]


def main():
    rows = []
    for shots in SHOTS:
        run = ROOT / f"{shots}shot"
        metrics = json.loads((run / "metrics.json").read_text())
        assert metrics["status"] == "complete" and metrics["classes"] == 96
        assert metrics["samples_per_monitored_class"] == shots
        assert metrics["samples_in_unmonitored_class"] == shots
        assert metrics["train_samples"] == 96 * shots
        assert all(math.isfinite(float(metrics[key])) for key in FIELDS[3:16])
        model = tf.keras.models.load_model(run / "model.keras", compile=False)
        assert model.count_params() == 17_814_208
        assert model.input_shape == (None, 5000, 1) and model.output_shape == (None, 96)
        del model
        tf.keras.backend.clear_session()
        rows.append(metrics)
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    verification = {
        "status": "verified",
        "parameters": 17_814_208,
        "shots": list(SHOTS),
        "unmonitored_training_samples": "same_as_monitored_shot",
    }
    (ROOT / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    print((ROOT / "summary.csv").read_text())


if __name__ == "__main__":
    main()
