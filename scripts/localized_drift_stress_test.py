#!/usr/bin/env python3
"""Localized-drift stress test for an aggregate probe-accuracy trigger."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from msffat.data import add_channel
from msffat.model import TemporalCropToMatch


SCENARIOS = ((0.05, "5%"), (0.20, "20%"), (0.50, "50%"))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--split-root", required=True)
    parser.add_argument("--cohort-json", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--threshold-pp", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--require-gpu", action="store_true")
    return parser.parse_args()


def load_npz(path: Path):
    with np.load(path, allow_pickle=True) as payload:
        return payload["data"], payload["labels"]


def load_split(path: Path, names: tuple[str, ...]) -> np.ndarray:
    with np.load(path) as payload:
        return np.concatenate([payload[name].astype(np.int64) for name in names])


def correctness_by_site(model, data, labels, indices, site_to_index, batch_size):
    x = data[indices]
    y = np.fromiter((site_to_index[label] for label in labels[indices]), np.int64, len(indices))
    prediction = model.predict(add_channel(x), batch_size=batch_size, verbose=0)
    correct = np.argmax(prediction, axis=1) == y
    per_site = []
    for site_index in range(len(site_to_index)):
        selected = correct[y == site_index]
        if not len(selected):
            raise AssertionError(f"No samples for site index {site_index}")
        per_site.append(selected)
    return per_site


def flatten_accuracy(per_site, site_indices=None):
    if site_indices is None:
        site_indices = range(len(per_site))
    values = [per_site[index] for index in site_indices]
    return float(np.mean(np.concatenate(values)))


def main():
    args = parse_args()
    started = time.perf_counter()
    if args.require_gpu and not tf.config.list_physical_devices("GPU"):
        raise RuntimeError("GPU is required but TensorFlow found none")

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root).resolve()
    split_root = Path(args.split_root).resolve()
    sites = json.loads(Path(args.cohort_json).read_text())["sites"]
    site_to_index = {site: index for index, site in enumerate(sites)}
    num_sites = len(sites)

    rng = np.random.default_rng(args.seed)
    permutation = rng.permutation(num_sites)
    scenario_counts = [int(round(ratio * num_sites)) for ratio, _ in SCENARIOS]
    drifted_sets = [permutation[:count] for count in scenario_counts]
    for previous, current in zip(drifted_sets, drifted_sets[1:]):
        if not set(previous).issubset(set(current)):
            raise AssertionError("Drifted-site sets are not nested")

    model = tf.keras.models.load_model(
        Path(args.model).resolve(),
        custom_objects={"TemporalCropToMatch": TemporalCropToMatch},
    )

    day0_data, day0_labels = load_npz(data_root / "tor_200w_2500tr_new.npz")
    day3_data, day3_labels = load_npz(data_root / "tor_200w_100tr_time_test3d.npz")
    day42_data, day42_labels = load_npz(data_root / "tor_200w_100tr_time_test6w.npz")

    day0_probe_idx = load_split(split_root / "day0_indices.npz", ("probe",))
    day0_remainder_idx = load_split(split_root / "day0_indices.npz", ("unused", "remainder"))
    day3_probe_idx = load_split(split_root / "day3_indices.npz", ("probe",))
    day3_remainder_idx = load_split(split_root / "day3_indices.npz", ("refresh", "remainder"))
    day42_probe_idx = load_split(split_root / "day42_indices.npz", ("probe",))
    day42_remainder_idx = load_split(split_root / "day42_indices.npz", ("refresh", "remainder"))

    expected_probe = num_sites * 5
    expected_remainder = num_sites * 95
    for name, indices, expected in (
        ("day0_probe", day0_probe_idx, expected_probe),
        ("day0_remainder", day0_remainder_idx, expected_remainder),
        ("day3_probe", day3_probe_idx, expected_probe),
        ("day3_remainder", day3_remainder_idx, expected_remainder),
        ("day42_probe", day42_probe_idx, expected_probe),
        ("day42_remainder", day42_remainder_idx, expected_remainder),
    ):
        if len(indices) != expected or len(np.unique(indices)) != expected:
            raise AssertionError(f"Unexpected or duplicate indices in {name}: {len(indices)}")

    day0_probe = correctness_by_site(
        model, day0_data, day0_labels, day0_probe_idx, site_to_index, args.batch_size
    )
    day0_remainder = correctness_by_site(
        model, day0_data, day0_labels, day0_remainder_idx, site_to_index, args.batch_size
    )
    del day0_data, day0_labels
    day3_probe = correctness_by_site(
        model, day3_data, day3_labels, day3_probe_idx, site_to_index, args.batch_size
    )
    day3_remainder = correctness_by_site(
        model, day3_data, day3_labels, day3_remainder_idx, site_to_index, args.batch_size
    )
    del day3_data, day3_labels
    day42_probe = correctness_by_site(
        model, day42_data, day42_labels, day42_probe_idx, site_to_index, args.batch_size
    )
    day42_remainder = correctness_by_site(
        model, day42_data, day42_labels, day42_remainder_idx, site_to_index, args.batch_size
    )
    del day42_data, day42_labels

    day0_probe_accuracy = flatten_accuracy(day0_probe)
    day0_remainder_accuracy = flatten_accuracy(day0_remainder)
    all_indices = np.arange(num_sites)
    rows = []
    scenario_details = []
    for (ratio, ratio_label), count, drifted in zip(SCENARIOS, scenario_counts, drifted_sets):
        drifted_set = set(drifted.tolist())
        stable = np.asarray([index for index in all_indices if index not in drifted_set], dtype=np.int64)
        mixed_probe = [day42_probe[index] if index in drifted_set else day3_probe[index]
                       for index in all_indices]
        mixed_remainder = [day42_remainder[index] if index in drifted_set else day3_remainder[index]
                           for index in all_indices]

        mixed_probe_accuracy = flatten_accuracy(mixed_probe)
        mixed_remainder_accuracy = flatten_accuracy(mixed_remainder)
        day0_drifted_accuracy = flatten_accuracy(day0_remainder, drifted)
        day42_drifted_accuracy = flatten_accuracy(day42_remainder, drifted)
        day0_stable_accuracy = flatten_accuracy(day0_remainder, stable)
        day3_stable_accuracy = flatten_accuracy(day3_remainder, stable)
        overall_drop = 100.0 * (day0_remainder_accuracy - mixed_remainder_accuracy)
        drifted_drop = 100.0 * (day0_drifted_accuracy - day42_drifted_accuracy)
        stable_drop = 100.0 * (day0_stable_accuracy - day3_stable_accuracy)
        probe_drop = 100.0 * (day0_probe_accuracy - mixed_probe_accuracy)

        row = {
            "drifted_sites": f"{ratio_label} ({count}/{num_sites})",
            "requested_ratio": ratio,
            "actual_ratio": count / num_sites,
            "overall_accuracy_drop_pp": overall_drop,
            "drifted_site_accuracy_drop_pp": drifted_drop,
            "stable_site_accuracy_drop_pp": stable_drop,
            "aggregate_need_update": overall_drop > args.threshold_pp,
            "local_need_update": drifted_drop > args.threshold_pp,
            "detector_atf": probe_drop > args.threshold_pp,
            "probe_accuracy_drop_pp": probe_drop,
            "mixed_probe_accuracy": mixed_probe_accuracy,
            "mixed_remainder_accuracy": mixed_remainder_accuracy,
        }
        rows.append(row)
        scenario_details.append({
            "drifted_sites": [sites[index] for index in drifted],
            "stable_site_count": int(len(stable)),
            "day0_drifted_remainder_accuracy": day0_drifted_accuracy,
            "day42_drifted_remainder_accuracy": day42_drifted_accuracy,
            "day0_stable_remainder_accuracy": day0_stable_accuracy,
            "day3_stable_remainder_accuracy": day3_stable_accuracy,
        })

    payload = {
        "model": str(Path(args.model).resolve()),
        "seed": args.seed,
        "threshold_pp": args.threshold_pp,
        "num_sites": num_sites,
        "probe_per_site": 5,
        "remainder_per_site": 95,
        "day0_probe_accuracy": day0_probe_accuracy,
        "day0_remainder_accuracy": day0_remainder_accuracy,
        "runtime_seconds": time.perf_counter() - started,
        "rows": rows,
        "scenario_details": scenario_details,
    }
    (output_root / "localized_drift_results.json").write_text(json.dumps(payload, indent=2) + "\n")

    csv_fields = [
        "drifted_sites", "overall_accuracy_drop_pp", "drifted_site_accuracy_drop_pp",
        "stable_site_accuracy_drop_pp", "aggregate_need_update", "local_need_update",
        "detector_atf", "probe_accuracy_drop_pp", "mixed_probe_accuracy",
        "mixed_remainder_accuracy",
    ]
    with (output_root / "localized_drift_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Localized-Drift Stress Test", "",
        f"Day-0 reference probe/remainder accuracy: {100*day0_probe_accuracy:.2f}% / "
        f"{100*day0_remainder_accuracy:.2f}%. Threshold: drop > {args.threshold_pp:.1f} pp.", "",
        "| Drifted sites | Overall Acc. drop | Drifted-site Acc. drop | Stable-site Acc. drop | Aggregate Need update | Local Need update | Detector ATF |",
        "|:---|---:|---:|---:|:---:|:---:|:---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['drifted_sites']} | {row['overall_accuracy_drop_pp']:.2f} pp | "
            f"{row['drifted_site_accuracy_drop_pp']:.2f} pp | "
            f"{row['stable_site_accuracy_drop_pp']:.2f} pp | "
            f"{'Yes' if row['aggregate_need_update'] else 'No'} | "
            f"{'Yes' if row['local_need_update'] else 'No'} | "
            f"{'Yes' if row['detector_atf'] else 'No'} |"
        )
    lines += ["", f"Total inference runtime: {payload['runtime_seconds']:.2f} s."]
    (output_root / "localized_drift_results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
