"""Compute realistic base-rate monitored-vs-unmonitored OW metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score


METHOD_ORDER = ("DF", "TikTok", "VarCNN", "NetCLR", "MSFFAT")
DISPLAY = {"DF": "DF", "TikTok": "Tik-Tok", "VarCNN": "Var-CNN", "NetCLR": "NetCLR", "MSFFAT": "MSFFAT"}
BASE_RATES = (0.10, 0.01)
UNMONITORED_LABEL = 95


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def base_rate_indices(y, rate, seed):
    rng = np.random.default_rng(seed)
    monitored = np.flatnonzero(y < UNMONITORED_LABEL)
    unmonitored = np.flatnonzero(y == UNMONITORED_LABEL)
    monitored_count = int(round(rate * len(unmonitored) / (1.0 - rate)))
    if monitored_count > len(monitored):
        raise ValueError(f"Need {monitored_count} monitored samples, only {len(monitored)} available")
    chosen_monitored = rng.choice(monitored, size=monitored_count, replace=False)
    selected = np.concatenate([chosen_monitored, unmonitored])
    rng.shuffle(selected)
    return selected


def compute_metrics(y_true, probs, indices):
    y = y_true[indices]
    p = probs[indices]
    pred = np.argmax(p, axis=1)
    true_positive = y < UNMONITORED_LABEL
    pred_positive = pred < UNMONITORED_LABEL
    score = 1.0 - p[:, UNMONITORED_LABEL]
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_positive.astype(int), pred_positive.astype(int), average="binary", zero_division=0
    )
    negatives = ~true_positive
    fp = np.sum(pred_positive & negatives)
    tn = np.sum((~pred_positive) & negatives)
    fpr = float(fp / (fp + tn)) if (fp + tn) else 0.0
    e2e_site_acc = float(np.mean(pred[true_positive] == y[true_positive])) if np.any(true_positive) else 0.0
    return {
        "Precision": float(precision),
        "Recall/TPR": float(recall),
        "FPR": fpr,
        "F1": float(f1),
        "ROC-AUC": float(roc_auc_score(true_positive.astype(int), score)),
        "PR-AUC": float(average_precision_score(true_positive.astype(int), score)),
        "E2E Site Acc.": e2e_site_acc,
        "monitored_samples": int(np.sum(true_positive)),
        "unmonitored_samples": int(np.sum(~true_positive)),
        "actual_base_rate": float(np.mean(true_positive)),
    }


def fmt(value):
    return f"{100.0 * value:.2f}%"


def markdown_table(title, rows):
    lines = [
        f"**{title}**",
        "",
        "| Method  | Precision | Recall/TPR |  FPR |   F1 | ROC-AUC | PR-AUC | E2E Site Acc. |",
        "| ------- | --------: | ---------: | ---: | ---: | ------: | -----: | ------------: |",
    ]
    for method in METHOD_ORDER:
        row = rows[method]
        lines.append(
            f"| {DISPLAY[method]:<7} | {fmt(row['Precision']):>9} | {fmt(row['Recall/TPR']):>10} | "
            f"{fmt(row['FPR']):>5} | {fmt(row['F1']):>5} | {fmt(row['ROC-AUC']):>7} | "
            f"{fmt(row['PR-AUC']):>6} | {fmt(row['E2E Site Acc.']):>13} |"
        )
    return "\n".join(lines)


def main():
    args = parse_args()
    pred_root = Path(args.pred_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_y = None
    predictions = {}
    for method in METHOD_ORDER:
        run = pred_root / method
        probs = np.load(run / "test_probs.npy")
        y = np.load(run / "test_y.npy")
        if probs.shape[1] != UNMONITORED_LABEL + 1:
            raise AssertionError(f"{method} expected 96 classes, got {probs.shape}")
        if reference_y is None:
            reference_y = y
        elif not np.array_equal(reference_y, y):
            raise AssertionError(f"{method} test labels differ from reference")
        predictions[method] = probs

    all_results = {}
    index_metadata = {}
    csv_rows = []
    md_parts = []
    for rate in BASE_RATES:
        indices = base_rate_indices(reference_y, rate, args.seed + int(rate * 1000))
        np.save(output_dir / f"indices_{int(rate * 100)}pct.npy", indices.astype("int64"))
        rate_key = f"{int(rate * 100)}%"
        all_results[rate_key] = {}
        index_metadata[rate_key] = {
            "target_base_rate": rate,
            "actual_base_rate": float(np.mean(reference_y[indices] < UNMONITORED_LABEL)),
            "monitored_samples": int(np.sum(reference_y[indices] < UNMONITORED_LABEL)),
            "unmonitored_samples": int(np.sum(reference_y[indices] == UNMONITORED_LABEL)),
            "total_samples": int(len(indices)),
        }
        for method, probs in predictions.items():
            metrics = compute_metrics(reference_y, probs, indices)
            all_results[rate_key][method] = metrics
            csv_rows.append({"base_rate": rate_key, "method": DISPLAY[method], **metrics})
        title = f"Open-world evaluation under {rate_key} monitored base rate."
        md_parts.append(markdown_table(title, all_results[rate_key]))

    with (output_dir / "base_rate_metrics.csv").open("w", newline="") as handle:
        fields = [
            "base_rate", "method", "Precision", "Recall/TPR", "FPR", "F1", "ROC-AUC", "PR-AUC",
            "E2E Site Acc.", "monitored_samples", "unmonitored_samples", "actual_base_rate",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(csv_rows)

    payload = {"seed": args.seed, "indices": index_metadata, "results": all_results}
    (output_dir / "base_rate_metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    markdown = "\n\n".join(md_parts) + "\n"
    (output_dir / "base_rate_tables.md").write_text(markdown)
    print(markdown)


if __name__ == "__main__":
    main()
