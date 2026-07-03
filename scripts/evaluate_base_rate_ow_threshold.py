"""Compute base-rate OW metrics using validation-selected Max-F1 thresholds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score


METHOD_ORDER = (
    "AWF", "AWF-LSTM", "AWF-SDAE", "DF", "TikTok", "VarCNN", "NetCLR",
    "RF", "TF", "ARES", "TMWF", "MSFFAT",
)
DISPLAY = {
    "AWF": "AWF", "AWF-LSTM": "AWF-LSTM", "AWF-SDAE": "AWF-SDAE",
    "DF": "DF", "TikTok": "Tik-Tok", "VarCNN": "Var-CNN", "NetCLR": "NetCLR",
    "RF": "RF", "TF": "TF", "ARES": "ARES", "TMWF": "TMWF", "MSFFAT": "MSFFAT",
}
BASE_RATES = (0.10, 0.01)
UNMONITORED_LABEL = 95


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--methods", nargs="+", choices=METHOD_ORDER, default=list(METHOD_ORDER))
    parser.add_argument("--markdown-name", default="base_rate_tables.md")
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


def positive_score(probs):
    return 1.0 - probs[:, UNMONITORED_LABEL]


def select_max_f1_threshold(y_true, scores):
    y = (y_true < UNMONITORED_LABEL).astype(bool)
    candidates = np.unique(scores)
    candidates = np.concatenate(([np.nextafter(0.0, -1.0)], candidates, [np.nextafter(1.0, 2.0)]))
    best = {"threshold": 0.5, "f1": -1.0, "precision": 0.0, "recall": 0.0}
    for threshold in candidates:
        pred = scores >= threshold
        precision, recall, f1, _ = precision_recall_fscore_support(
            y.astype(int), pred.astype(int), average="binary", zero_division=0
        )
        if f1 > best["f1"] or (np.isclose(f1, best["f1"]) and threshold > best["threshold"]):
            best = {
                "threshold": float(threshold),
                "f1": float(f1),
                "precision": float(precision),
                "recall": float(recall),
            }
    return best


def compute_metrics(y_true, probs, indices, threshold):
    y = y_true[indices]
    p = probs[indices]
    site_pred = np.argmax(p[:, :UNMONITORED_LABEL], axis=1)
    true_positive = y < UNMONITORED_LABEL
    score = positive_score(p)
    pred_positive = score >= threshold
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_positive.astype(int), pred_positive.astype(int), average="binary", zero_division=0
    )
    negatives = ~true_positive
    fp = np.sum(pred_positive & negatives)
    tn = np.sum((~pred_positive) & negatives)
    fpr = float(fp / (fp + tn)) if (fp + tn) else 0.0
    e2e_site_acc = float(np.mean(pred_positive[true_positive] & (site_pred[true_positive] == y[true_positive]))) if np.any(true_positive) else 0.0
    return {
        "Best Threshold": float(threshold),
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


def fmt_pct(value):
    return f"{100.0 * value:.2f}%"


def fmt_thr(value):
    return f"{value:.4f}"


def markdown_table(title, rows, methods):
    lines = [
        f"**{title}**",
        "",
        "| Method   | Best Threshold | Precision | Recall/TPR |  FPR |   F1 | ROC-AUC | PR-AUC | E2E Site Acc. |",
        "| -------- | -------------: | --------: | ---------: | ---: | ---: | ------: | -----: | ------------: |",
    ]
    for method in methods:
        row = rows[method]
        lines.append(
            f"| {DISPLAY[method]:<8} | {fmt_thr(row['Best Threshold']):>14} | {fmt_pct(row['Precision']):>9} | "
            f"{fmt_pct(row['Recall/TPR']):>10} | {fmt_pct(row['FPR']):>5} | {fmt_pct(row['F1']):>5} | "
            f"{fmt_pct(row['ROC-AUC']):>7} | {fmt_pct(row['PR-AUC']):>6} | {fmt_pct(row['E2E Site Acc.']):>13} |"
        )
    return "\n".join(lines)


def main():
    args = parse_args()
    pred_root = Path(args.pred_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ref_valid_y = None
    ref_test_y = None
    predictions = {}
    for method in args.methods:
        run = pred_root / method
        valid_probs = np.load(run / "valid_probs.npy")
        valid_y = np.load(run / "valid_y.npy")
        test_probs = np.load(run / "test_probs.npy")
        test_y = np.load(run / "test_y.npy")
        if valid_probs.shape[1] != UNMONITORED_LABEL + 1 or test_probs.shape[1] != UNMONITORED_LABEL + 1:
            raise AssertionError(f"{method} expected 96 classes, got valid={valid_probs.shape}, test={test_probs.shape}")
        if ref_valid_y is None:
            ref_valid_y = valid_y
            ref_test_y = test_y
        elif not (np.array_equal(ref_valid_y, valid_y) and np.array_equal(ref_test_y, test_y)):
            raise AssertionError(f"{method} labels differ from reference")
        predictions[method] = {"valid_probs": valid_probs, "test_probs": test_probs}

    all_results = {}
    threshold_details = {}
    index_metadata = {}
    csv_rows = []
    md_parts = []
    for rate in BASE_RATES:
        rate_key = f"{int(rate * 100)}%"
        valid_indices = base_rate_indices(ref_valid_y, rate, args.seed + int(rate * 1000) + 1)
        test_indices = base_rate_indices(ref_test_y, rate, args.seed + int(rate * 1000) + 2)
        np.save(output_dir / f"valid_indices_{int(rate * 100)}pct.npy", valid_indices.astype("int64"))
        np.save(output_dir / f"test_indices_{int(rate * 100)}pct.npy", test_indices.astype("int64"))
        index_metadata[rate_key] = {
            "target_base_rate": rate,
            "valid_actual_base_rate": float(np.mean(ref_valid_y[valid_indices] < UNMONITORED_LABEL)),
            "valid_monitored_samples": int(np.sum(ref_valid_y[valid_indices] < UNMONITORED_LABEL)),
            "valid_unmonitored_samples": int(np.sum(ref_valid_y[valid_indices] == UNMONITORED_LABEL)),
            "test_actual_base_rate": float(np.mean(ref_test_y[test_indices] < UNMONITORED_LABEL)),
            "test_monitored_samples": int(np.sum(ref_test_y[test_indices] < UNMONITORED_LABEL)),
            "test_unmonitored_samples": int(np.sum(ref_test_y[test_indices] == UNMONITORED_LABEL)),
        }
        all_results[rate_key] = {}
        threshold_details[rate_key] = {}
        for method, payload in predictions.items():
            valid_y = ref_valid_y[valid_indices]
            valid_scores = positive_score(payload["valid_probs"][valid_indices])
            threshold_info = select_max_f1_threshold(valid_y, valid_scores)
            metrics = compute_metrics(ref_test_y, payload["test_probs"], test_indices, threshold_info["threshold"])
            all_results[rate_key][method] = metrics
            threshold_details[rate_key][method] = threshold_info
            csv_rows.append({"base_rate": rate_key, "method": DISPLAY[method], **metrics})
        title = f"Open-world evaluation under {rate_key} monitored base rate."
        md_parts.append(markdown_table(title, all_results[rate_key], args.methods))

    with (output_dir / "base_rate_metrics.csv").open("w", newline="") as handle:
        fields = [
            "base_rate", "method", "Best Threshold", "Precision", "Recall/TPR", "FPR", "F1",
            "ROC-AUC", "PR-AUC", "E2E Site Acc.", "monitored_samples", "unmonitored_samples",
            "actual_base_rate",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(csv_rows)

    payload = {
        "seed": args.seed,
        "threshold_policy": "Max-F1 threshold on validation set; ties choose higher threshold",
        "indices": index_metadata,
        "validation_thresholds": threshold_details,
        "results": all_results,
    }
    (output_dir / "base_rate_metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    markdown = "\n\n".join(md_parts) + "\n"
    if {"AWF-LSTM", "AWF-SDAE"}.intersection(args.methods):
        markdown += (
            "\n*Implementation note: AWF-LSTM and AWF-SDAE are lightweight PyTorch "
            "reproductions based on the architectures released by the AWF authors.*\n"
        )
    (output_dir / args.markdown_name).write_text(markdown)
    print(markdown)


if __name__ == "__main__":
    main()
