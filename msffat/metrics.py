"""Metrics used by the multi-tab MSFFAT head."""

from __future__ import annotations

import numpy as np
import tensorflow as tf


def kplus1_classification_metrics(y_true, y_pred, unmonitored_label: int):
    """Compute multiclass and open-world detection metrics without sklearn."""
    y_true = np.asarray(y_true, dtype="int64")
    y_pred = np.asarray(y_pred, dtype="int64")
    confusion = np.zeros((unmonitored_label + 1, unmonitored_label + 1), dtype="int64")
    np.add.at(confusion, (y_true, y_pred), 1)
    tp = np.diag(confusion).astype("float64")
    support = confusion.sum(axis=1).astype("float64")
    predicted = confusion.sum(axis=0).astype("float64")
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted != 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) != 0)
    weights = support / support.sum()

    true_monitored = y_true != unmonitored_label
    pred_monitored = y_pred != unmonitored_label
    detection_tp = int(np.sum(true_monitored & pred_monitored))
    detection_fn = int(np.sum(true_monitored & ~pred_monitored))
    detection_fp = int(np.sum(~true_monitored & pred_monitored))
    detection_tn = int(np.sum(~true_monitored & ~pred_monitored))
    detection_precision = detection_tp / max(detection_tp + detection_fp, 1)
    detection_recall = detection_tp / max(detection_tp + detection_fn, 1)
    detection_f1 = 2 * detection_precision * detection_recall / max(detection_precision + detection_recall, 1e-12)
    metrics = {
        "accuracy": float(np.mean(y_true == y_pred)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "weighted_precision": float(np.sum(weights * precision)),
        "weighted_recall": float(np.sum(weights * recall)),
        "weighted_f1": float(np.sum(weights * f1)),
        "monitored_site_accuracy": float(np.mean(y_pred[true_monitored] == y_true[true_monitored])),
        "unmonitored_accuracy": float(np.mean(y_pred[~true_monitored] == unmonitored_label)),
        "detection_precision": float(detection_precision),
        "detection_tpr": float(detection_recall),
        "detection_fpr": float(detection_fp / max(detection_fp + detection_tn, 1)),
        "detection_f1": float(detection_f1),
    }
    return metrics, confusion


from .ranking_metrics import PrecisionAtK, AveragePrecisionAtK


def precision_at_k(k: int):
    return PrecisionAtK(k)


def average_precision_at_k(k: int):
    return AveragePrecisionAtK(k)
