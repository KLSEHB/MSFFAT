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


def precision_at_k(k: int):
    def metric(y_true, y_pred):
        top_k = tf.nn.top_k(y_pred, k=k).indices
        batch = tf.range(tf.shape(y_true)[0])
        batch = tf.tile(tf.expand_dims(batch, 1), [1, k])
        gather_idx = tf.stack([batch, top_k], axis=-1)
        hits = tf.gather_nd(y_true, gather_idx)
        return tf.reduce_mean(tf.reduce_mean(hits, axis=1))

    metric.__name__ = f"p@{k}"
    return metric


def average_precision_at_k(k: int):
    def metric(y_true, y_pred):
        top_k = tf.nn.top_k(y_pred, k=k, sorted=True).indices
        batch = tf.range(tf.shape(y_true)[0])
        batch = tf.tile(tf.expand_dims(batch, 1), [1, k])
        gather_idx = tf.stack([batch, top_k], axis=-1)
        rel = tf.gather_nd(y_true, gather_idx)
        cumsum = tf.cumsum(rel, axis=1)
        ranks = tf.cast(tf.range(1, k + 1), tf.float32)
        precision = cumsum / tf.expand_dims(ranks, 0)
        denom = tf.minimum(tf.reduce_sum(y_true, axis=1), tf.cast(k, tf.float32))
        ap = tf.reduce_sum(precision * rel, axis=1)
        ap = tf.where(denom > 0, ap / denom, 0.0)
        return tf.reduce_mean(ap)

    metric.__name__ = f"ap@{k}"
    return metric
