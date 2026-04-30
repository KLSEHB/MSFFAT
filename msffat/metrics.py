"""Metrics used by the multi-tab MSFFAT head."""

from __future__ import annotations

import tensorflow as tf


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
