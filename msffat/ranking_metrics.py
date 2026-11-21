"""Serializable, sample-weighted multilabel ranking metrics."""
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="msffat")
class PrecisionAtK(tf.keras.metrics.Metric):
    def __init__(self, k, name=None, **kwargs):
        super().__init__(name=name or f"p@{k}", **kwargs)
        if int(k) < 1:
            raise ValueError("k must be positive")
        self.k = int(k)
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def per_trace(self, y_true, y_pred):
        relevance = tf.gather(tf.cast(y_true, self.dtype), tf.math.top_k(y_pred, k=self.k).indices, batch_dims=1)
        return tf.reduce_mean(relevance, axis=1)

    def update_state(self, y_true, y_pred, sample_weight=None):
        values = self.per_trace(y_true, y_pred)
        weights = tf.ones_like(values) if sample_weight is None else tf.broadcast_to(tf.reshape(tf.cast(sample_weight, self.dtype), [-1]), tf.shape(values))
        self.total.assign_add(tf.reduce_sum(values * weights))
        self.count.assign_add(tf.reduce_sum(weights))

    def result(self):
        return tf.math.divide_no_nan(self.total, self.count)

    def reset_state(self):
        self.total.assign(0)
        self.count.assign(0)

    def get_config(self):
        return {**super().get_config(), "k": self.k}


@tf.keras.utils.register_keras_serializable(package="msffat")
class AveragePrecisionAtK(PrecisionAtK):
    def __init__(self, k, name=None, **kwargs):
        super().__init__(k, name=name or f"ap@{k}", **kwargs)

    def per_trace(self, y_true, y_pred):
        y_true = tf.cast(y_true, self.dtype)
        indices = tf.math.top_k(y_pred, k=self.k, sorted=True).indices
        relevance = tf.gather(y_true, indices, batch_dims=1)
        precision = tf.cumsum(relevance, axis=1) / tf.cast(tf.range(1, self.k + 1), self.dtype)
        denominator = tf.minimum(tf.reduce_sum(y_true, axis=1), tf.cast(self.k, self.dtype))
        return tf.math.divide_no_nan(tf.reduce_sum(precision * relevance, axis=1), denominator)
