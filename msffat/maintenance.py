"""Drift monitoring and attention-transfer helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import add_channel
from .deployment import detector_step


SYMBOLS = (-1, 0, 1)


def sequence_symbol_distribution(x: np.ndarray, symbols=SYMBOLS) -> np.ndarray:
    """Estimate position-wise distributions over {-1, 0, +1}.

    Returns an array of shape ``(T, len(symbols))``.
    """
    x = np.asarray(x)
    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[:, :, 0]
    if x.ndim != 2 or not x.shape[0] or not x.shape[1]:
        raise ValueError("Expected non-empty (N, T) direction sequences")
    if not np.isin(x, symbols).all():
        raise ValueError("Direction sequences must contain only -1, 0, +1")
    dist = np.zeros((x.shape[1], len(symbols)), dtype="float64")
    for i, symbol in enumerate(symbols):
        dist[:, i] = np.mean(x == symbol, axis=0)
    eps = 1e-12
    dist = np.clip(dist, eps, 1.0)
    dist /= dist.sum(axis=1, keepdims=True)
    return dist


def sequence_jsd(p: np.ndarray, q: np.ndarray) -> float:
    if p.shape != q.shape:
        raise ValueError(f"Distribution shape mismatch: {p.shape} vs {q.shape}")
    if p.ndim != 2 or not p.size or not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError("Expected finite non-empty position-wise distributions")
    if np.any(p < 0) or np.any(q < 0) or not np.allclose(p.sum(axis=1), 1) or not np.allclose(q.sum(axis=1), 1):
        raise ValueError("Each distribution row must sum to one")
    m = 0.5 * (p + q)
    with np.errstate(divide="ignore", invalid="ignore"):
        kl_pm = np.sum(np.where(p > 0, p * np.log2(p / m), 0), axis=1)
        kl_qm = np.sum(np.where(q > 0, q * np.log2(q / m), 0), axis=1)
    return float(np.mean(0.5 * kl_pm + 0.5 * kl_qm))


@dataclass
class DriftDecision:
    adapt: bool
    new_tau: float
    reason: str
    d_seq: float
    probe_accuracy: float | None


def drift_decision(
    train_distribution: np.ndarray,
    probe_x: np.ndarray,
    probe_y: np.ndarray,
    model,
    a_base: float,
    tau: float = 0.04,
    epsilon: float = 0.02,
    delta_tau: float = 0.005,
    tau_base: float = 0.04,
    boost_cap: float = 0.02,
    decay: float = 0.005,
    mode: str = "joint_dynamic",
) -> DriftDecision:
    p = sequence_symbol_distribution(probe_x)
    d_seq = sequence_jsd(p, train_distribution)
    def probe_accuracy():
        probe_pred = model.predict(add_channel(probe_x), verbose=0)
        return float(np.mean(np.argmax(probe_pred, axis=1) == probe_y))

    outcome = detector_step(jsd=d_seq, probe_accuracy=probe_accuracy,
                            reference_accuracy=a_base, q=tau, w_pp=100 * epsilon,
                            e=delta_tau, q_base=tau_base, boost_cap=boost_cap, decay=decay, mode=mode)
    return DriftDecision(outcome.trigger, outcome.q_after, outcome.reason, d_seq, outcome.probe_accuracy)


def attention_transfer_fit(
    model,
    train_x,
    train_y,
    valid_x=None,
    valid_y=None,
    batch_size: int = 8,
    epochs: int = 50,
    callbacks=None,
    learning_rate: float = 1e-3,
    seed: int = 2024,
    scope: str = "attention",
    patience: int = 10,
):
    """Adapt using refresh data alone; no probe/test feedback or extra labels.

    Partial scopes run with training=False so frozen BN moments and Dropout
    cannot change the backbone. Full tuning uses training=True. Best weights,
    LR reduction (factor sqrt(.1), patience 5) and early stopping use refresh
    cross-entropy only. This is fitting loss, not an estimate of generalization.
    """
    from .model import set_adaptation_scope
    if valid_x is not None or valid_y is not None or callbacks:
        raise ValueError("v2 adaptation accepts refresh samples only; external validation/callbacks are disabled")
    return fit_refresh_model(model, add_channel(train_x), train_y, configure_trainable=set_adaptation_scope,
                             batch_size=batch_size, epochs=epochs, learning_rate=learning_rate,
                             seed=seed, scope=scope, patience=patience)


def fit_refresh_model(model, inputs, labels, *, configure_trainable, batch_size=8,
                      epochs=50, learning_rate=1e-3, seed=2024, scope="attention",
                      patience=10, from_logits=False):
    """Common optimizer for native and externally supplied complete Keras models.

    The caller supplies only refresh inputs/labels. Multi-input preprocessing is
    supported; no validation/probe/remainder arrays are accepted by this API.
    """
    import tensorflow as tf

    if epochs < 1 or batch_size < 1 or learning_rate <= 0 or patience < 1:
        raise ValueError("Invalid adaptation optimization settings")
    configure_trainable(model, scope)
    x = tf.nest.map_structure(lambda value: np.asarray(value, dtype="float32"), inputs)
    n = len(tf.nest.flatten(x)[0])
    if any(len(value) != n for value in tf.nest.flatten(x)):
        raise ValueError("Input branches have different sample counts")
    take = lambda idx: tf.nest.map_structure(lambda value: value[idx], x)
    y = np.asarray(labels)
    if n == 0 or n != len(y):
        raise ValueError("Refresh data must be non-empty and aligned")
    if y.ndim == 1:
        y = tf.keras.utils.to_categorical(y, model.output_shape[-1])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=from_logits)
    trainable = model.trainable_variables
    if not trainable:
        raise ValueError("No trainable parameters in selected scope")
    before = frozen_weights_digest(model)
    initial_trainable = [v.numpy().copy() for v in trainable]

    def refresh_loss():
        total = 0.0
        for start in range(0, n, batch_size):
            bx, by = take(slice(start,start+batch_size)), y[start:start + batch_size]
            total += float(loss_fn(by, model(bx, training=False))) * len(by)
        return total / n

    @tf.function(reduce_retracing=True)
    def step(bx, by):
        with tf.GradientTape() as tape:
            loss = loss_fn(by, model(bx, training=(scope == "full")))
        gradients = tape.gradient(loss, trainable)
        pairs = [(g, v) for g, v in zip(gradients, trainable) if g is not None]
        if not pairs:
            raise ValueError("No adaptation gradients")
        tf.debugging.assert_all_finite(loss, "Non-finite adaptation loss")
        for gradient, _ in pairs:
            tf.debugging.assert_all_finite(gradient, "Non-finite adaptation gradient")
        optimizer.apply_gradients(pairs)

    bn_before = [(layer, layer.moving_mean.numpy().copy(), layer.moving_variance.numpy().copy())
                 for layer in iter_layers(model) if isinstance(layer, tf.keras.layers.BatchNormalization)]
    best_loss = refresh_loss()
    if not np.isfinite(best_loss):
        raise FloatingPointError("Non-finite initial refresh loss")
    initial_loss = best_loss
    best_weights = model.get_weights()
    stalled = lr_stalled = 0
    best_epoch = 0
    history = []
    rng = np.random.default_rng(seed)
    for epoch in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            step(take(idx), y[idx])
        current = refresh_loss()
        if not np.isfinite(current):
            raise FloatingPointError("Non-finite refresh loss")
        history.append({"epoch": epoch + 1, "refresh_loss": current,
                        "learning_rate": float(optimizer.learning_rate.numpy())})
        if current < best_loss - 1e-8:
            best_loss, best_weights, best_epoch = current, model.get_weights(), epoch + 1
            stalled = lr_stalled = 0
        else:
            stalled += 1
            lr_stalled += 1
        if lr_stalled >= 5:
            optimizer.learning_rate.assign(max(1e-6, float(optimizer.learning_rate.numpy()) * np.sqrt(0.1)))
            lr_stalled = 0
        if stalled >= patience:
            break
    model.set_weights(best_weights)
    if frozen_weights_digest(model) != before:
        raise AssertionError("Adaptation changed frozen weights")
    if scope != "full":
        for layer, mean, variance in bn_before:
            if not np.array_equal(layer.moving_mean.numpy(), mean) or not np.array_equal(layer.moving_variance.numpy(), variance):
                raise AssertionError("Partial adaptation changed BN running moments")
    delta = float(tf.linalg.global_norm([v - initial for v, initial in zip(trainable, initial_trainable)]))
    parameters = int(sum(np.prod(v.shape) for v in trainable))
    return {"scope": scope, "epochs_ran": len(history), "best_epoch": best_epoch,
            "initial_refresh_loss": initial_loss, "best_refresh_loss": best_loss,
            "trainable_parameters": parameters, "total_parameters":model.count_params(),
            "trainable_fraction":parameters/model.count_params(),
            "trainable_layers": [layer.name for layer in model.layers if layer.trainable],
            "frozen_digest_unchanged": True, "bn_moments_frozen": scope != "full", "trainable_weight_delta": delta,
            "history": history}


def frozen_weights_digest(model) -> str:
    import hashlib
    digest = hashlib.sha256()
    seen = set()
    for layer in iter_layers(model):
        if not layer.trainable:
            digest.update(layer.name.encode())
            for weight in layer.weights:
                if id(weight) not in seen:
                    digest.update(np.ascontiguousarray(weight.numpy()).tobytes())
                    seen.add(id(weight))
    return digest.hexdigest()


def iter_layers(model):
    """Public layer-tree traversal, including nested Models, with no duplicates."""
    seen = set()
    def visit(parent):
        for layer in getattr(parent, "layers", ()):
            if id(layer) not in seen:
                seen.add(id(layer))
                yield layer
                yield from visit(layer)
    yield from visit(model)
