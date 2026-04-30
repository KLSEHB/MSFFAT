"""Drift monitoring and attention-transfer helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import add_channel
from .model import set_attention_only_trainable


SYMBOLS = (-1, 0, 1)


def sequence_symbol_distribution(x: np.ndarray, symbols=SYMBOLS) -> np.ndarray:
    """Estimate position-wise distributions over {-1, 0, +1}.

    Returns an array of shape ``(T, len(symbols))``.
    """
    if x.ndim == 3:
        x = x[:, :, 0]
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
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log2(p / m), axis=1)
    kl_qm = np.sum(q * np.log2(q / m), axis=1)
    return float(np.mean(0.5 * kl_pm + 0.5 * kl_qm))


@dataclass
class DriftDecision:
    adapt: bool
    new_tau: float
    reason: str
    d_seq: float
    probe_accuracy: float


def drift_decision(
    train_distribution: np.ndarray,
    probe_x: np.ndarray,
    probe_y: np.ndarray,
    model,
    a_base: float,
    tau: float = 0.05,
    epsilon: float = 0.05,
    delta_tau: float = 0.005,
    tau_min: float = 0.0,
) -> DriftDecision:
    p = sequence_symbol_distribution(probe_x)
    d_seq = sequence_jsd(p, train_distribution)
    probe_pred = model.predict(add_channel(probe_x), verbose=0)
    probe_accuracy = float(np.mean(np.argmax(probe_pred, axis=1) == probe_y))

    if d_seq > tau:
        if probe_accuracy >= a_base - epsilon:
            return DriftDecision(True, tau + delta_tau, "jsd_only_shift", d_seq, probe_accuracy)
        return DriftDecision(True, tau, "jsd_and_accuracy_shift", d_seq, probe_accuracy)

    if probe_accuracy < a_base - epsilon:
        return DriftDecision(True, max(tau_min, tau - delta_tau), "accuracy_shift_without_jsd", d_seq, probe_accuracy)

    return DriftDecision(False, tau, "no_maintenance", d_seq, probe_accuracy)


def attention_transfer_fit(
    model,
    train_x,
    train_y,
    valid_x,
    valid_y,
    batch_size: int = 8,
    epochs: int = 50,
    callbacks=None,
):
    set_attention_only_trainable(model)
    return model.fit(
        add_channel(train_x),
        train_y,
        validation_data=(add_channel(valid_x), valid_y),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks or [],
        verbose=1,
    )
