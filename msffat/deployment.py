"""Pure helpers for stateful AWF-Time deployment simulations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class DetectorOutcome:
    trigger: bool
    q_after: float
    probe_drop_pp: float
    reason: str


@dataclass(frozen=True)
class ProbeDetectorOutcome:
    """Decision made solely from aggregate probe-accuracy degradation."""

    trigger: bool
    probe_drop_pp: float
    reason: str


def probe_only_detector_step(
    *,
    probe_accuracy: float,
    reference_accuracy: float,
    threshold_pp: float = 2.0,
) -> ProbeDetectorOutcome:
    """Trigger when aggregate probe accuracy drops strictly beyond a threshold."""
    if threshold_pp < 0:
        raise ValueError("threshold_pp must be non-negative")
    drop_pp = (reference_accuracy - probe_accuracy) * 100.0
    # Guard the strict boundary against binary floating-point representation
    # (e.g., 1.0 - 0.98 can be microscopically greater than 0.02).
    trigger = drop_pp > threshold_pp + 1e-12
    return ProbeDetectorOutcome(
        trigger=trigger,
        probe_drop_pp=drop_pp,
        reason="probe_accuracy_drop" if trigger else "probe_accuracy_stable",
    )


def detector_step(
    *,
    jsd: float,
    probe_accuracy: float,
    reference_accuracy: float,
    q: float,
    w_pp: float,
    e: float,
) -> DetectorOutcome:
    """Apply the experiment's JSD-gated detector policy."""
    drop_pp = (reference_accuracy - probe_accuracy) * 100.0
    if jsd <= q:
        return DetectorOutcome(False, q, drop_pp, "jsd_not_exceeded")
    if drop_pp > w_pp:
        return DetectorOutcome(True, max(0.0, q - e), drop_pp, "jsd_and_accuracy_drop")
    return DetectorOutcome(False, min(1.0, q + e), drop_pp, "jsd_only_false_alarm")


def stratified_indices(
    y: np.ndarray,
    labels: Sequence[int],
    sizes: Mapping[str, int],
    *,
    seed: int,
) -> dict[str, np.ndarray]:
    """Create deterministic, disjoint per-class index partitions."""
    output: dict[str, list[np.ndarray]] = {name: [] for name in sizes}
    required = sum(sizes.values())
    for label in labels:
        idx = np.flatnonzero(y == label)
        if len(idx) < required:
            raise ValueError(f"label {label} has {len(idx)} samples; need {required}")
        rng = np.random.default_rng(seed + 1009 * int(label))
        chosen = rng.permutation(idx)[:required]
        offset = 0
        for name, size in sizes.items():
            output[name].append(chosen[offset : offset + size])
            offset += size
    return {name: np.concatenate(parts).astype("int64") for name, parts in output.items()}


def find_matching_k(
    remainder_drops_pp: Iterable[float],
    detector_triggers: Iterable[bool],
    *,
    step_pp: float = 0.1,
    require_nondegenerate: bool = True,
) -> tuple[float | None, list[bool] | None]:
    """Return the smallest non-negative K whose strict oracle labels match."""
    drops = np.asarray(list(remainder_drops_pp), dtype="float64")
    triggers = np.asarray(list(detector_triggers), dtype=bool)
    if require_nondegenerate and (triggers.all() or (~triggers).all()):
        return None, None
    candidates = np.round(np.arange(0.0, 100.0 + step_pp / 2.0, step_pp), 10)
    for k in candidates:
        oracle = drops > k
        if np.array_equal(oracle, triggers):
            return float(k), oracle.tolist()
    return None, None


def best_k_accuracy(
    remainder_drops_pp: Iterable[float],
    detector_triggers: Iterable[bool],
    *,
    step_pp: float = 0.1,
) -> tuple[float, float, list[bool]]:
    """Fallback: find K with the highest detector/oracle agreement."""
    drops = np.asarray(list(remainder_drops_pp), dtype="float64")
    triggers = np.asarray(list(detector_triggers), dtype=bool)
    best = (-1.0, 0.0, [])
    for k in np.round(np.arange(0.0, 100.0 + step_pp / 2.0, step_pp), 10):
        oracle = drops > k
        accuracy = float(np.mean(oracle == triggers))
        if accuracy > best[0]:
            best = (accuracy, float(k), oracle.tolist())
    return best[1], best[0], best[2]
