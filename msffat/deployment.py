"""Pure helpers for stateful AWF-Time deployment simulations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class DetectorOutcome:
    trigger: bool
    q_after: float
    probe_drop_pp: float | None
    reason: str
    probe_accuracy: float | None = None
    accuracy_evaluated: bool = False
    boost_after: float = 0.0


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
    probe_accuracy: float | Callable[[], float] | None,
    reference_accuracy: float,
    q: float,
    w_pp: float,
    e: float,
    mode: str = "joint_dynamic",
    q_base: float = 0.04,
    boost_cap: float = 0.02,
    decay: float = 0.005,
) -> DetectorOutcome:
    """JS-first decision with a bounded temporary threshold boost.

    Pass a zero-argument accuracy callable to enforce lazy probe prediction.
    The current q is q_base + boost. Decay occurs only on a failed JS gate;
    stable gated accuracy raises the boost and a trigger preserves it.
    """
    if mode not in {"joint_dynamic", "joint_fixed", "accuracy_only", "js_only"}:
        raise ValueError(f"Unknown detector mode: {mode}")
    if not (0 <= q_base <= 1 and 0 <= q <= 1) or min(e,w_pp,boost_cap,decay) < 0:
        raise ValueError("Invalid detector thresholds")
    if not all(np.isfinite(v) for v in (jsd, q, q_base, w_pp, e, boost_cap, decay)):
        raise ValueError("Detector inputs must be finite")
    if not 0 <= jsd <= 1:
        raise ValueError("JSD must lie in [0, 1]")
    if mode == "joint_dynamic" and not q_base-1e-12 <= q <= min(1,q_base+boost_cap)+1e-12:
        raise ValueError("Dynamic threshold must equal base plus a bounded nonnegative boost")
    shifted = jsd > q + 1e-12
    if mode == "js_only":
        return DetectorOutcome(shifted, q, None, "js_only", boost_after=max(0,q-q_base))
    if mode != "accuracy_only" and not shifted:
        next_q = max(q_base,q-decay) if mode == "joint_dynamic" else q
        return DetectorOutcome(False,next_q,None,"js_gate_closed",boost_after=max(0,next_q-q_base))
    value = probe_accuracy() if callable(probe_accuracy) else probe_accuracy
    if value is None or not np.isfinite(value) or not np.isfinite(reference_accuracy):
        raise ValueError("A finite probe accuracy and reference are required after the JS gate")
    value = float(value)
    if not (0 <= value <= 1 and 0 <= reference_accuracy <= 1):
        raise ValueError("Accuracy must lie in [0, 1]")
    drop_pp = (reference_accuracy-value)*100.0
    degraded = drop_pp > w_pp + 1e-12
    next_q = min(1,q_base+boost_cap,q+e) if mode == "joint_dynamic" and not degraded else q
    reason = "accuracy_only" if mode == "accuracy_only" else "jsd_and_accuracy_drop" if degraded else "jsd_only_shift"
    return DetectorOutcome(degraded,next_q,drop_pp,reason,value,True,max(0,next_q-q_base))


def stratified_indices(
    y: np.ndarray,
    labels: Sequence[int],
    sizes: Mapping[str, int],
    *,
    seed: int,
) -> dict[str, np.ndarray]:
    """Create deterministic, disjoint per-class index partitions."""
    output: dict[str, list[np.ndarray]] = {name: [] for name in sizes}
    if not sizes or any(not isinstance(size, int) or size < 0 for size in sizes.values()):
        raise ValueError("Partition sizes must be non-negative integers")
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


def oracle_requires_update(drop_pp: float, threshold_pp: float = 2.0) -> bool:
    """Evaluation-only label under a fixed, predeclared threshold."""
    if not np.isfinite(drop_pp) or not np.isfinite(threshold_pp) or threshold_pp < 0:
        raise ValueError("Invalid oracle threshold/drop")
    return bool(drop_pp > threshold_pp + 1e-12)


def paired_time_indices(y, labels, *, probe=5, reserve=10, refresh=2, seed=2024, total=100):
    """Keep the held-out set fixed while taking nested refresh-budget prefixes."""
    if not 1 <= refresh <= reserve or probe < 1 or probe + reserve >= total:
        raise ValueError("Require 1 <= refresh <= reserve and a nonempty remainder")
    parts = stratified_indices(np.asarray(y), labels,
                               {"probe": probe, "refresh_pool": reserve, "remainder": total-probe-reserve}, seed=seed)
    return select_refresh_prefix(parts, np.asarray(y), labels, refresh)


def select_refresh_prefix(parts, y, labels, refresh):
    if not isinstance(refresh,(int,np.integer)) or refresh < 1:
        raise ValueError("Refresh budget must be a positive integer")
    pool = np.asarray(parts["refresh_pool"], dtype=np.int64)
    selected = []
    for label in labels:
        group = pool[y[pool] == label]
        if len(group) < refresh:
            raise ValueError("Shared split does not reserve enough refresh samples")
        selected.append(group[:refresh])
    result = {k: np.asarray(v, dtype=np.int64) for k,v in parts.items()}
    result["refresh"] = np.concatenate(selected)
    result["unused_refresh"] = np.setdiff1d(pool, result["refresh"])
    independent = [result[k] for k in ("probe", "refresh_pool", "remainder")]
    if len(np.unique(np.concatenate(independent))) != sum(map(len, independent)):
        raise ValueError("Probe, refresh pool and remainder must be disjoint")
    return result
