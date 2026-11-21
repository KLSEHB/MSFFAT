"""Stateful AWF-Time deployment simulation with oracle and detector labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

from msffat.data import add_channel, pad_or_truncate
from msffat.deployment import (
    detector_step,
    oracle_requires_update,
    stratified_indices, paired_time_indices, select_refresh_prefix,
)
from msffat.label_maps import dict_dict
from msffat.training import initial_callbacks, restore_selection
from msffat.maintenance import sequence_jsd, sequence_symbol_distribution, attention_transfer_fit
from msffat.model import (
    TemporalCropToMatch,
    build_msffat,
    validate_aligned_model, model_identity,
)


DAYS = ((3, "3d"), (10, "10d"), (14, "2w"), (28, "4w"), (42, "6w"))


@dataclass
class State:
    history: tuple[int, ...]
    model_path: Path
    distribution: np.ndarray
    probe_reference: float
    remainder_reference: float


def parse_args():
    parser = argparse.ArgumentParser(description="MSFFAT v3: one frozen protocol, no test-based tuning")
    parser.add_argument("--data-root", required=True, help="DLWF directory, or its parent")
    parser.add_argument("--output-root", required=True, help="New, empty run directory")
    parser.add_argument("--scratch-root", default=None)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--length", type=int, default=5000)
    parser.add_argument("--initial-epochs", type=int, default=100)
    parser.add_argument("--initial-batch-size", type=int, default=32)
    parser.add_argument("--resume-model", default=None, help="Explicit aligned v3 Day-0 checkpoint only")
    parser.add_argument("--resume-split-root", help="Original checkpoint's cohort/split/hash manifest")
    parser.add_argument("--resume-from-epoch", type=int, default=0)
    parser.add_argument("--resume-epochs", type=int, default=None)
    parser.add_argument("--checkpoint-every-epoch", action="store_true")
    parser.add_argument("--persistent-day0-model", default=None)
    parser.add_argument("--train-day0-only", action="store_true")
    parser.add_argument("--atf-epochs", type=int, default=50)
    parser.add_argument("--atf-batch-size", type=int, default=8)
    parser.add_argument("--atf-learning-rate", type=float, default=1e-3)
    parser.add_argument("--atf-train-scope", choices=("attention", "head", "attention_head", "full"), default="attention")
    parser.add_argument("--detector-mode", choices=("joint_dynamic", "joint_fixed", "accuracy_only", "js_only"), default="joint_dynamic")
    parser.add_argument("--variant", choices=["full", "ltf_only", "msf_only"], default="full")
    parser.add_argument("--fusion", choices=["proposed", "independent", "standard", "concat"], default="proposed")
    parser.add_argument("--update-policy", choices=["detector", "fixed", "never"], default="detector")
    parser.add_argument("--update-days", type=int, nargs="+", default=[14,28,42])
    parser.add_argument("--refresh-reserve", type=int, default=10, help="Common refresh pool/site/checkpoint; paper default 10 (explicit 2 for legacy single-budget runs)")
    parser.add_argument("--split-root", default=None, help="Reuse raw indices and cohort from a shared split")
    parser.add_argument("--n-probe", type=int, default=5)
    parser.add_argument("--m-refresh", type=int, default=2)
    parser.add_argument("--q", type=float, default=0.04)
    parser.add_argument("--w-pp", type=float, default=2.0)
    parser.add_argument("--e", type=float, default=0.005)
    parser.add_argument("--boost-cap", type=float, default=0.02)
    parser.add_argument("--boost-decay", type=float, default=0.005, help="Boost reduction per checkpoint with a closed JS gate")
    parser.add_argument("--fixed-k-pp", type=float, default=2.0, help="Evaluation-only oracle threshold; never searched")
    parser.add_argument("--cohort-limit", type=int, default=None, help="Engineering checks only; not the full cohort")
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()
    if args.n_probe < 1 or args.m_refresh < 1 or args.n_probe + args.m_refresh >= 100:
        parser.error("Require positive probe/refresh sizes with sum < 100")
    if args.fixed_k_pp < 0 or args.atf_epochs < 1 or args.length < 1:
        parser.error("Invalid oracle/optimization settings")
    if args.cohort_limit is not None and args.cohort_limit < 1:
        parser.error("cohort-limit must be positive")
    if any(day not in dict(DAYS) for day in args.update_days):
        parser.error("update-days must be AWF checkpoints")
    reserve = args.m_refresh if args.refresh_reserve is None else args.refresh_reserve
    if not args.m_refresh <= reserve or args.n_probe + reserve >= 100:
        parser.error("refresh-reserve must cover m-refresh and leave held-out traces")
    detector_step(jsd=0, probe_accuracy=1, reference_accuracy=1,
                  q=args.q, q_base=args.q, boost_cap=args.boost_cap, decay=args.boost_decay,
                  w_pp=args.w_pp, e=args.e, mode=args.detector_mode)
    return args


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_npz(path: Path):
    with np.load(path, allow_pickle=True) as payload:
        return payload["data"], payload["labels"]


def load_labels(path: Path):
    with np.load(path, allow_pickle=True) as payload:
        return payload["labels"]


def symbol_distribution_chunked(x: np.ndarray, chunk_size: int = 4096) -> np.ndarray:
    counts = np.zeros((x.shape[1], 3), dtype="float64")
    for start in range(0, len(x), chunk_size):
        chunk = x[start : start + chunk_size]
        counts[:, 0] += np.sum(chunk == -1, axis=0)
        counts[:, 1] += np.sum(chunk == 0, axis=0)
        counts[:, 2] += np.sum(chunk == 1, axis=0)
    counts = np.clip(counts, 1e-12, None)
    return counts / counts.sum(axis=1, keepdims=True)


def symbol_distribution_indexed(x: np.ndarray, indices: np.ndarray, chunk_size: int = 4096) -> np.ndarray:
    counts = np.zeros((x.shape[1], 3), dtype="float64")
    for start in range(0, len(indices), chunk_size):
        chunk = x[indices[start : start + chunk_size]]
        counts[:, 0] += np.sum(chunk == -1, axis=0)
        counts[:, 1] += np.sum(chunk == 0, axis=0)
        counts[:, 2] += np.sum(chunk == 1, axis=0)
    counts = np.clip(counts, 1e-12, None)
    return counts / counts.sum(axis=1, keepdims=True)


def accuracy(model, x: np.ndarray, y: np.ndarray, batch_size: int = 256) -> float:
    prediction = model.predict(add_channel(x), batch_size=batch_size, verbose=0)
    return float(np.mean(np.argmax(prediction, axis=1) == y))


def fit_attention(model, x, y, *, classes, args):
    return attention_transfer_fit(model, x, y, batch_size=args.atf_batch_size,
                                  epochs=args.atf_epochs, learning_rate=args.atf_learning_rate,
                                  seed=args.seed, scope=args.atf_train_scope)


def load_model(path: Path):
    model = tf.keras.models.load_model(path, custom_objects={"TemporalCropToMatch": TemporalCropToMatch}, compile=False)
    validate_aligned_model(model, allow_ablation=True)
    return model


def model_digest(model, include_attention: bool) -> str:
    digest = hashlib.sha256()
    for layer in model.layers:
        is_attention = "attention" in layer.name
        if is_attention != include_attention:
            continue
        digest.update(layer.name.encode())
        for weight in layer.get_weights():
            digest.update(np.ascontiguousarray(weight).tobytes())
    return digest.hexdigest()


def make_cohort(data_root: Path, limit: int | None):
    paths = [data_root / "tor_200w_2500tr_new.npz"] + [
        data_root / f"tor_200w_100tr_time_test{suffix}.npz" for _, suffix in DAYS
    ]
    label_counts = []
    for path in paths:
        labels = load_labels(path)
        unique, counts = np.unique(labels, return_counts=True)
        label_counts.append(dict(zip(unique.tolist(), counts.tolist())))
    complete = {
        label for label in label_counts[0]
        if label_counts[0][label] >= 2500 and all(counts.get(label, 0) >= 100 for counts in label_counts[1:])
    }
    ordered = [
        label for label, _ in sorted(dict_dict["200"].items(), key=lambda item: item[1]) if label in complete
    ]
    if not ordered:
        raise ValueError("No sites satisfy the complete-cohort data requirements")
    if limit is not None:
        ordered = ordered[:limit]
    return ordered, {label: idx for idx, label in enumerate(ordered)}


def filter_and_map(data, labels, label_map, length=5000):
    mask = np.fromiter((label in label_map for label in labels), dtype=bool, count=len(labels))
    x = pad_or_truncate(data[mask], maxlen=length)
    y = np.fromiter((label_map[label] for label in labels[mask]), dtype="int64", count=int(mask.sum()))
    return x, y, np.flatnonzero(mask).astype("int64")


def assert_balanced(indices, labels, classes, count):
    indices = np.asarray(indices)
    if np.any(indices < 0) or np.any(indices >= len(labels)) or len(np.unique(indices)) != len(indices):
        raise ValueError("Invalid or duplicate split indices")
    if not np.array_equal(np.bincount(labels[indices], minlength=classes), np.full(classes, count)):
        raise ValueError("Shared split has a different per-site budget")


def read_shared_parts(path, raw_indices, size):
    with np.load(path) as payload:
        result = {}
        for key in payload.files:
            raw = payload[key].astype(np.int64)
            idx = np.searchsorted(raw_indices, raw)
            if np.any(idx >= size) or not np.array_equal(raw_indices[idx], raw):
                raise ValueError("Shared split contains samples outside this cohort")
            result[key] = idx
    return result


def assert_base_parts(parts, y, classes, probe, remainder):
    for key, count in (("train",2280),("valid",120),("probe",probe),("unused",100-probe-remainder),("remainder",remainder)):
        assert_balanced(parts[key], y, classes, count)
    independent = [parts[k] for k in ("train","valid","probe","unused","remainder")]
    if len(np.unique(np.concatenate(independent))) != sum(map(len, independent)):
        raise ValueError("Day0 train/valid/probe/remainder overlap")


def prepare_data(args, output_root: Path):
    data_root = Path(args.data_root).resolve()
    if not (data_root / "tor_200w_2500tr_new.npz").exists():
        data_root = data_root / "dlwf"
    cohort, label_map = make_cohort(data_root, args.cohort_limit)
    (output_root / "cohort.json").write_text(json.dumps({"sites": cohort, "count": len(cohort)}, indent=2) + "\n")

    base_data, base_labels = load_npz(data_root / "tor_200w_2500tr_new.npz")
    base_x, base_y, base_raw = filter_and_map(base_data, base_labels, label_map, args.length)
    reserve = args.m_refresh if args.refresh_reserve is None else args.refresh_reserve
    remainder_size = 100 - args.n_probe - reserve
    if remainder_size <= 0:
        raise ValueError("n_probe + m_refresh must be less than 100")
    base_parts = stratified_indices(
        base_y,
        list(range(len(cohort))),
        {
            "train": 2280, "valid": 120, "probe": args.n_probe,
            "unused": reserve, "remainder": remainder_size,
        },
        seed=args.seed,
    )
    if args.split_root:
        source = Path(args.split_root)
        prior = json.loads((source/"run_manifest.json").read_text(encoding="utf-8"))
        if prior.get("status") not in {"complete","day0_complete"}:
            raise ValueError("Shared splits require a completed master run")
        for name, expected in prior.get("split_sha256",{}).items():
            if file_sha256(source/name) != expected:
                raise ValueError("Shared split content changed after it was recorded")
        if json.loads((source / "cohort.json").read_text(encoding="utf-8"))["sites"] != cohort:
            raise ValueError("Shared split cohort differs from current data")
        base_parts = read_shared_parts(source / "day0_indices.npz", base_raw, len(base_x))
        assert_base_parts(base_parts, base_y, len(cohort), args.n_probe, remainder_size)
    np.savez_compressed(
        output_root / "day0_indices.npz",
        **{name: base_raw[idx] for name, idx in base_parts.items()},
    )
    day_data = {}
    for day, suffix in DAYS:
        raw_x, raw_y = load_npz(data_root / f"tor_200w_100tr_time_test{suffix}.npz")
        x, y, raw_idx = filter_and_map(raw_x, raw_y, label_map, args.length)
        parts = paired_time_indices(y, list(range(len(cohort))), probe=args.n_probe,
                                    reserve=reserve, refresh=args.m_refresh, seed=args.seed + day*10000)
        if args.split_root:
            parts = read_shared_parts(Path(args.split_root) / f"day{day}_indices.npz", raw_idx, len(x))
            if "refresh_pool" not in parts:
                parts["refresh_pool"] = parts["refresh"]
            parts = select_refresh_prefix(parts, y, list(range(len(cohort))), args.m_refresh)
            for key, count in (("probe",args.n_probe),("refresh_pool",reserve),("remainder",remainder_size)):
                assert_balanced(parts[key], y, len(cohort), count)
        np.savez_compressed(
            output_root / f"day{day}_indices.npz",
            **{name: raw_idx[idx] for name, idx in parts.items()},
        )
        day_data[day] = {
            "probe_x": x[parts["probe"]], "probe_y": y[parts["probe"]],
            "refresh_pool_x": x[parts["refresh_pool"]],
            "refresh_x": x[parts["refresh"]], "refresh_y": y[parts["refresh"]],
            "remainder_x": x[parts["remainder"]], "remainder_y": y[parts["remainder"]],
        }
    return cohort, base_x, base_y, base_parts, day_data


def train_day0(args, run_root, model_root, classes, base_x, base_y, parts):
    model_path = model_root / "day0.keras"
    metadata_path = model_root / "day0.json"
    if model_path.exists():
        raise FileExistsError("Day0 output must be new; supply an explicit resume-model instead")

    if args.resume_model:
        from msffat.baseline_adapters import verify_initial_split
        origin = Path(args.resume_split_root) if args.resume_split_root else Path(args.resume_model).resolve().parent.parent
        cohort = json.loads((run_root/"cohort.json").read_text(encoding="utf-8"))["sites"]
        verify_initial_split({"class_order":cohort,"initial_split_root":str(origin)},Path.cwd(),run_root,cohort)
        prior = json.loads((origin/"run_manifest.json").read_text(encoding="utf-8"))
        if prior.get("method_version") != "MSFFAT-v3":
            raise ValueError("Resume requires a v3 initial run; retain v2 checkpoints separately")
        if prior.get("day0_model_sha256") != file_sha256(args.resume_model):
            raise ValueError("Resume model is not the recorded Day0 checkpoint; migrate and audit legacy models separately")
        model = load_model(Path(args.resume_model).resolve())
        if model.input_shape[1:] != (args.length, 1) or model.output_shape[-1] != classes:
            raise ValueError("Resume checkpoint shape/classes do not match the saved common-cohort protocol")
        identity = model_identity(model)
        if (identity["variant"], identity["fusion"]) != (args.variant, args.fusion):
            raise ValueError("Resume checkpoint differs from requested branch/fusion configuration")
        initial_epoch = args.resume_from_epoch
        target_epochs = initial_epoch + args.resume_epochs if args.resume_epochs is not None else args.initial_epochs
    else:
        model = build_msffat(
            input_shape=(args.length, 1), num_classes=classes, mode="single",
            attention_activation="sigmoid", variant=args.variant, fusion=args.fusion,
        )
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"], jit_compile=False)
        initial_epoch = 0
        target_epochs = args.initial_epochs
    for layer in model.layers:
        layer.trainable = True
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"], jit_compile=False)
    checkpoint_path, callbacks = initial_callbacks(model_path)
    if args.checkpoint_every_epoch:
        checkpoint_dir = model_root / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint_dir / "day0_epoch_{epoch:03d}.keras"),
                save_freq="epoch",
                save_weights_only=False,
                verbose=1,
            )
        )
    started = time.time()
    history = None
    if target_epochs > initial_epoch:
        history = model.fit(
            add_channel(base_x[parts["train"]]),
            to_categorical(base_y[parts["train"]], classes),
            validation_data=(add_channel(base_x[parts["valid"]]), to_categorical(base_y[parts["valid"]], classes)),
            batch_size=args.initial_batch_size,
            initial_epoch=initial_epoch,
            epochs=target_epochs,
            callbacks=callbacks,
            verbose=2,
        )
    if history is not None:
        selection = restore_selection(model, checkpoint_path, history)
    else:
        selection = {"selection_metric": "explicit_checkpoint", "best_epoch": initial_epoch}
    probe_acc = accuracy(model, base_x[parts["probe"]], base_y[parts["probe"]])
    remainder_acc = accuracy(model, base_x[parts["remainder"]], base_y[parts["remainder"]])
    # Equal-size reference/current probes avoid changing JS estimation sample size.
    distribution = sequence_symbol_distribution(base_x[parts["probe"]])
    model.save(model_path)
    if args.persistent_day0_model:
        persistent_path = Path(args.persistent_day0_model).resolve()
        persistent_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(persistent_path)
    np.save(model_root / "day0_distribution.npy", distribution)
    metadata = {
        "classes": classes, "model_identity": model_identity(model), "selection": selection,
        "probe_accuracy": probe_acc,
        "remainder_accuracy": remainder_acc,
        "best_epoch": (int(np.argmax(history.history["val_accuracy"]) + initial_epoch + 1)
                       if history is not None else initial_epoch),
        "initial_epoch": initial_epoch,
        "target_epochs": target_epochs,
        "epochs_ran": len(history.history["loss"]) if history is not None else 0,
        "runtime_seconds": time.time() - started,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return State((), model_path, distribution, probe_acc, remainder_acc)


def state_name(history):
    return "day0" if not history else "triggers_" + "_".join(map(str, history))


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    args = parse_args()
    if args.persistent_day0_model and Path(args.persistent_day0_model).exists():
        raise FileExistsError("Refusing to overwrite an existing persistent Day-0 checkpoint")
    set_seed(args.seed)
    if args.require_gpu and not tf.config.list_physical_devices("GPU"):
        raise RuntimeError("--require-gpu was set, but TensorFlow found no GPU")
    run_root = Path(args.output_root).resolve()
    scratch_root = Path(args.scratch_root).resolve() if args.scratch_root else run_root
    model_root = scratch_root / "models"
    if run_root.exists() and any(run_root.iterdir()):
        raise FileExistsError("Use a new empty output-root; cached states cannot be reused implicitly")
    if model_root.exists() and any(model_root.iterdir()):
        raise FileExistsError("Use an empty scratch model directory")
    for folder in (run_root, model_root):
        folder.mkdir(parents=True, exist_ok=True)
    code_root = Path(__file__).resolve().parents[1]
    source_files = list((code_root / "msffat").glob("*.py")) + list((code_root / "scripts").glob("*.py"))
    manifest = {"method_version": "MSFFAT-v3", "config": vars(args),
                "tensorflow": tf.__version__, "numpy": np.__version__,
                "source_sha256": {p.relative_to(code_root).as_posix(): file_sha256(p) for p in source_files},
                "status": "running", "reduced_cohort": args.cohort_limit is not None}
    (run_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    cohort, base_x, base_y, base_parts, day_data = prepare_data(args, run_root)
    data_root = Path(args.data_root).resolve()
    if not (data_root / "tor_200w_2500tr_new.npz").exists():
        data_root = data_root / "dlwf"
    manifest["data_sha256"] = {p.name: file_sha256(p) for p in
                               [data_root / "tor_200w_2500tr_new.npz"] +
                               [data_root / f"tor_200w_100tr_time_test{suffix}.npz" for _, suffix in DAYS]}
    if args.split_root:
        prior_path = Path(args.split_root) / "run_manifest.json"
        if not prior_path.exists():
            raise ValueError("Shared splits require their data-identity run manifest")
        prior = json.loads(prior_path.read_text(encoding="utf-8"))
        if prior.get("data_sha256") != manifest["data_sha256"]:
            raise ValueError("Shared indices belong to different dataset content")
    manifest["split_sha256"] = {p.name: file_sha256(p) for p in run_root.glob("*indices.npz")}
    manifest["cohort_size"] = len(cohort)
    if args.resume_model:
        manifest["resume_model_sha256"] = file_sha256(args.resume_model)
    (run_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    state = train_day0(args, run_root, model_root, len(cohort), base_x, base_y, base_parts)
    manifest["day0_model_sha256"] = file_sha256(state.model_path)
    if args.train_day0_only:
        manifest["status"] = "day0_complete"
        (run_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return
    del base_x, base_y
    model = load_model(state.model_path)
    q = args.q
    rows = []
    for day, suffix in DAYS:
        data = day_data[day]
        started = time.perf_counter()
        distribution = sequence_symbol_distribution(data["probe_x"])
        jsd = sequence_jsd(distribution, state.distribution)
        decision = detector_step(jsd=jsd,
                                 probe_accuracy=lambda: accuracy(model, data["probe_x"], data["probe_y"]),
                                 reference_accuracy=state.probe_reference,
                                 q=q, q_base=args.q, boost_cap=args.boost_cap, decay=args.boost_decay,
                                 w_pp=args.w_pp, e=args.e, mode=args.detector_mode)
        detector_seconds = time.perf_counter() - started
        # Remainder is read AFTER the decision, exclusively for reporting.
        remainder_acc = accuracy(model, data["remainder_x"], data["remainder_y"])
        drop_pp = (state.remainder_reference - remainder_acc) * 100
        oracle = oracle_requires_update(drop_pp, args.fixed_k_pp)
        row = {"day": day, "reference_day": state.history[-1] if state.history else 0,
               "probe_reference_accuracy": state.probe_reference,
               "remainder_reference_accuracy": state.remainder_reference,
               "probe_accuracy": decision.probe_accuracy, "probe_drop_pp": decision.probe_drop_pp,
               "probe_accuracy_evaluated": decision.accuracy_evaluated,
               "probe_labels_per_site": args.n_probe if decision.accuracy_evaluated else 0,
               "remainder_accuracy": remainder_acc, "remainder_drop_pp": drop_pp,
               "jsd": jsd, "q_base": args.q, "boost_before": max(0,q-args.q),
               "boost_after": decision.boost_after, "q_before": q, "q_after": decision.q_after,
               "detector_trigger": decision.trigger, "detector_reason": decision.reason,
               "oracle_should_update": oracle, "detector_correct": decision.trigger == oracle,
               "detector_decision_seconds": detector_seconds,
               "post_atf_probe_accuracy": None, "post_atf_remainder_accuracy": None,
               "atf_runtime_seconds": None, "peak_gpu_bytes": None, "refresh_labels_per_site": 0}
        q = decision.q_after
        update = decision.trigger if args.update_policy == "detector" else day in args.update_days if args.update_policy == "fixed" else False
        row["update_executed"] = update
        row["update_policy"] = args.update_policy
        if update:
            set_seed(args.seed + day)
            args_for_fit = argparse.Namespace(**vars(args))
            args_for_fit.seed += day
            started = time.perf_counter()
            if tf.config.list_physical_devices("GPU"):
                tf.config.experimental.reset_memory_stats("GPU:0")
            fit = fit_attention(model, data["refresh_x"], data["refresh_y"], classes=len(cohort), args=args_for_fit)
            row["atf_runtime_seconds"] = time.perf_counter() - started
            row["peak_gpu_bytes"] = (tf.config.experimental.get_memory_info("GPU:0")["peak"]
                                     if tf.config.list_physical_devices("GPU") else None)
            post_probe = (accuracy(model, data["probe_x"], data["probe_y"])
                          if args.detector_mode != "js_only" else None)
            post_remainder = accuracy(model, data["remainder_x"], data["remainder_y"])
            row.update(post_atf_probe_accuracy=post_probe, post_atf_remainder_accuracy=post_remainder,
                       refresh_labels_per_site=args.m_refresh,
                       probe_labels_per_site=args.n_probe if post_probe is not None else row["probe_labels_per_site"])
            model_path = model_root / f"day{day}.keras"
            model.save(model_path)
            metadata = {"day": day, "fit": fit, "model_sha256": file_sha256(model_path)}
            (run_root / f"day{day}_adaptation.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            state = State(state.history + (day,), model_path, distribution,
                          post_probe if post_probe is not None else state.probe_reference, post_remainder)
        rows.append(row)
        # Durable progress; no parameter selection is performed on these values.
        (run_root / "deployment_progress.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    tp = sum(r["oracle_should_update"] and r["detector_trigger"] for r in rows)
    tn = sum(not r["oracle_should_update"] and not r["detector_trigger"] for r in rows)
    fp = sum(not r["oracle_should_update"] and r["detector_trigger"] for r in rows)
    fn = sum(r["oracle_should_update"] and not r["detector_trigger"] for r in rows)
    triggers = sum(r["update_executed"] for r in rows)
    probe_cost = sum(r["probe_labels_per_site"] for r in rows)
    refresh_cost = args.m_refresh * triggers
    identity = model_identity(model)
    summary = {"method_version": "MSFFAT-v3", "result_kind": "engineering_check" if args.cohort_limit is not None or identity.get("engineering_fixture") else "measured_run", "seed": args.seed, "cohort_size": len(cohort),
               "detector_mode": args.detector_mode, "adaptation_scope": args.atf_train_scope,
               "update_policy": args.update_policy, "model_identity": identity,
               "detector_triggers": tp+fp,
               "tp": tp, "tn": tn, "fp": fp, "fn": fn,
               "detection_accuracy": (tp + tn) / len(rows), "triggers": triggers,
               "probe_labels_per_site": probe_cost, "refresh_labels_per_site": refresh_cost,
               "probe_accuracy_checks": sum(r["probe_accuracy_evaluated"] for r in rows),
               "monitoring_traces_per_site": args.n_probe*len(DAYS),
               "maximum_probe_label_budget_per_site": args.n_probe*len(DAYS),
               "maintenance_labels_per_site": probe_cost + refresh_cost,
               "maintenance_labels_total": len(cohort) * (probe_cost + refresh_cost),
               "reserved_refresh_traces_per_site": (args.m_refresh if args.refresh_reserve is None else args.refresh_reserve) * len(DAYS),
               "initial_training_and_validation_labels_per_site": 2400,
               "initial_reference_probe_labels_per_site": args.n_probe,
               "evaluation_only": True, "rows": rows}
    (run_root / "deployment_result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (run_root / "deployment_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Sequential deployment results", "",
             f"Method: MSFFAT-v3; detector={args.detector_mode}; scope={args.atf_train_scope}; sites={len(cohort)}.",
             f"TP/TN/FP/FN: {tp}/{tn}/{fp}/{fn}. New labels/site: {probe_cost + refresh_cost}.", "",
             "| Day | Reference day | JS | q before | q after | Probe drop (pp) | Detector | Updated | Oracle | Pre remainder | Post remainder |",
             "|---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|---:|---:|"]
    for row in rows:
        post = "N/A" if row["post_atf_remainder_accuracy"] is None else f"{100*row['post_atf_remainder_accuracy']:.2f}%"
        probe_drop = "N/A" if row["probe_drop_pp"] is None else f"{row['probe_drop_pp']:.2f}"
        lines.append(f"| {row['day']} | {row['reference_day']} | {row['jsd']:.5f} | {row['q_before']:.3f} | {row['q_after']:.3f} | {probe_drop} | {row['detector_trigger']} | {row['update_executed']} | {row['oracle_should_update']} | {100*row['remainder_accuracy']:.2f}% | {post} |")
    (run_root / "deployment_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest["status"] = "complete"
    (run_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
