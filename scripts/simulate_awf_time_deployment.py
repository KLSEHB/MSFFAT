"""Stateful AWF-Time deployment simulation with oracle and detector labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

from msffat.data import add_channel
from msffat.deployment import (
    best_k_accuracy,
    detector_step,
    find_matching_k,
    stratified_indices,
)
from msffat.label_maps import dict_dict
from msffat.maintenance import sequence_jsd
from msffat.model import (
    TemporalCropToMatch,
    build_msffat,
    set_attention_and_output_trainable,
    set_attention_only_trainable,
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--scratch-root", default=None, help="Node-local directory for large model checkpoints.")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--length", type=int, default=5000)
    parser.add_argument("--initial-epochs", type=int, default=100)
    parser.add_argument("--initial-batch-size", type=int, default=32)
    parser.add_argument("--attention-activation", choices=("hard_sigmoid", "sigmoid"), default="hard_sigmoid")
    parser.add_argument("--resume-model", default=None)
    parser.add_argument("--resume-from-epoch", type=int, default=0)
    parser.add_argument("--resume-epochs", type=int, default=None)
    parser.add_argument("--checkpoint-every-epoch", action="store_true")
    parser.add_argument("--persistent-day0-model", default=None)
    parser.add_argument("--train-day0-only", action="store_true")
    parser.add_argument("--atf-epochs", type=int, default=20)
    parser.add_argument("--atf-batch-size", type=int, default=8)
    parser.add_argument("--atf-learning-rate", type=float, default=1e-3)
    parser.add_argument("--atf-deterministic-backbone", action="store_true")
    parser.add_argument("--atf-target-accuracy", type=float, default=None)
    parser.add_argument("--atf-stall-patience", type=int, default=None)
    parser.add_argument(
        "--atf-train-scope", choices=("attention", "attention_head"), default="attention"
    )
    parser.add_argument("--detector-mode", choices=("jsd_probe", "probe_only"), default="jsd_probe")
    parser.add_argument("--detector-probe-drop-pp", type=float, default=2.0)
    parser.add_argument("--n-probe", type=int, default=5)
    parser.add_argument("--m-refresh", type=int, default=5)
    parser.add_argument("--q", type=float, default=0.05)
    parser.add_argument("--w-pp", type=float, default=3.0)
    parser.add_argument("--e", type=float, default=0.01)
    parser.add_argument("--k-step-pp", type=float, default=0.1)
    parser.add_argument("--fixed-k-pp", type=float, default=None)
    parser.add_argument("--cohort-limit", type=int, default=None)
    parser.add_argument("--skip-tuning", action="store_true")
    parser.add_argument("--force-tuning", action="store_true")
    parser.add_argument("--q-grid", type=float, nargs="+", default=None)
    parser.add_argument("--w-grid-pp", type=float, nargs="+", default=None)
    parser.add_argument("--e-grid", type=float, nargs="+", default=None)
    parser.add_argument("--require-gpu", action="store_true")
    return parser.parse_args()


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
    """Fit attention weights while optionally keeping frozen Dropout deterministic."""
    x = add_channel(x)
    y_onehot = to_categorical(y, classes)
    if not args.atf_deterministic_backbone:
        return model.fit(
            x, y_onehot, batch_size=args.atf_batch_size, epochs=args.atf_epochs,
            shuffle=True, verbose=2,
        )

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.atf_learning_rate)
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    trainable = model.trainable_variables
    if not trainable:
        raise RuntimeError("ATF has no trainable variables")
    gate_model = tf.keras.Model(model.input, model.get_layer("fusion_attention_weights").output)
    gate_values = gate_model.predict(x, batch_size=256, verbose=0)
    saturated = np.mean((gate_values <= 1e-7) | (gate_values >= 1.0 - 1e-7))
    print(f"ATF diagnostics: trainable_parameters={sum(np.prod(v.shape) for v in trainable)} "
          f"gate_saturation={saturated:.6f} gate_min={gate_values.min():.6f} "
          f"gate_max={gate_values.max():.6f}")
    initial_weights = [v.numpy().copy() for v in trainable]
    dataset = tf.data.Dataset.from_tensor_slices((x, y_onehot))
    dataset = dataset.shuffle(len(x), seed=args.seed, reshuffle_each_iteration=True)
    dataset = dataset.batch(args.atf_batch_size)
    best_acc = -1.0
    stalled = 0
    for epoch in range(args.atf_epochs):
        losses = []
        gradient_norms = []
        for batch_x, batch_y in dataset:
            with tf.GradientTape() as tape:
                # training=False disables frozen-backbone Dropout while gradients
                # still flow into the attention Dense layers.
                prediction = model(batch_x, training=False)
                loss = loss_fn(batch_y, prediction)
            gradients = tape.gradient(loss, trainable)
            gradient_norms.append(float(tf.linalg.global_norm([g for g in gradients if g is not None])))
            optimizer.apply_gradients((g, v) for g, v in zip(gradients, trainable) if g is not None)
            losses.append(float(loss))
        train_acc = accuracy(model, x[:, :, 0], y)
        weight_delta = float(tf.linalg.global_norm([
            variable - initial for variable, initial in zip(trainable, initial_weights)
        ]))
        print(f"ATF epoch {epoch + 1}/{args.atf_epochs}: loss={np.mean(losses):.6f} "
              f"refresh_accuracy={train_acc:.6f} gradient_norm={np.mean(gradient_norms):.6e} "
              f"weight_delta={weight_delta:.6e}")
        if args.atf_target_accuracy is not None and train_acc >= args.atf_target_accuracy:
            print(f"ATF reached target refresh accuracy {args.atf_target_accuracy:.6f}")
            break
        if train_acc > best_acc + 1e-12:
            best_acc, stalled = train_acc, 0
        else:
            stalled += 1
        if args.atf_stall_patience is not None and stalled >= args.atf_stall_patience:
            print(f"ATF stopped after {stalled} epochs without refresh-accuracy improvement")
            break


def load_model(path: Path):
    return tf.keras.models.load_model(path, custom_objects={"TemporalCropToMatch": TemporalCropToMatch})


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
    if len(ordered) != 186:
        raise AssertionError(f"Expected 186 complete sites, found {len(ordered)}")
    if limit is not None:
        ordered = ordered[:limit]
    return ordered, {label: idx for idx, label in enumerate(ordered)}


def filter_and_map(data, labels, label_map):
    mask = np.fromiter((label in label_map for label in labels), dtype=bool, count=len(labels))
    x = np.asarray(data[mask], dtype="float32")
    y = np.fromiter((label_map[label] for label in labels[mask]), dtype="int64", count=int(mask.sum()))
    return x, y, np.flatnonzero(mask).astype("int64")


def prepare_data(args, output_root: Path):
    data_root = Path(args.data_root).resolve()
    cohort, label_map = make_cohort(data_root, args.cohort_limit)
    (output_root / "cohort.json").write_text(json.dumps({"sites": cohort, "count": len(cohort)}, indent=2) + "\n")

    base_data, base_labels = load_npz(data_root / "tor_200w_2500tr_new.npz")
    base_x, base_y, base_raw = filter_and_map(base_data, base_labels, label_map)
    remainder_size = 100 - args.n_probe - args.m_refresh
    if remainder_size <= 0:
        raise ValueError("n_probe + m_refresh must be less than 100")
    base_parts = stratified_indices(
        base_y,
        list(range(len(cohort))),
        {
            "train": 2280, "valid": 120, "probe": args.n_probe,
            "unused": args.m_refresh, "remainder": remainder_size,
        },
        seed=args.seed,
    )
    np.savez_compressed(
        output_root / "day0_indices.npz",
        **{name: base_raw[idx] for name, idx in base_parts.items()},
    )
    day_data = {}
    for day, suffix in DAYS:
        raw_x, raw_y = load_npz(data_root / f"tor_200w_100tr_time_test{suffix}.npz")
        x, y, raw_idx = filter_and_map(raw_x, raw_y, label_map)
        parts = stratified_indices(
            y,
            list(range(len(cohort))),
            {"probe": args.n_probe, "refresh": args.m_refresh, "remainder": remainder_size},
            seed=args.seed + day * 10000,
        )
        np.savez_compressed(
            output_root / f"day{day}_indices.npz",
            **{name: raw_idx[idx] for name, idx in parts.items()},
        )
        day_data[day] = {
            "probe_x": x[parts["probe"]], "probe_y": y[parts["probe"]],
            "refresh_x": x[parts["refresh"]], "refresh_y": y[parts["refresh"]],
            "remainder_x": x[parts["remainder"]], "remainder_y": y[parts["remainder"]],
        }
    return cohort, base_x, base_y, base_parts, day_data


def train_day0(args, run_root, model_root, classes, base_x, base_y, parts):
    model_path = model_root / "day0.keras"
    metadata_path = model_root / "day0.json"
    if model_path.exists() and metadata_path.exists():
        payload = json.loads(metadata_path.read_text())
        distribution = np.load(model_root / "day0_distribution.npy")
        return State((), model_path, distribution, payload["probe_accuracy"], payload["remainder_accuracy"])

    if args.resume_model:
        model = load_model(Path(args.resume_model).resolve())
        initial_epoch = args.resume_from_epoch
        target_epochs = initial_epoch + args.resume_epochs if args.resume_epochs is not None else args.initial_epochs
    else:
        model = build_msffat(
            input_shape=(args.length, 1), num_classes=classes, mode="single",
            attention_activation=args.attention_activation,
        )
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"], jit_compile=False)
        initial_epoch = 0
        target_epochs = args.initial_epochs
    callbacks = [
        ReduceLROnPlateau(monitor="val_accuracy", factor=np.sqrt(0.1), patience=5, min_lr=1e-5, verbose=1),
        EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True),
    ]
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
    probe_acc = accuracy(model, base_x[parts["probe"]], base_y[parts["probe"]])
    remainder_acc = accuracy(model, base_x[parts["remainder"]], base_y[parts["remainder"]])
    development_idx = np.concatenate([parts["train"], parts["valid"]])
    distribution = (
        np.empty((0, 3), dtype="float64")
        if args.detector_mode == "probe_only"
        else symbol_distribution_indexed(base_x, development_idx)
    )
    model.save(model_path)
    if args.persistent_day0_model:
        persistent_path = Path(args.persistent_day0_model).resolve()
        persistent_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(persistent_path)
    np.save(model_root / "day0_distribution.npy", distribution)
    metadata = {
        "classes": classes,
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


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.require_gpu and not tf.config.list_physical_devices("GPU"):
        raise RuntimeError("--require-gpu was set, but TensorFlow found no GPU")
    run_root = Path(args.output_root).resolve()
    scratch_root = Path(args.scratch_root).resolve() if args.scratch_root else run_root
    model_root = scratch_root / "models"
    state_root = scratch_root / "states"
    for folder in (run_root, model_root, state_root):
        folder.mkdir(parents=True, exist_ok=True)

    cohort, base_x, base_y, base_parts, day_data = prepare_data(args, run_root)
    initial = train_day0(args, run_root, model_root, len(cohort), base_x, base_y, base_parts)
    if args.train_day0_only:
        summary = {
            "cohort_size": len(cohort),
            "seed": args.seed,
            "day0_probe_accuracy": initial.probe_reference,
            "day0_remainder_accuracy": initial.remainder_reference,
            "model_path": str(initial.model_path),
            "persistent_model_path": args.persistent_day0_model,
        }
        (run_root / "day0_training_result.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
        return
    del base_x, base_y
    states = {(): initial}
    metric_cache = {}

    def metrics_for(history, day):
        key = (history, day)
        if key not in metric_cache:
            state = states[history]
            model = load_model(state.model_path)
            data = day_data[day]
            detector_started = time.perf_counter()
            probe_accuracy = accuracy(model, data["probe_x"], data["probe_y"])
            probe_inference_seconds = time.perf_counter() - detector_started
            metric_cache[key] = {
                "probe_accuracy": probe_accuracy,
                "probe_inference_seconds": probe_inference_seconds,
                "remainder_accuracy": accuracy(model, data["remainder_x"], data["remainder_y"]),
                "probe_distribution": (
                    None if args.detector_mode == "probe_only"
                    else symbol_distribution_chunked(data["probe_x"])
                ),
            }
            tf.keras.backend.clear_session()
        return metric_cache[key]

    def trigger_state(history, day):
        child_history = history + (day,)
        if child_history in states:
            return states[child_history]
        parent = states[history]
        state_dir = state_root / state_name(child_history)
        state_dir.mkdir(parents=True, exist_ok=True)
        model_path = state_dir / "model.keras"
        metadata_path = state_dir / "metadata.json"
        distribution_path = state_dir / "distribution.npy"
        if model_path.exists() and metadata_path.exists() and distribution_path.exists():
            payload = json.loads(metadata_path.read_text())
            state = State(
                child_history, model_path, np.load(distribution_path),
                payload["post_probe_accuracy"], payload["post_remainder_accuracy"],
            )
            states[child_history] = state
            return state
        set_seed(args.seed + sum((i + 1) * value for i, value in enumerate(child_history)))
        model = load_model(parent.model_path)
        if args.atf_train_scope == "attention_head":
            set_attention_and_output_trainable(model)
        else:
            set_attention_only_trainable(model)
        frozen_before = hashlib.sha256()
        for layer in model.layers:
            if not layer.trainable:
                frozen_before.update(layer.name.encode())
                for weight in layer.get_weights():
                    frozen_before.update(np.ascontiguousarray(weight).tobytes())
        frozen_before = frozen_before.hexdigest()
        attention_before = model_digest(model, include_attention=True)
        trainable_names = [layer.name for layer in model.layers if layer.trainable]
        allowed = lambda name: "attention" in name or (
            args.atf_train_scope == "attention_head" and name == "single_label_output"
        )
        if not trainable_names or any(not allowed(name) for name in trainable_names):
            raise AssertionError(f"Unexpected trainable layers: {trainable_names}")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.atf_learning_rate),
            loss="categorical_crossentropy", metrics=["accuracy"],
        )
        data = day_data[day]
        started = time.time()
        fit_attention(model, data["refresh_x"], data["refresh_y"], classes=len(cohort), args=args)
        runtime = time.time() - started
        frozen_after = hashlib.sha256()
        for layer in model.layers:
            if not layer.trainable:
                frozen_after.update(layer.name.encode())
                for weight in layer.get_weights():
                    frozen_after.update(np.ascontiguousarray(weight).tobytes())
        if frozen_after.hexdigest() != frozen_before:
            raise AssertionError("ATF changed frozen weights")
        attention_after = model_digest(model, include_attention=True)
        post_probe = accuracy(model, data["probe_x"], data["probe_y"])
        post_remainder = accuracy(model, data["remainder_x"], data["remainder_y"])
        distribution = (
            np.empty((0, 3), dtype="float64")
            if args.detector_mode == "probe_only"
            else symbol_distribution_chunked(data["probe_x"])
        )
        model.save(model_path)
        np.save(distribution_path, distribution)
        payload = {
            "history": child_history,
            "post_probe_accuracy": post_probe,
            "post_remainder_accuracy": post_remainder,
            "atf_runtime_seconds": runtime,
            "trainable_layers": trainable_names,
            "frozen_digest_unchanged": True,
            "attention_digest_changed": attention_before != attention_after,
        }
        metadata_path.write_text(json.dumps(payload, indent=2) + "\n")
        state = State(child_history, model_path, distribution, post_probe, post_remainder)
        states[child_history] = state
        tf.keras.backend.clear_session()
        return state

    def simulate(q0, w_pp, e):
        history = ()
        q = q0
        rows = []
        for day, suffix in DAYS:
            state = states[history]
            current = metrics_for(history, day)
            decision_started = time.perf_counter()
            if args.detector_mode == "probe_only":
                jsd = None
                probe_drop_pp = (state.probe_reference - current["probe_accuracy"]) * 100.0
                detector_trigger = probe_drop_pp > args.detector_probe_drop_pp
                detector_reason = "probe_accuracy_drop" if detector_trigger else "probe_accuracy_stable"
                q_after = q
            else:
                jsd = sequence_jsd(current["probe_distribution"], state.distribution)
                decision = detector_step(
                    jsd=jsd,
                    probe_accuracy=current["probe_accuracy"],
                    reference_accuracy=state.probe_reference,
                    q=q,
                    w_pp=w_pp,
                    e=e,
                )
                probe_drop_pp = decision.probe_drop_pp
                detector_trigger = decision.trigger
                detector_reason = decision.reason
                q_after = decision.q_after
            detector_decision_seconds = (
                current["probe_inference_seconds"] + time.perf_counter() - decision_started
            )
            row = {
                "day": day, "suffix": suffix, "state_before": state_name(history),
                "reference_day": 0 if not history else history[-1],
                "q_before": q, "jsd": jsd,
                "probe_reference_accuracy": state.probe_reference,
                "probe_accuracy": current["probe_accuracy"],
                "probe_drop_pp": probe_drop_pp,
                "remainder_reference_accuracy": state.remainder_reference,
                "remainder_accuracy": current["remainder_accuracy"],
                "remainder_drop_pp": (state.remainder_reference - current["remainder_accuracy"]) * 100.0,
                "detector_trigger": detector_trigger,
                "detector_reason": detector_reason,
                "detector_decision_seconds": detector_decision_seconds,
                "q_after": q_after,
                "post_atf_probe_accuracy": None,
                "post_atf_remainder_accuracy": None,
                "atf_runtime_seconds": None,
            }
            q = q_after
            if detector_trigger:
                child = trigger_state(history, day)
                metadata = json.loads((child.model_path.parent / "metadata.json").read_text())
                row["post_atf_probe_accuracy"] = child.probe_reference
                row["post_atf_remainder_accuracy"] = child.remainder_reference
                row["atf_runtime_seconds"] = metadata["atf_runtime_seconds"]
                history = child.history
            rows.append(row)
        return {"q": q0, "w_pp": w_pp, "e": e, "rows": rows}

    def attach_k(result):
        drops = [row["remainder_drop_pp"] for row in result["rows"]]
        triggers = [row["detector_trigger"] for row in result["rows"]]
        if args.fixed_k_pp is not None:
            k = args.fixed_k_pp
            oracle = [drop > k for drop in drops]
            accuracy_value = float(np.mean(np.asarray(oracle) == np.asarray(triggers)))
            result["k_pp"] = k
            result["detection_accuracy"] = accuracy_value
            for row, label in zip(result["rows"], oracle):
                row["oracle_should_update"] = label
                row["detector_correct"] = label == row["detector_trigger"]
            if accuracy_value < 1.0 or all(oracle) or not any(oracle):
                result["k_pp"] = None
        else:
            k, oracle = find_matching_k(drops, triggers, step_pp=args.k_step_pp, require_nondegenerate=True)
            result["k_pp"] = k
            result["detection_accuracy"] = 1.0 if k is not None else None
            if oracle is not None:
                for row, label in zip(result["rows"], oracle):
                    row["oracle_should_update"] = label
                    row["detector_correct"] = label == row["detector_trigger"]
        return result

    default_result = attach_k(simulate(args.q, args.w_pp, args.e))
    all_results = [default_result]
    candidates = [default_result] if default_result["k_pp"] is not None else []
    searched = 1
    if (not candidates or args.force_tuning) and not args.skip_tuning:
        grid = itertools.product(
            args.q_grid or (0.03, 0.04, 0.05, 0.06, 0.07),
            args.w_grid_pp or (1.0, 2.0, 3.0, 4.0, 5.0),
            args.e_grid or (0.0, 0.005, 0.01, 0.015, 0.02),
        )
        for q0, w_pp, e in grid:
            if (q0, w_pp, e) == (args.q, args.w_pp, args.e):
                continue
            searched += 1
            result = attach_k(simulate(q0, w_pp, e))
            all_results.append(result)
            if result["k_pp"] is not None:
                candidates.append(result)
    if candidates:
        def rank(result):
            triggers = sum(row["detector_trigger"] for row in result["rows"])
            distance = abs(result["q"] - 0.05) / 0.01 + abs(result["w_pp"] - 3.0) + abs(result["e"] - 0.01) / 0.005
            return result["k_pp"], triggers, distance
        selected = min(candidates, key=rank)
    else:
        if args.fixed_k_pp is not None:
            selected = max(all_results, key=lambda result: result["detection_accuracy"])
            selected["k_pp"] = args.fixed_k_pp
        else:
            fallback = []
            for result in all_results:
                k, acc, oracle = best_k_accuracy(
                    [row["remainder_drop_pp"] for row in result["rows"]],
                    [row["detector_trigger"] for row in result["rows"]],
                    step_pp=args.k_step_pp,
                )
                fallback.append((acc, -k, result, k, oracle))
            _, _, selected, k, oracle = max(fallback, key=lambda item: (item[0], item[1]))
            selected["k_pp"] = k
            selected["detection_accuracy"] = max(item[0] for item in fallback)
            for row, label in zip(selected["rows"], oracle):
                row["oracle_should_update"] = label
                row["detector_correct"] = label == row["detector_trigger"]

    tp = sum(row["oracle_should_update"] and row["detector_trigger"] for row in selected["rows"])
    tn = sum(not row["oracle_should_update"] and not row["detector_trigger"] for row in selected["rows"])
    fp = sum(not row["oracle_should_update"] and row["detector_trigger"] for row in selected["rows"])
    fn = sum(row["oracle_should_update"] and not row["detector_trigger"] for row in selected["rows"])
    selected["confusion"] = {"tp": tp, "tn": tn, "fp": fp, "fn": fn}

    payload = {
        "cohort_size": len(cohort), "seed": args.seed,
        "n_probe": args.n_probe, "m_refresh": args.m_refresh,
        "remainder_per_site": 100 - args.n_probe - args.m_refresh,
        "detector_mode": args.detector_mode,
        "detector_probe_drop_pp": args.detector_probe_drop_pp,
        "atf_epochs": args.atf_epochs, "k_step_pp": args.k_step_pp,
        "day0_probe_accuracy": initial.probe_reference,
        "day0_remainder_accuracy": initial.remainder_reference,
        "scratch_root": str(scratch_root),
        "parameter_configurations_searched": searched,
        "default": default_result, "selected": selected,
    }
    (run_root / "deployment_results.json").write_text(json.dumps(payload, indent=2) + "\n")
    csv_fields = [
        "day", "suffix", "state_before", "reference_day", "q_before", "q_after", "jsd",
        "probe_reference_accuracy", "probe_accuracy", "probe_drop_pp",
        "remainder_reference_accuracy", "remainder_accuracy", "remainder_drop_pp",
        "oracle_should_update", "detector_trigger", "detector_correct", "detector_reason",
        "detector_decision_seconds",
        "post_atf_probe_accuracy", "post_atf_remainder_accuracy", "atf_runtime_seconds",
    ]
    with (run_root / "deployment_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selected["rows"])

    lines = [
        "# AWF-Time Stateful Drift-Trigger Simulation", "",
        f"Cohort: {len(cohort)} fixed websites; probe/refresh/remainder: "
        f"{args.n_probe}/{args.m_refresh}/{100-args.n_probe-args.m_refresh} traces per site.", "",
        f"Detection accuracy: {100.0 * selected['detection_accuracy']:.2f}% "
        f"(TP={tp}, TN={tn}, FP={fp}, FN={fn}; ATF triggers={tp + fp}).", "",
    ]
    if args.detector_mode == "probe_only":
        lines += [
            f"Detector threshold: probe drop > {args.detector_probe_drop_pp:.1f} pp; "
            f"Oracle threshold: remainder drop > {selected['k_pp']:.1f} pp.", "",
            "| Day | Ref. day | Need update | Detector ATF | Pre-ATF remainder Acc. | Remainder drop | Pre-ATF probe Acc. | Probe drop | Post-ATF probe Acc. | Post-ATF remainder Acc. | Detector time |",
            "|---:|:---:|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    else:
        lines += [
            f"Selected parameters: q={selected['q']:.3f}, w={selected['w_pp']:.1f} pp, "
            f"e={selected['e']:.3f}, K={selected['k_pp']:.1f} pp.", "",
            "| Day | Oracle update | Remainder ACC | Remainder drop | JSD | Probe ACC | Probe drop | q before→after | Detector ATF | Post-ATF probe ACC | Post-ATF remainder ACC |",
            "|---:|:---:|---:|---:|---:|---:|---:|---:|:---:|---:|---:|",
        ]
    for row in selected["rows"]:
        post_probe = "—" if row["post_atf_probe_accuracy"] is None else f"{100*row['post_atf_probe_accuracy']:.2f}%"
        post_rem = "—" if row["post_atf_remainder_accuracy"] is None else f"{100*row['post_atf_remainder_accuracy']:.2f}%"
        if args.detector_mode == "probe_only":
            lines.append(
                f"| {row['day']} | Day {row['reference_day']} | "
                f"{'Yes' if row['oracle_should_update'] else 'No'} | "
                f"{'Yes' if row['detector_trigger'] else 'No'} | "
                f"{100*row['remainder_accuracy']:.2f}% | {row['remainder_drop_pp']:.2f} pp | "
                f"{100*row['probe_accuracy']:.2f}% | {row['probe_drop_pp']:.2f} pp | "
                f"{post_probe} | {post_rem} | {1000*row['detector_decision_seconds']:.2f} ms |"
            )
        else:
            lines.append(
                f"| {row['day']} | {'Yes' if row['oracle_should_update'] else 'No'} | {100*row['remainder_accuracy']:.2f}% | "
                f"{row['remainder_drop_pp']:.2f} pp | {row['jsd']:.5f} | {100*row['probe_accuracy']:.2f}% | "
                f"{row['probe_drop_pp']:.2f} pp | {row['q_before']:.3f}→{row['q_after']:.3f} | "
                f"{'Yes' if row['detector_trigger'] else 'No'} | {post_probe} | {post_rem} |"
            )
    if args.detector_mode == "probe_only":
        lines += ["", "The Detector uses only probe accuracy; the held-out remainder set is used only for Oracle evaluation."]
    else:
        lines += ["", "The Oracle label uses only remainder accuracy; the Detector uses only probe JSD and probe accuracy."]
    (run_root / "deployment_results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
