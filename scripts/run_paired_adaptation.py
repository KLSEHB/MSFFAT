"""Equal-index static/fixed-schedule comparisons using complete checkpoints.

Initial training is a separate stage. All runs read the master Day0/temporal
indices; budget changes use nested prefixes of its reserved refresh pool.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import tensorflow as tf

from msffat.baseline_adapters import load_external_backend, verify_initial_split
from msffat.data import add_channel
from msffat.maintenance import attention_transfer_fit
from msffat.model import model_identity
from simulate_awf_time_deployment import DAYS, prepare_data, load_model, set_seed, file_sha256


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", required=True)
    p.add_argument("--split-root", required=True, help="Completed master split/run directory")
    p.add_argument("--output-root", required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", help="Complete aligned native Day0 checkpoint")
    group.add_argument("--baseline-spec", help="Original third-party model/config/provenance JSON")
    p.add_argument("--initial-split-root", help="Native checkpoint's original training split")
    p.add_argument("--scope", choices=["attention", "head", "attention_head", "published_partial", "full"], default="attention")
    p.add_argument("--policy", choices=["static", "fixed"], default="fixed")
    p.add_argument("--update-days", type=int, nargs="+", default=[14,28,42])
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--length", type=int, default=5000)
    p.add_argument("--n-probe", type=int, default=5)
    p.add_argument("--refresh-reserve", type=int, default=10)
    p.add_argument("--m-refresh", type=int, default=2)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--require-gpu", action="store_true")
    a = p.parse_args()
    if a.model and not a.initial_split_root:
        p.error("Native checkpoints require --initial-split-root for exact initial training provenance")
    if any(day not in dict(DAYS) for day in a.update_days) or len(set(a.update_days)) != len(a.update_days):
        p.error("update-days must be distinct AWF checkpoints")
    if a.model and a.scope == "published_partial":
        p.error("published_partial is an external-method scope")
    if not 1 <= a.m_refresh <= a.refresh_reserve or a.n_probe < 1 or a.n_probe + a.refresh_reserve >= 100:
        p.error("Invalid probe/refresh reservation")
    if a.epochs < 1 or a.batch_size < 1 or a.learning_rate <= 0:
        p.error("Invalid optimization budget")
    a.cohort_limit = None
    return a


def gpu_memory(reset=False):
    if not tf.config.list_physical_devices("GPU"):
        return None
    if reset:
        tf.config.experimental.reset_memory_stats("GPU:0")
    return tf.config.experimental.get_memory_info("GPU:0")["peak"]


def main():
    a = parse_args()
    set_seed(a.seed)
    out, shared = Path(a.output_root).resolve(), Path(a.split_root).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a new empty output directory")
    out.mkdir(parents=True, exist_ok=True)
    source = json.loads((shared / "run_manifest.json").read_text(encoding="utf-8"))
    if source.get("status") not in {"complete", "day0_complete"} or source.get("reduced_cohort"):
        raise ValueError("Master split must be a completed full-cohort run")
    cohort, base_x, base_y, base_parts, days = prepare_data(a, out)
    day0_x,day0_y = base_x[base_parts["remainder"]],base_y[base_parts["remainder"]]
    del base_x, base_y, base_parts
    raw = Path(a.data_root)
    if not (raw / "tor_200w_2500tr_new.npz").exists():
        raw /= "dlwf"
    hashes = {p.name: file_sha256(p) for p in [raw / "tor_200w_2500tr_new.npz"] +
              [raw / f"tor_200w_100tr_time_test{s}.npz" for _, s in DAYS]}
    if hashes != source.get("data_sha256"):
        raise ValueError("Shared splits belong to different dataset content")
    if a.baseline_spec:
        backend, provenance = load_external_backend(a.baseline_spec, classes=len(cohort), length=a.length,
                                                    cohort=cohort, shared_root=shared)
        predict = backend.predict
        fit = lambda x,y,seed: backend.fit_refresh(x,y,scope=a.scope,epochs=a.epochs,
                                                  batch_size=a.batch_size,learning_rate=a.learning_rate,seed=seed)
        save = backend.save
        identity = {"method": provenance["spec"]["method"], "external_complete_checkpoint": True}
        tf_backend = getattr(backend,"framework",None) == "tensorflow"
    else:
        verify_initial_split({"class_order": cohort, "initial_split_root": a.initial_split_root}, Path.cwd(), shared, cohort)
        initial_record = json.loads((Path(a.initial_split_root)/"run_manifest.json").read_text(encoding="utf-8"))
        if initial_record.get("method_version") != "MSFFAT-v3":
            raise ValueError("Native paired adaptation requires a v3 initial run")
        if initial_record.get("day0_model_sha256") != file_sha256(a.model):
            raise ValueError("Native checkpoint is not the recorded Day0 model of its training split")
        model = load_model(Path(a.model))
        identity = model_identity(model)
        if identity["classes"] != len(cohort) or identity["length"] != a.length or identity["mode"] != "single":
            raise ValueError("Initial model does not match the common task")
        provenance = {"checkpoint_sha256": file_sha256(a.model), "initial_split_root": a.initial_split_root}
        predict = lambda x: model.predict(add_channel(x), batch_size=256, verbose=0)
        fit = lambda x,y,seed: attention_transfer_fit(model,x,y,scope=a.scope,epochs=a.epochs,
                                                      batch_size=a.batch_size,learning_rate=a.learning_rate,seed=seed)
        save = lambda path: model.save(path)
        tf_backend = True
    if a.require_gpu:
        if tf_backend:
            if not tf.config.list_physical_devices("GPU"):
                raise RuntimeError("TensorFlow found no GPU")
        elif callable(getattr(backend,"assert_gpu",None)):
            backend.assert_gpu()
        else:
            raise ValueError("A custom backend must implement assert_gpu to verify --require-gpu")
    def accuracy(x,y):
        values = np.asarray(predict(x))
        if values.shape != (len(y),len(cohort)) or not np.isfinite(values).all():
            raise ValueError("Prediction/class-order contract violated")
        return float(np.mean(values.argmax(axis=1) == y))
    initial_accuracy = accuracy(day0_x,day0_y)
    del day0_x,day0_y
    root = Path(__file__).resolve().parents[1]
    manifest = {"method_version": "MSFFAT-v3", "status": "running", "experiment": "paired_adaptation",
                "config": vars(a), "model_identity": identity, "provenance": provenance,
                "tensorflow": tf.__version__, "devices": [d.name for d in tf.config.list_logical_devices()],
                "data_sha256": hashes, "cohort_size": len(cohort),
                "source_sha256": {p.relative_to(root).as_posix():file_sha256(p) for folder in ("msffat","scripts") for p in (root/folder).glob("*.py")},
                "split_sha256": {p.name:file_sha256(p) for p in out.glob("*indices.npz")}}
    (out/"run_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    rows = []
    for day,_ in DAYS:
        data = days[day]
        pre = accuracy(data["remainder_x"],data["remainder_y"])
        probe = accuracy(data["probe_x"],data["probe_y"])
        updated = a.policy == "fixed" and day in a.update_days
        row = {"day":day,"pre_accuracy":pre,"post_accuracy":pre,"probe_accuracy":probe,"post_probe_accuracy":probe,"update_executed":updated,
               "runtime_seconds":None,"peak_gpu_bytes":None,"fit":None}
        if updated:
            set_seed(a.seed+day)
            if tf_backend:
                gpu_memory(reset=True)
            started = time.perf_counter()
            row["fit"] = fit(data["refresh_x"],data["refresh_y"],a.seed+day)
            row["runtime_seconds"] = time.perf_counter()-started
            row["peak_gpu_bytes"] = gpu_memory() if tf_backend else row["fit"].get("peak_gpu_bytes")
            row["memory_backend"] = "tensorflow_allocator" if tf_backend else row["fit"].get("memory_backend","not_reported")
            row["post_accuracy"] = accuracy(data["remainder_x"],data["remainder_y"])
            row["post_probe_accuracy"] = accuracy(data["probe_x"],data["probe_y"])
            save(out/f"day{day}.keras")
        rows.append(row)
        (out/"paired_progress.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")
    result = {"method_version":"MSFFAT-v3","status":"complete","result_kind":"measured_run","experiment":"paired_adaptation",
              "model_identity":identity,"seed":a.seed,"scope":a.scope,"policy":a.policy,
              "day0_remainder_accuracy":initial_accuracy,
              "optimization":{"epochs":a.epochs,"batch_size":a.batch_size,"learning_rate":a.learning_rate},
              "update_days":a.update_days,
              "refresh_per_site_per_update":a.m_refresh,"reserved_pool_per_site":a.refresh_reserve,
              "reserved_refresh_traces_per_site":a.refresh_reserve*len(DAYS),
              "refresh_labels_per_site":a.m_refresh*sum(r["update_executed"] for r in rows),"rows":rows}
    result["probe_labels_per_site"] = a.n_probe*len(DAYS)
    result["maintenance_labels_per_site"] = result["probe_labels_per_site"]+result["refresh_labels_per_site"]
    (out/"paired_result.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    manifest["status"] = "complete"
    (out/"run_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
