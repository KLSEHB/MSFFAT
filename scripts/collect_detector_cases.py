"""Measure predeclared frozen-model temporal/localized/probe/tail-mask cases.

Intervention names do not assign oracle labels. No case is selected or discarded
based on its measured JS or accuracy. Bootstrap cases reuse traces and are
reported as dependent observations, not independent traffic collections.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from msffat.data import add_channel, pad_or_truncate
from msffat.maintenance import sequence_symbol_distribution, sequence_jsd
from msffat.replay import replay_trajectories
from simulate_awf_time_deployment import DAYS, load_model, load_npz, file_sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root",required=True)
    p.add_argument("--model",required=True)
    p.add_argument("--split-root",required=True)
    p.add_argument("--cohort-json",help="Optional compatibility check against master cohort")
    p.add_argument("--output-root",required=True)
    p.add_argument("--seed",type=int,default=2024)
    p.add_argument("--replicates",type=int,default=10)
    p.add_argument("--batch-size",type=int,default=256)
    p.add_argument("--require-gpu",action="store_true")
    a = p.parse_args()
    if a.replicates < 1:
        p.error("replicates must be positive")
    import tensorflow as tf
    if a.require_gpu and not tf.config.list_physical_devices("GPU"):
        raise RuntimeError("TensorFlow found no GPU")
    root,out,raw = Path(a.split_root),Path(a.output_root),Path(a.data_root)
    if not (raw/"tor_200w_2500tr_new.npz").exists():
        raw /= "dlwf"
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a new empty output directory")
    out.mkdir(parents=True,exist_ok=True)
    manifest = json.loads((root/"run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("method_version") != "MSFFAT-v3":
        raise ValueError("Detector cases require the v3 initial model and manifest")
    if manifest.get("status") not in {"complete","day0_complete"} or manifest.get("reduced_cohort"):
        raise ValueError("Require a completed full-cohort master run")
    for filename,digest in manifest["split_sha256"].items():
        if file_sha256(root/filename) != digest:
            raise ValueError("Saved probe/remainder split changed after initial training")
    if file_sha256(a.model) != manifest.get("day0_model_sha256"):
        raise ValueError("Replay requires the recorded unchanged Day0 model")
    sites = json.loads((root/"cohort.json").read_text(encoding="utf-8"))["sites"]
    if a.cohort_json and json.loads(Path(a.cohort_json).read_text(encoding="utf-8"))["sites"] != sites:
        raise ValueError("Class order differs from the shared cohort")
    mapping = {s:i for i,s in enumerate(sites)}
    model = load_model(Path(a.model))
    from msffat.model import validate_aligned_model
    validate_aligned_model(model)
    if model.output_shape[-1] != len(sites):
        raise ValueError("Checkpoint class count differs")
    length = model.input_shape[1]
    data = {}
    hashes = {}
    for day,suffix in ((0,None),)+DAYS:
        filename = "tor_200w_2500tr_new.npz" if day == 0 else f"tor_200w_100tr_time_test{suffix}.npz"
        path = raw/filename
        hashes[filename] = file_sha256(path)
        if hashes[filename] != manifest["data_sha256"][filename]:
            raise ValueError("Dataset content differs from the saved split")
        x,y = load_npz(path)
        with np.load(root/f"day{day}_indices.npz") as saved:
            probe,remainder = saved["probe"],saved["remainder"]
            if set(probe)&set(remainder) or len(np.unique(np.r_[probe,remainder])) != len(probe)+len(remainder):
                raise ValueError("Probe and remainder overlap")
            data[day] = {"probe_x":pad_or_truncate(x[probe],length),
                         "probe_y":np.array([mapping[v] for v in y[probe]]),
                         "remainder_x":pad_or_truncate(x[remainder],length),
                         "remainder_y":np.array([mapping[v] for v in y[remainder]])}
        del x,y
    def acc(x,y):
        return float(np.mean(model.predict(add_channel(x),batch_size=a.batch_size,verbose=0).argmax(axis=1)==y))
    reference = data[0]
    reference_probe = acc(reference["probe_x"],reference["probe_y"])
    def remainder_stats(case):
        correct = model.predict(add_channel(case["remainder_x"]),batch_size=a.batch_size,verbose=0).argmax(axis=1)==case["remainder_y"]
        return float(correct.mean()),[float(correct[case["remainder_y"]==label].mean()) for label in range(len(sites))]
    reference_remainder,reference_sites = remainder_stats(reference)
    reference_dist = sequence_symbol_distribution(reference["probe_x"])
    def event(case,checkpoint):
        probe = acc(case["probe_x"],case["probe_y"])
        remainder,per_site = remainder_stats(case)
        return {"checkpoint":checkpoint,"jsd":sequence_jsd(sequence_symbol_distribution(case["probe_x"]),reference_dist),
                "probe_accuracy":probe,"reference_probe_accuracy":reference_probe,
                "remainder_accuracy":remainder,"remainder_drop_pp":100*(reference_remainder-remainder),
                "per_site_remainder_accuracy":per_site,
                "per_site_drop_pp":[100*(r-c) for r,c in zip(reference_sites,per_site)]}
    trajectories = [{"id":"temporal","family":"temporal","events":[event(data[d],d) for d,_ in DAYS]}]
    # Interventions are fixed before model evaluation, not tuned to obtain a
    # harmless shift or a low-JS positive case. Oracle outcomes stay empirical.
    rng = np.random.default_rng(a.seed)
    for rep in range(a.replicates):
        order = rng.permutation(len(sites))
        for ratio in (.05,.20,.50):
            chosen = order[:max(1,round(ratio*len(sites)))]
            mixed = {}
            for part in ("probe","remainder"):
                pieces_x,pieces_y = [],[]
                for label in range(len(sites)):
                    source = data[42] if label in chosen else data[3]
                    keep = source[f"{part}_y"] == label
                    pieces_x.append(source[f"{part}_x"][keep])
                    pieces_y.append(source[f"{part}_y"][keep])
                mixed[f"{part}_x"],mixed[f"{part}_y"] = np.concatenate(pieces_x),np.concatenate(pieces_y)
            trajectories.append({"id":f"localized-{rep}-{ratio}","family":"localized","sites":chosen.tolist(),
                                 "events":[event(mixed,ratio)]})
        boot = []
        for day,_ in DAYS:
            source = data[day]
            idx = np.concatenate([rng.choice(np.flatnonzero(source["probe_y"]==label),
                                           size=np.sum(source["probe_y"]==label),replace=True) for label in range(len(sites))])
            case = {**source,"probe_x":source["probe_x"][idx],"probe_y":source["probe_y"][idx]}
            boot.append(event(case,day))
        trajectories.append({"id":f"probe-bootstrap-{rep}","family":"probe_bootstrap","events":boot})
    for fraction in (.001,.01,.05):
        case = {k:v.copy() for k,v in data[3].items()}
        for part in ("probe_x","remainder_x"):
            case[part][:,-max(1,round(length*fraction)):] = 0
        trajectories.append({"id":f"tail-mask-{fraction}","family":"tail_mask","events":[event(case,fraction)]})
    prefix = file_sha256(a.model)[:16]+f"-seed{a.seed}-"
    for trajectory in trajectories:
        trajectory["id"] = prefix+trajectory["id"]
    payload = {"method_version":"MSFFAT-v3","result_kind":"measured_frozen_replay_inputs","config":vars(a),
               "model_sha256":file_sha256(a.model),"data_sha256":hashes,
               "split_sha256":{p.name:file_sha256(p) for p in root.glob("*indices.npz")},
               "dependent_reuse":True,"reference_probe_accuracy":reference_probe,
               "class_order":sites,"reference_per_site_remainder_accuracy":reference_sites,
               "reference_remainder_accuracy":reference_remainder,"trajectories":trajectories}
    (out/"frozen_cases.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    result = {"method_version":"MSFFAT-v3","result_kind":"measured_detector_replay","config":{"q":.04,"w_pp":2.,"e":.005,"boost_cap":.02,"boost_decay":.005,"oracle_pp":2.},
              "input_sha256":file_sha256(out/"frozen_cases.json"),"policies":replay_trajectories(trajectories)}
    (out/"replay_result.json").write_text(json.dumps(result,indent=2),encoding="utf-8")


if __name__ == "__main__":
    main()
