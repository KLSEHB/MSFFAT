"""Replay all detector policies on measured, unchanged model histories."""
import argparse
import hashlib
import json
from pathlib import Path

from msffat.replay import replay_trajectories
from msffat.baseline_adapters import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", nargs="+", required=True, help="Frozen replay inputs or never-update deployment_result.json")
    p.add_argument("--output", required=True)
    p.add_argument("--q", type=float, default=.04)
    p.add_argument("--w-pp", type=float, default=2.)
    p.add_argument("--e", type=float, default=.005)
    p.add_argument("--boost-cap", type=float, default=.02)
    p.add_argument("--boost-decay", type=float, default=.005)
    p.add_argument("--oracle-pp", type=float, default=2.)
    a = p.parse_args()
    target = Path(a.output)
    if target.exists():
        raise FileExistsError("Replay output must be new")
    trajectories, hashes = [], {}
    for filename in a.inputs:
        path = Path(filename).resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("method_version") != "MSFFAT-v3":
            raise ValueError("Use v3 frozen-model inputs; older model histories are a separate comparison")
        hashes[str(path)] = sha256(path)
        if payload.get("result_kind") == "measured_frozen_replay_inputs":
            trajectories.extend(payload["trajectories"])
        elif payload.get("result_kind") == "measured_run" and payload.get("update_policy") == "never":
            if any(row["probe_accuracy"] is None for row in payload["rows"]):
                raise ValueError("Gated deployment omits some probe accuracies; use complete offline collect_detector_cases records for all-policy replay")
            trajectories.append({"id":str(path),"events":[
                {"checkpoint":r["day"],"jsd":r["jsd"],"probe_accuracy":r["probe_accuracy"],
                 "reference_probe_accuracy":r["probe_reference_accuracy"],"remainder_drop_pp":r["remainder_drop_pp"]}
                for r in payload["rows"]]})
        else:
            raise ValueError("Only measured frozen histories are accepted; target tables/stateful adaptive histories are incompatible")
    if len({t["id"] for t in trajectories}) != len(trajectories):
        raise ValueError("Duplicate trajectory IDs")
    result = {"method_version":"MSFFAT-v3","result_kind":"measured_detector_replay","config":vars(a),"input_sha256":hashes,
              "trajectory_count":len(trajectories),"policies":replay_trajectories(
                  trajectories,q=a.q,w_pp=a.w_pp,e=a.e,boost_cap=a.boost_cap,decay=a.boost_decay,oracle_pp=a.oracle_pp)}
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result,indent=2),encoding="utf-8")


if __name__ == "__main__":
    main()
