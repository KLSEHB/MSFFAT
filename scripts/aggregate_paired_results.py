"""Aggregate only actual paired-run records with matching per-seed test indices."""
import argparse
import json
from pathlib import Path
import numpy as np

from msffat.baseline_adapters import sha256
from msffat.results import aggregate_paired


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results",nargs="+",required=True,help="Explicit paired_result.json paths")
    p.add_argument("--output",required=True)
    p.add_argument("--seeds",type=int,nargs="+",default=[2024,2025,2026,2027,2028])
    a = p.parse_args()
    out = Path(a.output)
    if out.exists():
        raise FileExistsError("Aggregation output must be new")
    records,master,source_hashes = [],{},{}
    protocol = None
    for name in a.results:
        path = Path(name).resolve()
        rec = json.loads(path.read_text(encoding="utf-8"))
        manifest = json.loads((path.parent/"run_manifest.json").read_text(encoding="utf-8"))
        if manifest.get("status") != "complete" or manifest.get("reduced_cohort"):
            raise ValueError("Incomplete/engineering run manifest")
        if manifest.get("config",{}).get("seed") != rec.get("seed"):
            raise ValueError("Result seed differs from its run manifest")
        sources = {name.replace("\\","/"):value for name,value in manifest["source_sha256"].items()}
        current = {name:sources[name] for name in
                   ("msffat/maintenance.py","msffat/deployment.py","scripts/run_paired_adaptation.py")}
        if protocol is not None and current != protocol:
            raise ValueError("Paired runs used different protocol source versions")
        protocol = current
        fingerprints = {}
        for filename,digest in manifest["split_sha256"].items():
            part = path.parent/filename
            if sha256(part) != digest:
                raise ValueError("Saved split was modified after the run")
            with np.load(part) as arrays:
                fingerprints[filename] = {k:arrays[k].tobytes() for k in arrays.files if k in ("train","valid","probe","refresh_pool","remainder")}
        identity = (manifest["data_sha256"],json.loads((path.parent/"cohort.json").read_text(encoding="utf-8")),fingerprints)
        seed = rec["seed"]
        if seed in master and master[seed] != identity:
            raise ValueError("Methods/budgets do not share the same per-seed held-out indices/data/cohort")
        master[seed] = identity
        source_hashes[str(path)] = sha256(path)
        records.append(rec)
    result = {"method_version":"MSFFAT-v3","result_kind":"aggregate_of_measured_paired_runs","input_sha256":source_hashes,
              "groups":aggregate_paired(records,a.seeds)}
    reference = next((g for g in result["groups"] if (g["method"],g["variant"],g["fusion"],g["scope"],g["policy"],g["refresh"]) ==
                      ("MSFFAT","full","proposed","attention","fixed",2)),None)
    result["paired_day42_differences"] = []
    if reference:
        for other in result["groups"]:
            if other is reference or other["policy"]!="fixed" or other["refresh"]!=2:
                continue
            values = np.array([reference["per_seed_day42"][str(s)]-other["per_seed_day42"][str(s)] for s in a.seeds])
            result["paired_day42_differences"].append({"reference":"MSFFAT/full/proposed/attention",
                "other":{k:other[k] for k in ("method","variant","fusion","scope")},"n":len(values),
                "mean_accuracy_difference":float(values.mean()),"std":float(values.std(ddof=1)) if len(values)>1 else None})
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2),encoding="utf-8")


if __name__ == "__main__":
    main()
