"""Aggregation of paired measured runs; targets and incomplete grids fail closed."""
from collections import defaultdict
import numpy as np


def aggregate_paired(records, expected_seeds=(2024,2025,2026,2027,2028)):
    if not expected_seeds or len(expected_seeds)!=len(set(expected_seeds)):
        raise ValueError("Expected seeds must be nonempty and unique")
    groups = defaultdict(dict)
    for record in records:
        if record.get("result_kind") != "measured_run" or record.get("status") != "complete" or record.get("experiment") != "paired_adaptation":
            raise ValueError("Expected completed measured paired runs, not targets/fixtures")
        if record.get("method_version") != "MSFFAT-v3":
            raise ValueError("Expected v3 records; independent-MLP v2 results cannot be relabeled as shared-MLP results")
        identity = record["model_identity"]
        if identity.get("engineering_fixture"):
            raise ValueError("Engineering results cannot enter a paper table")
        method = identity.get("method","MSFFAT")
        key = (method,identity.get("variant"),identity.get("fusion"),record["scope"],record["policy"],record["refresh_per_site_per_update"])
        seed = record["seed"]
        if seed in groups[key]:
            raise ValueError("Duplicate run for the same method/configuration/seed")
        groups[key][seed] = record
    def stats(values):
        values = np.asarray(values,dtype=float)
        if not len(values) or not np.isfinite(values).all():
            raise ValueError("Invalid measured values")
        return {"n":len(values),"mean":float(values.mean()),"std":float(values.std(ddof=1)) if len(values)>1 else None}
    output = []
    for key,runs in sorted(groups.items(),key=lambda item:str(item[0])):
        if set(runs) != set(expected_seeds):
            raise ValueError(f"Incomplete seed grid for {key}: {sorted(runs)}")
        ordered = [runs[seed] for seed in expected_seeds]
        if any(r.get("optimization") != ordered[0].get("optimization") or
               r.get("update_days") != ordered[0].get("update_days") or
               r.get("reserved_pool_per_site") != ordered[0].get("reserved_pool_per_site") for r in ordered):
            raise ValueError("Optimization/schedule/reservation differs across paired seeds")
        if any([r["day"] for r in rec["rows"]] != [3,10,14,28,42] for rec in ordered):
            raise ValueError("Incomplete/nonmatching checkpoint grid")
        rows = []
        for index,day in enumerate((3,10,14,28,42)):
            rows.append({"day":day,"pre_accuracy":stats([r["rows"][index]["pre_accuracy"] for r in ordered]),
                         "post_accuracy":stats([r["rows"][index]["post_accuracy"] for r in ordered])})
        per_run_time = [[row["runtime_seconds"] for row in rec["rows"] if row["update_executed"]] for rec in ordered]
        summary = {"method":key[0],"variant":key[1],"fusion":key[2],"scope":key[3],"policy":key[4],"refresh":key[5],
                   "seeds":list(expected_seeds),"day0_accuracy":stats([r["day0_remainder_accuracy"] for r in ordered]),
                   "days":rows,"mean_seconds_per_update":stats([np.mean(v) for v in per_run_time]) if all(per_run_time) else None,
                   "per_seed_day42":{str(s):runs[s]["rows"][-1]["post_accuracy"] for s in expected_seeds}}
        output.append(summary)
    if not output:
        raise ValueError("No measured runs")
    return output
