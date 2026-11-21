"""JS diagnostic using the exact saved reference/current probe indices."""
import argparse
import json
from pathlib import Path
import numpy as np
from msffat.data import pad_or_truncate
from msffat.maintenance import sequence_jsd, sequence_symbol_distribution
from simulate_awf_time_deployment import DAYS, file_sha256, load_npz


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root",required=True)
    p.add_argument("--split-root",required=True)
    p.add_argument("--suffix",required=True,choices=[s for _,s in DAYS])
    p.add_argument("--reference-day",type=int,default=0,choices=[0,*[d for d,_ in DAYS]])
    a = p.parse_args()
    root,raw = Path(a.split_root),Path(a.data_root)
    if not (raw/"tor_200w_2500tr_new.npz").exists():
        raw /= "dlwf"
    manifest = json.loads((root/"run_manifest.json").read_text(encoding="utf-8"))
    for filename,digest in manifest["split_sha256"].items():
        if file_sha256(root/filename) != digest:
            raise ValueError("Saved probe indices changed after they were recorded")
    length = manifest["config"]["length"]
    def distribution(day):
        filename = "tor_200w_2500tr_new.npz" if day==0 else f"tor_200w_100tr_time_test{dict(DAYS)[day]}.npz"
        path = raw/filename
        if file_sha256(path) != manifest["data_sha256"][filename]:
            raise ValueError("Dataset differs from saved probe indices")
        x,_ = load_npz(path)
        with np.load(root/f"day{day}_indices.npz") as saved:
            return sequence_symbol_distribution(pad_or_truncate(x[saved["probe"]],length))
    day = {suffix:day for day,suffix in DAYS}[a.suffix]
    print(json.dumps({"protocol":"saved_probe_reference","reference_day":a.reference_day,"day":day,
                      "d_seq":sequence_jsd(distribution(day),distribution(a.reference_day))}))


if __name__ == "__main__":
    main()
