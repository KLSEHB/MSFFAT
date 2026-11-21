"""Print the paper's dependency-ordered matrix; --execute runs it in fresh paths."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

CONFIGS = (("full","proposed"),("ltf_only","proposed"),("msf_only","proposed"),
           ("full","independent"),("full","standard"),("full","concat"))


def build_commands(a):
    output = Path(a.output_root).resolve()
    script = Path(__file__).resolve().parent
    jobs = []
    def add(entry,*args):
        jobs.append([sys.executable,str(script/entry),*map(str,args),*(["--require-gpu"] if a.require_gpu else [])])
    external = []
    if a.external_specs:
        file = Path(a.external_specs).resolve()
        external = json.loads(file.read_text(encoding="utf-8"))["runs"]
        for item in external:
            item["spec"] = str((file.parent/item["spec"]).resolve())
    for seed in a.seeds:
        base = output/f"seed{seed}"
        master = base/"day0_full_proposed"
        for variant,fusion in CONFIGS:
            initial = base/f"day0_{variant}_{fusion}"
            args = ["--data-root",a.data_root,"--output-root",initial,"--seed",seed,"--length",5000,
                    "--variant",variant,"--fusion",fusion,"--refresh-reserve",10,"--train-day0-only"]
            if initial != master:
                args += ["--split-root",master]
            add("simulate_awf_time_deployment.py",*args)
            add("run_paired_adaptation.py","--data-root",a.data_root,"--split-root",master,
                "--model",initial/"models/day0.keras","--initial-split-root",initial,"--seed",seed,
                "--policy","static","--output-root",base/f"static_{variant}_{fusion}")
            if a.cw_root:
                for shots in ((5,10,20,50) if fusion=="proposed" else (10,)):
                    add("train_single.py","--data-root",a.cw_root,"--dataset","df-cw","--samples",shots,
                        "--classes",95,"--variant",variant,"--fusion",fusion,"--seed",seed,
                        "--output",base/f"cw_{variant}_{fusion}_{shots}/model.keras")
            if a.include_scale and fusion=="proposed":
                for sites in (100,200,500,900):
                    add("train_single.py","--data-root",a.data_root,"--dataset","awf-cw","--awf-part",sites,
                        "--awf-traces",2500,"--classes",sites,"--variant",variant,"--fusion",fusion,
                        "--seed",seed,"--output",base/f"scale_{variant}_{fusion}_{sites}/model.keras")
        for scope in ("attention","head","attention_head","full"):
            for budget in ((1,2,5,10) if scope == "attention" else (2,)):
                add("run_paired_adaptation.py","--data-root",a.data_root,"--split-root",master,
                    "--model",master/"models/day0.keras","--initial-split-root",master,"--seed",seed,
                    "--scope",scope,"--m-refresh",budget,"--output-root",base/f"paired_{scope}_m{budget}")
        for mode in ("accuracy_only","js_only","joint_fixed","joint_dynamic"):
            add("simulate_awf_time_deployment.py","--data-root",a.data_root,"--split-root",master,
                "--resume-model",master/"models/day0.keras","--resume-split-root",master,
                "--resume-epochs",0,"--refresh-reserve",10,"--detector-mode",mode,"--seed",seed,
                "--output-root",base/f"sequential_{mode}")
        add("collect_detector_cases.py","--data-root",a.data_root,"--model",master/"models/day0.keras",
            "--split-root",master,"--seed",seed,"--output-root",base/"frozen_replay")
        for index,item in enumerate(external):
            if item["seed"] != seed:
                continue
            scopes = item["scopes"]
            if not scopes or any(s not in ("head","published_partial","full") for s in scopes):
                raise ValueError("External scopes must name the original head/partial/full strategy")
            for scope in ["static",*scopes]:
                vary = scope == "published_partial" or scope == "full" and "published_partial" not in scopes
                for budget in ((1,2,5,10) if vary else (2,)):
                    add("run_paired_adaptation.py","--data-root",a.data_root,"--split-root",master,
                        "--baseline-spec",item["spec"],"--seed",seed,"--scope","full" if scope == "static" else scope,
                        "--policy","static" if scope == "static" else "fixed","--m-refresh",budget,
                        "--output-root",base/f"external{index}_{scope}_m{budget}")
    return jobs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root",required=True)
    p.add_argument("--output-root",required=True)
    p.add_argument("--cw-root",help="Raw DF data root for paired low-shot branch/fusion training")
    p.add_argument("--include-scale",action="store_true")
    p.add_argument("--seeds",type=int,nargs="+",default=[2024,2025,2026,2027,2028])
    p.add_argument("--external-specs",help="JSON runs list: seed/spec/scopes. Requires separately trained original baseline checkpoints.")
    p.add_argument("--require-gpu",action="store_true")
    p.add_argument("--execute",action="store_true")
    a = p.parse_args()
    out = Path(a.output_root)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a new output root; this matrix never resumes implicitly")
    if len(a.seeds) != len(set(a.seeds)):
        p.error("Duplicate seeds")
    jobs = build_commands(a)
    for command in jobs:
        print(subprocess.list2cmdline(command) if os.name == "nt" else shlex.join(command),flush=True)
        if a.execute:
            subprocess.run(command,check=True)


if __name__ == "__main__":
    main()
