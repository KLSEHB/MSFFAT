# Experiment Mapping

This document maps paper-side MSFFAT experiments to cleaned scripts.

| Paper experiment | Script | Notes |
| --- | --- | --- |
| DF-CW low-shot | `scripts/train_single.py --dataset df-cw --samples K --classes 95` | Repeat `K in {5,10,20,50}`. |
| Operational OW | `prepare_wfl_ow.py`, `train_open_world_kplus1.py`, `predict_msffat_ow.py`, `evaluate_base_rate_ow_threshold.py` | Validation Max-F1 binary detection; site accuracy reported separately. |
| AWF-Time static drift | `scripts/run_paired_adaptation.py --policy static` | Recorded Day0 model/common-cohort mapping; same85-trace remainder as adaptation. |
| AWF-Time stateful trigger/ATF | `scripts/simulate_awf_time_deployment.py` | v3 JS-first detector with bounded temporary boost and decay; shared-MLP adaptation; probe labels queried only after JS gate; 2 refresh traces/site/update; evaluation-only fixed oracle. |
| Frozen detector cases | `scripts/collect_detector_cases.py`, `scripts/replay_detectors.py` | Four policies, no fitting; temporal/localized/bootstrap/tail-mask cases; dependent reuse disclosed. |
| Sequence-level JSD | `scripts/compute_jsd.py --split-root MASTER --suffix S` | Same saved probes, log2 and position-wise averaging as maintenance; reference day is explicit. |
| WFL-CW few-shot | `scripts/prepare_wfl_cw.py`, `scripts/run_cw_fewshot.py` | Deterministic 5/10/20/50-shot protocol. |
| WFL-OW 95+1 | `scripts/prepare_wfl_ow.py`, `scripts/train_open_world_kplus1.py` | K+1 training and open-world metrics. |
| Large monitored sets | `scripts/train_single.py --dataset awf-cw --awf-part N` | Repeat `N in {100,200,500,900}`. |
| Multi-tab | `scripts/train_multitab.py --tabs K` | Repeat `K in {2,3,4,5}`. |
| Model size | `scripts/model_summary.py` | Prints model parameter count. |
| Inference throughput | `scripts/benchmark_a40_throughput.py` | Peak throughput and maximum feasible batch size. |
| Unified latency/throughput | `scripts/benchmark_table10_torch.py` | Batch-1 latency, batch-128 throughput, and peak stable throughput. |

Suffix mapping for AWF-Time:

- `3d`: 3 days
- `10d`: 10 days
- `2w`: 14 days
- `4w`: 28 days
- `6w`: 42 days


The scope/budget/baseline controls use `run_paired_adaptation.py`.
The complete native grid is printed/executed by `run_paper_experiments.py`.
See EXPERIMENT_PROTOCOL.md for the paired data, optimization and baseline interfaces.
`evaluate_single.py` and `finetune_atf.py` remain independent-task tools; they do not
reproduce the common-cohort sequential or matched-budget protocol by themselves.
