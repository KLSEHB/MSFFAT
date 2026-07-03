# Experiment Mapping

This document maps paper-side MSFFAT experiments to cleaned scripts.

| Paper experiment | Script | Notes |
| --- | --- | --- |
| DF-CW low-shot | `scripts/train_single.py --dataset df-cw --samples K --classes 95` | Repeat `K in {5,10,20,50}`. |
| DF-OW low-shot | `scripts/train_open_world.py --samples K` | Verify output class encoding locally. |
| AWF-Time static drift | `scripts/evaluate_single.py --dataset awf-time --suffix S` | Requires a trained AWF-CW200 model. |
| AWF-Time stateful trigger/ATF | `scripts/simulate_awf_time_deployment.py` | Final aggregate probe-accuracy detector with stateful updates. |
| Localized drift | `scripts/localized_drift_stress_test.py` | Site-level Day-42 replacement stress test; does not execute ATF. |
| Sequence-level JSD | `scripts/compute_jsd.py --suffix S` | Computes averaged position-wise JS divergence. |
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
