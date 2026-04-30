# Experiment Mapping

This document maps paper-side MSFFAT experiments to cleaned scripts.

| Paper experiment | Script | Notes |
| --- | --- | --- |
| DF-CW low-shot | `scripts/train_single.py --dataset df-cw --samples K --classes 95` | Repeat `K in {5,10,20,50}`. |
| DF-OW low-shot | `scripts/train_open_world.py --samples K` | Verify output class encoding locally. |
| AWF-Time static drift | `scripts/evaluate_single.py --dataset awf-time --suffix S` | Requires a trained AWF-CW200 model. |
| AWF-Time ATF | `scripts/finetune_atf.py --suffix S --traces 2` | Applies ATF at a selected interval. |
| Drift monitoring | `scripts/monitor_and_adapt.py` | Implements calibration-based trigger logic. |
| Sequence-level JSD | `scripts/compute_jsd.py --suffix S` | Computes averaged position-wise JS divergence. |
| Large monitored sets | `scripts/train_single.py --dataset awf-cw --awf-part N` | Repeat `N in {100,200,500,900}`. |
| Multi-tab | `scripts/train_multitab.py --tabs K` | Repeat `K in {2,3,4,5}`. |
| Model size | `scripts/model_summary.py` | Prints model parameter count. |

Suffix mapping for AWF-Time:

- `3d`: 3 days
- `10d`: 10 days
- `2w`: 14 days
- `4w`: 28 days
- `6w`: 42 days

