# Next-Gen Website Fingerprinting: Sample-Efficient Modeling and Attention Transfer for Robust Performance under Concept Drift

Official implementation of **MSFFAT** (Multi-Scale Spatiotemporal Feature Fusion
with Attention Transfer).

MSFFAT uses a shared attention MLP to generate separate channel weights for its
LTF and MSF branches. A JS-gated accuracy check controls attention-only updates.
The architecture and detector are specified in [docs/METHOD_SPEC.md](docs/METHOD_SPEC.md).
The paired experiment protocol is described in [docs/EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md).

This repository provides an implementation of MSFFAT and reproducible entry
points for website-fingerprinting experiments.

The code supports native MSFFAT experiments and matched adaptation of explicitly supplied complete external models:

- data-scarce closed-world training;
- K+1 open-world training and validation-threshold operational evaluation;
- large monitored-set training on AWF-CW subsets;
- temporal concept-drift evaluation;
- stateful joint JS/accuracy monitoring with JS-gated accuracy checks and a capped, decaying temporary threshold boost;
- attention-only adaptation, with explicit head/attention+head/full ablation scopes;
- localized-drift stress testing;
- GPU inference latency, throughput, and feasible-batch-size benchmarking;
- sequence-level JS divergence analysis;
- multi-tab concurrent traffic with a sigmoid multi-label head.

Third-party baseline repositories are **not** bundled here. Lightweight AWF
variants and evaluation wrappers used by these experiments are included;
external methods must still be obtained from their original releases.

## Repository Layout

```text
.
├── msffat/                  # MSFFAT model, data loaders, metrics, maintenance helpers
├── scripts/                 # command-line entry points
├── tests/                   # detector/splitting regression tests
├── configs/                 # example dataset/config files
├── docs/                    # data and experiment documentation
├── requirements.txt         # Python dependencies
├── pyproject.toml           # package metadata
└── README.md
```

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Install the optional PyTorch dependency for throughput comparisons:

```bash
pip install -e ".[benchmarks]"
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

## Data

Raw datasets are not included.  Put datasets under a local directory and pass it with `--data-root`, or set:

```bash
export MSFFAT_DATA_ROOT=/path/to/data
```

See [docs/DATA.md](docs/DATA.md) for the expected layout.

## Feature Visualization Export

Export native branch activations, branch-specific channel gates from the shared MLP and weighted fusion
from a complete MSFFAT v3 checkpoint, using predetermined sample indices:

```powershell
python scripts/export_feature_visualization.py --model checkpoint.keras --inputs prepared_inputs.npz --x-key x --indices 0 1 --output feature_export.npz
```

Inputs must contain canonical signed traces of length 5,000. The exporter uses
inference mode, verifies the gate multiplication and channel concatenation, and
records checkpoint/input hashes and sample indices in a companion JSON file.
It requires a new output path and does not train or select examples by accuracy.

## Quick Sanity Checks

Check Python syntax:

```bash
python -m py_compile msffat/*.py scripts/*.py
```

Run the engineering checks (including topology, serialization, frozen-weight
and sample-budget checks when TensorFlow is installed):

```bash
python -m unittest discover -s tests -v
```

Check whether the expected dataset files are present:

```bash
python scripts/check_dataset.py --data-root /path/to/data --profile core
```

Print a model summary:

```bash
python scripts/model_summary.py --classes 95 --length 5000 --mode single
```

## Inference Throughput

Measure MSFFAT peak throughput and maximum feasible batch size with synthetic
inputs on the current GPU:

```bash
python scripts/benchmark_a40_throughput.py \
  --method MSFFAT \
  --msffat-backend tensorflow \
  --seq-len 5000 --num-classes 100 \
  --start-batch 1 --max-batch 65536 \
  --binary-search \
  --output results/gpu_throughput.csv
```

External baselines are loaded from a local Website-Fingerprinting-Library
checkout supplied explicitly:

```bash
python scripts/benchmark_a40_throughput.py \
  --method DF \
  --baseline-root /path/to/Website-Fingerprinting-Library \
  --output results/gpu_throughput.csv
```

The unified Table-10-style benchmark reports batch-1 latency, batch-128
throughput, peak stable throughput, parameter count, and GPU model:

```bash
python scripts/benchmark_table10_torch.py \
  --method AWF-CNN --classes 100 \
  --wfl-root /path/to/Website-Fingerprinting-Library \
  --output results/table10_throughput.csv
```

## Paired Paper Evaluation

[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) maps the experiments to their command-line entry points.
Print the dependency-ordered, five-seed grid before executing it:

```bash
python scripts/run_paper_experiments.py --data-root /path/to/data --cw-root /path/to/data --include-scale --output-root results/new-paper-grid
```

Add `--execute --require-gpu` to run the grid. It trains each branch/fusion variant
from scratch; all temporal runs use a shared five-probe, ten-candidate refresh pool
and 85-trace remainder per site/checkpoint. Budgets use nested refresh prefixes.
All output paths must be new. Existing experiments/checkpoints are never changed.

For an individual matched run, starting from a recorded full-cohort master:

```bash
python scripts/run_paired_adaptation.py --data-root /path/to/data --split-root results/master --model results/master/models/day0.keras --initial-split-root results/master --scope attention --m-refresh 2 --output-root results/new-paired
```

Use `--scope head/attention_head/full`, `--policy static`, or
`--baseline-spec original-method.json` for the named controls. An external spec
binds a complete original model, original preprocessing/revision, class order,
initial train/validation indices and trainable parameter names. See
[configs/external-baseline.example.json](configs/external-baseline.example.json).
For non-Keras models, supply the original-model adapter through `backend_factory`.

`aggregate_paired_results.py` summarizes explicit run-result paths across paired seeds, using the common test indices and class order.

## Main Commands

Run commands from the repository root.

### Data-Scarce Closed World

```bash
python scripts/train_single.py \
  --data-root /path/to/data \
  --dataset df-cw \
  --samples 10 \
  --classes 95 \
  --output models/msffat_df_cw_10shot.hdf5
```

Repeat `--samples` with `5`, `10`, `20`, and `50`.

For Website-Fingerprinting-Library closed-world splits:

```bash
python scripts/prepare_wfl_cw.py \
  --dataset-dir /path/to/WFL/CW \
  --output-dir /path/to/prepared-cw
python scripts/run_cw_fewshot.py \
  --prepared-root /path/to/prepared-cw \
  --output-root results/wfl-cw \
  --require-gpu
```

### Open World

```bash
python scripts/train_open_world.py \
  --data-root /path/to/data \
  --samples 10 \
  --classes 96 \
  --output models/msffat_df_ow_10shot.hdf5
```

Set `--classes` and label encoding to match the prepared dataset.

The reproducible WFL 95+1 protocol is implemented separately:

```bash
python scripts/prepare_wfl_ow.py \
  --dataset-dir /path/to/WFL/OW \
  --output-dir /path/to/prepared-ow \
  --unmonitored-shots -1
python scripts/train_open_world_kplus1.py \
  --prepared-root /path/to/prepared-ow \
  --samples 5 \
  --output-dir results/wfl-ow/5shot \
  --require-gpu
```

### Large Monitored Sets

```bash
python scripts/train_single.py \
  --data-root /path/to/data \
  --dataset awf-cw \
  --awf-part 200 \
  --awf-traces 2500 \
  --classes 200 \
  --output models/msffat_awf_cw200.hdf5
```

Repeat with `--awf-part 100`, `200`, `500`, and `900`, changing `--classes` accordingly.

### Temporal Drift Evaluation

Evaluate a static model:

```bash
python scripts/evaluate_single.py \
  --data-root /path/to/data \
  --model models/msffat_awf_cw200.hdf5 \
  --dataset awf-time \
  --suffix 6w \
  --classes 200
```

Temporal collection suffixes:

- `3d` = 3 days
- `10d` = 10 days
- `2w` = 14 days
- `4w` = 28 days
- `6w` = 42 days

### Stateful Deployment and Attention Transfer

```bash
python scripts/simulate_awf_time_deployment.py \
  --data-root /path/to/data \
  --output-root results/awf-time-deployment \
  --n-probe 5 --m-refresh 2 \
  --detector-mode joint_dynamic \
  --q 0.04 --w-pp 2 --e 0.005 --boost-cap 0.02 --boost-decay 0.005 \
  --fixed-k-pp 2 \
  --atf-train-scope attention \
  --require-gpu
```

Use a new empty output directory. This entry accepts either the DLWF directory
itself or its parent. It trains Day 0 from scratch unless an explicitly supplied
aligned checkpoint with the shared-MLP architecture is supplied with `--resume-model`. The complete common site cohort is derived from data and saved in
`cohort.json`, rather than assumed to contain 200 sites.

The detector checks JS first. If JS does not exceed q, no probe prediction or
accuracy-label query is made; the temporary boost decays by 0.005 toward zero.
If JS exceeds q and probe accuracy is stable, the boost increases by 0.005,
capped at 0.02. If accuracy drops by more than two percentage points, the model
and references update while the boost is preserved. q = 0.04 + boost. Decay
occurs per closed-gate checkpoint, not per elapsed day. Configure this with
`--boost-cap` and `--boost-decay`. The remainder is evaluation-only.

Two refresh traces/site means two labeled fitting traces with no extra validation
labels. Best refresh loss selects the adaptation checkpoint. Five checkpoints
collect 25 monitoring traces/site. With G open JS gates and U updates, the main
policy uses 5G probe labels for accuracy checks and 2U refresh labels for fitting per site, excluding the initial reference. Forced
updates also evaluate that checkpoint's probes to refresh the reference. The temporal experiment samples labeled traces from the dataset at the corresponding
collection day. In active collection, website identities are known for all collected
traces; the JS gate reduces probe inference, while the label-use counts describe
the subset accessed by accuracy checks and fitting.

`--detector-mode` also accepts `accuracy_only`, `js_only`, `joint_fixed`.
`--atf-train-scope` also accepts `head`, `attention_head`, `full` as ablations.
Partial scopes disable Dropout and keep BN running moments fixed. Full tuning
uses training mode. All scopes share the refresh-only optimizer/stopping policy.
The output includes configuration, code/data/split/checkpoint hashes, every
checkpoint's JS/thresholds/accuracy, trainable layers and frozen-weight checks.
`monitor_and_adapt.py` is an alias for this same stateful entry.

For an independent interval (a distinct protocol), use:

```bash
python scripts/finetune_atf.py \
  --data-root /path/to/data --model models/v3_day0_200.keras \
  --suffix 6w --classes 200 --traces 2 \
  --output models/v3_independent_6w.keras
```

The independent entry requires matching class counts/labels. A common-cohort
checkpoint from the sequential entry must stay with that entry's saved mapping.

### Localized-Drift Stress Test

```bash
python scripts/localized_drift_stress_test.py \
  --data-root /path/to/data \
  --model results/awf-time-deployment/models/day0.keras \
  --split-root results/awf-time-deployment \
  --cohort-json results/awf-time-deployment/cohort.json \
  --output-root results/localized-drift \
  --seed 2024 --replicates 10 \
  --require-gpu
```

This entry measures temporal, nested 5/20/50% site replacement,
probe-bootstrap and predeclared tail-mask cases on the unchanged Day0 model.
It compares all four detectors, keeps the original shared remainder partitions,
and does not execute ATF. Reused traces are dependent observations.
The model must be the Day0 checkpoint recorded by the supplied split root.

### Sequence-Level JS Divergence

```bash
python scripts/compute_jsd.py \
  --data-root /path/to/data \
  --split-root results/awf-time-deployment \
  --suffix 6w
```

### Multi-Tab Concurrent Traffic

```bash
python scripts/train_multitab.py \
  --data-root /path/to/data \
  --tabs 2 \
  --classes 100 \
  --output models/msffat_2tab.hdf5
```

Repeat with `--tabs 2`, `3`, `4`, and `5`.

## Run a Batch of Commands

To print individual-task examples without executing them (a separate protocol from the paired paper matrix):

```bash
python scripts/run_experiments.py --data-root /path/to/data --dry-run
```

Add `--execute` to run the commands.

## Dataset Provenance and Downloads

The raw datasets are not redistributed in this repository. Obtain them from the original authors or the public mirrors below, and arrange them according to [docs/DATA.md](docs/DATA.md).

| Dataset used in this project | Source paper | Public repository / data source |
| --- | --- | --- |
| DF-CW and DF-OW | [Deep Fingerprinting: Undermining Website Fingerprinting Defenses with Deep Learning (CCS 2018)](https://doi.org/10.1145/3243734.3243768) | [deep-fingerprinting/df](https://github.com/deep-fingerprinting/df) |
| AWF-CW100/200/500/900 and AWF-Time (3d/10d/14d/28d/42d) | [Automated Website Fingerprinting through Deep Learning (NDSS 2018)](https://www.ndss-symposium.org/ndss-paper/automated-website-fingerprinting-through-deep-learning/) | [DistriNet/DLWF](https://github.com/DistriNet/DLWF) |
| WTF-PAD defended traffic | [WTF-PAD: Toward an Efficient Website Fingerprinting Defense for Tor](https://arxiv.org/abs/1512.00524); the pickle-form evaluation dataset was released with [Deep Fingerprinting](https://doi.org/10.1145/3243734.3243768) | [deep-fingerprinting/df](https://github.com/deep-fingerprinting/df) |
| Walkie-Talkie defended traffic | [Walkie-Talkie: An Efficient Defense Against Passive Website Fingerprinting Attacks (USENIX Security 2017)](https://www.usenix.org/conference/usenixsecurity17/technical-sessions/presentation/wang-tao); the pickle-form evaluation dataset was released with Deep Fingerprinting | [deep-fingerprinting/df](https://github.com/deep-fingerprinting/df) |
| Front defended traffic | [Zero-delay Lightweight Defenses against Website Fingerprinting (USENIX Security 2020)](https://www.usenix.org/conference/usenixsecurity20/presentation/gong) | [FIND-Lab/Website-Fingerprinting-Library](https://github.com/FIND-Lab/Website-Fingerprinting-Library) and its [Zenodo dataset release](https://zenodo.org/records/13732130) |
| TrafficSliver defended traffic | [TrafficSliver: Fighting Website Fingerprinting Attacks with Traffic Splitting (CCS 2020)](https://doi.org/10.1145/3372297.3423351) | [FIND-Lab/Website-Fingerprinting-Library](https://github.com/FIND-Lab/Website-Fingerprinting-Library) and its [Zenodo dataset release](https://zenodo.org/records/13732130) |
| ARES open-world 2/3/4/5-tab traffic | [Robust Multi-tab Website Fingerprinting Attacks in the Wild (IEEE S&P 2023)](https://doi.org/10.1109/SP46215.2023.10179464) | [Xinhao-Deng/Multitab-WF-Datasets](https://github.com/Xinhao-Deng/Multitab-WF-Datasets) |

The [Website-Fingerprinting-Library dataset release](https://zenodo.org/records/13732130) also provides a convenient unified mirror for several single-tab and defended datasets. Users should still cite the original paper associated with each dataset or defense.

## Development

Run the unit tests from the repository root:

```bash
python -m unittest discover -s tests -v
```
