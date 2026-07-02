# MSFFAT: Multi-Scale Spatiotemporal Feature Fusion with Attention Transfer

This repository contains a cleaned implementation of MSFFAT for website fingerprinting research.  It was reconstructed from the available project code and organized for anonymous review release.

The code supports MSFFAT-only experiments:

- data-scarce closed-world training;
- best-effort open-world training/evaluation;
- large monitored-set training on AWF-CW subsets;
- temporal concept-drift evaluation;
- calibration-based drift monitoring;
- attention-transfer fine-tuning;
- sequence-level JS divergence analysis;
- multi-tab concurrent traffic with a sigmoid multi-label head.

Baseline implementations are **not** bundled here.  Baseline results in the paper should be treated as either prior released results or separately reproduced experiments.

## Repository Layout

```text
.
├── msffat/                  # MSFFAT model, data loaders, metrics, maintenance helpers
├── scripts/                 # command-line entry points
├── configs/                 # example dataset/config files
├── docs/                    # data and experiment documentation
├── requirements.txt         # Python dependencies
├── pyproject.toml           # package metadata
└── README.md
```

## Installation

Python 3.9+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
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

## Quick Sanity Checks

Check Python syntax:

```bash
python -m py_compile msffat/*.py scripts/*.py
```

Check whether the expected dataset files are present:

```bash
python scripts/check_dataset.py --data-root /path/to/data --profile core
```

Print a model summary:

```bash
python scripts/model_summary.py --classes 95 --length 5000 --mode single
```

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

### Open World

```bash
python scripts/train_open_world.py \
  --data-root /path/to/data \
  --samples 10 \
  --classes 96 \
  --output models/msffat_df_ow_10shot.hdf5
```

The recovered legacy code did not contain a fully verified open-world decision protocol.  This script is a clean training/evaluation entry point; verify `--classes` and label encoding against your local dataset.

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

Legacy suffix mapping:

- `3d` = 3 days
- `10d` = 10 days
- `2w` = 14 days
- `4w` = 28 days
- `6w` = 42 days

### Attention Transfer

```bash
python scripts/finetune_atf.py \
  --data-root /path/to/data \
  --model models/msffat_awf_cw200.hdf5 \
  --suffix 6w \
  --traces 2 \
  --classes 200 \
  --output models/msffat_awf_cw200_6w_atf.hdf5
```

### Calibration-Based Monitor plus Optional ATF

```bash
python scripts/monitor_and_adapt.py \
  --data-root /path/to/data \
  --model models/msffat_awf_cw200.hdf5 \
  --suffix 6w \
  --a-base 0.9992 \
  --refresh-traces 2 \
  --output models/msffat_awf_cw200_6w_monitor_atf.hdf5
```

This implements the deployment-oriented trigger policy.  It should not be confused with the paper's adaptation-effect table, where ATF is applied at each interval to isolate the adaptation outcome.

### Sequence-Level JS Divergence

```bash
python scripts/compute_jsd.py \
  --data-root /path/to/data \
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

To print commands for a paper-style run without executing them:

```bash
python scripts/run_experiments.py --data-root /path/to/data --dry-run
```

Use `--execute` only after verifying paths, GPU availability, and storage.

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
