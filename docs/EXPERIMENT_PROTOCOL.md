# Paired Experiment Protocol

## Common data and seeds

Use seeds 2024, 2025, 2026, 2027 and 2028. Pair methods by website cohort, raw trace indices, class order and initial training budget. Preserve the dataset version and preprocessing configuration with each run. Report means and sample standard deviations across seeds.

Choose initial checkpoints using the initial validation split. Select adapted checkpoints using refresh-set cross-entropy, including the starting weights. Fix detector settings before held-out evaluation: q0=0.04, w=2pp, e=0.005, boost cap=0.02, closed-gate decay=0.005, oracle margin=2pp.

Temporal traces come from the public dataset's corresponding collection day. Sample probes and refresh traces per website using the supplied identities. In deployment this corresponds to visiting the monitored websites for fresh labeled traffic. Distinguish collected traces, probe labels used for accuracy checks, and refresh labels used for fitting.

## Fixed temporal partitions

Use 2,280 training and 120 validation traces per website on Day 0. Each temporal checkpoint reserves 5 probes, 10 refresh candidates and 85 remainder traces per site. Apply the same allocation to the initial held-out pool. The update budgets 1/2/5/10 select nested prefixes from the refresh pool; the remainder stays unchanged.

Monitor at Days 3/10/14/28/42. The two-refresh policy uses 5G probe labels in G accuracy checks and 2U refresh labels in U updates, per website. Initial reference probes are accounted for separately. All 25 monitoring traces per site have known website identities under active collection; G counts when those labels enter accuracy evaluation.

## Adaptation scope controls

Compare `head`, `attention`, `attention_head` and `full` from the same initial model, using the same refresh indices and fixed Days 14/28/42 update schedule. Report pre/post held-out accuracy, trainable parameters, update seconds, peak GPU memory, epochs and sample use.

Use `run_paired_adaptation.py --scope S --policy fixed` for these controls. Use `--policy static` for retention without adaptation. The stateful entry additionally supports `--update-policy fixed/never/detector`.

## External baselines

Prepare complete original-method models with their original preprocessing and paired initial train/validation indices. Supply each model through `--baseline-spec`, including its repository revision, class order, checkpoint and exact trainable parameter names. The configuration schema is illustrated in `configs/external-baseline.example.json`.

For non-Keras models, provide `backend_factory`. Its adapter implements `predict`, `fit_refresh` and `save`, follows the `refresh-ce-v1` fitting protocol, and records its framework/device. Compare published partial and full scopes under matched refresh budgets. Account for external pretraining and unlabeled adaptation data separately.

## Branch and attention controls

Train `--variant full/ltf_only/msf_only` and `--fusion proposed/independent/standard/concat` from scratch with matching seeds and fitting data. `independent` keeps the same concatenated average/maximum statistics as `proposed` and uses separate branch MLPs. `standard` applies the shared MLP separately to average and maximum statistics before summation and sigmoid.

## Detector comparisons

Compare `accuracy_only`, `js_only`, `joint_fixed` and `joint_dynamic` on common frozen probe/reference trajectories and, separately, in stateful deployment. `collect_detector_cases.py` builds temporal, localized replacement, bootstrap and tail-mask cases. `replay_detectors.py` evaluates stored measurements under each policy.

Report confusion counts, false-alarm/miss rates, checkpoint delay, accuracy checks, updates, final accuracy and computation. Bootstrap and recombination preserve their source-trace grouping in analysis. Replay ACC-check counts measure accesses to stored probe accuracy. Stateful results are computed within each seed before aggregation.

## Commands

Print the dependency-ordered matrix:

```bash
python scripts/run_paper_experiments.py --data-root /data --cw-root /data --include-scale --output-root results/paper-grid
```

Add `--execute --require-gpu` to run it. The matrix covers initial training, static retention, adaptation scopes, sample-budget curves, branch/fusion variants and the four detector policies.

External initial training uses the method's source repository. Pass all seed-specific configurations with `--external-specs baseline-runs.json`:

```json
{"runs":[{"seed":2024,"spec":"apwf-seed2024.json","scopes":["published_partial","full"]}]}
```

Provide one entry for every paired seed. Aggregate explicit result files with:

```bash
python scripts/aggregate_paired_results.py --results RUN_2024/paired_result.json RUN_2025/paired_result.json RUN_2026/paired_result.json RUN_2027/paired_result.json RUN_2028/paired_result.json --output summary.json
```

The summary includes paired differences, means and sample standard deviations under the shared data/split configuration.
