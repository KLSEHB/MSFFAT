# MSFFAT Method Specification

## Input and architecture

- Convert each Tor cell-direction trace to its sign and truncate or pad it to 5,000 positions. Symbols are `{-1, 0, +1}`, where 0 denotes padding.
- SFED feeds the LTF and MSF branches. Each branch returns a tensor of shape `(batch, 18, 512)`.
- For a branch tensor F with C channels, concatenate its temporal average and maximum: `z = concat(GAP(F), GMP(F))`.
- Both branches share the same biased two-layer MLP: `s = sigmoid(W2 ReLU(W1 z + b1) + b2)`. Its dimensions are `2C -> max(C//16,1) -> C`.
- Each branch supplies its own statistics and receives its own weights. Concatenate the reweighted LTF and MSF tensors in that order.
- The classifier is Flatten, Dense(512), BN, ReLU, Dropout(0.5), and the output layer. Single-label classification uses softmax; multi-label classification uses sigmoid and top-k selection with known tab count k.
- The 100-class model has 17,733,796 parameters. The shared attention MLP has 49,696 trainable parameters (0.28%).

## Feature extractors

SFED applies Conv32/64 (kernel 8, same padding), each followed by BN, ELU and MaxPool(4,2), then Dropout(0.1).

LTF uses four stages of widths 64/128/256/512. Each stage has two residual units with dilation pairs (1,2)/(4,8), causal kernel 5, and final MaxPool(4,2). The first convolution of stages 2 and 3 has stride 2. The first unit has a 1x1 projected shortcut with BN; the second uses an identity shortcut. Residual addition precedes the final ReLU.

MSF has six blocks with kernels 3/5/7/9. Their respective width tuples are (8,8,16,32), (16,16,32,64), (16,16,32,64), (32,32,64,128), (32,32,64,128), and (64,64,128,256). Apply ReLU within each branch, then concatenate in kernel order [9,7,5,3]. Blocks 1/3/5 add their shortcut before BN. Each pair ends with MaxPool(8,4) and Dropout(0.2).

## JS-gated maintenance

At a checkpoint, estimate the distribution of {-1,0,+1} independently at each trace position. Compute JS divergence with base-two logarithms at each position, then average over positions. Padding participates as a symbol.

The effective threshold is `q = q0 + b`, with defaults `q0=0.04`, `b=0`, accuracy-drop margin `w=2` percentage points, increment `e=0.005`, cap `B=0.02`, and decay `rho=0.005`.

| Checkpoint condition | Model action | Increment for the next checkpoint |
| --- | --- | --- |
| JS <= q | Skip probe prediction and accuracy evaluation | max(0, b-rho) |
| JS > q, accuracy drop <= w | Keep the model | min(B, b+e) |
| JS > q, accuracy drop > w | Adapt the shared MLP and update references | Keep b |

Use the current threshold for the current decision. Decay is per closed-gate checkpoint. Strict comparisons use tolerance 1e-12.

`detector_step` accepts a callable for deferred probe-accuracy evaluation. Unchecked accuracy and drop fields are `None`. The alternative policies are `accuracy_only`, `js_only`, and `joint_fixed`.

## Probe data and references

Temporal probes and refresh traces are sampled per website from the dataset's corresponding collection day, using its supplied website labels. This represents active collection by visiting the monitored websites. Website identities are available from collection; the JS gate controls when the algorithm predicts probes and uses those identities for an accuracy check.

Initial references use five held-out probes per site. After adaptation, save the current probe distribution and the updated model's probe accuracy as the new references. Preserve both references at checkpoints without adaptation. The increment b is preserved after adaptation.

Probe, refresh and remainder indices are disjoint. Probes are used for checking accuracy and setting references; refresh traces supply adaptation gradients; the remainder supplies held-out evaluation.

## Adaptation and optimization

The main adaptation scope is the shared MLP's Dense weights and biases. Freeze SFED, LTF, MSF and the full classifier. Partial adaptation uses `training=False`, keeping BN moments fixed and Dropout disabled. Record hashes of frozen weights before and after fitting.

Each update uses two refresh traces per website. Start a fresh Adam optimizer with learning rate 1e-3, batch size 8 and at most 50 epochs. Select the minimum refresh cross-entropy, including the starting checkpoint. After five epochs without improvement, multiply the learning rate by sqrt(0.1), down to 1e-6. Stop after ten epochs without improvement. No additional validation labels are used for adaptation.

Scope controls are `head`, `attention_head`, and `full`. The head scope includes Dense(512), BN affine parameters and the output layer. Full tuning uses training mode, including BN moments and Dropout. All scopes use the same refresh budget and optimization limits.

Initial training uses validation accuracy for single-label tasks and validation AP@k for multi-label tasks. The saved checkpoint is the one evaluated. Multi-label training uses sigmoid BCE; P@k and AP@k are averaged over traces, and ROC-AUC over classes.

## Temporal protocol and entry points

The checkpoints are Days 3/10/14/28/42. The paired protocol uses the common complete website cohort across all days. Day 0 has 2,280 training and 120 validation traces per website. Reserve 5 probes, a 10-trace refresh pool and 85 remainder traces per site; refresh budgets 1/2/5/10 use nested prefixes of that pool.

Five checkpoints collect 25 monitoring traces per site. With G open gates and U updates, the two-refresh policy uses 5G probe labels for accuracy checks and 2U refresh labels for fitting. These are algorithmic label-use counts. Under active collection, identities are known for all 25 monitoring traces, and closed gates save probe inference. Account for initial fitting/reference data and unused reserve traces separately.

The evaluation oracle marks a checkpoint degraded when held-out accuracy drops strictly more than 2pp from its separate reference. Detector decisions precede remainder evaluation. Shared seeds, indices, class order and budgets define paired comparisons.

- Stateful deployment: `simulate_awf_time_deployment.py` (`monitor_and_adapt.py` is an alias).
- Independent-interval adaptation: `finetune_atf.py`.
- Paired scope/baseline controls: `run_paired_adaptation.py`.
- Complete command matrix: `run_paper_experiments.py`.
- Experiment entry-point mapping: [EXPERIMENTS.md](EXPERIMENTS.md).
- Paired protocol: [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md).
