# Selective Plasticity for Continual Learning in Spiking Neural Networks

Code for the bachelor's thesis of the same title by Madhusudhan Dudapu
(Institute for Pervasive Computing, Johannes Kepler University Linz, 2026).

A spiking network learns five tasks one after the other. Synaptic Intelligence
protects what was learned before. On top of it, a rule-based **decision
module** assigns one of four plasticity actions to every training sample:
*update*, *consolidate*, *protect* or *ignore*. An ignored sample gets loss
weight zero. The decision uses only quantities that the training forward pass
computes anyway. The experiments measure how many updates can be withheld in
this way, what that does to retention, and what it saves.

## Method in brief

| Part | Implementation |
|---|---|
| Input | Poisson rate coding, T = 20 timesteps |
| Network | 784-256-256 spiking MLP, LIF or adaptive LIF (ALIF) neurons, SuperSpike surrogate gradient, 269,322 parameters |
| Retention | Synaptic Intelligence (Zenke et al., 2017) |
| Signals | predictive entropy `H_i` and attention `a_i` (entropy times normalised firing rate), both per sample |
| Thresholds | quantiles of a sliding window (4096 values) of each signal |
| Cost | synaptic operations under a dense and an event-driven cost model |

The rule is evaluated per sample in this order:

| Action | Condition | Loss weight |
|---|---|---|
| ignore | attention below its 25 % quantile | 0 |
| update | entropy above its 75 % quantile | 1.0 |
| protect | entropy below its 40 % quantile, and the importance gate is open | 0.1 |
| consolidate | otherwise | 0.3 |

The loss of a mini-batch is the weighted mean of the per-sample losses plus the
Synaptic Intelligence penalty. The penalty is scaled up with the share of
protected samples in the batch (factor 3 if all samples are protected). If every
sample of a batch is ignored, the backward pass is skipped.

The **random-skip control** ignores the same share of samples, chosen at
random. It uses only *update* and *ignore*, so it is matched to the decision
module in the share of ignored samples and in nothing else.

## Results in brief

Split-MNIST, task-incremental, ten epochs per task, mean ± sample standard
deviation over three seeds. Cost in giga synaptic operations.

| Configuration | Accuracy | Forgetting | Samples learned | Dense cost | Event-driven cost |
|---|---|---|---|---|---|
| LIF | 0.7178 ± 0.0604 | 0.3479 ± 0.0750 | 100 % | | |
| ALIF | 0.7196 ± 0.0692 | 0.3461 ± 0.0861 | 100 % | | |
| ALIF + SI | 0.9913 ± 0.0003 | 0.0022 ± 0.0016 | 100 % | 1756.5 ± 23.8 | 1756.5 ± 23.8 |
| ALIF + SI + decision module | 0.9907 ± 0.0026 | 0.0015 ± 0.0010 | 70.7 % | 1656.1 ± 60.3 | 1324.9 ± 50.1 |
| ALIF + SI + random skipping | 0.9863 ± 0.0040 | 0.0061 ± 0.0060 | 70.7 % | 1742.0 ± 40.5 | 1401.3 ± 33.1 |

What these numbers show, and what they do not:

- Synaptic Intelligence removes almost all forgetting on this benchmark.
- With the decision module, 29.3 % of the samples are ignored and retention
  does not change measurably. The benchmark is close to saturation, so it cannot
  show an improvement in retention.
- The decision module forgot less than random skipping in all three seeds. The
  difference is not statistically significant (Welch's |t| = 1.31).
- Skipping saves about 20 % of the synaptic operations under the event-driven
  cost model. On dense hardware such as a GPU it saves nothing, because a
  masked sample is still computed. The differences between the dense costs in
  the table come from different firing activity, not from skipping.
- The backward cost is a model (twice the forward cost), not a measurement.

The thesis reports further experiments with one seed: a sweep of the share of
ignored samples up to 77 %, and the class-incremental setting, in which every
configuration fails.

## Repository layout

```
dgcl_snn/               the library
    neurons.py          LIF and ALIF neurons, surrogate gradient
    network.py          spiking MLP with per-sample spike statistics
    encoding.py         Poisson rate coding
    si.py               Synaptic Intelligence
    decision.py         decision module and random-skip control
    quantile.py         sliding-window quantiles for the thresholds
    accounting.py       synaptic operations under two cost models
    isi.py              firing-regularity importance (analysis only)
    roles.py            neuron roles (analysis only)
    data.py             Split-MNIST
    training.py         training of one task, evaluation
    experiment.py       configuration and one complete run
    diagnostics.py      data for the diagnostic figures
    metrics.py          accuracy, forgetting, backward transfer, statistics
    utils.py            seeding, device selection
scripts/
    run_experiments.py  runs the experiment blocks, one JSON file per run
    make_tables.py      LaTeX tables and a plain-text summary
    make_figures.py     figures
tests/                  unit tests and end-to-end runs on synthetic data
results/                result files of the thesis runs
```

## Installation

Python 3.9 or newer.

```bash
pip install -e .            # installs torch, torchvision, numpy, matplotlib
pip install -e ".[dev]"     # additionally pytest and ruff
```

The thesis runs used Python 3.12, PyTorch 2.3.1 with CUDA 12.1 and torchvision
0.18.1 on Kaggle:

```bash
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
```

## Quick check

The tests need no dataset and no GPU and take about twenty seconds:

```bash
pytest
```

A complete pass through runner, tables and figures on a small synthetic dataset
takes one to two minutes on a CPU. The numbers of such a run have no meaning; it
only shows that the pipeline works.

```bash
python scripts/run_experiments.py --preset quick --synthetic --device cpu --out output/smoke
python scripts/make_tables.py  --results output/smoke/results --out output/smoke
python scripts/make_figures.py --results output/smoke/results --out output/smoke
```

## Reproducing the experiments

MNIST is downloaded on first use. One run of one configuration over the five
tasks takes five to six minutes on a Tesla P100.

```bash
# Blocks 1 and 2 with three seeds: ablation ladder and random-skip control
python scripts/run_experiments.py --preset taskil --seeds 0 1 2 --out output/three_seeds

# All four blocks with one seed: adds the sweep and the class-incremental ladder
python scripts/run_experiments.py --preset full --seeds 0 --out output/single_seed

# Tables and figures
python scripts/make_tables.py  --results output/three_seeds/results --out output/three_seeds
python scripts/make_figures.py --results output/single_seed/results --out output/single_seed
```

Every finished run is written to `<out>/results/` immediately, and a run whose
file exists is not repeated. An interrupted session is continued by starting
the same command again.

`--device auto` (the default) tests the GPU with a small computation and falls
back to the CPU if the installed PyTorch build does not support the card. This
can happen with recent PyTorch builds on older cards such as the Tesla P100.

Runs are deterministic for a fixed seed on the same hardware and library
versions. On other GPUs or PyTorch versions individual runs can differ, so
reruns should be compared with the thesis at the level of means and standard
deviations.

### Where the numbers of the thesis come from

| Thesis | Source | Output |
|---|---|---|
| Table 7.1, ablation ladder | Block 1, three seeds | `tables/ladder_task_il.tex` |
| Table 7.2, control and cost | Blocks 1 and 2, three seeds | `tables/control.tex` |
| Table 7.3, sweep | Block 3, one seed | `tables/sweep.tex` |
| Table 7.4, action shares | Block 1, one seed | `tables/actions.tex` |
| Table 7.5, neuron roles | Block 1, one seed | `tables/roles.tex` |
| Section 7.4, class-incremental | Block 4, one seed | `tables/ladder_class_il.tex` |
| Figure 7.1, accuracy matrices | Block 1, one seed | `figures/fig2_accuracy_matrices` |
| Figure 7.2, predictive entropy | diagnostics, one seed | `figures/fig15_uncertainty_diagnostics` |
| Figure 7.3, spike activity | diagnostics, one seed | `figures/fig13_spike_activity` |
| Figure 7.4, firing regularity | diagnostics, one seed | `figures/fig18_isi_cv_map` |
| Cohen's d, Welch's t | Blocks 1 and 2, three seeds | `RESULTS_SUMMARY.txt` |

The results of the thesis come from two sessions. The three-seed session
covered Blocks 1 and 2. The single-seed session covered all four blocks and
was run first; the data loading was changed between the sessions, which
changes the order of the mini-batches. The two sessions are therefore not
numerically identical for seed 0, and a rerun of the single-seed blocks with
this code does not reproduce their numbers digit for digit.

## Result files

`run_experiments.py` writes one file per run,
`<block>__<configuration>__s<seed>.json`, with these entries:

| Key | Content |
|---|---|
| `acc_matrix` | accuracy on task j after training through task i |
| `avg_acc`, `avg_forgetting`, `bwt` | metrics computed from the matrix |
| `action_by_task`, `action_global` | share of each action |
| `compute` | synaptic operations (forward, dense backward, event-driven backward), number of skipped batches, wall-clock time |
| `omega_trace` | mean Synaptic Intelligence importance after each task |
| `isi_trace`, `memory_trace` | firing-regularity importance and neuron roles after each task |
| `epoch_logs` | loss, accuracy, entropy and active fraction per epoch |
| `diagnostics` | per-sample signals and spike rasters of the trained model (first seed of the ladder configurations) |

## Hyperparameters

| Parameter | Value | Parameter | Value |
|---|---|---|---|
| Timesteps | 20 | SI strength λ | 0.5 |
| Membrane time constant | 20 ms | SI damping ξ | 0.001 |
| Adaptation time constant | 80 ms | Protect boost κ | 3.0 |
| Baseline threshold | 0.3 | Loss weights | 1.0, 0.3, 0.1, 0 |
| Adaptation strength β | 0.6 (LIF: 0) | Rate bounds | 0.05, 0.95 |
| Surrogate steepness | 100 | Ignore quantile | 0.25 |
| Hidden layers | 256, 256 | Entropy quantiles | 0.40, 0.75 |
| Optimiser | Adam, learning rate 0.001, new for every task | Gate quantile | 0.70 |
| Batch size | 64 | Quantile window, warm-up | 4096, 512 |
| Epochs per task | 10 | Minimum number of intervals | 2 |
| Gradient clipping (norm) | 1.0 | Importance weights (SI, firing regularity) | 0.7, 0.3 |

All of them are fields of `ExpConfig` and `DecisionConfig` with these values
as defaults.

## Relation to the notebook used for the experiments

The experiments were run on Kaggle from one notebook that contained the
library, the runner, the table code and the figure code as four cells. This
repository is that code as a package with scripts. The following differs:

- The library is split into modules and its comments and docstrings are
  rewritten. The code of 27 of its 32 functions and classes is unchanged. The
  other five differ as follows: the device is passed as an argument instead of
  being read from a notebook variable (`get_device`, `build_split_mnist`,
  `run_experiment`), one error message was reworded (`_load_mnist_tensors`),
  `aggregate` uses the sample standard deviation, and an unused import and an
  unused variable were removed.
- Standard deviations over seeds are sample standard deviations (n - 1), as in
  the thesis. The notebook printed population standard deviations.
- The tables have additional columns (samples learned, cost, share of ignored
  samples) and descriptive file names.
- Figure titles are neutral. The scatter plot of attention against entropy
  (`fig16`) paired the two quantities in different sample orders in the
  notebook; it now pairs them per sample. The class-incremental ladder
  (`fig1b`) is now drawn; the notebook looked for it under the wrong
  configuration names.
- `scripts/make_tables.py` also writes Cohen's d and Welch's t.

The package and the notebook code were run side by side on synthetic data: the
complete suite of four blocks (26 runs) and eight separate configurations with
two seeds each. All result files were identical. The four figures that appear
in the thesis are identical pixel for pixel when drawn from the same result
files.

## Limitations

- One benchmark, close to saturation in the task-incremental setting; three
  seeds for the main comparison, one seed for the sweep, the class-incremental
  setting and all diagnostics.
- The random control matches the share of ignored samples only. It differs
  from the decision module also in the loss weights of the remaining samples
  and in the strength of the penalty.
- The attention signal carried almost no information beyond entropy in these
  experiments, because the firing rate hardly varied between samples.
- The importance gate compares the mean importance with a quantile of its own
  recent values. It is therefore open only for the first part of every task
  after the first. This dependence on time was not intended.
- Firing-regularity importance and neuron roles are computed after each task
  for analysis. They do not influence training.
- The event-driven cost is a model estimate. Nothing was measured on
  neuromorphic hardware.

## Use of AI assistance

Large parts of this code were written with an AI assistant under the author's
direction, and were run, tested and checked by the author. The thesis contains
a declaration of AI usage.

## Citation

```bibtex
@thesis{dudapu2026selective,
  author      = {Dudapu Madhusudhan},
  title       = {Selective Plasticity for Continual Learning in Spiking Neural Networks},
  type        = {Bachelor's thesis},
  institution = {Johannes Kepler University Linz},
  year        = {2026}
}
```

## License

MIT, see [LICENSE](LICENSE).
