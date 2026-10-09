#!/usr/bin/env python3
"""Run the experiment blocks and write one JSON file per run.

Blocks
    1  ablation ladder, task-incremental:  LIF, ALIF, ALIF+SI, ALIF+SI+DM
    2  random-skip control at the share of ignored samples measured in block 1
    3  sweep of the target share of ignored samples (10, 25, 50, 75 %), one seed
    4  ablation ladder, class-incremental

Presets
    taskil   blocks 1 and 2
    classil  block 4
    sweep    block 3
    full     all four blocks
    quick    blocks 1 and 2 with one seed and three epochs (for testing)

Every finished run is stored as `<out>/results/<block>__<config>__s<seed>.json`.
A run whose file exists is not repeated, so an interrupted session can be
continued by starting the script again with the same arguments.

Examples
    python scripts/run_experiments.py --preset taskil --seeds 0 1 2
    python scripts/run_experiments.py --preset full --seeds 0
    python scripts/run_experiments.py --preset quick --synthetic --device cpu
"""

import argparse
import json
import os
import pathlib
import sys
import time
from dataclasses import replace

import numpy as np
import torch

# Make the package importable when the script is run from a clone that has not
# been installed with pip.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dgcl_snn import (  # noqa: E402
    DecisionConfig,
    ExpConfig,
    build_split_mnist,
    mean_sd,
    run_experiment,
    select_device,
)

# The ablation ladder: every configuration adds one mechanism to the previous one.
LADDER = [
    ("LIF",        dict(adaptive=False, use_si=False, use_decision=False)),
    ("ALIF",       dict(adaptive=True,  use_si=False, use_decision=False)),
    ("ALIF+SI",    dict(adaptive=True,  use_si=True,  use_decision=False)),
    ("ALIF+SI+DM", dict(adaptive=True,  use_si=True,  use_decision=True)),
]
LADDER_NAMES = tuple(name for name, _ in LADDER)

# Configurations of the central comparison. They can be given more seeds than
# the rest with --key-seeds.
KEY_CONFIGS = {"ALIF+SI", "ALIF+SI+DM", "ALIF+SI+Random"}

SWEEP_RATES = (0.10, 0.25, 0.50, 0.75)

PRESETS = {
    "full": (1, 2, 4, 3),
    "taskil": (1, 2),
    "classil": (4,),
    "sweep": (3,),
    "quick": (1, 2),
}


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", choices=sorted(PRESETS), default="taskil")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--key-seeds", type=int, nargs="+", default=None,
                   help="seeds for ALIF+SI, ALIF+SI+DM and the random control "
                        "(default: the same as --seeds)")
    p.add_argument("--epochs", type=int, default=10, help="epochs per task")
    p.add_argument("--out", default="output", help="output directory")
    p.add_argument("--data-dir", default="./data", help="where MNIST is stored")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--synthetic", action="store_true",
                   help="use a small random dataset instead of MNIST (smoke test only)")
    p.add_argument("--n-synth", type=int, default=512,
                   help="samples per task of the synthetic dataset")
    return p.parse_args()


def save_run(path, result):
    def default(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=default)


class Runner:
    """Runs configurations over seeds, with a file per run as cache."""

    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.results_dir = os.path.join(args.out, "results")
        os.makedirs(self.results_dir, exist_ok=True)
        self._tasks = {}

    def tasks(self, multi_head, batch_size):
        """The task sequence is built once per setting and shared by all runs."""
        key = (multi_head, batch_size)
        if key not in self._tasks:
            print(f"  building dataset (multi_head={multi_head}) ...")
            self._tasks[key] = build_split_mnist(
                data_dir=self.args.data_dir, batch_size=batch_size,
                multi_head=multi_head, synthetic=self.args.synthetic,
                n_synth=self.args.n_synth, device=self.device)
        return self._tasks[key]

    def run_block(self, block, cfgs, seeds, key_seeds=None):
        """Run every configuration of a block. Returns {name: [result, ...]}."""
        out = {}
        for cfg in cfgs:
            use = key_seeds if (key_seeds is not None and cfg.name in KEY_CONFIGS) else seeds
            runs = []
            for seed in use:
                tag = f"{block}__{cfg.name.replace('/', '-')}__s{seed}"
                path = os.path.join(self.results_dir, f"{tag}.json")
                if os.path.exists(path):
                    with open(path) as f:
                        runs.append(json.load(f))
                    print(f"  [cached] {tag}")
                    continue
                print(f"\n  >>> {cfg.name}  seed={seed}")
                t0 = time.time()
                # The diagnostic pass is only needed once per ladder
                # configuration, so it runs for the first seed only.
                run_cfg = replace(cfg, collect_diagnostics=(
                    seed == use[0] and cfg.name in LADDER_NAMES))
                r = run_experiment(run_cfg, seed=seed,
                                   tasks=self.tasks(cfg.multi_head, cfg.batch_size),
                                   device=self.device)
                r["runtime_s"] = time.time() - t0
                save_run(path, r)
                runs.append(r)
                c = r["compute"]
                print(f"      acc={r['avg_acc']:.4f}  forget={r['avg_forgetting']:.4f}  "
                      f"bwt={r['bwt']:+.4f}  ({r['runtime_s'] / 60:.1f} min)")
                print(f"      synops dense={c['synops_total_G']:.1f}G "
                      f"event={c['synops_total_event_G']:.1f}G  "
                      f"learned={c['backward_sample_fraction']:.1%}")
            out[cfg.name] = runs
        return out


def print_table(block, title):
    line = "=" * 92
    print(f"\n{line}\n  {title}\n{line}")
    print(f"{'Configuration':<24}{'Seeds':>6}{'Avg accuracy':>22}{'Avg forgetting':>24}"
          f"{'Learned':>11}")
    print("-" * 92)
    for name, runs in block.items():
        a_m, a_s = mean_sd([r["avg_acc"] for r in runs])
        f_m, f_s = mean_sd([r["avg_forgetting"] for r in runs])
        learned = np.mean([r["compute"]["backward_sample_fraction"] for r in runs])
        print(f"{name:<24}{len(runs):>6}{a_m:>13.4f} +- {a_s:<6.4f}"
              f"{f_m:>15.4f} +- {f_s:<6.4f}{learned:>10.1%}")
    print(line)


def banner(text):
    print("\n" + "#" * 92 + f"\n#  {text}\n" + "#" * 92)


def main():
    args = parse_args()
    seeds, epochs = list(args.seeds), args.epochs
    if args.preset == "quick":
        seeds, epochs = seeds[:1], 3
    key_seeds = list(args.key_seeds) if args.key_seeds else seeds
    # The sweep traces a trend over the share of ignored samples and uses one seed.
    sweep_seeds = seeds[:1]

    device = select_device() if args.device == "auto" else torch.device(args.device)
    print(f"preset={args.preset}  seeds={seeds}  epochs={epochs}  device={device}  "
          f"out={args.out}")

    base = dict(t_steps=20, epochs=epochs, batch_size=64, lr=1e-3,
                lambda_si=0.5, beta=0.6, h1=256, h2=256, verbose=True)
    runner = Runner(args, device)
    blocks = PRESETS[args.preset]
    results = {}
    t_start = time.time()

    if 1 in blocks:
        banner("BLOCK 1: ablation ladder (task-incremental)")
        cfgs = [ExpConfig(name=n, multi_head=True, **kw, **base) for n, kw in LADDER]
        b1 = runner.run_block("block1_taskIL", cfgs, seeds, key_seeds)
        results.update(b1)
        print_table(b1, "BLOCK 1: task-incremental ablation ladder")

    if 2 in blocks:
        banner("BLOCK 2: random-skip control")
        rate = float(np.mean([r["action_global"].get("IGNORE", 0.0)
                              for r in results["ALIF+SI+DM"]]))
        print(f"\n  share of samples ignored by the decision module: {rate:.1%}")
        print("  the random control ignores the same share\n")
        ctrl = [ExpConfig(name="ALIF+SI+Random", multi_head=True, adaptive=True,
                          use_si=True, use_decision=False, random_skip_rate=rate, **base)]
        b2 = runner.run_block("block2_control", ctrl, seeds, key_seeds)
        results.update(b2)
        print_table({k: results[k] for k in ("ALIF+SI", "ALIF+SI+DM", "ALIF+SI+Random")},
                    "BLOCK 2: decision-guided and random skipping")

    if 4 in blocks:
        banner("BLOCK 4: ablation ladder (class-incremental)")
        cfgs = [ExpConfig(name=f"{n}/classIL", multi_head=False, **kw, **base)
                for n, kw in LADDER]
        b4 = runner.run_block("block4_classIL", cfgs, seeds, key_seeds)
        results.update(b4)
        print_table(b4, "BLOCK 4: class-incremental ablation ladder")

    if 3 in blocks:
        banner("BLOCK 3: sweep of the share of ignored samples (one seed)")
        cfgs = []
        for r in SWEEP_RATES:
            cfgs.append(ExpConfig(name=f"DM@ignore{int(r * 100)}", multi_head=True,
                                  adaptive=True, use_si=True, use_decision=True,
                                  decision=DecisionConfig(p_attention_low=r), **base))
            cfgs.append(ExpConfig(name=f"Rand@ignore{int(r * 100)}", multi_head=True,
                                  adaptive=True, use_si=True, use_decision=False,
                                  random_skip_rate=r, **base))
        b3 = runner.run_block("block3_sweep", cfgs, sweep_seeds)
        results.update(b3)
        print_table(b3, "BLOCK 3: accuracy against the share of ignored samples")

    print(f"\nTotal wall time: {(time.time() - t_start) / 3600:.2f} h")
    print(f"Results in {runner.results_dir}/")


if __name__ == "__main__":
    main()
