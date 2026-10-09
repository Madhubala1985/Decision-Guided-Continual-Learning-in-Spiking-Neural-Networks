# Result files of the thesis runs

One JSON file per run, as written by `scripts/run_experiments.py`
(`<block>__<configuration>__s<seed>.json`; the format is described in the main
README).

| Directory | Session | Content |
|---|---|---|
| `three_seeds/` | Blocks 1 and 2, seeds 0, 1 and 2 | ablation ladder and random-skip control (15 runs) |
| `single_seed/` | Blocks 1 to 4, seed 0 | ladder, control, sweep and class-incremental ladder (17 runs) |

Tables and figures are produced from these files with

```bash
python scripts/make_tables.py  --results results/three_seeds --out output/three_seeds
python scripts/make_tables.py  --results results/single_seed --out output/single_seed
python scripts/make_figures.py --results results/single_seed --out output/single_seed
```
